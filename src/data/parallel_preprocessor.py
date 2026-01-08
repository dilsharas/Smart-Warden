#!/usr/bin/env python3
"""
Parallel Preprocessing Pipeline for SmartBugs Wild Dataset.
Implements multi-threaded preprocessing with memory optimization.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterator, Callable
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing import Manager, Queue
import time
import gc
import psutil
from dataclasses import dataclass
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import pickle

# Add src to path for imports
import sys
sys.path.insert(0, 'src')

from data.efficient_data_loader import ContractData, SmartBugsDataLoader
from features.feature_extractor import SolidityFeatureExtractor

logger = logging.getLogger(__name__)

@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing pipeline."""
    normalize_features: bool = True
    feature_selection: bool = True
    max_features: int = 50
    remove_duplicates: bool = True
    handle_missing: str = 'mean'  # 'mean', 'median', 'drop'
    encoding_method: str = 'label'  # 'label', 'onehot'
    validation_split: float = 0.2
    test_split: float = 0.1
    random_state: int = 42

@dataclass
class PreprocessingStats:
    """Statistics for preprocessing performance."""
    total_samples: int = 0
    processed_samples: int = 0
    removed_duplicates: int = 0
    removed_invalid: int = 0
    feature_count_original: int = 0
    feature_count_final: int = 0
    processing_time: float = 0.0
    memory_usage_mb: float = 0.0

class ParallelPreprocessor:
    """
    Parallel preprocessing pipeline with memory optimization and smart caching.
    """
    
    def __init__(self, 
                 config: PreprocessingConfig = None,
                 n_jobs: int = -1,
                 chunk_size: int = 1000,
                 cache_dir: str = "data/cache/preprocessing"):
        """
        Initialize the parallel preprocessor.
        
        Args:
            config: Preprocessing configuration
            n_jobs: Number of parallel jobs (-1 for all cores)
            chunk_size: Size of processing chunks
            cache_dir: Directory for caching preprocessed data
        """
        self.config = config or PreprocessingConfig()
        self.n_jobs = n_jobs if n_jobs != -1 else psutil.cpu_count()
        self.chunk_size = chunk_size
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize feature extractor
        self.feature_extractor = SolidityFeatureExtractor()
        
        # Preprocessing components
        self.scaler = None
        self.feature_selector = None
        self.label_encoder = None
        
        # Statistics
        self.stats = PreprocessingStats()
        
        logger.info(f"Initialized ParallelPreprocessor with {self.n_jobs} workers")
    
    def preprocess_dataset(self, 
                          contracts: List[ContractData],
                          target_column: str = 'is_vulnerable') -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Preprocess complete dataset with parallel feature extraction.
        
        Args:
            contracts: List of contract data
            target_column: Target variable column name
            
        Returns:
            Tuple of (features, labels, metadata)
        """
        logger.info(f"🔄 Preprocessing {len(contracts)} contracts with {self.n_jobs} workers...")
        start_time = time.time()
        
        # Extract features in parallel
        features_df = self._extract_features_parallel(contracts)
        
        # Create labels
        labels = self._create_labels(contracts, target_column)
        
        # Data cleaning and validation
        features_df, labels = self._clean_data(features_df, labels)
        
        # Feature engineering
        features_df = self._engineer_features(features_df)
        
        # Feature selection
        if self.config.feature_selection:
            features_df = self._select_features(features_df, labels)
        
        # Normalization
        if self.config.normalize_features:
            features_df = self._normalize_features(features_df)
        
        # Convert to numpy arrays
        X = features_df.values.astype(np.float32)
        y = labels.astype(np.int32)
        
        # Update statistics
        self.stats.total_samples = len(contracts)
        self.stats.processed_samples = len(X)
        self.stats.feature_count_final = X.shape[1]
        self.stats.processing_time = time.time() - start_time
        self.stats.memory_usage_mb = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Create metadata
        metadata = {
            'feature_names': list(features_df.columns),
            'preprocessing_config': self.config,
            'statistics': self.stats,
            'data_shape': X.shape,
            'class_distribution': np.bincount(y)
        }
        
        logger.info(f"✅ Preprocessing complete: {X.shape} in {self.stats.processing_time:.2f}s")
        
        return X, y, metadata
    
    def _extract_features_parallel(self, contracts: List[ContractData]) -> pd.DataFrame:
        """Extract features from contracts using parallel processing."""
        logger.info("🔧 Extracting features in parallel...")
        
        # Split contracts into chunks
        chunks = [contracts[i:i + self.chunk_size] 
                 for i in range(0, len(contracts), self.chunk_size)]
        
        all_features = []
        
        # Process chunks in parallel
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            # Submit all chunks
            future_to_chunk = {
                executor.submit(self._process_feature_chunk, chunk): chunk 
                for chunk in chunks
            }
            
            # Collect results
            for future in as_completed(future_to_chunk):
                try:
                    chunk_features = future.result()
                    all_features.extend(chunk_features)
                except Exception as e:
                    logger.error(f"Error processing feature chunk: {e}")
        
        # Create DataFrame
        if all_features:
            features_df = pd.DataFrame(all_features)
            logger.info(f"✅ Extracted {len(features_df.columns)} features from {len(features_df)} contracts")
            return features_df
        else:
            raise ValueError("No features extracted from contracts")
    
    def _process_feature_chunk(self, contracts: List[ContractData]) -> List[Dict]:
        """Process a chunk of contracts for feature extraction."""
        chunk_features = []
        
        for contract in contracts:
            try:
                # Extract features
                features = self.feature_extractor.extract_features(contract.code)
                
                # Add metadata features
                features.update({
                    'contract_size': contract.size,
                    'filename_length': len(contract.filename),
                    'has_vulnerabilities': len(contract.vulnerabilities) > 0,
                    'vulnerability_count': len(contract.vulnerabilities)
                })
                
                # Add contract identifier
                features['contract_hash'] = contract.hash
                
                chunk_features.append(features)
                
            except Exception as e:
                logger.warning(f"Error extracting features from {contract.filename}: {e}")
                continue
        
        return chunk_features
    
    def _create_labels(self, contracts: List[ContractData], target_column: str) -> np.ndarray:
        """Create label array from contracts."""
        labels = []
        
        for contract in contracts:
            if target_column == 'is_vulnerable':
                labels.append(1 if contract.is_vulnerable else 0)
            elif target_column == 'primary_vulnerability':
                # Multi-class classification
                if contract.vulnerabilities:
                    labels.append(contract.vulnerabilities[0])
                else:
                    labels.append('safe')
            else:
                raise ValueError(f"Unknown target column: {target_column}")
        
        # Encode labels if necessary
        if target_column == 'primary_vulnerability':
            if self.label_encoder is None:
                self.label_encoder = LabelEncoder()
                encoded_labels = self.label_encoder.fit_transform(labels)
            else:
                encoded_labels = self.label_encoder.transform(labels)
            return encoded_labels
        
        return np.array(labels)
    
    def _clean_data(self, features_df: pd.DataFrame, labels: np.ndarray) -> Tuple[pd.DataFrame, np.ndarray]:
        """Clean and validate data."""
        logger.info("🧹 Cleaning data...")
        
        original_size = len(features_df)
        
        # Remove rows with all NaN values
        features_df = features_df.dropna(how='all')
        
        # Handle missing values
        if self.config.handle_missing == 'mean':
            features_df = features_df.fillna(features_df.mean())
        elif self.config.handle_missing == 'median':
            features_df = features_df.fillna(features_df.median())
        elif self.config.handle_missing == 'drop':
            features_df = features_df.dropna()
        
        # Remove infinite values
        features_df = features_df.replace([np.inf, -np.inf], np.nan)
        features_df = features_df.fillna(0)
        
        # Remove duplicates if configured
        if self.config.remove_duplicates:
            # Keep track of which rows to keep
            keep_mask = ~features_df.duplicated()
            features_df = features_df[keep_mask]
            labels = labels[keep_mask.values]
            
            duplicates_removed = original_size - len(features_df)
            self.stats.removed_duplicates = duplicates_removed
            
            if duplicates_removed > 0:
                logger.info(f"Removed {duplicates_removed} duplicate samples")
        
        # Remove constant features (zero variance)
        constant_features = features_df.columns[features_df.var() == 0]
        if len(constant_features) > 0:
            features_df = features_df.drop(columns=constant_features)
            logger.info(f"Removed {len(constant_features)} constant features")
        
        self.stats.removed_invalid = original_size - len(features_df)
        
        return features_df, labels
    
    def _engineer_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Engineer additional features."""
        logger.info("⚙️ Engineering features...")
        
        # Create interaction features for key security metrics
        if 'external_calls' in features_df.columns and 'state_changes' in features_df.columns:
            features_df['external_calls_state_ratio'] = (
                features_df['external_calls'] / (features_df['state_changes'] + 1)
            )
        
        if 'function_count' in features_df.columns and 'lines_of_code' in features_df.columns:
            features_df['functions_per_line'] = (
                features_df['function_count'] / (features_df['lines_of_code'] + 1)
            )
        
        # Create complexity features
        complexity_features = [col for col in features_df.columns 
                             if any(keyword in col.lower() for keyword in 
                                   ['complexity', 'depth', 'nesting'])]
        
        if len(complexity_features) > 1:
            features_df['total_complexity'] = features_df[complexity_features].sum(axis=1)
            features_df['avg_complexity'] = features_df[complexity_features].mean(axis=1)
        
        # Create security pattern features
        security_features = [col for col in features_df.columns 
                           if any(keyword in col.lower() for keyword in 
                                 ['dangerous', 'external', 'call', 'transfer'])]
        
        if len(security_features) > 1:
            features_df['security_risk_score'] = features_df[security_features].sum(axis=1)
        
        logger.info(f"Engineered features: {len(features_df.columns)} total features")
        
        return features_df
    
    def _select_features(self, features_df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
        """Select most important features."""
        logger.info(f"🎯 Selecting top {self.config.max_features} features...")
        
        # Skip non-numeric columns for feature selection
        numeric_columns = features_df.select_dtypes(include=[np.number]).columns
        non_numeric_columns = features_df.select_dtypes(exclude=[np.number]).columns
        
        if len(numeric_columns) == 0:
            logger.warning("No numeric features found for selection")
            return features_df
        
        # Apply feature selection to numeric features only
        X_numeric = features_df[numeric_columns].values
        
        # Use mutual information for feature selection
        if self.feature_selector is None:
            self.feature_selector = SelectKBest(
                score_func=mutual_info_classif,
                k=min(self.config.max_features, len(numeric_columns))
            )
            X_selected = self.feature_selector.fit_transform(X_numeric, labels)
        else:
            X_selected = self.feature_selector.transform(X_numeric)
        
        # Get selected feature names
        selected_mask = self.feature_selector.get_support()
        selected_features = numeric_columns[selected_mask]
        
        # Create new DataFrame with selected features
        selected_df = pd.DataFrame(X_selected, columns=selected_features, index=features_df.index)
        
        # Add back non-numeric columns if any
        if len(non_numeric_columns) > 0:
            for col in non_numeric_columns:
                selected_df[col] = features_df[col]
        
        logger.info(f"Selected {len(selected_features)} features from {len(numeric_columns)} numeric features")
        
        return selected_df
    
    def _normalize_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Normalize feature values."""
        logger.info("📏 Normalizing features...")
        
        # Only normalize numeric columns
        numeric_columns = features_df.select_dtypes(include=[np.number]).columns
        non_numeric_columns = features_df.select_dtypes(exclude=[np.number]).columns
        
        if len(numeric_columns) == 0:
            return features_df
        
        # Apply normalization
        if self.scaler is None:
            self.scaler = StandardScaler()
            normalized_values = self.scaler.fit_transform(features_df[numeric_columns])
        else:
            normalized_values = self.scaler.transform(features_df[numeric_columns])
        
        # Create normalized DataFrame
        normalized_df = pd.DataFrame(
            normalized_values, 
            columns=numeric_columns, 
            index=features_df.index
        )
        
        # Add back non-numeric columns
        for col in non_numeric_columns:
            normalized_df[col] = features_df[col]
        
        return normalized_df
    
    def create_data_splits(self, 
                          X: np.ndarray, 
                          y: np.ndarray,
                          stratify: bool = True) -> Tuple[np.ndarray, ...]:
        """
        Create train/validation/test splits with stratification.
        
        Args:
            X: Feature matrix
            y: Labels
            stratify: Whether to stratify splits
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        from sklearn.model_selection import train_test_split
        
        logger.info("📊 Creating data splits...")
        
        # First split: train+val vs test
        test_size = self.config.test_split
        val_size = self.config.validation_split / (1 - test_size)
        
        stratify_y = y if stratify else None
        
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, 
            test_size=test_size,
            stratify=stratify_y,
            random_state=self.config.random_state
        )
        
        # Second split: train vs val
        stratify_y_temp = y_temp if stratify else None
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size,
            stratify=stratify_y_temp,
            random_state=self.config.random_state
        )
        
        logger.info(f"Data splits created:")
        logger.info(f"  Train: {X_train.shape[0]} samples")
        logger.info(f"  Validation: {X_val.shape[0]} samples")
        logger.info(f"  Test: {X_test.shape[0]} samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def save_preprocessing_artifacts(self, output_dir: str):
        """Save preprocessing components for later use."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save scaler
        if self.scaler is not None:
            joblib.dump(self.scaler, output_path / 'scaler.pkl')
        
        # Save feature selector
        if self.feature_selector is not None:
            joblib.dump(self.feature_selector, output_path / 'feature_selector.pkl')
        
        # Save label encoder
        if self.label_encoder is not None:
            joblib.dump(self.label_encoder, output_path / 'label_encoder.pkl')
        
        # Save configuration
        with open(output_path / 'preprocessing_config.pkl', 'wb') as f:
            pickle.dump(self.config, f)
        
        logger.info(f"Preprocessing artifacts saved to {output_path}")
    
    def load_preprocessing_artifacts(self, input_dir: str):
        """Load preprocessing components."""
        input_path = Path(input_dir)
        
        # Load scaler
        scaler_path = input_path / 'scaler.pkl'
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)
        
        # Load feature selector
        selector_path = input_path / 'feature_selector.pkl'
        if selector_path.exists():
            self.feature_selector = joblib.load(selector_path)
        
        # Load label encoder
        encoder_path = input_path / 'label_encoder.pkl'
        if encoder_path.exists():
            self.label_encoder = joblib.load(encoder_path)
        
        # Load configuration
        config_path = input_path / 'preprocessing_config.pkl'
        if config_path.exists():
            with open(config_path, 'rb') as f:
                self.config = pickle.load(f)
        
        logger.info(f"Preprocessing artifacts loaded from {input_path}")
    
    def get_preprocessing_report(self) -> Dict:
        """Generate comprehensive preprocessing report."""
        return {
            'configuration': {
                'normalize_features': self.config.normalize_features,
                'feature_selection': self.config.feature_selection,
                'max_features': self.config.max_features,
                'remove_duplicates': self.config.remove_duplicates,
                'handle_missing': self.config.handle_missing
            },
            'statistics': {
                'total_samples': self.stats.total_samples,
                'processed_samples': self.stats.processed_samples,
                'removed_duplicates': self.stats.removed_duplicates,
                'removed_invalid': self.stats.removed_invalid,
                'feature_count_final': self.stats.feature_count_final,
                'processing_time': self.stats.processing_time,
                'memory_usage_mb': self.stats.memory_usage_mb
            },
            'performance_metrics': {
                'samples_per_second': self.stats.processed_samples / max(1, self.stats.processing_time),
                'memory_per_sample': self.stats.memory_usage_mb / max(1, self.stats.processed_samples),
                'data_quality_score': 1.0 - (self.stats.removed_invalid / max(1, self.stats.total_samples))
            }
        }


def main():
    """Example usage of ParallelPreprocessor."""
    # Load sample data
    loader = SmartBugsDataLoader("dataset/smartbugs-wild-master")
    contracts = loader.load_contracts_parallel(max_contracts=500)
    
    # Configure preprocessing
    config = PreprocessingConfig(
        normalize_features=True,
        feature_selection=True,
        max_features=30,
        remove_duplicates=True,
        validation_split=0.2,
        test_split=0.1
    )
    
    # Initialize preprocessor
    preprocessor = ParallelPreprocessor(config=config, n_jobs=4)
    
    # Preprocess dataset
    X, y, metadata = preprocessor.preprocess_dataset(contracts)
    
    # Create data splits
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.create_data_splits(X, y)
    
    # Generate report
    report = preprocessor.get_preprocessing_report()
    print("📊 Preprocessing Report:")
    for section, data in report.items():
        print(f"\n{section.upper()}:")
        for key, value in data.items():
            print(f"  {key}: {value}")
    
    # Save preprocessing artifacts
    preprocessor.save_preprocessing_artifacts("data/preprocessing_artifacts")
    
    print(f"\n✅ Preprocessing complete!")
    print(f"Final dataset shape: {X.shape}")
    print(f"Class distribution: {np.bincount(y)}")


if __name__ == "__main__":
    main()