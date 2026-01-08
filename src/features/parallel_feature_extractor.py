#!/usr/bin/env python3
"""
Parallel Feature Extraction System for SmartBugs Wild Dataset.
Implements memory-efficient streaming feature extraction with intelligent caching.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterator, Callable, Union
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing import Manager, Queue, shared_memory
import time
import gc
import psutil
from dataclasses import dataclass, asdict
import joblib
import pickle
import hashlib
import sqlite3
from functools import lru_cache
import threading
from collections import defaultdict

# Add src to path for imports
import sys
sys.path.insert(0, 'src')

from features.feature_extractor import SolidityFeatureExtractor, FeatureConfig
from data.efficient_data_loader import ContractData

logger = logging.getLogger(__name__)

@dataclass
class ExtractionConfig:
    """Configuration for parallel feature extraction."""
    n_jobs: int = -1
    chunk_size: int = 100
    enable_caching: bool = True
    cache_features: bool = True
    feature_selection: bool = False
    max_features: int = 50
    memory_limit_mb: int = 4096
    use_shared_memory: bool = True
    batch_processing: bool = True

@dataclass
class ExtractionStats:
    """Statistics for feature extraction performance."""
    total_contracts: int = 0
    processed_contracts: int = 0
    failed_contracts: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    extraction_time: float = 0.0
    memory_usage_mb: float = 0.0
    features_extracted: int = 0
    processing_rate: float = 0.0

class FeatureCache:
    """Intelligent caching system for extracted features."""
    
    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / "feature_cache.db"
        self._init_cache_db()
        self._lock = threading.Lock()
    
    def _init_cache_db(self):
        """Initialize SQLite cache database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feature_cache (
                    contract_hash TEXT PRIMARY KEY,
                    features BLOB,
                    feature_names BLOB,
                    extraction_time REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_contract_hash 
                ON feature_cache(contract_hash)
            """)
            conn.commit()
    
    def get_features(self, contract_hash: str) -> Optional[Dict[str, float]]:
        """Get cached features for a contract."""
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute(
                        "SELECT features, feature_names FROM feature_cache WHERE contract_hash = ?",
                        (contract_hash,)
                    )
                    row = cursor.fetchone()
                    
                    if row:
                        features_blob, names_blob = row
                        features = pickle.loads(features_blob)
                        return features
            except Exception as e:
                logger.debug(f"Cache lookup failed for {contract_hash}: {e}")
        
        return None
    
    def cache_features(self, contract_hash: str, features: Dict[str, float], extraction_time: float):
        """Cache extracted features."""
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        INSERT OR REPLACE INTO feature_cache 
                        (contract_hash, features, feature_names, extraction_time)
                        VALUES (?, ?, ?, ?)
                    """, (
                        contract_hash,
                        pickle.dumps(features),
                        pickle.dumps(list(features.keys())),
                        extraction_time
                    ))
                    conn.commit()
            except Exception as e:
                logger.debug(f"Failed to cache features for {contract_hash}: {e}")
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM feature_cache")
            total_cached = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT AVG(extraction_time) FROM feature_cache")
            avg_extraction_time = cursor.fetchone()[0] or 0
            
            return {
                'total_cached': total_cached,
                'avg_extraction_time': avg_extraction_time,
                'cache_size_mb': self.db_path.stat().st_size / (1024 * 1024)
            }

class ParallelFeatureExtractor:
    """
    Parallel feature extraction system with memory optimization and smart caching.
    """
    
    def __init__(self, 
                 config: ExtractionConfig = None,
                 feature_config: FeatureConfig = None,
                 cache_dir: str = "data/cache/features"):
        """
        Initialize the parallel feature extractor.
        
        Args:
            config: Extraction configuration
            feature_config: Feature extraction configuration
            cache_dir: Directory for feature caching
        """
        self.config = config or ExtractionConfig()
        self.feature_config = feature_config or FeatureConfig()
        
        # Set number of workers
        if self.config.n_jobs == -1:
            self.config.n_jobs = min(32, (psutil.cpu_count() or 1) + 4)
        
        # Initialize feature extractor
        self.base_extractor = SolidityFeatureExtractor(self.feature_config)
        
        # Initialize cache
        self.cache = FeatureCache(cache_dir) if self.config.enable_caching else None
        
        # Statistics
        self.stats = ExtractionStats()
        
        # Feature selection components
        self.feature_selector = None
        self.selected_features = None
        
        logger.info(f"Initialized ParallelFeatureExtractor with {self.config.n_jobs} workers")
    
    def extract_features_parallel(self, 
                                contracts: List[ContractData],
                                progress_callback: Optional[Callable] = None) -> pd.DataFrame:
        """
        Extract features from contracts using parallel processing.
        
        Args:
            contracts: List of contract data
            progress_callback: Optional callback for progress updates
            
        Returns:
            DataFrame with extracted features
        """
        logger.info(f"🔧 Extracting features from {len(contracts)} contracts in parallel...")
        start_time = time.time()
        
        self.stats.total_contracts = len(contracts)
        
        # Split contracts into chunks
        chunks = self._create_chunks(contracts)
        
        # Process chunks in parallel
        all_features = []
        processed_count = 0
        
        with ProcessPoolExecutor(max_workers=self.config.n_jobs) as executor:
            # Submit all chunks
            future_to_chunk = {
                executor.submit(self._process_chunk_wrapper, chunk, chunk_idx): chunk 
                for chunk_idx, chunk in enumerate(chunks)
            }
            
            # Collect results with progress tracking
            for future in as_completed(future_to_chunk):
                try:
                    chunk_features = future.result()
                    all_features.extend(chunk_features)
                    processed_count += len(chunk_features)
                    
                    # Update progress
                    if progress_callback:
                        progress_callback(processed_count, len(contracts))
                    
                    # Memory management
                    if processed_count % (self.config.chunk_size * 5) == 0:
                        gc.collect()
                
                except Exception as e:
                    logger.error(f"Error processing chunk: {e}")
                    chunk = future_to_chunk[future]
                    self.stats.failed_contracts += len(chunk)
        
        # Create DataFrame
        if all_features:
            features_df = pd.DataFrame(all_features)
            
            # Apply feature selection if configured
            if self.config.feature_selection and len(features_df) > 0:
                features_df = self._apply_feature_selection(features_df)
            
            # Update statistics
            self.stats.processed_contracts = len(features_df)
            self.stats.features_extracted = len(features_df.columns) if len(features_df) > 0 else 0
            self.stats.extraction_time = time.time() - start_time
            self.stats.memory_usage_mb = psutil.Process().memory_info().rss / 1024 / 1024
            self.stats.processing_rate = self.stats.processed_contracts / max(1, self.stats.extraction_time)
            
            logger.info(f"✅ Feature extraction complete: {features_df.shape} in {self.stats.extraction_time:.2f}s")
            logger.info(f"📊 Processing rate: {self.stats.processing_rate:.1f} contracts/second")
            
            return features_df
        else:
            raise ValueError("No features extracted from contracts")
    
    def extract_features_streaming(self, 
                                 contracts: Iterator[ContractData],
                                 batch_size: int = 1000) -> Iterator[pd.DataFrame]:
        """
        Extract features using streaming processing for memory efficiency.
        
        Args:
            contracts: Iterator of contract data
            batch_size: Size of processing batches
            
        Yields:
            DataFrames with extracted features
        """
        logger.info(f"🌊 Starting streaming feature extraction with batch size {batch_size}")
        
        batch = []
        batch_count = 0
        
        for contract in contracts:
            batch.append(contract)
            
            if len(batch) >= batch_size:
                # Process batch
                try:
                    features_df = self.extract_features_parallel(batch)
                    batch_count += 1
                    
                    logger.info(f"Processed batch {batch_count}: {features_df.shape}")
                    yield features_df
                    
                    # Clear batch and force garbage collection
                    batch = []
                    gc.collect()
                
                except Exception as e:
                    logger.error(f"Error processing batch {batch_count}: {e}")
                    batch = []
        
        # Process remaining contracts
        if batch:
            try:
                features_df = self.extract_features_parallel(batch)
                batch_count += 1
                logger.info(f"Processed final batch {batch_count}: {features_df.shape}")
                yield features_df
            except Exception as e:
                logger.error(f"Error processing final batch: {e}")
    
    def _create_chunks(self, contracts: List[ContractData]) -> List[List[ContractData]]:
        """Create processing chunks from contracts."""
        chunks = []
        for i in range(0, len(contracts), self.config.chunk_size):
            chunk = contracts[i:i + self.config.chunk_size]
            chunks.append(chunk)
        return chunks
    
    def _process_chunk_wrapper(self, chunk: List[ContractData], chunk_idx: int) -> List[Dict[str, float]]:
        """Wrapper for chunk processing to handle serialization."""
        return self._process_chunk(chunk, chunk_idx)
    
    def _process_chunk(self, chunk: List[ContractData], chunk_idx: int) -> List[Dict[str, float]]:
        """Process a chunk of contracts for feature extraction."""
        # Create a new extractor instance for this process
        extractor = SolidityFeatureExtractor(self.feature_config)
        chunk_features = []
        
        for contract in chunk:
            try:
                # Check cache first
                if self.cache:
                    cached_features = self.cache.get_features(contract.hash)
                    if cached_features:
                        chunk_features.append(cached_features)
                        self.stats.cache_hits += 1
                        continue
                
                # Extract features
                start_time = time.time()
                features = extractor.extract_features(contract.code)
                extraction_time = time.time() - start_time
                
                # Add metadata
                features.update({
                    'contract_hash': contract.hash,
                    'filename': contract.filename,
                    'contract_size': contract.size,
                    'is_vulnerable': contract.is_vulnerable,
                    'vulnerability_count': len(contract.vulnerabilities)
                })
                
                # Cache features
                if self.cache:
                    self.cache.cache_features(contract.hash, features, extraction_time)
                    self.stats.cache_misses += 1
                
                chunk_features.append(features)
                
            except Exception as e:
                logger.warning(f"Error extracting features from {contract.filename}: {e}")
                continue
        
        return chunk_features
    
    def _apply_feature_selection(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Apply feature selection to reduce dimensionality."""
        if self.selected_features is not None:
            # Use previously selected features
            available_features = [f for f in self.selected_features if f in features_df.columns]
            metadata_cols = ['contract_hash', 'filename', 'is_vulnerable', 'vulnerability_count']
            selected_cols = available_features + [c for c in metadata_cols if c in features_df.columns]
            return features_df[selected_cols]
        
        # Perform feature selection
        from sklearn.feature_selection import SelectKBest, f_classif
        
        # Separate features from metadata
        metadata_cols = ['contract_hash', 'filename', 'is_vulnerable', 'vulnerability_count']
        feature_cols = [col for col in features_df.columns if col not in metadata_cols]
        
        if len(feature_cols) <= self.config.max_features:
            return features_df
        
        # Prepare data for feature selection
        X = features_df[feature_cols].fillna(0)
        y = features_df['is_vulnerable'].astype(int)
        
        # Apply feature selection
        if self.feature_selector is None:
            self.feature_selector = SelectKBest(
                score_func=f_classif,
                k=min(self.config.max_features, len(feature_cols))
            )
            X_selected = self.feature_selector.fit_transform(X, y)
        else:
            X_selected = self.feature_selector.transform(X)
        
        # Get selected feature names
        selected_mask = self.feature_selector.get_support()
        self.selected_features = [feature_cols[i] for i, selected in enumerate(selected_mask) if selected]
        
        # Create new DataFrame with selected features
        selected_df = pd.DataFrame(X_selected, columns=self.selected_features, index=features_df.index)
        
        # Add back metadata columns
        for col in metadata_cols:
            if col in features_df.columns:
                selected_df[col] = features_df[col]
        
        logger.info(f"Feature selection: {len(feature_cols)} -> {len(self.selected_features)} features")
        
        return selected_df
    
    def extract_single_contract_features(self, contract: ContractData) -> Dict[str, float]:
        """
        Extract features from a single contract with caching.
        
        Args:
            contract: Contract data
            
        Returns:
            Dictionary of extracted features
        """
        # Check cache first
        if self.cache:
            cached_features = self.cache.get_features(contract.hash)
            if cached_features:
                return cached_features
        
        # Extract features
        start_time = time.time()
        features = self.base_extractor.extract_features(contract.code)
        extraction_time = time.time() - start_time
        
        # Add metadata
        features.update({
            'contract_hash': contract.hash,
            'filename': contract.filename,
            'contract_size': contract.size,
            'is_vulnerable': contract.is_vulnerable,
            'vulnerability_count': len(contract.vulnerabilities)
        })
        
        # Cache features
        if self.cache:
            self.cache.cache_features(contract.hash, features, extraction_time)
        
        return features
    
    def get_feature_importance(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate feature importance using multiple methods.
        
        Args:
            features_df: DataFrame with extracted features
            
        Returns:
            DataFrame with feature importance scores
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.feature_selection import mutual_info_classif
        
        # Prepare data
        metadata_cols = ['contract_hash', 'filename', 'is_vulnerable', 'vulnerability_count']
        feature_cols = [col for col in features_df.columns if col not in metadata_cols]
        
        X = features_df[feature_cols].fillna(0)
        y = features_df['is_vulnerable'].astype(int)
        
        importance_scores = []
        
        # Random Forest importance
        try:
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X, y)
            rf_importance = rf.feature_importances_
            
            for i, feature in enumerate(feature_cols):
                importance_scores.append({
                    'feature': feature,
                    'rf_importance': rf_importance[i],
                    'method': 'random_forest'
                })
        except Exception as e:
            logger.warning(f"Random Forest importance calculation failed: {e}")
        
        # Mutual information
        try:
            mi_scores = mutual_info_classif(X, y, random_state=42)
            
            for i, feature in enumerate(feature_cols):
                if i < len(importance_scores):
                    importance_scores[i]['mi_score'] = mi_scores[i]
                else:
                    importance_scores.append({
                        'feature': feature,
                        'mi_score': mi_scores[i],
                        'method': 'mutual_info'
                    })
        except Exception as e:
            logger.warning(f"Mutual information calculation failed: {e}")
        
        # Create DataFrame
        importance_df = pd.DataFrame(importance_scores)
        
        # Calculate combined score
        if 'rf_importance' in importance_df.columns and 'mi_score' in importance_df.columns:
            # Normalize scores
            importance_df['rf_importance_norm'] = (
                importance_df['rf_importance'] / importance_df['rf_importance'].max()
            )
            importance_df['mi_score_norm'] = (
                importance_df['mi_score'] / importance_df['mi_score'].max()
            )
            
            # Combined score (weighted average)
            importance_df['combined_score'] = (
                0.6 * importance_df['rf_importance_norm'] + 
                0.4 * importance_df['mi_score_norm']
            )
        
        # Sort by combined score or available score
        sort_column = 'combined_score' if 'combined_score' in importance_df.columns else 'rf_importance'
        importance_df = importance_df.sort_values(sort_column, ascending=False)
        
        return importance_df
    
    def save_extraction_artifacts(self, output_dir: str):
        """Save extraction components and statistics."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save feature selector
        if self.feature_selector is not None:
            joblib.dump(self.feature_selector, output_path / 'feature_selector.pkl')
        
        # Save selected features
        if self.selected_features is not None:
            with open(output_path / 'selected_features.txt', 'w') as f:
                for feature in self.selected_features:
                    f.write(f"{feature}\n")
        
        # Save configuration
        with open(output_path / 'extraction_config.pkl', 'wb') as f:
            pickle.dump(self.config, f)
        
        # Save statistics
        with open(output_path / 'extraction_stats.json', 'w') as f:
            import json
            json.dump(asdict(self.stats), f, indent=2)
        
        logger.info(f"Extraction artifacts saved to {output_path}")
    
    def get_extraction_report(self) -> Dict:
        """Generate comprehensive extraction performance report."""
        cache_stats = self.cache.get_cache_stats() if self.cache else {}
        
        return {
            'configuration': asdict(self.config),
            'statistics': asdict(self.stats),
            'cache_statistics': cache_stats,
            'performance_metrics': {
                'contracts_per_second': self.stats.processing_rate,
                'memory_per_contract': self.stats.memory_usage_mb / max(1, self.stats.processed_contracts),
                'cache_hit_rate': self.stats.cache_hits / max(1, self.stats.cache_hits + self.stats.cache_misses),
                'success_rate': self.stats.processed_contracts / max(1, self.stats.total_contracts),
                'features_per_contract': self.stats.features_extracted
            }
        }


def main():
    """Example usage of ParallelFeatureExtractor."""
    from data.efficient_data_loader import SmartBugsDataLoader
    
    # Load sample contracts
    loader = SmartBugsDataLoader("dataset/smartbugs-wild-master")
    contracts = loader.load_contracts_parallel(max_contracts=200)
    
    # Configure extraction
    extraction_config = ExtractionConfig(
        n_jobs=4,
        chunk_size=50,
        enable_caching=True,
        feature_selection=True,
        max_features=30
    )
    
    feature_config = FeatureConfig(
        include_basic_metrics=True,
        include_function_analysis=True,
        include_dangerous_patterns=True,
        include_control_flow=True,
        include_access_control=True,
        include_arithmetic=True,
        include_randomness=True
    )
    
    # Initialize extractor
    extractor = ParallelFeatureExtractor(
        config=extraction_config,
        feature_config=feature_config
    )
    
    # Extract features
    def progress_callback(processed, total):
        print(f"Progress: {processed}/{total} ({processed/total*100:.1f}%)")
    
    features_df = extractor.extract_features_parallel(contracts, progress_callback)
    
    print(f"\n✅ Feature extraction complete!")
    print(f"Dataset shape: {features_df.shape}")
    print(f"Features extracted: {len([c for c in features_df.columns if c not in ['contract_hash', 'filename', 'is_vulnerable']])}")
    
    # Calculate feature importance
    importance_df = extractor.get_feature_importance(features_df)
    print(f"\nTop 10 Most Important Features:")
    print(importance_df.head(10)[['feature', 'combined_score']])
    
    # Generate report
    report = extractor.get_extraction_report()
    print(f"\n📊 Extraction Report:")
    print(f"Processing rate: {report['performance_metrics']['contracts_per_second']:.1f} contracts/sec")
    print(f"Cache hit rate: {report['performance_metrics']['cache_hit_rate']:.1%}")
    print(f"Success rate: {report['performance_metrics']['success_rate']:.1%}")
    
    # Save artifacts
    extractor.save_extraction_artifacts("data/extraction_artifacts")
    
    # Example of streaming extraction
    print(f"\n🌊 Testing streaming extraction...")
    batch_count = 0
    for batch_df in extractor.extract_features_streaming(iter(contracts[:100]), batch_size=25):
        batch_count += 1
        print(f"Batch {batch_count}: {batch_df.shape}")
        if batch_count >= 3:  # Limit for demo
            break


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    main()