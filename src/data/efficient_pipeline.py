#!/usr/bin/env python3
"""
Complete Efficient Data Loading and Preprocessing Pipeline.
Integrates all components for SmartBugs Wild dataset processing.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import time
import json
from dataclasses import asdict

# Add src to path for imports
import sys
sys.path.insert(0, 'src')

from data.efficient_data_loader import SmartBugsDataLoader, ContractData
from data.parallel_preprocessor import ParallelPreprocessor, PreprocessingConfig
from data.data_validator import DataValidator, ValidationResult

logger = logging.getLogger(__name__)

class EfficientDataPipeline:
    """
    Complete efficient data pipeline for SmartBugs Wild dataset.
    Combines loading, validation, preprocessing, and caching.
    """
    
    def __init__(self, 
                 dataset_path: str,
                 output_dir: str = "data/processed",
                 cache_dir: str = "data/cache",
                 max_workers: int = None,
                 enable_validation: bool = True,
                 min_quality_score: float = 0.8):
        """
        Initialize the complete pipeline.
        
        Args:
            dataset_path: Path to SmartBugs Wild dataset
            output_dir: Directory for processed outputs
            cache_dir: Directory for caching
            max_workers: Number of parallel workers
            enable_validation: Enable data validation
            min_quality_score: Minimum quality score for validation
        """
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.cache_dir = Path(cache_dir)
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.loader = SmartBugsDataLoader(
            dataset_path=str(self.dataset_path),
            cache_dir=str(self.cache_dir),
            max_workers=max_workers,
            enable_caching=True
        )
        
        self.validator = DataValidator(
            min_quality_score=min_quality_score
        ) if enable_validation else None
        
        self.preprocessor = None  # Will be initialized with config
        
        # Pipeline statistics
        self.pipeline_stats = {
            'total_time': 0.0,
            'loading_time': 0.0,
            'validation_time': 0.0,
            'preprocessing_time': 0.0,
            'contracts_loaded': 0,
            'contracts_processed': 0,
            'quality_score': 0.0
        }
        
        logger.info("Initialized EfficientDataPipeline")
    
    def run_complete_pipeline(self,
                            max_contracts: Optional[int] = None,
                            sample_ratio: float = 1.0,
                            preprocessing_config: Optional[PreprocessingConfig] = None,
                            target_column: str = 'is_vulnerable',
                            save_outputs: bool = True) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Run the complete data pipeline from raw data to processed features.
        
        Args:
            max_contracts: Maximum contracts to process
            sample_ratio: Ratio of dataset to sample
            preprocessing_config: Configuration for preprocessing
            target_column: Target variable column
            save_outputs: Save intermediate and final outputs
            
        Returns:
            Tuple of (features, labels, metadata)
        """
        logger.info("🚀 Starting complete efficient data pipeline...")
        pipeline_start = time.time()
        
        # Step 1: Load contracts
        logger.info("📁 Step 1: Loading contracts...")
        load_start = time.time()
        contracts = self.loader.load_contracts_parallel(
            max_contracts=max_contracts,
            sample_ratio=sample_ratio
        )
        self.pipeline_stats['loading_time'] = time.time() - load_start
        self.pipeline_stats['contracts_loaded'] = len(contracts)
        
        if not contracts:
            raise ValueError("No contracts loaded from dataset")
        
        # Step 2: Validate data quality
        validation_result = None
        if self.validator:
            logger.info("🔍 Step 2: Validating data quality...")
            validation_start = time.time()
            validation_result = self.validator.validate_dataset(contracts)
            self.pipeline_stats['validation_time'] = time.time() - validation_start
            self.pipeline_stats['quality_score'] = validation_result.quality_score
            
            # Save validation report
            if save_outputs:
                report_path = self.output_dir / 'validation_report.txt'
                self.validator.generate_validation_report(validation_result, str(report_path))
                
                # Create quality visualizations
                viz_dir = self.output_dir / 'validation_plots'
                self.validator.create_quality_visualization(validation_result, str(viz_dir))
            
            # Check if validation passed
            if not validation_result.is_valid:
                logger.warning(f"⚠️ Data validation failed (score: {validation_result.quality_score:.3f})")
                logger.warning("Proceeding with processing but results may be affected")
        
        # Step 3: Preprocess data
        logger.info("⚙️ Step 3: Preprocessing data...")
        preprocess_start = time.time()
        
        # Initialize preprocessor with config
        if preprocessing_config is None:
            preprocessing_config = PreprocessingConfig(
                normalize_features=True,
                feature_selection=True,
                max_features=50,
                remove_duplicates=True,
                validation_split=0.2,
                test_split=0.1
            )
        
        self.preprocessor = ParallelPreprocessor(
            config=preprocessing_config,
            n_jobs=self.loader.max_workers,
            cache_dir=str(self.cache_dir / 'preprocessing')
        )
        
        # Preprocess dataset
        X, y, preprocessing_metadata = self.preprocessor.preprocess_dataset(
            contracts, target_column
        )
        
        self.pipeline_stats['preprocessing_time'] = time.time() - preprocess_start
        self.pipeline_stats['contracts_processed'] = len(X)
        
        # Step 4: Create data splits
        logger.info("📊 Step 4: Creating data splits...")
        X_train, X_val, X_test, y_train, y_val, y_test = self.preprocessor.create_data_splits(X, y)
        
        # Step 5: Save outputs
        if save_outputs:
            logger.info("💾 Step 5: Saving outputs...")
            self._save_pipeline_outputs(
                X_train, X_val, X_test, y_train, y_val, y_test,
                preprocessing_metadata, validation_result
            )
        
        # Compile final metadata
        self.pipeline_stats['total_time'] = time.time() - pipeline_start
        
        final_metadata = {
            'pipeline_stats': self.pipeline_stats,
            'preprocessing_metadata': preprocessing_metadata,
            'validation_result': asdict(validation_result) if validation_result else None,
            'dataset_info': self.loader.get_dataset_statistics(),
            'data_splits': {
                'train_shape': X_train.shape,
                'val_shape': X_val.shape,
                'test_shape': X_test.shape,
                'class_distribution_train': np.bincount(y_train).tolist(),
                'class_distribution_val': np.bincount(y_val).tolist(),
                'class_distribution_test': np.bincount(y_test).tolist()
            }
        }
        
        logger.info("✅ Pipeline complete!")
        self._print_pipeline_summary(final_metadata)
        
        return (X_train, X_val, X_test, y_train, y_val, y_test), final_metadata
    
    def _save_pipeline_outputs(self,
                             X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray,
                             y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray,
                             preprocessing_metadata: Dict,
                             validation_result: Optional[ValidationResult]):
        """Save all pipeline outputs."""
        
        # Save data splits
        np.savez_compressed(
            self.output_dir / 'data_splits.npz',
            X_train=X_train, X_val=X_val, X_test=X_test,
            y_train=y_train, y_val=y_val, y_test=y_test
        )
        
        # Save preprocessing artifacts
        artifacts_dir = self.output_dir / 'preprocessing_artifacts'
        self.preprocessor.save_preprocessing_artifacts(str(artifacts_dir))
        
        # Save metadata
        metadata = {
            'pipeline_stats': self.pipeline_stats,
            'preprocessing_metadata': preprocessing_metadata,
            'validation_result': asdict(validation_result) if validation_result else None
        }
        
        with open(self.output_dir / 'pipeline_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        # Save feature names for reference
        feature_names = preprocessing_metadata.get('feature_names', [])
        with open(self.output_dir / 'feature_names.txt', 'w') as f:
            for i, name in enumerate(feature_names):
                f.write(f"{i}: {name}\n")
        
        logger.info(f"Pipeline outputs saved to {self.output_dir}")
    
    def _print_pipeline_summary(self, metadata: Dict):
        """Print a summary of pipeline execution."""
        stats = metadata['pipeline_stats']
        
        print("\n" + "=" * 60)
        print("EFFICIENT DATA PIPELINE SUMMARY")
        print("=" * 60)
        
        print(f"📊 Dataset Statistics:")
        print(f"  Contracts loaded: {stats['contracts_loaded']}")
        print(f"  Contracts processed: {stats['contracts_processed']}")
        print(f"  Success rate: {stats['contracts_processed']/stats['contracts_loaded']*100:.1f}%")
        
        print(f"\n⏱️ Performance Metrics:")
        print(f"  Total time: {stats['total_time']:.2f}s")
        print(f"  Loading time: {stats['loading_time']:.2f}s")
        print(f"  Validation time: {stats['validation_time']:.2f}s")
        print(f"  Preprocessing time: {stats['preprocessing_time']:.2f}s")
        
        print(f"\n📈 Data Quality:")
        if stats['quality_score'] > 0:
            print(f"  Quality score: {stats['quality_score']:.3f}")
            quality_status = "EXCELLENT" if stats['quality_score'] > 0.9 else \
                           "GOOD" if stats['quality_score'] > 0.8 else \
                           "ACCEPTABLE" if stats['quality_score'] > 0.7 else "POOR"
            print(f"  Quality status: {quality_status}")
        
        print(f"\n🎯 Final Dataset:")
        splits = metadata['data_splits']
        print(f"  Training set: {splits['train_shape']}")
        print(f"  Validation set: {splits['val_shape']}")
        print(f"  Test set: {splits['test_shape']}")
        
        print(f"\n📋 Class Distribution (Train):")
        class_dist = splits['class_distribution_train']
        for i, count in enumerate(class_dist):
            print(f"  Class {i}: {count} samples ({count/sum(class_dist)*100:.1f}%)")
        
        print("=" * 60)
    
    def load_processed_data(self, data_dir: Optional[str] = None) -> Tuple[Tuple[np.ndarray, ...], Dict]:
        """
        Load previously processed data.
        
        Args:
            data_dir: Directory containing processed data (uses output_dir if None)
            
        Returns:
            Tuple of (data_splits, metadata)
        """
        if data_dir is None:
            data_dir = self.output_dir
        else:
            data_dir = Path(data_dir)
        
        # Load data splits
        data_file = data_dir / 'data_splits.npz'
        if not data_file.exists():
            raise FileNotFoundError(f"Processed data not found at {data_file}")
        
        data = np.load(data_file)
        data_splits = (
            data['X_train'], data['X_val'], data['X_test'],
            data['y_train'], data['y_val'], data['y_test']
        )
        
        # Load metadata
        metadata_file = data_dir / 'pipeline_metadata.json'
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {}
        
        # Load preprocessing artifacts
        artifacts_dir = data_dir / 'preprocessing_artifacts'
        if artifacts_dir.exists() and self.preprocessor is None:
            self.preprocessor = ParallelPreprocessor()
            self.preprocessor.load_preprocessing_artifacts(str(artifacts_dir))
        
        logger.info(f"Loaded processed data from {data_dir}")
        
        return data_splits, metadata
    
    def get_pipeline_report(self) -> Dict:
        """Get comprehensive pipeline performance report."""
        return {
            'pipeline_configuration': {
                'dataset_path': str(self.dataset_path),
                'output_dir': str(self.output_dir),
                'cache_dir': str(self.cache_dir),
                'max_workers': self.loader.max_workers,
                'validation_enabled': self.validator is not None
            },
            'performance_metrics': self.pipeline_stats,
            'loader_stats': self.loader.get_dataset_statistics(),
            'preprocessing_report': self.preprocessor.get_preprocessing_report() if self.preprocessor else None
        }


def main():
    """Example usage of EfficientDataPipeline."""
    # Configuration
    DATASET_PATH = "dataset/smartbugs-wild-master"
    OUTPUT_DIR = "data/processed_efficient"
    
    # Initialize pipeline
    pipeline = EfficientDataPipeline(
        dataset_path=DATASET_PATH,
        output_dir=OUTPUT_DIR,
        max_workers=8,
        enable_validation=True,
        min_quality_score=0.7
    )
    
    # Configure preprocessing
    preprocessing_config = PreprocessingConfig(
        normalize_features=True,
        feature_selection=True,
        max_features=40,
        remove_duplicates=True,
        handle_missing='mean',
        validation_split=0.2,
        test_split=0.1,
        random_state=42
    )
    
    # Run complete pipeline
    try:
        data_splits, metadata = pipeline.run_complete_pipeline(
            max_contracts=2000,  # Process 2000 contracts for demo
            sample_ratio=1.0,
            preprocessing_config=preprocessing_config,
            target_column='is_vulnerable',
            save_outputs=True
        )
        
        X_train, X_val, X_test, y_train, y_val, y_test = data_splits
        
        print(f"\n🎉 Pipeline completed successfully!")
        print(f"Training data shape: {X_train.shape}")
        print(f"Validation data shape: {X_val.shape}")
        print(f"Test data shape: {X_test.shape}")
        
        # Generate comprehensive report
        report = pipeline.get_pipeline_report()
        print(f"\n📊 Pipeline Report:")
        print(f"Total processing time: {report['performance_metrics']['total_time']:.2f}s")
        print(f"Quality score: {report['performance_metrics']['quality_score']:.3f}")
        
        # Example: Load processed data later
        print(f"\n🔄 Testing data loading...")
        loaded_data, loaded_metadata = pipeline.load_processed_data()
        print(f"Successfully loaded processed data: {loaded_data[0].shape}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    main()