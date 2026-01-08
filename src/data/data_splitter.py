"""
Data splitting utilities for train/validation/test splits with stratification.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class DataSplitter:
    """
    Handles stratified splitting of smart contract datasets into train/validation/test sets.
    
    Features:
    - Stratified splitting to maintain vulnerability distribution
    - Configurable split ratios
    - Validation of split quality
    - Save/load split datasets
    """
    
    def __init__(self, 
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.15,
                 test_ratio: float = 0.15,
                 random_state: int = 42):
        """
        Initialize the data splitter.
        
        Args:
            train_ratio: Proportion for training set
            val_ratio: Proportion for validation set  
            test_ratio: Proportion for test set
            random_state: Random seed for reproducibility
        """
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0")
        
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_state = random_state
        self.split_info = {}
        
    def split_dataset(self, df: pd.DataFrame, 
                     stratify_column: str = 'vulnerability') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split dataset into train/validation/test sets with stratification.
        
        Args:
            df: DataFrame with contract data
            stratify_column: Column to use for stratification
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        logger.info(f"Splitting dataset of {len(df)} contracts")
        logger.info(f"Split ratios: train={self.train_ratio}, val={self.val_ratio}, test={self.test_ratio}")
        
        if stratify_column not in df.columns:
            raise ValueError(f"Stratify column '{stratify_column}' not found in DataFrame")
        
        # First split: separate test set
        train_val_df, test_df = train_test_split(
            df,
            test_size=self.test_ratio,
            stratify=df[stratify_column],
            random_state=self.random_state
        )
        
        # Second split: separate train and validation from remaining data
        # Adjust validation ratio for the remaining data
        val_ratio_adjusted = self.val_ratio / (self.train_ratio + self.val_ratio)
        
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_ratio_adjusted,
            stratify=train_val_df[stratify_column],
            random_state=self.random_state
        )
        
        # Store split information
        self.split_info = {
            'total_samples': len(df),
            'train_samples': len(train_df),
            'val_samples': len(val_df),
            'test_samples': len(test_df),
            'stratify_column': stratify_column,
            'train_ratio_actual': len(train_df) / len(df),
            'val_ratio_actual': len(val_df) / len(df),
            'test_ratio_actual': len(test_df) / len(df)
        }
        
        # Validate split quality
        self._validate_split_quality(df, train_df, val_df, test_df, stratify_column)
        
        logger.info(f"Split completed: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
        
        return train_df, val_df, test_df
    
    def _validate_split_quality(self, original_df: pd.DataFrame,
                               train_df: pd.DataFrame,
                               val_df: pd.DataFrame, 
                               test_df: pd.DataFrame,
                               stratify_column: str):
        """
        Validate the quality of the data split.
        
        Args:
            original_df: Original dataset
            train_df: Training set
            val_df: Validation set
            test_df: Test set
            stratify_column: Column used for stratification
        """
        logger.info("Validating split quality...")
        
        # Check for data leakage (overlapping samples)
        train_hashes = set(train_df['code_hash']) if 'code_hash' in train_df.columns else set()
        val_hashes = set(val_df['code_hash']) if 'code_hash' in val_df.columns else set()
        test_hashes = set(test_df['code_hash']) if 'code_hash' in test_df.columns else set()
        
        if train_hashes & val_hashes:
            logger.warning(f"Data leakage detected: {len(train_hashes & val_hashes)} samples overlap between train and validation")
        
        if train_hashes & test_hashes:
            logger.warning(f"Data leakage detected: {len(train_hashes & test_hashes)} samples overlap between train and test")
        
        if val_hashes & test_hashes:
            logger.warning(f"Data leakage detected: {len(val_hashes & test_hashes)} samples overlap between validation and test")
        
        # Check stratification quality
        original_dist = original_df[stratify_column].value_counts(normalize=True).sort_index()
        train_dist = train_df[stratify_column].value_counts(normalize=True).sort_index()
        val_dist = val_df[stratify_column].value_counts(normalize=True).sort_index()
        test_dist = test_df[stratify_column].value_counts(normalize=True).sort_index()
        
        logger.info("Distribution comparison:")
        logger.info(f"{'Category':<20} {'Original':<10} {'Train':<10} {'Val':<10} {'Test':<10}")
        logger.info("-" * 60)
        
        for category in original_dist.index:
            orig_pct = original_dist.get(category, 0) * 100
            train_pct = train_dist.get(category, 0) * 100
            val_pct = val_dist.get(category, 0) * 100
            test_pct = test_dist.get(category, 0) * 100
            
            logger.info(f"{category:<20} {orig_pct:<10.1f} {train_pct:<10.1f} {val_pct:<10.1f} {test_pct:<10.1f}")
        
        # Calculate maximum deviation from original distribution
        max_deviation = 0
        for category in original_dist.index:
            orig_pct = original_dist.get(category, 0)
            for split_dist in [train_dist, val_dist, test_dist]:
                split_pct = split_dist.get(category, 0)
                deviation = abs(orig_pct - split_pct)
                max_deviation = max(max_deviation, deviation)
        
        logger.info(f"Maximum distribution deviation: {max_deviation * 100:.2f}%")
        
        if max_deviation > 0.05:  # 5% threshold
            logger.warning("Large distribution deviation detected - stratification may not be optimal")
    
    def save_splits(self, train_df: pd.DataFrame,
                   val_df: pd.DataFrame,
                   test_df: pd.DataFrame,
                   output_dir: str = "data/processed"):
        """
        Save train/validation/test splits to CSV files.
        
        Args:
            train_df: Training set
            val_df: Validation set
            test_df: Test set
            output_dir: Directory to save the files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save datasets
        train_path = output_path / "train_contracts.csv"
        val_path = output_path / "val_contracts.csv"
        test_path = output_path / "test_contracts.csv"
        
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        logger.info(f"Saved splits to {output_dir}:")
        logger.info(f"  Training: {train_path} ({len(train_df)} samples)")
        logger.info(f"  Validation: {val_path} ({len(val_df)} samples)")
        logger.info(f"  Test: {test_path} ({len(test_df)} samples)")
        
        # Save split metadata
        metadata_path = output_path / "split_metadata.json"
        import json
        with open(metadata_path, 'w') as f:
            json.dump(self.split_info, f, indent=2)
        
        logger.info(f"Saved split metadata to {metadata_path}")
    
    def load_splits(self, input_dir: str = "data/processed") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load previously saved train/validation/test splits.
        
        Args:
            input_dir: Directory containing the split files
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        input_path = Path(input_dir)
        
        train_path = input_path / "train_contracts.csv"
        val_path = input_path / "val_contracts.csv"
        test_path = input_path / "test_contracts.csv"
        
        if not all(path.exists() for path in [train_path, val_path, test_path]):
            raise FileNotFoundError("One or more split files not found")
        
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)
        
        logger.info(f"Loaded splits from {input_dir}:")
        logger.info(f"  Training: {len(train_df)} samples")
        logger.info(f"  Validation: {len(val_df)} samples")
        logger.info(f"  Test: {len(test_df)} samples")
        
        return train_df, val_df, test_df
    
    def get_split_info(self) -> Dict:
        """
        Get information about the last split operation.
        
        Returns:
            Dictionary with split information
        """
        return self.split_info.copy()
    
    def create_balanced_split(self, df: pd.DataFrame,
                            stratify_column: str = 'vulnerability',
                            min_samples_per_class: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create a balanced split ensuring minimum samples per class in each split.
        
        Args:
            df: DataFrame with contract data
            stratify_column: Column to use for stratification
            min_samples_per_class: Minimum samples per class in each split
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        logger.info("Creating balanced split with minimum samples per class")
        
        # Check if we have enough samples for each class
        class_counts = df[stratify_column].value_counts()
        min_total_needed = min_samples_per_class * 3  # train + val + test
        
        insufficient_classes = class_counts[class_counts < min_total_needed]
        if len(insufficient_classes) > 0:
            logger.warning(f"Classes with insufficient samples: {insufficient_classes.to_dict()}")
            logger.warning("Consider reducing min_samples_per_class or collecting more data")
        
        # Perform stratified split
        return self.split_dataset(df, stratify_column)


def main():
    """Example usage of DataSplitter."""
    # Load and clean data
    from .data_loader import ContractDataLoader
    from .data_cleaner import DataCleaner
    
    # Load data
    loader = ContractDataLoader("data/raw")
    df = loader.load_contracts()
    
    # Clean data
    cleaner = DataCleaner()
    df_clean = cleaner.clean_dataset(df)
    
    print(f"Dataset size: {len(df_clean)} contracts")
    print(f"Vulnerability distribution:\n{df_clean['vulnerability'].value_counts()}")
    
    # Split data
    splitter = DataSplitter(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    train_df, val_df, test_df = splitter.split_dataset(df_clean)
    
    # Save splits
    splitter.save_splits(train_df, val_df, test_df)
    
    # Get split information
    split_info = splitter.get_split_info()
    print("\nSplit Information:")
    for key, value in split_info.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()