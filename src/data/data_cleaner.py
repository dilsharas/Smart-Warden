"""
Data cleaning and preprocessing utilities for smart contract datasets.
"""

import pandas as pd
import numpy as np
import re
import hashlib
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class DataCleaner:
    """
    Handles data cleaning and preprocessing for smart contract datasets.
    
    Features:
    - Remove duplicate contracts
    - Handle encoding issues
    - Validate Solidity syntax
    - Filter by contract size
    - Clean and normalize contract code
    """
    
    def __init__(self, min_lines: int = 10, max_lines: int = 5000):
        """
        Initialize the data cleaner.
        
        Args:
            min_lines: Minimum number of lines for valid contracts
            max_lines: Maximum number of lines for valid contracts
        """
        self.min_lines = min_lines
        self.max_lines = max_lines
        self.cleaning_stats = {}
        
    def clean_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all cleaning steps to the dataset.
        
        Args:
            df: DataFrame with contract data
            
        Returns:
            Cleaned DataFrame
        """
        logger.info(f"Starting data cleaning on {len(df)} contracts")
        original_count = len(df)
        
        # Initialize cleaning stats
        self.cleaning_stats = {
            'original_count': original_count,
            'removed_duplicates': 0,
            'removed_invalid_syntax': 0,
            'removed_size_filter': 0,
            'removed_encoding_issues': 0,
            'final_count': 0
        }
        
        # Step 1: Remove duplicates
        df = self.remove_duplicates(df)
        
        # Step 2: Handle encoding issues
        df = self.handle_encoding_issues(df)
        
        # Step 3: Validate Solidity syntax
        df = self.validate_solidity_syntax(df)
        
        # Step 4: Filter by size
        df = self.filter_by_size(df)
        
        # Step 5: Clean and normalize code
        df = self.clean_contract_code(df)
        
        self.cleaning_stats['final_count'] = len(df)
        
        logger.info(f"Data cleaning completed: {original_count} -> {len(df)} contracts")
        self._log_cleaning_stats()
        
        return df
    
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate contracts based on code hash.
        
        Args:
            df: DataFrame with contract data
            
        Returns:
            DataFrame without duplicates
        """
        logger.info("Removing duplicate contracts...")
        original_count = len(df)
        
        # If code_hash column doesn't exist, create it
        if 'code_hash' not in df.columns:
            df['code_hash'] = df['code'].apply(self._calculate_hash)
        
        # Remove duplicates based on code hash
        df_clean = df.drop_duplicates(subset=['code_hash'], keep='first')
        
        removed_count = original_count - len(df_clean)
        self.cleaning_stats['removed_duplicates'] = removed_count
        
        logger.info(f"Removed {removed_count} duplicate contracts")
        return df_clean
    
    def handle_encoding_issues(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle encoding issues in contract code.
        
        Args:
            df: DataFrame with contract data
            
        Returns:
            DataFrame with encoding issues resolved
        """
        logger.info("Handling encoding issues...")
        original_count = len(df)
        
        def clean_encoding(code: str) -> Optional[str]:
            """Clean encoding issues in contract code."""
            if not isinstance(code, str):
                return None
            
            try:
                # Remove non-printable characters except newlines and tabs
                cleaned = re.sub(r'[^\x20-\x7E\n\t]', '', code)
                
                # Remove excessive whitespace
                cleaned = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned)
                
                # Ensure it's valid UTF-8
                cleaned.encode('utf-8')
                
                return cleaned.strip()
            except Exception:
                return None
        
        # Apply encoding cleaning
        df['code_cleaned'] = df['code'].apply(clean_encoding)
        
        # Remove contracts with encoding issues
        df_clean = df[df['code_cleaned'].notna()].copy()
        df_clean['code'] = df_clean['code_cleaned']
        df_clean = df_clean.drop('code_cleaned', axis=1)
        
        removed_count = original_count - len(df_clean)
        self.cleaning_stats['removed_encoding_issues'] = removed_count
        
        logger.info(f"Removed {removed_count} contracts with encoding issues")
        return df_clean
    
    def validate_solidity_syntax(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate basic Solidity syntax.
        
        Args:
            df: DataFrame with contract data
            
        Returns:
            DataFrame with only syntactically valid contracts
        """
        logger.info("Validating Solidity syntax...")
        original_count = len(df)
        
        def is_valid_solidity(code: str) -> bool:
            """Check if contract has basic valid Solidity syntax."""
            if not isinstance(code, str) or len(code.strip()) < 10:
                return False
            
            # Check for pragma statement
            if not re.search(r'pragma\s+solidity', code, re.IGNORECASE):
                return False
            
            # Check for contract keyword
            if not re.search(r'\bcontract\s+\w+', code, re.IGNORECASE):
                return False
            
            # Check for balanced braces
            open_braces = code.count('{')
            close_braces = code.count('}')
            if open_braces != close_braces or open_braces == 0:
                return False
            
            # Check for balanced parentheses
            open_parens = code.count('(')
            close_parens = code.count(')')
            if open_parens != close_parens:
                return False
            
            return True
        
        # Filter valid contracts
        valid_mask = df['code'].apply(is_valid_solidity)
        df_clean = df[valid_mask].copy()
        
        removed_count = original_count - len(df_clean)
        self.cleaning_stats['removed_invalid_syntax'] = removed_count
        
        logger.info(f"Removed {removed_count} contracts with invalid syntax")
        return df_clean
    
    def filter_by_size(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter contracts by size (number of lines).
        
        Args:
            df: DataFrame with contract data
            
        Returns:
            DataFrame with contracts within size limits
        """
        logger.info(f"Filtering contracts by size ({self.min_lines}-{self.max_lines} lines)...")
        original_count = len(df)
        
        # Calculate line counts
        df['line_count'] = df['code'].str.count('\n') + 1
        
        # Filter by size
        size_mask = (df['line_count'] >= self.min_lines) & (df['line_count'] <= self.max_lines)
        df_clean = df[size_mask].copy()
        
        # Remove temporary column
        df_clean = df_clean.drop('line_count', axis=1)
        
        removed_count = original_count - len(df_clean)
        self.cleaning_stats['removed_size_filter'] = removed_count
        
        logger.info(f"Removed {removed_count} contracts outside size limits")
        return df_clean
    
    def clean_contract_code(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and normalize contract code.
        
        Args:
            df: DataFrame with contract data
            
        Returns:
            DataFrame with cleaned code
        """
        logger.info("Cleaning and normalizing contract code...")
        
        def normalize_code(code: str) -> str:
            """Normalize contract code formatting."""
            if not isinstance(code, str):
                return ""
            
            # Remove excessive whitespace
            code = re.sub(r'[ \t]+', ' ', code)  # Multiple spaces/tabs to single space
            code = re.sub(r'\n\s*\n\s*\n+', '\n\n', code)  # Multiple newlines to double
            
            # Normalize line endings
            code = code.replace('\r\n', '\n').replace('\r', '\n')
            
            # Remove trailing whitespace from lines
            lines = [line.rstrip() for line in code.split('\n')]
            code = '\n'.join(lines)
            
            return code.strip()
        
        # Apply code normalization
        df['code'] = df['code'].apply(normalize_code)
        
        # Recalculate hash after cleaning
        df['code_hash'] = df['code'].apply(self._calculate_hash)
        
        return df
    
    def get_cleaning_stats(self) -> Dict:
        """
        Get statistics about the cleaning process.
        
        Returns:
            Dictionary with cleaning statistics
        """
        return self.cleaning_stats.copy()
    
    def _calculate_hash(self, code: str) -> str:
        """Calculate SHA-256 hash of contract code."""
        if not isinstance(code, str):
            return ""
        return hashlib.sha256(code.encode('utf-8')).hexdigest()
    
    def _log_cleaning_stats(self):
        """Log detailed cleaning statistics."""
        stats = self.cleaning_stats
        logger.info("Data cleaning statistics:")
        logger.info(f"  Original contracts: {stats['original_count']}")
        logger.info(f"  Removed duplicates: {stats['removed_duplicates']}")
        logger.info(f"  Removed encoding issues: {stats['removed_encoding_issues']}")
        logger.info(f"  Removed invalid syntax: {stats['removed_invalid_syntax']}")
        logger.info(f"  Removed size filter: {stats['removed_size_filter']}")
        logger.info(f"  Final contracts: {stats['final_count']}")
        
        if stats['original_count'] > 0:
            retention_rate = (stats['final_count'] / stats['original_count']) * 100
            logger.info(f"  Retention rate: {retention_rate:.1f}%")


def main():
    """Example usage of DataCleaner."""
    # Load sample data
    from .data_loader import ContractDataLoader
    
    loader = ContractDataLoader("data/raw")
    df = loader.load_contracts()
    
    print(f"Original dataset: {len(df)} contracts")
    
    # Clean the data
    cleaner = DataCleaner(min_lines=10, max_lines=1000)
    df_clean = cleaner.clean_dataset(df)
    
    print(f"Cleaned dataset: {len(df_clean)} contracts")
    
    # Get cleaning statistics
    stats = cleaner.get_cleaning_stats()
    print("\nCleaning Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Save cleaned data
    output_path = "data/processed/cleaned_contracts.csv"
    df_clean.to_csv(output_path, index=False)
    print(f"\nSaved cleaned data to {output_path}")


if __name__ == "__main__":
    main()