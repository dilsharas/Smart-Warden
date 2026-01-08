"""
Unit tests for DataPreprocessor
"""

import numpy as np
import pandas as pd
import pytest

from src.fraud_detection.preprocessor import DataPreprocessor


@pytest.fixture
def preprocessor():
    """Create a preprocessor instance."""
    return DataPreprocessor()


@pytest.fixture
def sample_data():
    """Create sample transaction data."""
    return pd.DataFrame({
        'sender': ['0x123', '0x456', '0x789', '0x123'],
        'receiver': ['0xabc', '0xdef', '0xghi', '0xabc'],
        'value': [100.0, 200.0, 150.0, 120.0],
        'gas_used': [21000, 21000, 21000, 21000],
        'timestamp': [1000000, 1000001, 1000002, 1000003],
        'category': ['A', 'B', 'A', 'B'],
    })


class TestDataPreprocessor:
    """Test cases for DataPreprocessor."""

    def test_handle_missing_values_drop(self, sample_data):
        """Test handling missing values with drop strategy."""
        preprocessor = DataPreprocessor(missing_strategy='drop')
        df = sample_data.copy()
        df.loc[0, 'value'] = np.nan
        
        result = preprocessor.handle_missing_values(df)
        assert len(result) == 3  # One row dropped

    def test_handle_missing_values_mean(self, sample_data):
        """Test handling missing values with mean strategy."""
        preprocessor = DataPreprocessor(missing_strategy='mean')
        df = sample_data.copy()
        df.loc[0, 'value'] = np.nan
        
        result = preprocessor.handle_missing_values(df)
        assert len(result) == 4  # No rows dropped
        assert not result['value'].isnull().any()

    def test_normalize_features(self, preprocessor, sample_data):
        """Test feature normalization."""
        result = preprocessor.normalize_features(sample_data, fit=True)
        
        # Check that values are in [0, 1] range
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            assert result[col].min() >= 0
            assert result[col].max() <= 1

    def test_encode_categorical(self, preprocessor, sample_data):
        """Test categorical encoding."""
        result = preprocessor.encode_categorical(sample_data, fit=True)
        
        # Check that categorical columns are now numeric
        assert result['category'].dtype in [np.int64, np.int32]
        assert result['sender'].dtype in [np.int64, np.int32]

    def test_preprocess_complete(self, preprocessor, sample_data):
        """Test complete preprocessing pipeline."""
        result = preprocessor.preprocess(sample_data, fit=True)
        
        # Check shape is preserved
        assert len(result) == len(sample_data)
        
        # Check all values are numeric
        assert result.select_dtypes(include=[np.number]).shape[1] == len(result.columns)
        
        # Check no NaN values
        assert not result.isnull().any().any()

    def test_preprocessor_fitted_flag(self, preprocessor, sample_data):
        """Test that fitted flag is set correctly."""
        assert preprocessor.fitted is False
        preprocessor.preprocess(sample_data, fit=True)
        assert preprocessor.fitted is True

    def test_get_preprocessing_info(self, preprocessor, sample_data):
        """Test getting preprocessing information."""
        preprocessor.preprocess(sample_data, fit=True)
        info = preprocessor.get_preprocessing_info()
        
        assert info['fitted'] is True
        assert 'numeric_columns' in info
        assert 'categorical_columns' in info

    def test_save_and_load_preprocessor(self, preprocessor, sample_data):
        """Test saving and loading preprocessor state."""
        import tempfile
        import os
        
        # Fit preprocessor
        preprocessor.preprocess(sample_data, fit=True)
        
        # Save
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
            temp_path = f.name
        
        try:
            preprocessor.save_preprocessor(temp_path)
            
            # Load into new preprocessor
            new_preprocessor = DataPreprocessor()
            new_preprocessor.load_preprocessor(temp_path)
            
            assert new_preprocessor.fitted is True
            assert new_preprocessor.numeric_columns == preprocessor.numeric_columns
        finally:
            os.unlink(temp_path)
