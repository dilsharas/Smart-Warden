"""
Unit tests for TransactionFeatureExtractor
"""

import numpy as np
import pandas as pd
import pytest

from src.fraud_detection.feature_extractor import TransactionFeatureExtractor


@pytest.fixture
def extractor():
    """Create a feature extractor instance."""
    return TransactionFeatureExtractor()


@pytest.fixture
def sample_transactions():
    """Create sample transaction data."""
    return pd.DataFrame({
        'sender': ['0x123', '0x456', '0x789', '0x123', '0x456'],
        'receiver': ['0xabc', '0xdef', '0xghi', '0xabc', '0xdef'],
        'value': [100.0, 200.0, 150.0, 120.0, 180.0],
        'gas_used': [21000, 21000, 21000, 21000, 21000],
        'timestamp': [1000000, 1000001, 1000002, 1000003, 1000004],
    })


class TestTransactionFeatureExtractor:
    """Test cases for TransactionFeatureExtractor."""

    def test_extract_transaction_value_stats(self, extractor, sample_transactions):
        """Test transaction value statistics extraction."""
        features = extractor.extract_transaction_value_stats(sample_transactions)
        
        assert 'transaction_value' in features.columns
        assert 'value_mean' in features.columns
        assert 'value_std' in features.columns
        assert len(features) == len(sample_transactions)

    def test_extract_transaction_frequency(self, extractor, sample_transactions):
        """Test transaction frequency extraction."""
        features = extractor.extract_transaction_frequency(sample_transactions)
        
        assert 'transaction_frequency' in features.columns
        assert len(features) == len(sample_transactions)

    def test_extract_gas_usage_patterns(self, extractor, sample_transactions):
        """Test gas usage pattern extraction."""
        features = extractor.extract_gas_usage_patterns(sample_transactions)
        
        assert 'avg_gas_used' in features.columns
        assert 'gas_anomaly_score' in features.columns
        assert len(features) == len(sample_transactions)

    def test_extract_time_intervals(self, extractor, sample_transactions):
        """Test time interval extraction."""
        features = extractor.extract_time_intervals(sample_transactions)
        
        assert 'time_interval_mean' in features.columns
        assert 'time_interval_std' in features.columns
        assert len(features) == len(sample_transactions)

    def test_extract_activity_levels(self, extractor, sample_transactions):
        """Test activity level extraction."""
        features = extractor.extract_activity_levels(sample_transactions)
        
        assert 'sender_activity_level' in features.columns
        assert 'receiver_activity_level' in features.columns
        assert len(features) == len(sample_transactions)

    def test_extract_value_to_gas_ratio(self, extractor, sample_transactions):
        """Test value-to-gas ratio extraction."""
        features = extractor.extract_value_to_gas_ratio(sample_transactions)
        
        assert 'value_to_gas_ratio' in features.columns
        assert len(features) == len(sample_transactions)

    def test_extract_temporal_features(self, extractor, sample_transactions):
        """Test temporal feature extraction."""
        features = extractor.extract_temporal_features(sample_transactions)
        
        assert 'hour_of_day' in features.columns
        assert 'day_of_week' in features.columns
        assert 'is_weekend' in features.columns
        assert len(features) == len(sample_transactions)

    def test_extract_interaction_count(self, extractor, sample_transactions):
        """Test interaction count extraction."""
        features = extractor.extract_interaction_count(sample_transactions)
        
        assert 'sender_receiver_interaction_count' in features.columns
        assert len(features) == len(sample_transactions)

    def test_extract_features_complete(self, extractor, sample_transactions):
        """Test complete feature extraction."""
        features = extractor.extract_features(sample_transactions)
        
        # Check shape
        assert len(features) == len(sample_transactions)
        assert features.shape[1] == extractor.feature_count
        
        # Check no NaN values
        assert not features.isnull().any().any()
        
        # Check all values are numeric
        assert features.select_dtypes(include=[np.number]).shape[1] == len(features.columns)

    def test_feature_dimensionality_consistency(self, extractor, sample_transactions):
        """Test that all transactions produce same dimensionality."""
        features = extractor.extract_features(sample_transactions)
        
        # All rows should have same number of features
        assert all(features.notna().sum(axis=1) == features.shape[1])

    def test_get_feature_names(self, extractor):
        """Test getting feature names."""
        names = extractor.get_feature_names()
        
        assert isinstance(names, list)
        assert len(names) == extractor.feature_count

    def test_get_feature_count(self, extractor):
        """Test getting feature count."""
        count = extractor.get_feature_count()
        
        assert count == 15
