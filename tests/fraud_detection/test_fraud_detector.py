"""
Unit tests for FraudDetector
"""

import numpy as np
import pytest
import tempfile
import os

from src.fraud_detection.fraud_detector import FraudDetector


@pytest.fixture
def detector():
    """Create a fraud detector instance."""
    return FraudDetector()


@pytest.fixture
def sample_training_data():
    """Create sample training data."""
    np.random.seed(42)
    X_train = np.random.randn(100, 15)
    y_train = np.random.randint(0, 2, 100)
    return X_train, y_train


@pytest.fixture
def sample_test_data():
    """Create sample test data."""
    np.random.seed(43)
    X_test = np.random.randn(20, 15)
    y_test = np.random.randint(0, 2, 20)
    return X_test, y_test


class TestFraudDetector:
    """Test cases for FraudDetector."""

    def test_initialization(self, detector):
        """Test detector initialization."""
        assert detector.trained is False
        assert detector.n_estimators == 100
        assert detector.max_depth == 20

    def test_train(self, detector, sample_training_data):
        """Test model training."""
        X_train, y_train = sample_training_data
        result = detector.train(X_train, y_train)
        
        assert result['status'] == 'success'
        assert 'train_accuracy' in result
        assert detector.trained is True
        assert result['train_accuracy'] > 0

    def test_predict(self, detector, sample_training_data, sample_test_data):
        """Test prediction."""
        X_train, y_train = sample_training_data
        X_test, _ = sample_test_data
        
        detector.train(X_train, y_train)
        predictions = detector.predict(X_test)
        
        assert len(predictions) == len(X_test)
        assert all(p in [0, 1] for p in predictions)

    def test_predict_proba(self, detector, sample_training_data, sample_test_data):
        """Test probability prediction."""
        X_train, y_train = sample_training_data
        X_test, _ = sample_test_data
        
        detector.train(X_train, y_train)
        probabilities = detector.predict_proba(X_test)
        
        assert probabilities.shape == (len(X_test), 2)
        assert np.all(probabilities >= 0)
        assert np.all(probabilities <= 1)
        assert np.allclose(probabilities.sum(axis=1), 1.0)

    def test_predict_without_training(self, detector, sample_test_data):
        """Test error when predicting without training."""
        X_test, _ = sample_test_data
        
        with pytest.raises(ValueError, match="Model must be trained"):
            detector.predict(X_test)

    def test_get_feature_importance(self, detector, sample_training_data):
        """Test getting feature importance."""
        X_train, y_train = sample_training_data
        
        detector.train(X_train, y_train)
        detector.set_feature_names([f'feature_{i}' for i in range(15)])
        importance = detector.get_feature_importance()
        
        assert isinstance(importance, dict)
        assert len(importance) == 15
        # Check that importances sum to approximately 1.0
        assert abs(sum(importance.values()) - 1.0) < 0.01

    def test_save_and_load_model(self, detector, sample_training_data):
        """Test saving and loading model."""
        X_train, y_train = sample_training_data
        
        # Train model
        detector.train(X_train, y_train)
        detector.set_feature_names([f'feature_{i}' for i in range(15)])
        
        # Save
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
            temp_path = f.name
        
        try:
            detector.save_model(temp_path)
            
            # Load into new detector
            new_detector = FraudDetector()
            new_detector.load_model(temp_path)
            
            assert new_detector.trained is True
            assert new_detector.feature_names == detector.feature_names
        finally:
            os.unlink(temp_path)

    def test_set_feature_names(self, detector):
        """Test setting feature names."""
        feature_names = [f'feature_{i}' for i in range(15)]
        detector.set_feature_names(feature_names)
        
        assert detector.feature_names == feature_names

    def test_get_model_info(self, detector, sample_training_data):
        """Test getting model information."""
        X_train, y_train = sample_training_data
        
        detector.train(X_train, y_train)
        info = detector.get_model_info()
        
        assert info['trained'] is True
        assert 'classes' in info
        assert 'n_estimators' in info

    def test_empty_training_data(self, detector):
        """Test error with empty training data."""
        X_train = np.array([]).reshape(0, 15)
        y_train = np.array([])
        
        with pytest.raises(ValueError, match="Training data cannot be empty"):
            detector.train(X_train, y_train)

    def test_mismatched_data_shapes(self, detector):
        """Test error with mismatched data shapes."""
        X_train = np.random.randn(100, 15)
        y_train = np.random.randint(0, 2, 50)  # Wrong size
        
        with pytest.raises(ValueError, match="must have the same number of samples"):
            detector.train(X_train, y_train)
