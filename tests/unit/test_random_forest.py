"""
Unit tests for the RandomForestVulnerabilityDetector class.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from sklearn.metrics import accuracy_score
from models.random_forest import RandomForestVulnerabilityDetector


class TestRandomForestVulnerabilityDetector:
    """Test cases for RandomForestVulnerabilityDetector."""
    
    def test_initialization(self):
        """Test model initialization with default parameters."""
        detector = RandomForestVulnerabilityDetector()
        
        assert detector.model is not None
        assert detector.scaler is not None
        assert detector.label_encoder is not None
        assert detector.is_trained == False
        assert detector.feature_columns is None
    
    def test_initialization_with_custom_params(self):
        """Test model initialization with custom parameters."""
        detector = RandomForestVulnerabilityDetector(
            n_estimators=50,
            max_depth=10,
            random_state=123
        )
        
        assert detector.model.n_estimators == 50
        assert detector.model.max_depth == 10
        assert detector.model.random_state == 123
    
    def test_train_with_synthetic_data(self, binary_classifier, synthetic_ml_dataset):
        """Test training with synthetic data."""
        X = synthetic_ml_dataset['X']
        y = synthetic_ml_dataset['y_binary']
        
        # Split data
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Train model
        training_results = binary_classifier.train(X_train, y_train, X_val, y_val)
        
        assert binary_classifier.is_trained == True
        assert binary_classifier.feature_columns == X_train.columns.tolist()
        assert 'training_time' in training_results
        assert 'train_accuracy' in training_results
        assert 'val_accuracy' in training_results
        assert training_results['train_accuracy'] >= 0.0
        assert training_results['val_accuracy'] >= 0.0
    
    def test_train_without_validation(self, binary_classifier, synthetic_ml_dataset):
        """Test training without validation data."""
        X = synthetic_ml_dataset['X']
        y = synthetic_ml_dataset['y_binary']
        
        training_results = binary_classifier.train(X, y)
        
        assert binary_classifier.is_trained == True
        assert training_results['val_accuracy'] is None
    
    def test_predict_before_training(self, binary_classifier, synthetic_ml_dataset):
        """Test that prediction fails before training."""
        X = synthetic_ml_dataset['X']
        
        with pytest.raises(ValueError, match="Model must be trained"):
            binary_classifier.predict(X)
    
    def test_predict_after_training(self, trained_binary_model):
        """Test prediction after training."""
        model = trained_binary_model['model']
        X_test = trained_binary_model['X_test']
        
        predictions, probabilities = model.predict(X_test)
        
        assert len(predictions) == len(X_test)
        assert len(probabilities) == len(X_test)
        assert probabilities.shape[1] == 2  # Binary classification
        assert all(pred in ['safe', 'vulnerable'] for pred in predictions)
    
    def test_predict_single(self, trained_binary_model):
        """Test single sample prediction."""
        model = trained_binary_model['model']
        X_test = trained_binary_model['X_test']
        
        # Get first sample as dictionary
        sample_features = X_test.iloc[0].to_dict()
        
        prediction, confidence = model.predict_single(sample_features)
        
        assert prediction in ['safe', 'vulnerable']
        assert 0.0 <= confidence <= 1.0
    
    def test_evaluate(self, trained_binary_model):
        """Test model evaluation."""
        model = trained_binary_model['model']
        X_test = trained_binary_model['X_test']
        y_test = trained_binary_model['y_test']
        
        metrics = model.evaluate(X_test, y_test)
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 'roc_auc' in metrics
        assert 'confusion_matrix' in metrics
        assert 'predictions' in metrics
        assert 'probabilities' in metrics
        
        # Check metric ranges
        assert 0.0 <= metrics['accuracy'] <= 1.0
        assert 0.0 <= metrics['precision'] <= 1.0
        assert 0.0 <= metrics['recall'] <= 1.0
        assert 0.0 <= metrics['f1_score'] <= 1.0
        assert 0.0 <= metrics['roc_auc'] <= 1.0
    
    def test_get_feature_importance(self, trained_binary_model):
        """Test feature importance extraction."""
        model = trained_binary_model['model']
        
        importance_df = model.get_feature_importance(top_n=10)
        
        assert isinstance(importance_df, pd.DataFrame)
        assert len(importance_df) <= 10
        assert 'feature' in importance_df.columns
        assert 'importance' in importance_df.columns
        assert all(importance_df['importance'] >= 0)
        
        # Should be sorted by importance (descending)
        importances = importance_df['importance'].values
        assert all(importances[i] >= importances[i+1] for i in range(len(importances)-1))
    
    def test_get_feature_importance_before_training(self, binary_classifier):
        """Test that feature importance fails before training."""
        with pytest.raises(ValueError, match="Model must be trained"):
            binary_classifier.get_feature_importance()
    
    def test_hyperparameter_tuning(self, binary_classifier, synthetic_ml_dataset):
        """Test hyperparameter tuning."""
        X = synthetic_ml_dataset['X']
        y = synthetic_ml_dataset['y_binary']
        
        # Split data
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Run hyperparameter tuning (with limited grid for speed)
        results = binary_classifier.hyperparameter_tuning(X_train, y_train, X_val, y_val)
        
        assert 'best_params' in results
        assert 'best_score' in results
        assert 'cv_results' in results
        assert binary_classifier.is_trained == True
    
    def test_cross_validate(self, binary_classifier, synthetic_ml_dataset):
        """Test cross-validation."""
        X = synthetic_ml_dataset['X']
        y = synthetic_ml_dataset['y_binary']
        
        cv_results = binary_classifier.cross_validate(X, y, cv=3)
        
        assert 'cv_scores' in cv_results
        assert 'mean_score' in cv_results
        assert 'std_score' in cv_results
        assert 'min_score' in cv_results
        assert 'max_score' in cv_results
        
        assert len(cv_results['cv_scores']) == 3
        assert 0.0 <= cv_results['mean_score'] <= 1.0
    
    def test_explain_prediction(self, trained_binary_model):
        """Test prediction explanation."""
        model = trained_binary_model['model']
        X_test = trained_binary_model['X_test']
        
        # Get first sample as dictionary
        sample_features = X_test.iloc[0].to_dict()
        
        explanation = model.explain_prediction(sample_features, top_n=5)
        
        assert isinstance(explanation, dict)
        assert len(explanation) <= 5
        
        # All values should be non-negative (importance * feature value)
        assert all(value >= 0 for value in explanation.values())
    
    def test_save_and_load_model(self, trained_binary_model):
        """Test model saving and loading."""
        model = trained_binary_model['model']
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            model_path = f.name
        
        try:
            # Save model
            model.save_model(model_path)
            assert os.path.exists(model_path)
            
            # Load model
            loaded_model = RandomForestVulnerabilityDetector.load_model(model_path)
            
            assert loaded_model.is_trained == True
            assert loaded_model.feature_columns == model.feature_columns
            
            # Test that loaded model can make predictions
            X_test = trained_binary_model['X_test']
            predictions1, _ = model.predict(X_test)
            predictions2, _ = loaded_model.predict(X_test)
            
            # Predictions should be identical
            assert all(p1 == p2 for p1, p2 in zip(predictions1, predictions2))
            
        finally:
            # Cleanup
            if os.path.exists(model_path):
                os.unlink(model_path)
    
    def test_save_untrained_model(self, binary_classifier):
        """Test that saving untrained model fails."""
        with tempfile.NamedTemporaryFile(suffix='.pkl') as f:
            with pytest.raises(ValueError, match="Cannot save untrained model"):
                binary_classifier.save_model(f.name)
    
    def test_get_model_info(self, trained_binary_model):
        """Test getting model information."""
        model = trained_binary_model['model']
        
        info = model.get_model_info()
        
        assert 'is_trained' in info
        assert 'training_history' in info
        assert 'model_metadata' in info
        assert 'feature_count' in info
        
        assert info['is_trained'] == True
        assert info['feature_count'] > 0
    
    def test_get_model_info_untrained(self, binary_classifier):
        """Test getting model info for untrained model."""
        info = binary_classifier.get_model_info()
        
        assert info['is_trained'] == False
        assert info['feature_count'] == 0
    
    def test_predict_with_mismatched_features(self, trained_binary_model):
        """Test prediction with mismatched feature columns."""
        model = trained_binary_model['model']
        
        # Create DataFrame with different columns
        wrong_features = pd.DataFrame({
            'wrong_feature_1': [1, 2, 3],
            'wrong_feature_2': [4, 5, 6]
        })
        
        # Should handle gracefully (with warning)
        with pytest.warns(UserWarning):
            predictions, probabilities = model.predict(wrong_features)
    
    def test_string_labels(self, binary_classifier):
        """Test training with string labels."""
        # Create data with string labels
        X = pd.DataFrame(np.random.randn(50, 10), columns=[f'feature_{i}' for i in range(10)])
        y = pd.Series(['vulnerable', 'safe'] * 25)
        
        # Train model
        binary_classifier.train(X, y)
        
        # Make predictions
        predictions, probabilities = binary_classifier.predict(X[:5])
        
        assert all(pred in ['safe', 'vulnerable'] for pred in predictions)
    
    def test_numeric_labels(self, binary_classifier):
        """Test training with numeric labels."""
        # Create data with numeric labels
        X = pd.DataFrame(np.random.randn(50, 10), columns=[f'feature_{i}' for i in range(10)])
        y = pd.Series([0, 1] * 25)
        
        # Train model
        binary_classifier.train(X, y)
        
        # Make predictions
        predictions, probabilities = binary_classifier.predict(X[:5])
        
        # Predictions should be numeric when trained with numeric labels
        assert all(isinstance(pred, (int, np.integer)) for pred in predictions)
    
    def test_empty_dataset(self, binary_classifier):
        """Test training with empty dataset."""
        X = pd.DataFrame(columns=['feature_1', 'feature_2'])
        y = pd.Series(dtype=str)
        
        with pytest.raises(ValueError):
            binary_classifier.train(X, y)
    
    def test_single_class_dataset(self, binary_classifier):
        """Test training with single class dataset."""
        X = pd.DataFrame(np.random.randn(10, 5), columns=[f'feature_{i}' for i in range(5)])
        y = pd.Series(['safe'] * 10)  # All same class
        
        # Should train but may have warnings
        binary_classifier.train(X, y)
        
        # Should still be able to make predictions
        predictions, probabilities = binary_classifier.predict(X[:3])
        assert len(predictions) == 3