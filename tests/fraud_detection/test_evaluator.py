"""
Unit tests for ModelEvaluator
"""

import numpy as np
import pytest

from src.fraud_detection.evaluator import ModelEvaluator


@pytest.fixture
def evaluator():
    """Create an evaluator instance."""
    return ModelEvaluator()


@pytest.fixture
def sample_predictions():
    """Create sample predictions."""
    np.random.seed(42)
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 0, 1, 1, 1, 0, 1])
    y_proba = np.random.rand(10, 2)
    y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)
    return y_true, y_pred, y_proba


class TestModelEvaluator:
    """Test cases for ModelEvaluator."""

    def test_calculate_accuracy(self, evaluator, sample_predictions):
        """Test accuracy calculation."""
        y_true, y_pred, _ = sample_predictions
        accuracy = evaluator.calculate_accuracy(y_true, y_pred)
        
        assert 0 <= accuracy <= 1
        assert 'accuracy' in evaluator.metrics

    def test_calculate_precision(self, evaluator, sample_predictions):
        """Test precision calculation."""
        y_true, y_pred, _ = sample_predictions
        precision = evaluator.calculate_precision(y_true, y_pred)
        
        assert 0 <= precision <= 1
        assert 'precision' in evaluator.metrics

    def test_calculate_recall(self, evaluator, sample_predictions):
        """Test recall calculation."""
        y_true, y_pred, _ = sample_predictions
        recall = evaluator.calculate_recall(y_true, y_pred)
        
        assert 0 <= recall <= 1
        assert 'recall' in evaluator.metrics

    def test_calculate_f1_score(self, evaluator, sample_predictions):
        """Test F1 score calculation."""
        y_true, y_pred, _ = sample_predictions
        f1 = evaluator.calculate_f1_score(y_true, y_pred)
        
        assert 0 <= f1 <= 1
        assert 'f1_score' in evaluator.metrics

    def test_calculate_roc_auc(self, evaluator, sample_predictions):
        """Test ROC-AUC calculation."""
        y_true, _, y_proba = sample_predictions
        roc_auc = evaluator.calculate_roc_auc(y_true, y_proba)
        
        assert 0 <= roc_auc <= 1
        assert 'roc_auc' in evaluator.metrics

    def test_generate_confusion_matrix(self, evaluator, sample_predictions):
        """Test confusion matrix generation."""
        y_true, y_pred, _ = sample_predictions
        cm = evaluator.generate_confusion_matrix(y_true, y_pred)
        
        assert cm.shape == (2, 2)
        assert np.all(cm >= 0)
        assert cm.sum() == len(y_true)

    def test_get_confusion_matrix_stats(self, evaluator, sample_predictions):
        """Test confusion matrix statistics."""
        y_true, y_pred, _ = sample_predictions
        cm = evaluator.generate_confusion_matrix(y_true, y_pred)
        stats = evaluator.get_confusion_matrix_stats(cm)
        
        assert 'true_negatives' in stats
        assert 'false_positives' in stats
        assert 'false_negatives' in stats
        assert 'true_positives' in stats

    def test_measure_latency(self, evaluator):
        """Test latency measurement."""
        def dummy_predict(X):
            return np.zeros(len(X))
        
        X = np.random.randn(10, 15)
        latency_stats = evaluator.measure_latency(dummy_predict, X, iterations=10)
        
        assert 'mean_ms' in latency_stats
        assert 'std_ms' in latency_stats
        assert 'p50_ms' in latency_stats
        assert 'p95_ms' in latency_stats
        assert 'p99_ms' in latency_stats
        assert latency_stats['mean_ms'] > 0

    def test_get_metrics_summary(self, evaluator, sample_predictions):
        """Test getting metrics summary."""
        y_true, y_pred, _ = sample_predictions
        evaluator.calculate_accuracy(y_true, y_pred)
        evaluator.calculate_precision(y_true, y_pred)
        
        summary = evaluator.get_metrics_summary()
        
        assert 'accuracy' in summary
        assert 'precision' in summary

    def test_get_evaluation_report(self, evaluator, sample_predictions):
        """Test evaluation report generation."""
        y_true, y_pred, y_proba = sample_predictions
        
        evaluator.calculate_accuracy(y_true, y_pred)
        evaluator.calculate_precision(y_true, y_pred)
        evaluator.calculate_recall(y_true, y_pred)
        evaluator.calculate_f1_score(y_true, y_pred)
        evaluator.calculate_roc_auc(y_true, y_proba)
        evaluator.generate_confusion_matrix(y_true, y_pred)
        
        report = evaluator.get_evaluation_report()
        
        assert isinstance(report, str)
        assert 'ACCURACY' in report
        assert 'PRECISION' in report
        assert 'RECALL' in report
