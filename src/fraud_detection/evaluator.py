"""
Model Evaluation Module

Calculates performance metrics and generates evaluation reports.
"""

import logging
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    auc,
)

from .config import LATENCY_THRESHOLD_MS, LATENCY_ITERATIONS

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Evaluates fraud detection model performance.
    
    Calculates metrics, confusion matrix, ROC-AUC, and latency measurements.
    """

    def __init__(self):
        """Initialize the evaluator."""
        self.metrics = {}
        self.confusion_matrix_result = None
        self.latency_measurements = []

    def calculate_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate accuracy score.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Accuracy score (0-1)
        """
        accuracy = accuracy_score(y_true, y_pred)
        self.metrics['accuracy'] = accuracy
        logger.info(f"Accuracy: {accuracy:.4f}")
        return accuracy

    def calculate_precision(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate precision score.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Precision score (0-1)
        """
        precision = precision_score(y_true, y_pred, zero_division=0)
        self.metrics['precision'] = precision
        logger.info(f"Precision: {precision:.4f}")
        return precision

    def calculate_recall(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate recall score.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Recall score (0-1)
        """
        recall = recall_score(y_true, y_pred, zero_division=0)
        self.metrics['recall'] = recall
        logger.info(f"Recall: {recall:.4f}")
        return recall

    def calculate_f1_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate F1 score.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            F1 score (0-1)
        """
        f1 = f1_score(y_true, y_pred, zero_division=0)
        self.metrics['f1_score'] = f1
        logger.info(f"F1-Score: {f1:.4f}")
        return f1

    def calculate_roc_auc(self, y_true: np.ndarray, y_proba: np.ndarray) -> float:
        """
        Calculate ROC-AUC score.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            
        Returns:
            ROC-AUC score (0-1)
        """
        try:
            # Handle multi-class case
            if len(np.unique(y_true)) > 2:
                roc_auc = roc_auc_score(y_true, y_proba, multi_class='ovr')
            else:
                # Binary case - use probability of positive class
                if y_proba.ndim > 1:
                    y_proba = y_proba[:, 1]
                roc_auc = roc_auc_score(y_true, y_proba)
            
            self.metrics['roc_auc'] = roc_auc
            logger.info(f"ROC-AUC: {roc_auc:.4f}")
            return roc_auc
        except Exception as e:
            logger.warning(f"Could not calculate ROC-AUC: {e}. Setting to 0.0")
            self.metrics['roc_auc'] = 0.0
            return 0.0

    def generate_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Generate confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Confusion matrix (2x2 for binary classification)
        """
        cm = confusion_matrix(y_true, y_pred)
        self.confusion_matrix_result = cm
        
        logger.info(f"Confusion Matrix:\n{cm}")
        return cm

    def get_confusion_matrix_stats(self, cm: np.ndarray) -> Dict:
        """
        Extract statistics from confusion matrix.
        
        Args:
            cm: Confusion matrix
            
        Returns:
            Dictionary with TP, FP, FN, TN
        """
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            return {
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp),
            }
        else:
            return {"matrix": cm.tolist()}

    def measure_latency(self, predict_func, X: np.ndarray, iterations: int = LATENCY_ITERATIONS) -> Dict:
        """
        Measure prediction latency.
        
        Args:
            predict_func: Function to measure (should accept X and return predictions)
            X: Input features
            iterations: Number of iterations for measurement
            
        Returns:
            Dictionary with latency statistics
        """
        latencies = []
        
        for _ in range(iterations):
            start_time = time.time()
            predict_func(X)
            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
        
        self.latency_measurements = latencies
        
        latency_stats = {
            "mean_ms": np.mean(latencies),
            "std_ms": np.std(latencies),
            "min_ms": np.min(latencies),
            "max_ms": np.max(latencies),
            "p50_ms": np.percentile(latencies, 50),
            "p95_ms": np.percentile(latencies, 95),
            "p99_ms": np.percentile(latencies, 99),
            "iterations": iterations,
        }
        
        self.metrics['latency'] = latency_stats
        
        logger.info(f"Latency - Mean: {latency_stats['mean_ms']:.2f}ms, "
                   f"P95: {latency_stats['p95_ms']:.2f}ms, "
                   f"P99: {latency_stats['p99_ms']:.2f}ms")
        
        if latency_stats['mean_ms'] > LATENCY_THRESHOLD_MS:
            logger.warning(f"Average latency {latency_stats['mean_ms']:.2f}ms exceeds threshold {LATENCY_THRESHOLD_MS}ms")
        
        return latency_stats

    def evaluate(self, model, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Perform complete model evaluation.
        
        Args:
            model: Trained model with predict and predict_proba methods
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary with all evaluation metrics
        """
        logger.info(f"Starting model evaluation on {len(X_test)} samples")
        
        # Get predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        
        # Calculate metrics
        self.calculate_accuracy(y_test, y_pred)
        self.calculate_precision(y_test, y_pred)
        self.calculate_recall(y_test, y_pred)
        self.calculate_f1_score(y_test, y_pred)
        self.calculate_roc_auc(y_test, y_proba)
        
        # Generate confusion matrix
        cm = self.generate_confusion_matrix(y_test, y_pred)
        cm_stats = self.get_confusion_matrix_stats(cm)
        
        # Measure latency
        latency_stats = self.measure_latency(model.predict, X_test)
        
        # Compile results
        results = {
            "metrics": self.metrics.copy(),
            "confusion_matrix": cm.tolist(),
            "confusion_matrix_stats": cm_stats,
            "latency": latency_stats,
        }
        
        logger.info("Model evaluation complete")
        return results

    def get_metrics_summary(self) -> Dict:
        """
        Get summary of all calculated metrics.
        
        Returns:
            Dictionary with metrics summary
        """
        return self.metrics.copy()

    def get_evaluation_report(self) -> str:
        """
        Generate a text report of evaluation results.
        
        Returns:
            Formatted evaluation report
        """
        report = "=" * 60 + "\n"
        report += "MODEL EVALUATION REPORT\n"
        report += "=" * 60 + "\n\n"
        
        report += "CLASSIFICATION METRICS:\n"
        report += "-" * 60 + "\n"
        for metric, value in self.metrics.items():
            if metric != 'latency':
                report += f"{metric.upper():20s}: {value:.4f}\n"
        
        if 'latency' in self.metrics:
            report += "\nLATENCY METRICS (ms):\n"
            report += "-" * 60 + "\n"
            latency = self.metrics['latency']
            report += f"{'Mean':20s}: {latency['mean_ms']:.2f}\n"
            report += f"{'Std Dev':20s}: {latency['std_ms']:.2f}\n"
            report += f"{'P50':20s}: {latency['p50_ms']:.2f}\n"
            report += f"{'P95':20s}: {latency['p95_ms']:.2f}\n"
            report += f"{'P99':20s}: {latency['p99_ms']:.2f}\n"
        
        if self.confusion_matrix_result is not None:
            report += "\nCONFUSION MATRIX:\n"
            report += "-" * 60 + "\n"
            report += str(self.confusion_matrix_result) + "\n"
        
        report += "=" * 60 + "\n"
        return report
