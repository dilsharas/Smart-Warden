"""
Blockchain Fraud Detection Module

This module provides transaction-level fraud detection using supervised machine learning.
It integrates with the Smart Contract AI Analyzer to provide comprehensive blockchain security analysis.
"""

__version__ = "1.0.0"
__author__ = "Smart Contract AI Analyzer Team"

from .data_loader import TransactionDataLoader
from .preprocessor import DataPreprocessor
from .feature_extractor import TransactionFeatureExtractor
from .fraud_detector import FraudDetector
from .evaluator import ModelEvaluator
from .visualization import VisualizationEngine
from .report_generator import ReportGenerator

__all__ = [
    "TransactionDataLoader",
    "DataPreprocessor",
    "TransactionFeatureExtractor",
    "FraudDetector",
    "ModelEvaluator",
    "VisualizationEngine",
    "ReportGenerator",
]
