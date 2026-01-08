"""
Fraud Detection Model Module

Implements the RandomForest-based fraud detection classifier.
"""

import logging
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from .config import N_ESTIMATORS, MAX_DEPTH, CLASS_WEIGHT, RANDOM_STATE, MIN_ACCURACY

logger = logging.getLogger(__name__)


class FraudDetector:
    """
    RandomForest-based fraud detection classifier.
    
    Supports binary and multi-class classification with class weighting for imbalanced data.
    """

    def __init__(self, n_estimators: int = N_ESTIMATORS, max_depth: int = MAX_DEPTH):
        """
        Initialize the fraud detector.
        
        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of trees
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight=CLASS_WEIGHT,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        self.trained = False
        self.feature_names = None
        self.classes = None

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict:
        """
        Train the fraud detection model.
        
        Args:
            X_train: Training features (n_samples, n_features)
            y_train: Training labels (n_samples,)
            
        Returns:
            Dictionary with training results
            
        Raises:
            ValueError: If training data is invalid
        """
        if len(X_train) == 0:
            raise ValueError("Training data cannot be empty")
        
        if len(X_train) != len(y_train):
            raise ValueError("X_train and y_train must have the same number of samples")
        
        logger.info(f"Training RandomForest model with {len(X_train)} samples")
        
        # Train the model
        self.model.fit(X_train, y_train)
        self.trained = True
        self.classes = np.unique(y_train)
        
        # Calculate training accuracy
        y_pred = self.model.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_pred)
        
        logger.info(f"Model trained. Training accuracy: {train_accuracy:.4f}")
        
        if train_accuracy < MIN_ACCURACY:
            logger.warning(f"Training accuracy {train_accuracy:.4f} is below minimum {MIN_ACCURACY}")
        
        return {
            "status": "success",
            "train_accuracy": train_accuracy,
            "n_samples": len(X_train),
            "n_features": X_train.shape[1],
            "classes": self.classes.tolist(),
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict fraud labels for transactions.
        
        Args:
            X: Features (n_samples, n_features)
            
        Returns:
            Predicted labels (n_samples,)
            
        Raises:
            ValueError: If model is not trained
        """
        if not self.trained:
            raise ValueError("Model must be trained before making predictions")
        
        predictions = self.model.predict(X)
        return predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict fraud probabilities for transactions.
        
        Args:
            X: Features (n_samples, n_features)
            
        Returns:
            Probability predictions (n_samples, n_classes)
            
        Raises:
            ValueError: If model is not trained
        """
        if not self.trained:
            raise ValueError("Model must be trained before making predictions")
        
        probabilities = self.model.predict_proba(X)
        return probabilities

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores from the trained model.
        
        Returns:
            Dictionary mapping feature names to importance scores
            
        Raises:
            ValueError: If model is not trained
        """
        if not self.trained:
            raise ValueError("Model must be trained before getting feature importance")
        
        importances = self.model.feature_importances_
        
        # Normalize to sum to 1.0
        importances_normalized = importances / importances.sum()
        
        # Create feature importance dictionary
        if self.feature_names is not None:
            feature_importance = dict(zip(self.feature_names, importances_normalized))
        else:
            feature_importance = {f"feature_{i}": imp for i, imp in enumerate(importances_normalized)}
        
        return feature_importance

    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
            
        Raises:
            ValueError: If model is not trained
        """
        if not self.trained:
            raise ValueError("Cannot save untrained model")
        
        model_data = {
            "model": self.model,
            "trained": self.trained,
            "feature_names": self.feature_names,
            "classes": self.classes,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to load the model from
        """
        model_data = joblib.load(filepath)
        
        self.model = model_data["model"]
        self.trained = model_data["trained"]
        self.feature_names = model_data["feature_names"]
        self.classes = model_data["classes"]
        self.n_estimators = model_data["n_estimators"]
        self.max_depth = model_data["max_depth"]
        
        logger.info(f"Model loaded from {filepath}")

    def set_feature_names(self, feature_names: list) -> None:
        """
        Set feature names for the model.
        
        Args:
            feature_names: List of feature names
        """
        self.feature_names = feature_names
        logger.info(f"Set {len(feature_names)} feature names")

    def get_model_info(self) -> Dict:
        """
        Get information about the model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "trained": self.trained,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "classes": self.classes.tolist() if self.classes is not None else None,
            "feature_names": self.feature_names,
        }
