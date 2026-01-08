#!/usr/bin/env python3
"""
Advanced Ensemble System with Model Distillation.
Implements ensemble classifier with dynamic weighting, confidence scoring, and model distillation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.base import BaseEstimator, ClassifierMixin
import joblib
import time
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

@dataclass
class EnsembleConfig:
    """Configuration for ensemble system."""
    # Base models
    include_random_forest: bool = True
    include_gradient_boosting: bool = True
    include_logistic_regression: bool = True
    include_svm: bool = False  # Slower for large datasets
    include_neural_network: bool = True
    
    # Ensemble settings
    voting_method: str = 'soft'  # 'hard', 'soft', 'weighted'
    weighting_strategy: str = 'performance'  # 'equal', 'performance', 'diversity'
    confidence_threshold: float = 0.7
    
    # Model distillation
    enable_distillation: bool = True
    distillation_temperature: float = 3.0
    distillation_alpha: float = 0.7
    student_model_type: str = 'random_forest'  # 'random_forest', 'logistic', 'neural_network'
    
    # Cross-validation
    cv_folds: int = 5
    random_state: int = 42

class BaseEnsembleMember(ABC):
    """Abstract base class for ensemble members."""
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train the model."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get model name."""
        pass

class RandomForestMember(BaseEnsembleMember):
    """Random Forest ensemble member."""
    
    def __init__(self, **kwargs):
        self.model = RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
            **kwargs
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)
    
    def get_name(self) -> str:
        return "RandomForest"

class GradientBoostingMember(BaseEnsembleMember):
    """Gradient Boosting ensemble member."""
    
    def __init__(self, **kwargs):
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            **kwargs
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)
    
    def get_name(self) -> str:
        return "GradientBoosting"

class LogisticRegressionMember(BaseEnsembleMember):
    """Logistic Regression ensemble member."""
    
    def __init__(self, **kwargs):
        self.model = LogisticRegression(
            class_weight='balanced',
            random_state=42,
            max_iter=1000,
            **kwargs
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)
    
    def get_name(self) -> str:
        return "LogisticRegression"

class NeuralNetworkMember(BaseEnsembleMember):
    """Neural Network ensemble member."""
    
    def __init__(self, **kwargs):
        self.model = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            alpha=0.001,
            learning_rate='adaptive',
            max_iter=500,
            random_state=42,
            **kwargs
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)
    
    def get_name(self) -> str:
        return "NeuralNetwork"

class SVMMember(BaseEnsembleMember):
    """SVM ensemble member."""
    
    def __init__(self, **kwargs):
        self.model = SVC(
            kernel='rbf',
            class_weight='balanced',
            probability=True,
            random_state=42,
            **kwargs
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)
    
    def get_name(self) -> str:
        return "SVM"

class AdvancedEnsembleClassifier(BaseEstimator, ClassifierMixin):
    """
    Advanced ensemble classifier with dynamic weighting and confidence scoring.
    """
    
    def __init__(self, config: EnsembleConfig = None):
        """
        Initialize the ensemble classifier.
        
        Args:
            config: Ensemble configuration
        """
        self.config = config or EnsembleConfig()
        self.models = []
        self.model_weights = None
        self.model_performances = {}
        self.is_fitted = False
        
        # Initialize base models
        self._initialize_models()
        
        logger.info(f"Initialized AdvancedEnsembleClassifier with {len(self.models)} models")
    
    def _initialize_models(self):
        """Initialize base models based on configuration."""
        if self.config.include_random_forest:
            self.models.append(RandomForestMember())
        
        if self.config.include_gradient_boosting:
            self.models.append(GradientBoostingMember())
        
        if self.config.include_logistic_regression:
            self.models.append(LogisticRegressionMember())
        
        if self.config.include_neural_network:
            self.models.append(NeuralNetworkMember())
        
        if self.config.include_svm:
            self.models.append(SVMMember())
    
    def fit(self, X: np.ndarray, y: np.ndarray, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
        """
        Train all ensemble members and calculate weights.
        
        Args:
            X: Training features
            y: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
        """
        logger.info(f"🚀 Training ensemble with {len(self.models)} models...")
        
        # Train each model
        for i, model in enumerate(self.models):
            logger.info(f"Training {model.get_name()}...")
            start_time = time.time()
            
            try:
                model.fit(X, y)
                training_time = time.time() - start_time
                
                # Evaluate model performance
                if X_val is not None and y_val is not None:
                    val_predictions = model.predict(X_val)
                    val_probabilities = model.predict_proba(X_val)[:, 1]
                    
                    performance = {
                        'accuracy': accuracy_score(y_val, val_predictions),
                        'f1_score': f1_score(y_val, val_predictions, average='binary'),
                        'roc_auc': roc_auc_score(y_val, val_probabilities),
                        'training_time': training_time
                    }
                else:
                    # Use cross-validation for performance estimation
                    cv_scores = cross_val_score(
                        model.model, X, y, 
                        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
                        scoring='f1'
                    )
                    
                    performance = {
                        'accuracy': cv_scores.mean(),
                        'f1_score': cv_scores.mean(),
                        'roc_auc': cv_scores.mean(),  # Approximation
                        'training_time': training_time
                    }
                
                self.model_performances[model.get_name()] = performance
                
                logger.info(f"✅ {model.get_name()} trained - F1: {performance['f1_score']:.3f}, "
                           f"Time: {training_time:.1f}s")
                
            except Exception as e:
                logger.error(f"❌ Failed to train {model.get_name()}: {e}")
                # Remove failed model
                self.models.pop(i)
        
        # Calculate model weights
        self._calculate_model_weights()
        
        self.is_fitted = True
        
        return self
    
    def _calculate_model_weights(self):
        """Calculate weights for ensemble members."""
        if self.config.weighting_strategy == 'equal':
            self.model_weights = np.ones(len(self.models)) / len(self.models)
        
        elif self.config.weighting_strategy == 'performance':
            # Weight by F1-score performance
            f1_scores = [self.model_performances[model.get_name()]['f1_score'] 
                        for model in self.models]
            f1_scores = np.array(f1_scores)
            
            # Normalize to sum to 1
            self.model_weights = f1_scores / f1_scores.sum()
        
        elif self.config.weighting_strategy == 'diversity':
            # Weight by diversity (simplified approach)
            # In practice, this would involve calculating prediction diversity
            self.model_weights = np.ones(len(self.models)) / len(self.models)
        
        logger.info(f"Model weights: {dict(zip([m.get_name() for m in self.models], self.model_weights))}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make ensemble predictions."""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before making predictions")
        
        if self.config.voting_method == 'hard':
            return self._predict_hard_voting(X)
        elif self.config.voting_method == 'soft':
            return self._predict_soft_voting(X)
        elif self.config.voting_method == 'weighted':
            return self._predict_weighted_voting(X)
        else:
            raise ValueError(f"Unknown voting method: {self.config.voting_method}")
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before making predictions")
        
        # Get probabilities from all models
        all_probabilities = []
        for model in self.models:
            probabilities = model.predict_proba(X)
            all_probabilities.append(probabilities)
        
        # Weighted average of probabilities
        ensemble_probabilities = np.zeros_like(all_probabilities[0])
        for i, probabilities in enumerate(all_probabilities):
            ensemble_probabilities += self.model_weights[i] * probabilities
        
        return ensemble_probabilities
    
    def predict_with_confidence(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with confidence scores.
        
        Args:
            X: Input features
            
        Returns:
            Tuple of (predictions, confidence_scores)
        """
        probabilities = self.predict_proba(X)
        predictions = np.argmax(probabilities, axis=1)
        
        # Calculate confidence as max probability
        confidence_scores = np.max(probabilities, axis=1)
        
        return predictions, confidence_scores
    
    def _predict_hard_voting(self, X: np.ndarray) -> np.ndarray:
        """Hard voting prediction."""
        all_predictions = []
        for model in self.models:
            predictions = model.predict(X)
            all_predictions.append(predictions)
        
        all_predictions = np.array(all_predictions)
        
        # Majority vote
        ensemble_predictions = []
        for i in range(X.shape[0]):
            votes = all_predictions[:, i]
            unique, counts = np.unique(votes, return_counts=True)
            majority_class = unique[np.argmax(counts)]
            ensemble_predictions.append(majority_class)
        
        return np.array(ensemble_predictions)
    
    def _predict_soft_voting(self, X: np.ndarray) -> np.ndarray:
        """Soft voting prediction."""
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)
    
    def _predict_weighted_voting(self, X: np.ndarray) -> np.ndarray:
        """Weighted voting prediction."""
        # Same as soft voting but with weights (already incorporated in predict_proba)
        return self._predict_soft_voting(X)
    
    def get_model_performances(self) -> Dict:
        """Get performance metrics for all models."""
        return self.model_performances
    
    def get_feature_importance(self) -> Dict[str, np.ndarray]:
        """Get feature importance from models that support it."""
        importance_dict = {}
        
        for model in self.models:
            if hasattr(model.model, 'feature_importances_'):
                importance_dict[model.get_name()] = model.model.feature_importances_
            elif hasattr(model.model, 'coef_'):
                # For linear models, use absolute coefficients
                importance_dict[model.get_name()] = np.abs(model.model.coef_[0])
        
        return importance_dict
    
    def save_ensemble(self, filepath: str):
        """Save the ensemble model."""
        ensemble_data = {
            'config': self.config,
            'models': self.models,
            'model_weights': self.model_weights,
            'model_performances': self.model_performances,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(ensemble_data, filepath)
        logger.info(f"Ensemble saved to {filepath}")
    
    def load_ensemble(self, filepath: str):
        """Load a saved ensemble model."""
        ensemble_data = joblib.load(filepath)
        
        self.config = ensemble_data['config']
        self.models = ensemble_data['models']
        self.model_weights = ensemble_data['model_weights']
        self.model_performances = ensemble_data['model_performances']
        self.is_fitted = ensemble_data['is_fitted']
        
        logger.info(f"Ensemble loaded from {filepath}")

class ModelDistiller:
    """
    Model distillation system to create lightweight models from ensemble.
    """
    
    def __init__(self, temperature: float = 3.0, alpha: float = 0.7):
        """
        Initialize the model distiller.
        
        Args:
            temperature: Temperature for softmax in distillation
            alpha: Weight for distillation loss vs hard target loss
        """
        self.temperature = temperature
        self.alpha = alpha
        self.student_model = None
        self.teacher_model = None
        
    def distill_model(self, 
                     teacher_model: AdvancedEnsembleClassifier,
                     X_train: np.ndarray,
                     y_train: np.ndarray,
                     student_model_type: str = 'random_forest') -> BaseEstimator:
        """
        Distill knowledge from teacher ensemble to student model.
        
        Args:
            teacher_model: Trained ensemble model (teacher)
            X_train: Training features
            y_train: Training labels
            student_model_type: Type of student model
            
        Returns:
            Trained student model
        """
        logger.info(f"🎓 Distilling ensemble knowledge to {student_model_type} model...")
        
        self.teacher_model = teacher_model
        
        # Get soft targets from teacher
        soft_targets = teacher_model.predict_proba(X_train)
        
        # Initialize student model
        if student_model_type == 'random_forest':
            self.student_model = RandomForestClassifier(
                n_estimators=50,  # Smaller than teacher components
                max_depth=10,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
        elif student_model_type == 'logistic':
            self.student_model = LogisticRegression(
                class_weight='balanced',
                random_state=42,
                max_iter=1000
            )
        elif student_model_type == 'neural_network':
            self.student_model = MLPClassifier(
                hidden_layer_sizes=(50,),  # Smaller than teacher
                activation='relu',
                solver='adam',
                alpha=0.001,
                max_iter=300,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown student model type: {student_model_type}")
        
        # For sklearn models, we use a simplified distillation approach
        # Train on soft targets by using them as sample weights
        if hasattr(self.student_model, 'fit'):
            # Use confidence scores as sample weights
            confidence_scores = np.max(soft_targets, axis=1)
            
            # Train student model
            self.student_model.fit(X_train, y_train, sample_weight=confidence_scores)
        
        logger.info("✅ Model distillation complete")
        
        return self.student_model
    
    def evaluate_distillation(self, 
                            X_test: np.ndarray, 
                            y_test: np.ndarray) -> Dict:
        """
        Evaluate the distilled model against the teacher.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Comparison metrics
        """
        if self.student_model is None or self.teacher_model is None:
            raise ValueError("Both teacher and student models must be available")
        
        # Teacher predictions
        teacher_predictions = self.teacher_model.predict(X_test)
        teacher_probabilities = self.teacher_model.predict_proba(X_test)[:, 1]
        
        # Student predictions
        student_predictions = self.student_model.predict(X_test)
        student_probabilities = self.student_model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        teacher_metrics = {
            'accuracy': accuracy_score(y_test, teacher_predictions),
            'f1_score': f1_score(y_test, teacher_predictions, average='binary'),
            'roc_auc': roc_auc_score(y_test, teacher_probabilities)
        }
        
        student_metrics = {
            'accuracy': accuracy_score(y_test, student_predictions),
            'f1_score': f1_score(y_test, student_predictions, average='binary'),
            'roc_auc': roc_auc_score(y_test, student_probabilities)
        }
        
        # Calculate performance retention
        performance_retention = {
            'accuracy_retention': student_metrics['accuracy'] / teacher_metrics['accuracy'],
            'f1_retention': student_metrics['f1_score'] / teacher_metrics['f1_score'],
            'roc_auc_retention': student_metrics['roc_auc'] / teacher_metrics['roc_auc']
        }
        
        return {
            'teacher_metrics': teacher_metrics,
            'student_metrics': student_metrics,
            'performance_retention': performance_retention
        }


def main():
    """Example usage of AdvancedEnsembleClassifier and ModelDistiller."""
    # Create sample data
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    X, y = make_classification(
        n_samples=2000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_clusters_per_class=1,
        weights=[0.7, 0.3],
        random_state=42
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
    )
    
    # Configure ensemble
    config = EnsembleConfig(
        include_random_forest=True,
        include_gradient_boosting=True,
        include_logistic_regression=True,
        include_neural_network=True,
        voting_method='weighted',
        weighting_strategy='performance',
        enable_distillation=True
    )
    
    # Train ensemble
    ensemble = AdvancedEnsembleClassifier(config)
    ensemble.fit(X_train, y_train, X_val, y_val)
    
    # Make predictions
    predictions = ensemble.predict(X_test)
    probabilities = ensemble.predict_proba(X_test)
    predictions_with_conf, confidence_scores = ensemble.predict_with_confidence(X_test)
    
    # Evaluate ensemble
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average='binary')
    roc_auc = roc_auc_score(y_test, probabilities[:, 1])
    
    print("🎯 Ensemble Results:")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"F1-Score: {f1:.3f}")
    print(f"ROC-AUC: {roc_auc:.3f}")
    print(f"Average Confidence: {confidence_scores.mean():.3f}")
    
    # Show model performances
    performances = ensemble.get_model_performances()
    print("\n📊 Individual Model Performances:")
    for model_name, perf in performances.items():
        print(f"{model_name}: F1={perf['f1_score']:.3f}, Time={perf['training_time']:.1f}s")
    
    # Model distillation
    print("\n🎓 Performing model distillation...")
    distiller = ModelDistiller(temperature=3.0, alpha=0.7)
    student_model = distiller.distill_model(ensemble, X_train, y_train, 'random_forest')
    
    # Evaluate distillation
    distillation_results = distiller.evaluate_distillation(X_test, y_test)
    
    print("📈 Distillation Results:")
    print(f"Teacher Accuracy: {distillation_results['teacher_metrics']['accuracy']:.3f}")
    print(f"Student Accuracy: {distillation_results['student_metrics']['accuracy']:.3f}")
    print(f"Accuracy Retention: {distillation_results['performance_retention']['accuracy_retention']:.1%}")
    
    # Save models
    ensemble.save_ensemble("models/advanced_ensemble.pkl")
    joblib.dump(student_model, "models/distilled_student.pkl")
    
    print("\n✅ Ensemble and distilled models saved!")


if __name__ == "__main__":
    main()