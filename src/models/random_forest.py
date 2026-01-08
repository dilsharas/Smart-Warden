"""
Random Forest vulnerability detector for smart contracts.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support, roc_auc_score, roc_curve
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class RandomForestVulnerabilityDetector:
    """
    Random Forest classifier for smart contract vulnerability detection.
    
    Features:
    - Binary classification (vulnerable vs safe)
    - Multi-class classification (specific vulnerability types)
    - Hyperparameter tuning with GridSearchCV
    - Feature importance analysis
    - Model persistence and loading
    - Comprehensive evaluation metrics
    """
    
    def __init__(self, 
                 n_estimators: int = 100,
                 max_depth: Optional[int] = 20,
                 min_samples_split: int = 5,
                 min_samples_leaf: int = 2,
                 max_features: str = 'sqrt',
                 class_weight: str = 'balanced',
                 random_state: int = 42,
                 n_jobs: int = -1):
        """
        Initialize the Random Forest vulnerability detector.
        
        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of trees
            min_samples_split: Minimum samples required to split a node
            min_samples_leaf: Minimum samples required at a leaf node
            max_features: Number of features to consider for best split
            class_weight: Weights associated with classes
            random_state: Random state for reproducibility
            n_jobs: Number of jobs to run in parallel
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.class_weight = class_weight
        self.random_state = random_state
        self.n_jobs = n_jobs
        
        # Initialize models
        self.binary_model = None
        self.multiclass_model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Model metadata
        self.feature_names = None
        self.is_trained = False
        self.training_history = []
        
        # Initialize the Random Forest model
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=n_jobs
        )
        
        logger.info("RandomForestVulnerabilityDetector initialized")
        
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = None
        self.is_trained = False
        self.training_history = {}
        self.model_metadata = {}
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              X_val: Optional[pd.DataFrame] = None, 
              y_val: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Train the Random Forest model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Dictionary with training results
        """
        logger.info("Starting Random Forest training...")
        
        # Store feature columns
        self.feature_columns = X_train.columns.tolist()
        
        # Encode labels if they are strings
        if y_train.dtype == 'object':
            y_train_encoded = self.label_encoder.fit_transform(y_train)
            if y_val is not None:
                y_val_encoded = self.label_encoder.transform(y_val)
        else:
            y_train_encoded = y_train.values
            y_val_encoded = y_val.values if y_val is not None else None
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val) if X_val is not None else None
        
        # Train the model
        start_time = datetime.now()
        self.model.fit(X_train_scaled, y_train_encoded)
        training_time = (datetime.now() - start_time).total_seconds()
        
        self.is_trained = True
        
        # Evaluate on training set
        train_predictions = self.model.predict(X_train_scaled)
        train_accuracy = accuracy_score(y_train_encoded, train_predictions)
        
        # Evaluate on validation set if provided
        val_accuracy = None
        if X_val is not None and y_val is not None:
            val_predictions = self.model.predict(X_val_scaled)
            val_accuracy = accuracy_score(y_val_encoded, val_predictions)
        
        # Store training history
        self.training_history = {
            'training_time': training_time,
            'train_accuracy': train_accuracy,
            'val_accuracy': val_accuracy,
            'n_samples': len(X_train),
            'n_features': len(self.feature_columns),
            'timestamp': datetime.now().isoformat()
        }
        
        # Store model metadata
        self.model_metadata = {
            'model_type': 'RandomForestClassifier',
            'n_estimators': self.model.n_estimators,
            'max_depth': self.model.max_depth,
            'min_samples_split': self.model.min_samples_split,
            'min_samples_leaf': self.model.min_samples_leaf,
            'max_features': self.model.max_features,
            'class_weight': self.model.class_weight,
            'random_state': self.model.random_state,
            'feature_columns': self.feature_columns,
            'classes': self.label_encoder.classes_.tolist() if hasattr(self.label_encoder, 'classes_') else None
        }
        
        logger.info(f"Training completed in {training_time:.2f} seconds")
        logger.info(f"Training accuracy: {train_accuracy:.4f}")
        if val_accuracy is not None:
            logger.info(f"Validation accuracy: {val_accuracy:.4f}")
        
        return self.training_history
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on input data.
        
        Args:
            X: Input features
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Ensure feature columns match
        if self.feature_columns and list(X.columns) != self.feature_columns:
            logger.warning("Feature columns don't match training data")
            X = X[self.feature_columns]
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        
        # Decode labels if necessary
        if hasattr(self.label_encoder, 'classes_'):
            predictions = self.label_encoder.inverse_transform(predictions)
        
        return predictions, probabilities
    
    def predict_single(self, features: Dict[str, float]) -> Tuple[Any, float]:
        """
        Make prediction for a single sample.
        
        Args:
            features: Dictionary of feature values
            
        Returns:
            Tuple of (prediction, confidence)
        """
        # Convert to DataFrame
        X = pd.DataFrame([features])
        
        # Make prediction
        predictions, probabilities = self.predict(X)
        
        # Get confidence (max probability)
        confidence = np.max(probabilities[0])
        
        return predictions[0], confidence
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Evaluate model performance on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary containing evaluation metrics
        """
        logger.info("Evaluating model performance...")
        
        # Make predictions
        predictions, probabilities = self.predict(X_test)
        
        # Encode test labels if necessary
        if y_test.dtype == 'object':
            y_test_encoded = self.label_encoder.transform(y_test)
            predictions_encoded = self.label_encoder.transform(predictions)
        else:
            y_test_encoded = y_test.values
            predictions_encoded = predictions
        
        # Calculate metrics
        accuracy = accuracy_score(y_test_encoded, predictions_encoded)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test_encoded, predictions_encoded, average='binary'
        )
        
        # ROC AUC (for binary classification)
        if len(np.unique(y_test_encoded)) == 2:
            roc_auc = roc_auc_score(y_test_encoded, probabilities[:, 1])
        else:
            roc_auc = roc_auc_score(y_test_encoded, probabilities, multi_class='ovr')
        
        # Confusion matrix
        cm = confusion_matrix(y_test_encoded, predictions_encoded)
        
        # Classification report
        class_report = classification_report(
            y_test_encoded, predictions_encoded,
            target_names=self.label_encoder.classes_ if hasattr(self.label_encoder, 'classes_') else None,
            output_dict=True
        )
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
            'classification_report': class_report,
            'predictions': predictions,
            'probabilities': probabilities,
            'test_samples': len(X_test)
        }
        
        # Log results
        logger.info("Evaluation Results:")
        logger.info(f"  Accuracy:  {accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall:    {recall:.4f}")
        logger.info(f"  F1-Score:  {f1:.4f}")
        logger.info(f"  ROC AUC:   {roc_auc:.4f}")
        
        return metrics
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance scores.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame with features and their importance scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        importances = self.model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return feature_importance_df.head(top_n)
    
    def plot_feature_importance(self, top_n: int = 20, save_path: Optional[str] = None):
        """
        Create bar chart of feature importances.
        
        Args:
            top_n: Number of top features to plot
            save_path: Path to save the plot
        """
        importance_df = self.get_feature_importance(top_n)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(data=importance_df, x='importance', y='feature', palette='viridis')
        plt.title(f'Top {top_n} Most Important Features')
        plt.xlabel('Importance Score')
        plt.ylabel('Feature')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_confusion_matrix(self, cm: np.ndarray, save_path: Optional[str] = None):
        """
        Create heatmap of confusion matrix.
        
        Args:
            cm: Confusion matrix
            save_path: Path to save the plot
        """
        plt.figure(figsize=(8, 6))
        
        # Get class names
        if hasattr(self.label_encoder, 'classes_'):
            class_names = self.label_encoder.classes_
        else:
            class_names = ['Safe', 'Vulnerable']
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names,
                   yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_roc_curve(self, y_test: np.ndarray, probabilities: np.ndarray, 
                      save_path: Optional[str] = None):
        """
        Plot ROC curve.
        
        Args:
            y_test: True labels
            probabilities: Predicted probabilities
            save_path: Path to save the plot
        """
        if len(np.unique(y_test)) != 2:
            logger.warning("ROC curve plotting only supported for binary classification")
            return
        
        fpr, tpr, _ = roc_curve(y_test, probabilities[:, 1])
        auc = roc_auc_score(y_test, probabilities[:, 1])
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def hyperparameter_tuning(self, X_train: pd.DataFrame, y_train: pd.Series,
                            cv: int = 5, scoring: str = 'f1') -> Dict[str, Any]:
        """
        Perform hyperparameter tuning using GridSearchCV.
        
        Args:
            X_train: Training features
            y_train: Training labels
            cv: Number of cross-validation folds
            scoring: Scoring metric for optimization
            
        Returns:
            Dictionary with tuning results
        """
        logger.info("Starting hyperparameter tuning...")
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }
        
        # Encode labels if necessary
        if y_train.dtype == 'object':
            y_train_encoded = self.label_encoder.fit_transform(y_train)
        else:
            y_train_encoded = y_train.values
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Perform grid search
        grid_search = GridSearchCV(
            estimator=RandomForestClassifier(random_state=42),
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train_scaled, y_train_encoded)
        
        # Update model with best parameters
        self.model = grid_search.best_estimator_
        self.is_trained = True
        
        # Store feature columns
        self.feature_columns = X_train.columns.tolist()
        
        results = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
        
        logger.info("Hyperparameter tuning completed")
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        return results
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series, 
                      cv: int = 5, scoring: str = 'accuracy') -> Dict[str, Any]:
        """
        Perform cross-validation.
        
        Args:
            X: Features
            y: Labels
            cv: Number of cross-validation folds
            scoring: Scoring metric
            
        Returns:
            Dictionary with cross-validation results
        """
        logger.info(f"Performing {cv}-fold cross-validation...")
        
        # Encode labels if necessary
        if y.dtype == 'object':
            y_encoded = self.label_encoder.fit_transform(y)
        else:
            y_encoded = y.values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Perform cross-validation
        cv_scores = cross_val_score(self.model, X_scaled, y_encoded, 
                                   cv=cv, scoring=scoring, n_jobs=-1)
        
        results = {
            'cv_scores': cv_scores,
            'mean_score': np.mean(cv_scores),
            'std_score': np.std(cv_scores),
            'min_score': np.min(cv_scores),
            'max_score': np.max(cv_scores)
        }
        
        logger.info(f"Cross-validation results ({scoring}):")
        logger.info(f"  Mean: {results['mean_score']:.4f} (+/- {results['std_score'] * 2:.4f})")
        logger.info(f"  Range: [{results['min_score']:.4f}, {results['max_score']:.4f}]")
        
        return results
    
    def explain_prediction(self, features: Dict[str, float], top_n: int = 10) -> Dict[str, float]:
        """
        Explain a prediction by showing feature contributions.
        
        Args:
            features: Dictionary of feature values
            top_n: Number of top contributing features to return
            
        Returns:
            Dictionary of feature contributions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        # Get feature importances
        importances = self.model.feature_importances_
        
        # Calculate feature contributions (importance * feature value)
        contributions = {}
        for feature, value in features.items():
            if feature in self.feature_columns:
                idx = self.feature_columns.index(feature)
                contributions[feature] = importances[idx] * abs(value)
        
        # Sort by contribution and return top N
        sorted_contributions = dict(sorted(contributions.items(), 
                                         key=lambda x: x[1], reverse=True)[:top_n])
        
        return sorted_contributions
    
    def save_model(self, filepath: str):
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_columns': self.feature_columns,
            'training_history': self.training_history,
            'model_metadata': self.model_metadata,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str) -> 'RandomForestVulnerabilityDetector':
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded RandomForestVulnerabilityDetector instance
        """
        model_data = joblib.load(filepath)
        
        # Create new instance
        detector = cls()
        
        # Restore model state
        detector.model = model_data['model']
        detector.scaler = model_data['scaler']
        detector.label_encoder = model_data['label_encoder']
        detector.feature_columns = model_data['feature_columns']
        detector.training_history = model_data['training_history']
        detector.model_metadata = model_data['model_metadata']
        detector.is_trained = model_data['is_trained']
        
        logger.info(f"Model loaded from {filepath}")
        return detector
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the trained model.
        
        Returns:
            Dictionary with model information
        """
        return {
            'is_trained': self.is_trained,
            'training_history': self.training_history,
            'model_metadata': self.model_metadata,
            'feature_count': len(self.feature_columns) if self.feature_columns else 0
        }


def main():
    """Example usage of RandomForestVulnerabilityDetector."""
    # This would typically use real data from the feature extractor
    # For demonstration, we'll create synthetic data
    
    np.random.seed(42)
    
    # Create synthetic feature data
    n_samples = 1000
    n_features = 30
    
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    # Create synthetic labels (vulnerable vs safe)
    y = pd.Series(np.random.choice(['safe', 'vulnerable'], n_samples))
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Initialize and train model
    detector = RandomForestVulnerabilityDetector()
    
    print("Training Random Forest model...")
    training_results = detector.train(X_train, y_train)
    print(f"Training completed: {training_results}")
    
    # Evaluate model
    print("\nEvaluating model...")
    evaluation_results = detector.evaluate(X_test, y_test)
    print(f"Test accuracy: {evaluation_results['accuracy']:.4f}")
    
    # Show feature importance
    print("\nTop 10 most important features:")
    importance_df = detector.get_feature_importance(10)
    print(importance_df)
    
    # Make a single prediction
    sample_features = X_test.iloc[0].to_dict()
    prediction, confidence = detector.predict_single(sample_features)
    print(f"\nSample prediction: {prediction} (confidence: {confidence:.4f})")
    
    # Explain the prediction
    explanation = detector.explain_prediction(sample_features, top_n=5)
    print("Top contributing features:")
    for feature, contribution in explanation.items():
        print(f"  {feature}: {contribution:.4f}")
    
    # Save model
    detector.save_model("models/random_forest_detector.pkl")
    print("\nModel saved successfully")


if __name__ == "__main__":
    main()