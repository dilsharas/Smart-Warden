"""
Multi-class vulnerability classifier for smart contracts.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support, multilabel_confusion_matrix
)
from sklearn.model_selection import GridSearchCV
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class MultiClassVulnerabilityClassifier:
    """
    Multi-class classifier for identifying specific vulnerability types in smart contracts.
    
    Supported vulnerability types:
    - Reentrancy
    - Access Control
    - Arithmetic Overflow/Underflow
    - Unchecked External Calls
    - Denial of Service
    - Bad Randomness
    - Safe (no vulnerabilities)
    """
    
    VULNERABILITY_TYPES = [
        'safe',
        'reentrancy', 
        'access_control',
        'arithmetic',
        'unchecked_calls',
        'denial_of_service',
        'bad_randomness'
    ]
    
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
        Initialize the multi-class vulnerability classifier.
        
        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of trees
            min_samples_split: Minimum samples required to split a node
            min_samples_leaf: Minimum samples required at a leaf node
            max_features: Number of features to consider for best split
            class_weight: Weights associated with classes
            random_state: Random seed for reproducibility
            n_jobs: Number of jobs to run in parallel
        """
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
        
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = None
        self.is_trained = False
        self.training_history = {}
        self.model_metadata = {}
        self.class_distribution = {}
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: Optional[pd.DataFrame] = None,
              y_val: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Train the multi-class vulnerability classifier.
        
        Args:
            X_train: Training features
            y_train: Training labels (vulnerability types)
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Dictionary with training results
        """
        logger.info("Starting multi-class vulnerability classifier training...")
        
        # Store feature columns
        self.feature_columns = X_train.columns.tolist()
        
        # Validate and encode labels
        y_train_clean = self._validate_labels(y_train)
        y_train_encoded = self.label_encoder.fit_transform(y_train_clean)
        
        if y_val is not None:
            y_val_clean = self._validate_labels(y_val)
            y_val_encoded = self.label_encoder.transform(y_val_clean)
        else:
            y_val_encoded = None
        
        # Store class distribution
        self.class_distribution = pd.Series(y_train_clean).value_counts().to_dict()
        
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
        
        # Calculate per-class metrics for training set
        train_precision, train_recall, train_f1, _ = precision_recall_fscore_support(
            y_train_encoded, train_predictions, average=None, zero_division=0
        )
        
        # Evaluate on validation set if provided
        val_accuracy = None
        val_precision = None
        val_recall = None
        val_f1 = None
        
        if X_val is not None and y_val is not None:
            val_predictions = self.model.predict(X_val_scaled)
            val_accuracy = accuracy_score(y_val_encoded, val_predictions)
            val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(
                y_val_encoded, val_predictions, average=None, zero_division=0
            )
        
        # Store training history
        self.training_history = {
            'training_time': training_time,
            'train_accuracy': train_accuracy,
            'train_precision_per_class': train_precision.tolist(),
            'train_recall_per_class': train_recall.tolist(),
            'train_f1_per_class': train_f1.tolist(),
            'val_accuracy': val_accuracy,
            'val_precision_per_class': val_precision.tolist() if val_precision is not None else None,
            'val_recall_per_class': val_recall.tolist() if val_recall is not None else None,
            'val_f1_per_class': val_f1.tolist() if val_f1 is not None else None,
            'n_samples': len(X_train),
            'n_features': len(self.feature_columns),
            'n_classes': len(self.label_encoder.classes_),
            'class_distribution': self.class_distribution,
            'timestamp': datetime.now().isoformat()
        }
        
        # Store model metadata
        self.model_metadata = {
            'model_type': 'MultiClassRandomForestClassifier',
            'n_estimators': self.model.n_estimators,
            'max_depth': self.model.max_depth,
            'min_samples_split': self.model.min_samples_split,
            'min_samples_leaf': self.model.min_samples_leaf,
            'max_features': self.model.max_features,
            'class_weight': self.model.class_weight,
            'random_state': self.model.random_state,
            'feature_columns': self.feature_columns,
            'classes': self.label_encoder.classes_.tolist(),
            'vulnerability_types': self.VULNERABILITY_TYPES
        }
        
        logger.info(f"Training completed in {training_time:.2f} seconds")
        logger.info(f"Training accuracy: {train_accuracy:.4f}")
        logger.info(f"Number of classes: {len(self.label_encoder.classes_)}")
        logger.info(f"Class distribution: {self.class_distribution}")
        
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
        predictions_encoded = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        
        # Decode labels
        predictions = self.label_encoder.inverse_transform(predictions_encoded)
        
        return predictions, probabilities
    
    def predict_single(self, features: Dict[str, float]) -> Tuple[str, float, Dict[str, float]]:
        """
        Make prediction for a single sample with detailed probabilities.
        
        Args:
            features: Dictionary of feature values
            
        Returns:
            Tuple of (prediction, confidence, class_probabilities)
        """
        # Convert to DataFrame
        X = pd.DataFrame([features])
        
        # Make prediction
        predictions, probabilities = self.predict(X)
        
        # Get confidence (max probability)
        confidence = np.max(probabilities[0])
        
        # Create class probability dictionary
        class_probabilities = {}
        for i, class_name in enumerate(self.label_encoder.classes_):
            class_probabilities[class_name] = probabilities[0][i]
        
        return predictions[0], confidence, class_probabilities
    
    def predict_top_k(self, X: pd.DataFrame, k: int = 3) -> List[List[Tuple[str, float]]]:
        """
        Get top-k predictions for each sample.
        
        Args:
            X: Input features
            k: Number of top predictions to return
            
        Returns:
            List of top-k predictions with probabilities for each sample
        """
        _, probabilities = self.predict(X)
        
        results = []
        for prob_row in probabilities:
            # Get top-k indices
            top_k_indices = np.argsort(prob_row)[-k:][::-1]
            
            # Create list of (class_name, probability) tuples
            top_k_predictions = []
            for idx in top_k_indices:
                class_name = self.label_encoder.classes_[idx]
                probability = prob_row[idx]
                top_k_predictions.append((class_name, probability))
            
            results.append(top_k_predictions)
        
        return results
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Evaluate model performance on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary containing evaluation metrics
        """
        logger.info("Evaluating multi-class model performance...")
        
        # Make predictions
        predictions, probabilities = self.predict(X_test)
        
        # Clean and encode test labels
        y_test_clean = self._validate_labels(y_test)
        y_test_encoded = self.label_encoder.transform(y_test_clean)
        predictions_encoded = self.label_encoder.transform(predictions)
        
        # Calculate overall metrics
        accuracy = accuracy_score(y_test_encoded, predictions_encoded)
        
        # Calculate per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test_encoded, predictions_encoded, average=None, zero_division=0
        )
        
        # Calculate macro and micro averages
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_test_encoded, predictions_encoded, average='macro', zero_division=0
        )
        precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
            y_test_encoded, predictions_encoded, average='micro', zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_test_encoded, predictions_encoded)
        
        # Classification report
        class_report = classification_report(
            y_test_encoded, predictions_encoded,
            target_names=self.label_encoder.classes_,
            output_dict=True,
            zero_division=0
        )
        
        # Per-class analysis
        per_class_metrics = {}
        for i, class_name in enumerate(self.label_encoder.classes_):
            per_class_metrics[class_name] = {
                'precision': precision[i],
                'recall': recall[i],
                'f1_score': f1[i],
                'support': support[i]
            }
        
        metrics = {
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_micro': precision_micro,
            'recall_micro': recall_micro,
            'f1_micro': f1_micro,
            'per_class_metrics': per_class_metrics,
            'confusion_matrix': cm,
            'classification_report': class_report,
            'predictions': predictions,
            'probabilities': probabilities,
            'test_samples': len(X_test),
            'class_names': self.label_encoder.classes_.tolist()
        }
        
        # Log results
        logger.info("Multi-class Evaluation Results:")
        logger.info(f"  Overall Accuracy: {accuracy:.4f}")
        logger.info(f"  Macro F1-Score:   {f1_macro:.4f}")
        logger.info(f"  Micro F1-Score:   {f1_micro:.4f}")
        
        logger.info("Per-class Performance:")
        for class_name, class_metrics in per_class_metrics.items():
            logger.info(f"  {class_name:<15}: F1={class_metrics['f1_score']:.3f}, "
                       f"Precision={class_metrics['precision']:.3f}, "
                       f"Recall={class_metrics['recall']:.3f}, "
                       f"Support={class_metrics['support']}")
        
        return metrics
    
    def get_feature_importance_per_class(self, top_n: int = 10) -> Dict[str, pd.DataFrame]:
        """
        Get feature importance for each vulnerability class.
        
        Args:
            top_n: Number of top features per class
            
        Returns:
            Dictionary mapping class names to feature importance DataFrames
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        # Get overall feature importances
        importances = self.model.feature_importances_
        
        # Create feature importance DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # For multi-class Random Forest, we can't easily get per-class importances
        # So we return the overall importance for each class
        per_class_importance = {}
        for class_name in self.label_encoder.classes_:
            per_class_importance[class_name] = importance_df.head(top_n).copy()
        
        return per_class_importance
    
    def plot_confusion_matrix(self, cm: np.ndarray, save_path: Optional[str] = None):
        """
        Create heatmap of multi-class confusion matrix.
        
        Args:
            cm: Confusion matrix
            save_path: Path to save the plot
        """
        plt.figure(figsize=(10, 8))
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_)
        
        plt.title('Multi-Class Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_class_distribution(self, save_path: Optional[str] = None):
        """
        Plot the distribution of vulnerability classes in training data.
        
        Args:
            save_path: Path to save the plot
        """
        plt.figure(figsize=(10, 6))
        
        classes = list(self.class_distribution.keys())
        counts = list(self.class_distribution.values())
        
        bars = plt.bar(classes, counts, color='skyblue', edgecolor='navy', alpha=0.7)
        plt.title('Vulnerability Class Distribution in Training Data')
        plt.xlabel('Vulnerability Type')
        plt.ylabel('Number of Samples')
        plt.xticks(rotation=45)
        
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_per_class_metrics(self, metrics: Dict[str, Any], save_path: Optional[str] = None):
        """
        Plot per-class precision, recall, and F1-score.
        
        Args:
            metrics: Evaluation metrics dictionary
            save_path: Path to save the plot
        """
        per_class = metrics['per_class_metrics']
        
        classes = list(per_class.keys())
        precision_scores = [per_class[cls]['precision'] for cls in classes]
        recall_scores = [per_class[cls]['recall'] for cls in classes]
        f1_scores = [per_class[cls]['f1_score'] for cls in classes]
        
        x = np.arange(len(classes))
        width = 0.25
        
        plt.figure(figsize=(12, 6))
        
        plt.bar(x - width, precision_scores, width, label='Precision', alpha=0.8)
        plt.bar(x, recall_scores, width, label='Recall', alpha=0.8)
        plt.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8)
        
        plt.xlabel('Vulnerability Type')
        plt.ylabel('Score')
        plt.title('Per-Class Performance Metrics')
        plt.xticks(x, classes, rotation=45)
        plt.legend()
        plt.ylim(0, 1.1)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def _validate_labels(self, labels: pd.Series) -> pd.Series:
        """
        Validate and clean vulnerability labels.
        
        Args:
            labels: Series of vulnerability labels
            
        Returns:
            Cleaned labels series
        """
        # Convert to lowercase and strip whitespace
        cleaned_labels = labels.astype(str).str.lower().str.strip()
        
        # Map common variations to standard names
        label_mapping = {
            'reentrancy': 'reentrancy',
            're-entrancy': 'reentrancy',
            'reentrant': 'reentrancy',
            'access_control': 'access_control',
            'access-control': 'access_control',
            'accesscontrol': 'access_control',
            'arithmetic': 'arithmetic',
            'overflow': 'arithmetic',
            'underflow': 'arithmetic',
            'integer_overflow': 'arithmetic',
            'unchecked_calls': 'unchecked_calls',
            'unchecked-calls': 'unchecked_calls',
            'unchecked_call': 'unchecked_calls',
            'denial_of_service': 'denial_of_service',
            'denial-of-service': 'denial_of_service',
            'dos': 'denial_of_service',
            'bad_randomness': 'bad_randomness',
            'bad-randomness': 'bad_randomness',
            'randomness': 'bad_randomness',
            'safe': 'safe',
            'no_vulnerability': 'safe',
            'clean': 'safe'
        }
        
        # Apply mapping
        mapped_labels = cleaned_labels.map(label_mapping)
        
        # Check for unmapped labels
        unmapped = mapped_labels.isna()
        if unmapped.any():
            logger.warning(f"Found {unmapped.sum()} unmapped labels: {cleaned_labels[unmapped].unique()}")
            # Set unmapped labels to 'safe' as default
            mapped_labels = mapped_labels.fillna('safe')
        
        return mapped_labels
    
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
            'class_distribution': self.class_distribution,
            'is_trained': self.is_trained,
            'vulnerability_types': self.VULNERABILITY_TYPES
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Multi-class model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str) -> 'MultiClassVulnerabilityClassifier':
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded MultiClassVulnerabilityClassifier instance
        """
        model_data = joblib.load(filepath)
        
        # Create new instance
        classifier = cls()
        
        # Restore model state
        classifier.model = model_data['model']
        classifier.scaler = model_data['scaler']
        classifier.label_encoder = model_data['label_encoder']
        classifier.feature_columns = model_data['feature_columns']
        classifier.training_history = model_data['training_history']
        classifier.model_metadata = model_data['model_metadata']
        classifier.class_distribution = model_data['class_distribution']
        classifier.is_trained = model_data['is_trained']
        
        logger.info(f"Multi-class model loaded from {filepath}")
        return classifier
    
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
            'class_distribution': self.class_distribution,
            'feature_count': len(self.feature_columns) if self.feature_columns else 0,
            'class_count': len(self.label_encoder.classes_) if hasattr(self.label_encoder, 'classes_') else 0,
            'supported_vulnerability_types': self.VULNERABILITY_TYPES
        }


def main():
    """Example usage of MultiClassVulnerabilityClassifier."""
    # Create synthetic multi-class data
    np.random.seed(42)
    
    n_samples = 1000
    n_features = 30
    
    # Create synthetic feature data
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    # Create synthetic multi-class labels
    vulnerability_types = ['safe', 'reentrancy', 'access_control', 'arithmetic', 'unchecked_calls']
    y = pd.Series(np.random.choice(vulnerability_types, n_samples))
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Initialize and train classifier
    classifier = MultiClassVulnerabilityClassifier()
    
    print("Training multi-class vulnerability classifier...")
    training_results = classifier.train(X_train, y_train)
    print(f"Training completed: {training_results['train_accuracy']:.4f}")
    
    # Evaluate model
    print("\nEvaluating model...")
    evaluation_results = classifier.evaluate(X_test, y_test)
    print(f"Test accuracy: {evaluation_results['accuracy']:.4f}")
    print(f"Macro F1-score: {evaluation_results['f1_macro']:.4f}")
    
    # Make a single prediction with probabilities
    sample_features = X_test.iloc[0].to_dict()
    prediction, confidence, class_probs = classifier.predict_single(sample_features)
    print(f"\nSample prediction: {prediction} (confidence: {confidence:.4f})")
    print("Class probabilities:")
    for class_name, prob in class_probs.items():
        print(f"  {class_name}: {prob:.4f}")
    
    # Get top-3 predictions for first few samples
    top_k_predictions = classifier.predict_top_k(X_test.head(3), k=3)
    print("\nTop-3 predictions for first 3 samples:")
    for i, predictions in enumerate(top_k_predictions):
        print(f"Sample {i+1}:")
        for j, (class_name, prob) in enumerate(predictions):
            print(f"  {j+1}. {class_name}: {prob:.4f}")
    
    # Save model
    classifier.save_model("models/multiclass_vulnerability_classifier.pkl")
    print("\nMulti-class model saved successfully")


if __name__ == "__main__":
    main()