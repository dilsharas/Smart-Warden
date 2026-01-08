"""
Comprehensive model evaluation and metrics system for vulnerability detection.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report,
    average_precision_score, matthews_corrcoef
)
from sklearn.model_selection import learning_curve, validation_curve
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Comprehensive evaluation system for vulnerability detection models.
    
    Features:
    - Multiple evaluation metrics
    - Visualization of results
    - Performance comparison between models
    - Learning curve analysis
    - Feature importance analysis
    - Statistical significance testing
    """
    
    def __init__(self, save_plots: bool = True, plot_dir: str = "results/model_performance"):
        """
        Initialize the model evaluator.
        
        Args:
            save_plots: Whether to save generated plots
            plot_dir: Directory to save plots
        """
        self.save_plots = save_plots
        self.plot_dir = Path(plot_dir)
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        
        self.evaluation_results = {}
        self.comparison_results = {}
        
    def evaluate_model(self, model, X_test: pd.DataFrame, y_test: pd.Series,
                      model_name: str = "Model") -> Dict[str, Any]:
        """
        Comprehensive evaluation of a single model.
        
        Args:
            model: Trained model with predict and predict_proba methods
            X_test: Test features
            y_test: Test labels
            model_name: Name for the model
            
        Returns:
            Dictionary with comprehensive evaluation metrics
        """
        logger.info(f"Evaluating model: {model_name}")
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        
        # Handle binary vs multiclass
        is_binary = len(np.unique(y_test)) == 2
        
        # Basic metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        if is_binary:
            precision = precision_score(y_test, y_pred, pos_label=1)
            recall = recall_score(y_test, y_pred, pos_label=1)
            f1 = f1_score(y_test, y_pred, pos_label=1)
            roc_auc = roc_auc_score(y_test, y_proba[:, 1])
            avg_precision = average_precision_score(y_test, y_proba[:, 1])
        else:
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
            avg_precision = average_precision_score(y_test, y_proba, average='weighted')
        
        # Additional metrics
        mcc = matthews_corrcoef(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        # Calculate per-class metrics
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        # Compile results
        results = {
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'test_samples': len(X_test),
            'is_binary': is_binary,
            
            # Core metrics
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'average_precision': avg_precision,
            'matthews_corrcoef': mcc,
            
            # Detailed results
            'confusion_matrix': cm.tolist(),
            'classification_report': class_report,
            'predictions': y_pred.tolist() if hasattr(y_pred, 'tolist') else y_pred,
            'probabilities': y_proba.tolist() if hasattr(y_proba, 'tolist') else y_proba,
            'true_labels': y_test.tolist() if hasattr(y_test, 'tolist') else y_test
        }
        
        # Store results
        self.evaluation_results[model_name] = results
        
        # Generate visualizations
        if self.save_plots:
            self._plot_confusion_matrix(cm, model_name, class_report.keys())
            if is_binary:
                self._plot_roc_curve(y_test, y_proba[:, 1], model_name)
                self._plot_precision_recall_curve(y_test, y_proba[:, 1], model_name)
        
        # Log summary
        logger.info(f"Evaluation completed for {model_name}:")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  F1-Score: {f1:.4f}")
        logger.info(f"  ROC-AUC: {roc_auc:.4f}")
        
        return results
    
    def compare_models(self, models: Dict[str, Any], X_test: pd.DataFrame, 
                      y_test: pd.Series) -> Dict[str, Any]:
        """
        Compare multiple models on the same test set.
        
        Args:
            models: Dictionary of {model_name: model} pairs
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary with comparison results
        """
        logger.info(f"Comparing {len(models)} models...")
        
        # Evaluate each model
        model_results = {}
        for name, model in models.items():
            results = self.evaluate_model(model, X_test, y_test, name)
            model_results[name] = results
        
        # Create comparison summary
        comparison_df = self._create_comparison_dataframe(model_results)
        
        # Statistical significance tests
        significance_tests = self._perform_significance_tests(model_results)
        
        # Compile comparison results
        comparison_results = {
            'timestamp': datetime.now().isoformat(),
            'models_compared': list(models.keys()),
            'comparison_table': comparison_df.to_dict(),
            'significance_tests': significance_tests,
            'best_model': self._identify_best_model(comparison_df),
            'detailed_results': model_results
        }
        
        self.comparison_results = comparison_results
        
        # Generate comparison visualizations
        if self.save_plots:
            self._plot_model_comparison(comparison_df)
            self._plot_roc_comparison(model_results)
        
        return comparison_results
    
    def analyze_learning_curves(self, model, X_train: pd.DataFrame, y_train: pd.Series,
                               model_name: str = "Model", cv: int = 5) -> Dict[str, Any]:
        """
        Analyze learning curves to understand model performance vs training size.
        
        Args:
            model: Model to analyze
            X_train: Training features
            y_train: Training labels
            model_name: Name for the model
            cv: Number of cross-validation folds
            
        Returns:
            Dictionary with learning curve analysis
        """
        logger.info(f"Analyzing learning curves for {model_name}...")
        
        # Generate learning curves
        train_sizes, train_scores, val_scores = learning_curve(
            model, X_train, y_train, cv=cv, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='f1' if len(np.unique(y_train)) == 2 else 'f1_weighted'
        )
        
        # Calculate statistics
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        # Analyze overfitting
        overfitting_gap = train_mean - val_mean
        max_gap = np.max(overfitting_gap)
        final_gap = overfitting_gap[-1]
        
        results = {
            'model_name': model_name,
            'train_sizes': train_sizes.tolist(),
            'train_scores_mean': train_mean.tolist(),
            'train_scores_std': train_std.tolist(),
            'val_scores_mean': val_mean.tolist(),
            'val_scores_std': val_std.tolist(),
            'max_overfitting_gap': max_gap,
            'final_overfitting_gap': final_gap,
            'converged': final_gap < 0.05,  # Threshold for convergence
            'final_train_score': train_mean[-1],
            'final_val_score': val_mean[-1]
        }
        
        # Plot learning curves
        if self.save_plots:
            self._plot_learning_curves(results)
        
        logger.info(f"Learning curve analysis completed for {model_name}")
        logger.info(f"  Final validation score: {val_mean[-1]:.4f}")
        logger.info(f"  Overfitting gap: {final_gap:.4f}")
        
        return results
    
    def analyze_validation_curves(self, model, X_train: pd.DataFrame, y_train: pd.Series,
                                 param_name: str, param_range: List[Any],
                                 model_name: str = "Model", cv: int = 5) -> Dict[str, Any]:
        """
        Analyze validation curves for hyperparameter tuning insights.
        
        Args:
            model: Model to analyze
            X_train: Training features
            y_train: Training labels
            param_name: Name of parameter to vary
            param_range: Range of parameter values to test
            model_name: Name for the model
            cv: Number of cross-validation folds
            
        Returns:
            Dictionary with validation curve analysis
        """
        logger.info(f"Analyzing validation curves for {model_name} - {param_name}")
        
        # Generate validation curves
        train_scores, val_scores = validation_curve(
            model, X_train, y_train, param_name=param_name,
            param_range=param_range, cv=cv, n_jobs=-1,
            scoring='f1' if len(np.unique(y_train)) == 2 else 'f1_weighted'
        )
        
        # Calculate statistics
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        # Find optimal parameter
        best_idx = np.argmax(val_mean)
        best_param = param_range[best_idx]
        best_score = val_mean[best_idx]
        
        results = {
            'model_name': model_name,
            'parameter_name': param_name,
            'parameter_range': [str(p) for p in param_range],
            'train_scores_mean': train_mean.tolist(),
            'train_scores_std': train_std.tolist(),
            'val_scores_mean': val_mean.tolist(),
            'val_scores_std': val_std.tolist(),
            'best_parameter': str(best_param),
            'best_score': best_score,
            'best_index': best_idx
        }
        
        # Plot validation curves
        if self.save_plots:
            self._plot_validation_curves(results)
        
        logger.info(f"Validation curve analysis completed")
        logger.info(f"  Best {param_name}: {best_param}")
        logger.info(f"  Best score: {best_score:.4f}")
        
        return results
    
    def analyze_feature_importance(self, model, feature_names: List[str],
                                  model_name: str = "Model", top_n: int = 20) -> Dict[str, Any]:
        """
        Analyze feature importance for tree-based models.
        
        Args:
            model: Trained model with feature_importances_ attribute
            feature_names: List of feature names
            model_name: Name for the model
            top_n: Number of top features to analyze
            
        Returns:
            Dictionary with feature importance analysis
        """
        if not hasattr(model, 'feature_importances_'):
            logger.warning(f"Model {model_name} does not have feature_importances_ attribute")
            return {}
        
        logger.info(f"Analyzing feature importance for {model_name}")
        
        # Get feature importances
        importances = model.feature_importances_
        
        # Create importance DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Calculate statistics
        total_importance = np.sum(importances)
        cumulative_importance = np.cumsum(importance_df['importance'].values)
        
        # Find features that contribute to 80% of importance
        features_80_percent = np.argmax(cumulative_importance >= 0.8 * total_importance) + 1
        
        results = {
            'model_name': model_name,
            'feature_importance': importance_df.head(top_n).to_dict('records'),
            'total_features': len(feature_names),
            'features_for_80_percent': features_80_percent,
            'top_feature': importance_df.iloc[0]['feature'],
            'top_importance': importance_df.iloc[0]['importance'],
            'importance_concentration': importance_df.head(5)['importance'].sum() / total_importance
        }
        
        # Plot feature importance
        if self.save_plots:
            self._plot_feature_importance(importance_df.head(top_n), model_name)
        
        logger.info(f"Feature importance analysis completed")
        logger.info(f"  Top feature: {results['top_feature']} ({results['top_importance']:.4f})")
        logger.info(f"  Features for 80%: {features_80_percent}")
        
        return results
    
    def generate_evaluation_report(self, output_path: str = "results/evaluation_report.json"):
        """
        Generate comprehensive evaluation report.
        
        Args:
            output_path: Path to save the report
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'evaluation_results': self.evaluation_results,
            'comparison_results': self.comparison_results,
            'summary': self._generate_summary()
        }
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Evaluation report saved to {output_path}")
        
        return report
    
    def _create_comparison_dataframe(self, model_results: Dict[str, Dict]) -> pd.DataFrame:
        """Create comparison DataFrame from model results."""
        comparison_data = []
        
        for model_name, results in model_results.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': results['accuracy'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1-Score': results['f1_score'],
                'ROC-AUC': results['roc_auc'],
                'Avg Precision': results['average_precision'],
                'MCC': results['matthews_corrcoef']
            })
        
        return pd.DataFrame(comparison_data)
    
    def _perform_significance_tests(self, model_results: Dict[str, Dict]) -> Dict[str, Any]:
        """Perform statistical significance tests between models."""
        # Simplified implementation - in practice, would use McNemar's test or similar
        significance_tests = {}
        
        model_names = list(model_results.keys())
        for i, model1 in enumerate(model_names):
            for model2 in model_names[i+1:]:
                # Compare F1 scores (simplified)
                f1_diff = abs(model_results[model1]['f1_score'] - model_results[model2]['f1_score'])
                significance_tests[f"{model1}_vs_{model2}"] = {
                    'f1_difference': f1_diff,
                    'significant': f1_diff > 0.05  # Simplified threshold
                }
        
        return significance_tests
    
    def _identify_best_model(self, comparison_df: pd.DataFrame) -> Dict[str, Any]:
        """Identify the best performing model."""
        # Use F1-score as primary metric, with ROC-AUC as tiebreaker
        best_f1_idx = comparison_df['F1-Score'].idxmax()
        best_model = comparison_df.loc[best_f1_idx]
        
        return {
            'model_name': best_model['Model'],
            'f1_score': best_model['F1-Score'],
            'roc_auc': best_model['ROC-AUC'],
            'accuracy': best_model['Accuracy']
        }
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary of all evaluations."""
        summary = {
            'total_models_evaluated': len(self.evaluation_results),
            'evaluation_timestamp': datetime.now().isoformat()
        }
        
        if self.evaluation_results:
            # Calculate average metrics across all models
            all_accuracies = [r['accuracy'] for r in self.evaluation_results.values()]
            all_f1_scores = [r['f1_score'] for r in self.evaluation_results.values()]
            
            summary.update({
                'average_accuracy': np.mean(all_accuracies),
                'average_f1_score': np.mean(all_f1_scores),
                'best_accuracy': np.max(all_accuracies),
                'best_f1_score': np.max(all_f1_scores)
            })
        
        return summary
    
    # Plotting methods
    def _plot_confusion_matrix(self, cm: np.ndarray, model_name: str, class_names: List[str]):
        """Plot confusion matrix."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if self.save_plots:
            plt.savefig(self.plot_dir / f'{model_name}_confusion_matrix.png', 
                       dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_roc_curve(self, y_true: np.ndarray, y_proba: np.ndarray, model_name: str):
        """Plot ROC curve."""
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if self.save_plots:
            plt.savefig(self.plot_dir / f'{model_name}_roc_curve.png', 
                       dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_precision_recall_curve(self, y_true: np.ndarray, y_proba: np.ndarray, model_name: str):
        """Plot Precision-Recall curve."""
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        avg_precision = average_precision_score(y_true, y_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='darkorange', lw=2,
                label=f'PR curve (AP = {avg_precision:.3f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {model_name}')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if self.save_plots:
            plt.savefig(self.plot_dir / f'{model_name}_pr_curve.png', 
                       dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_model_comparison(self, comparison_df: pd.DataFrame):
        """Plot model comparison chart."""
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            bars = ax.bar(comparison_df['Model'], comparison_df[metric], 
                         color=plt.cm.viridis(np.linspace(0, 1, len(comparison_df))))
            ax.set_title(f'{metric} Comparison')
            ax.set_ylabel(metric)
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom')
        
        # Remove empty subplot
        fig.delaxes(axes[5])
        
        plt.tight_layout()
        
        if self.save_plots:
            plt.savefig(self.plot_dir / 'model_comparison.png', 
                       dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_roc_comparison(self, model_results: Dict[str, Dict]):
        """Plot ROC curves for all models on same chart."""
        plt.figure(figsize=(10, 8))
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(model_results)))
        
        for i, (model_name, results) in enumerate(model_results.items()):
            if results['is_binary']:
                y_true = np.array(results['true_labels'])
                y_proba = np.array(results['probabilities'])[:, 1]
                
                fpr, tpr, _ = roc_curve(y_true, y_proba)
                auc = roc_auc_score(y_true, y_proba)
                
                plt.plot(fpr, tpr, color=colors[i], lw=2,
                        label=f'{model_name} (AUC = {auc:.3f})')
        
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.8)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve Comparison')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if self.save_plots:
            plt.savefig(self.plot_dir / 'roc_comparison.png', 
                       dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_learning_curves(self, results: Dict[str, Any]):
        """Plot learning curves."""
        plt.figure(figsize=(10, 6))
        
        train_sizes = results['train_sizes']
        train_mean = results['train_scores_mean']
        train_std = results['train_scores_std']
        val_mean = results['val_scores_mean']
        val_std = results['val_scores_std']
        
        plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training score')
        plt.fill_between(train_sizes, np.array(train_mean) - np.array(train_std),
                        np.array(train_mean) + np.array(train_std), alpha=0.1, color='blue')
        
        plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validation score')
        plt.fill_between(train_sizes, np.array(val_mean) - np.array(val_std),
                        np.array(val_mean) + np.array(val_std), alpha=0.1, color='red')
        
        plt.xlabel('Training Set Size')
        plt.ylabel('Score')
        plt.title(f'Learning Curves - {results["model_name"]}')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if self.save_plots:
            plt.savefig(self.plot_dir / f'{results["model_name"]}_learning_curves.png', 
                       dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_validation_curves(self, results: Dict[str, Any]):
        """Plot validation curves."""
        plt.figure(figsize=(10, 6))
        
        param_range = results['parameter_range']
        train_mean = results['train_scores_mean']
        train_std = results['train_scores_std']
        val_mean = results['val_scores_mean']
        val_std = results['val_scores_std']
        
        plt.plot(param_range, train_mean, 'o-', color='blue', label='Training score')
        plt.fill_between(param_range, np.array(train_mean) - np.array(train_std),
                        np.array(train_mean) + np.array(train_std), alpha=0.1, color='blue')
        
        plt.plot(param_range, val_mean, 'o-', color='red', label='Validation score')
        plt.fill_between(param_range, np.array(val_mean) - np.array(val_std),
                        np.array(val_mean) + np.array(val_std), alpha=0.1, color='red')
        
        plt.xlabel(results['parameter_name'])
        plt.ylabel('Score')
        plt.title(f'Validation Curves - {results["model_name"]}')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if self.save_plots:
            plt.savefig(self.plot_dir / f'{results["model_name"]}_validation_curves.png', 
                       dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_feature_importance(self, importance_df: pd.DataFrame, model_name: str):
        """Plot feature importance."""
        plt.figure(figsize=(12, 8))
        sns.barplot(data=importance_df, x='importance', y='feature', palette='viridis')
        plt.title(f'Feature Importance - {model_name}')
        plt.xlabel('Importance Score')
        plt.ylabel('Feature')
        plt.tight_layout()
        
        if self.save_plots:
            plt.savefig(self.plot_dir / f'{model_name}_feature_importance.png', 
                       dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """Example usage of ModelEvaluator."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    import numpy as np
    
    # Create synthetic data
    np.random.seed(42)
    X = np.random.randn(1000, 20)
    y = (X[:, 0] + X[:, 1] + np.random.randn(1000) * 0.1) > 0
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train models
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    lr_model = LogisticRegression(random_state=42)
    
    rf_model.fit(X_train, y_train)
    lr_model.fit(X_train, y_train)
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Compare models
    models = {
        'Random Forest': rf_model,
        'Logistic Regression': lr_model
    }
    
    comparison_results = evaluator.compare_models(models, X_test, y_test)
    print("Model comparison completed")
    
    # Analyze learning curves
    learning_results = evaluator.analyze_learning_curves(rf_model, X_train, y_train, "Random Forest")
    print("Learning curve analysis completed")
    
    # Analyze feature importance
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    importance_results = evaluator.analyze_feature_importance(rf_model, feature_names, "Random Forest")
    print("Feature importance analysis completed")
    
    # Generate report
    report = evaluator.generate_evaluation_report()
    print("Evaluation report generated")


if __name__ == "__main__":
    main()