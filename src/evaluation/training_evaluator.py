#!/usr/bin/env python3
"""
Comprehensive Training Evaluation and Benchmarking System.
Implements automated model evaluation, cross-validation, and performance benchmarking.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
import logging
import time
from sklearn.model_selection import (
    cross_val_score, StratifiedKFold, learning_curve, validation_curve
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve
)
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, asdict
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class EvaluationConfig:
    """Configuration for training evaluation."""
    # Cross-validation
    cv_folds: int = 5
    cv_scoring: List[str] = None
    stratified: bool = True
    
    # Learning curves
    generate_learning_curves: bool = True
    learning_curve_train_sizes: List[float] = None
    
    # Validation curves
    generate_validation_curves: bool = True
    validation_params: Dict[str, List] = None
    
    # Benchmarking
    enable_benchmarking: bool = True
    baseline_models: List[str] = None
    
    # Statistical testing
    statistical_significance: bool = True
    significance_level: float = 0.05
    
    # Visualization
    generate_plots: bool = True
    save_plots: bool = True
    plot_dir: str = "evaluation_plots"
    
    # Performance tracking
    track_training_time: bool = True
    track_memory_usage: bool = True
    
    random_state: int = 42

class ModelEvaluator:
    """Comprehensive model evaluation system."""
    
    def __init__(self, config: EvaluationConfig = None):
        """
        Initialize the model evaluator.
        
        Args:
            config: Evaluation configuration
        """
        self.config = config or EvaluationConfig()
        
        # Set default values
        if self.config.cv_scoring is None:
            self.config.cv_scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        
        if self.config.learning_curve_train_sizes is None:
            self.config.learning_curve_train_sizes = np.linspace(0.1, 1.0, 10)
        
        if self.config.baseline_models is None:
            self.config.baseline_models = ['random_forest', 'logistic_regression']
        
        if self.config.validation_params is None:
            self.config.validation_params = {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [5, 10, 15, 20, None]
            }
        
        # Create plot directory
        if self.config.save_plots:
            Path(self.config.plot_dir).mkdir(parents=True, exist_ok=True)
        
        # Evaluation results
        self.evaluation_results = {}
        
        logger.info("Initialized ModelEvaluator")
    
    def evaluate_model_comprehensive(self,
                                   model: Any,
                                   X_train: np.ndarray,
                                   y_train: np.ndarray,
                                   X_test: np.ndarray,
                                   y_test: np.ndarray,
                                   model_name: str = "model") -> Dict:
        """
        Perform comprehensive model evaluation.
        
        Args:
            model: Trained model to evaluate
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            model_name: Name for the model
            
        Returns:
            Comprehensive evaluation results
        """
        logger.info(f"📊 Starting comprehensive evaluation for {model_name}...")
        
        evaluation_start = time.time()
        results = {'model_name': model_name}
        
        # 1. Basic performance metrics
        results['basic_metrics'] = self._evaluate_basic_metrics(
            model, X_test, y_test
        )
        
        # 2. Cross-validation evaluation
        results['cross_validation'] = self._evaluate_cross_validation(
            model, X_train, y_train
        )
        
        # 3. Learning curves
        if self.config.generate_learning_curves:
            results['learning_curves'] = self._generate_learning_curves(
                model, X_train, y_train, model_name
            )
        
        # 4. Validation curves
        if self.config.generate_validation_curves:
            results['validation_curves'] = self._generate_validation_curves(
                model, X_train, y_train, model_name
            )
        
        # 5. Statistical analysis
        if self.config.statistical_significance:
            results['statistical_analysis'] = self._perform_statistical_analysis(
                model, X_train, y_train
            )
        
        # 6. Performance visualization
        if self.config.generate_plots:
            results['visualizations'] = self._create_performance_visualizations(
                model, X_test, y_test, model_name
            )
        
        # 7. Training efficiency metrics
        if self.config.track_training_time or self.config.track_memory_usage:
            results['efficiency_metrics'] = self._measure_efficiency_metrics(
                model, X_train, y_train
            )
        
        evaluation_time = time.time() - evaluation_start
        results['evaluation_time'] = evaluation_time
        
        # Store results
        self.evaluation_results[model_name] = results
        
        logger.info(f"✅ Evaluation complete for {model_name} in {evaluation_time:.1f}s")
        
        return results
    
    def _evaluate_basic_metrics(self, model: Any, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Evaluate basic performance metrics."""
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, predictions),
            'precision': precision_score(y_test, predictions, average='binary'),
            'recall': recall_score(y_test, predictions, average='binary'),
            'f1_score': f1_score(y_test, predictions, average='binary'),
            'roc_auc': roc_auc_score(y_test, probabilities),
            'confusion_matrix': confusion_matrix(y_test, predictions).tolist(),
            'classification_report': classification_report(y_test, predictions, output_dict=True)
        }
        
        return metrics
    
    def _evaluate_cross_validation(self, model: Any, X: np.ndarray, y: np.ndarray) -> Dict:
        """Perform cross-validation evaluation."""
        cv_results = {}
        
        # Set up cross-validation
        if self.config.stratified:
            cv = StratifiedKFold(
                n_splits=self.config.cv_folds,
                shuffle=True,
                random_state=self.config.random_state
            )
        else:
            cv = self.config.cv_folds
        
        # Evaluate each scoring metric
        for scoring in self.config.cv_scoring:
            try:
                scores = cross_val_score(
                    model, X, y,
                    cv=cv,
                    scoring=scoring,
                    n_jobs=-1
                )
                
                cv_results[scoring] = {
                    'mean': scores.mean(),
                    'std': scores.std(),
                    'scores': scores.tolist(),
                    'confidence_interval': self._calculate_confidence_interval(scores)
                }
                
            except Exception as e:
                logger.warning(f"Failed to compute {scoring}: {e}")
                cv_results[scoring] = None
        
        return cv_results
    
    def _generate_learning_curves(self, 
                                model: Any, 
                                X: np.ndarray, 
                                y: np.ndarray,
                                model_name: str) -> Dict:
        """Generate learning curves."""
        logger.info("📈 Generating learning curves...")
        
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y,
            train_sizes=self.config.learning_curve_train_sizes,
            cv=self.config.cv_folds,
            scoring='f1',
            n_jobs=-1,
            random_state=self.config.random_state
        )
        
        learning_curve_data = {
            'train_sizes': train_sizes.tolist(),
            'train_scores_mean': train_scores.mean(axis=1).tolist(),
            'train_scores_std': train_scores.std(axis=1).tolist(),
            'val_scores_mean': val_scores.mean(axis=1).tolist(),
            'val_scores_std': val_scores.std(axis=1).tolist()
        }
        
        # Plot learning curves
        if self.config.generate_plots:
            self._plot_learning_curves(learning_curve_data, model_name)
        
        return learning_curve_data
    
    def _generate_validation_curves(self,
                                  model: Any,
                                  X: np.ndarray,
                                  y: np.ndarray,
                                  model_name: str) -> Dict:
        """Generate validation curves for hyperparameters."""
        logger.info("📊 Generating validation curves...")
        
        validation_curves = {}
        
        for param_name, param_values in self.config.validation_params.items():
            if hasattr(model, param_name):
                try:
                    train_scores, val_scores = validation_curve(
                        model, X, y,
                        param_name=param_name,
                        param_range=param_values,
                        cv=self.config.cv_folds,
                        scoring='f1',
                        n_jobs=-1
                    )
                    
                    validation_curves[param_name] = {
                        'param_values': param_values,
                        'train_scores_mean': train_scores.mean(axis=1).tolist(),
                        'train_scores_std': train_scores.std(axis=1).tolist(),
                        'val_scores_mean': val_scores.mean(axis=1).tolist(),
                        'val_scores_std': val_scores.std(axis=1).tolist()
                    }
                    
                    # Plot validation curves
                    if self.config.generate_plots:
                        self._plot_validation_curves(
                            validation_curves[param_name], param_name, model_name
                        )
                
                except Exception as e:
                    logger.warning(f"Failed to generate validation curve for {param_name}: {e}")
        
        return validation_curves
    
    def _perform_statistical_analysis(self, model: Any, X: np.ndarray, y: np.ndarray) -> Dict:
        """Perform statistical significance analysis."""
        from scipy import stats
        
        # Perform multiple CV runs for statistical testing
        n_runs = 10
        cv_scores = []
        
        for run in range(n_runs):
            cv = StratifiedKFold(
                n_splits=self.config.cv_folds,
                shuffle=True,
                random_state=self.config.random_state + run
            )
            
            scores = cross_val_score(model, X, y, cv=cv, scoring='f1')
            cv_scores.extend(scores)
        
        cv_scores = np.array(cv_scores)
        
        # Statistical tests
        # Test if performance is significantly better than random (0.5 for binary classification)
        t_stat, p_value = stats.ttest_1samp(cv_scores, 0.5)
        
        statistical_analysis = {
            'mean_cv_score': cv_scores.mean(),
            'std_cv_score': cv_scores.std(),
            'n_samples': len(cv_scores),
            't_statistic': t_stat,
            'p_value': p_value,
            'significantly_better_than_random': p_value < self.config.significance_level,
            'confidence_interval_95': self._calculate_confidence_interval(cv_scores, confidence=0.95)
        }
        
        return statistical_analysis
    
    def _create_performance_visualizations(self,
                                         model: Any,
                                         X_test: np.ndarray,
                                         y_test: np.ndarray,
                                         model_name: str) -> Dict:
        """Create performance visualization plots."""
        logger.info("📊 Creating performance visualizations...")
        
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]
        
        visualizations = {}
        
        # 1. Confusion Matrix
        cm = confusion_matrix(y_test, predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if self.config.save_plots:
            plt.savefig(f"{self.config.plot_dir}/confusion_matrix_{model_name}.png", 
                       dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. ROC Curve
        fpr, tpr, _ = roc_curve(y_test, probabilities)
        roc_auc = roc_auc_score(y_test, probabilities)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        
        if self.config.save_plots:
            plt.savefig(f"{self.config.plot_dir}/roc_curve_{model_name}.png", 
                       dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_test, probabilities)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {model_name}')
        plt.grid(True)
        
        if self.config.save_plots:
            plt.savefig(f"{self.config.plot_dir}/pr_curve_{model_name}.png", 
                       dpi=300, bbox_inches='tight')
        plt.close()
        
        visualizations = {
            'confusion_matrix_plot': f"confusion_matrix_{model_name}.png",
            'roc_curve_plot': f"roc_curve_{model_name}.png",
            'precision_recall_plot': f"pr_curve_{model_name}.png"
        }
        
        return visualizations
    
    def _measure_efficiency_metrics(self, model: Any, X: np.ndarray, y: np.ndarray) -> Dict:
        """Measure training and prediction efficiency."""
        import psutil
        import os
        
        efficiency_metrics = {}
        
        # Measure training time
        if self.config.track_training_time:
            start_time = time.time()
            model_copy = type(model)(**model.get_params())
            model_copy.fit(X, y)
            training_time = time.time() - start_time
            
            efficiency_metrics['training_time'] = training_time
        
        # Measure prediction time
        start_time = time.time()
        _ = model.predict(X[:100])  # Sample for prediction time
        prediction_time = (time.time() - start_time) / 100  # Per sample
        
        efficiency_metrics['prediction_time_per_sample'] = prediction_time
        
        # Memory usage
        if self.config.track_memory_usage:
            process = psutil.Process(os.getpid())
            memory_usage = process.memory_info().rss / 1024 / 1024  # MB
            efficiency_metrics['memory_usage_mb'] = memory_usage
        
        return efficiency_metrics
    
    def _plot_learning_curves(self, learning_curve_data: Dict, model_name: str):
        """Plot learning curves."""
        plt.figure(figsize=(10, 6))
        
        train_sizes = learning_curve_data['train_sizes']
        train_mean = learning_curve_data['train_scores_mean']
        train_std = learning_curve_data['train_scores_std']
        val_mean = learning_curve_data['val_scores_mean']
        val_std = learning_curve_data['val_scores_std']
        
        plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training score')
        plt.fill_between(train_sizes, 
                        np.array(train_mean) - np.array(train_std),
                        np.array(train_mean) + np.array(train_std),
                        alpha=0.1, color='blue')
        
        plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validation score')
        plt.fill_between(train_sizes,
                        np.array(val_mean) - np.array(val_std),
                        np.array(val_mean) + np.array(val_std),
                        alpha=0.1, color='red')
        
        plt.xlabel('Training Set Size')
        plt.ylabel('F1 Score')
        plt.title(f'Learning Curves - {model_name}')
        plt.legend(loc='best')
        plt.grid(True)
        
        if self.config.save_plots:
            plt.savefig(f"{self.config.plot_dir}/learning_curves_{model_name}.png", 
                       dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_validation_curves(self, validation_data: Dict, param_name: str, model_name: str):
        """Plot validation curves."""
        plt.figure(figsize=(10, 6))
        
        param_values = validation_data['param_values']
        train_mean = validation_data['train_scores_mean']
        train_std = validation_data['train_scores_std']
        val_mean = validation_data['val_scores_mean']
        val_std = validation_data['val_scores_std']
        
        plt.plot(param_values, train_mean, 'o-', color='blue', label='Training score')
        plt.fill_between(param_values,
                        np.array(train_mean) - np.array(train_std),
                        np.array(train_mean) + np.array(train_std),
                        alpha=0.1, color='blue')
        
        plt.plot(param_values, val_mean, 'o-', color='red', label='Validation score')
        plt.fill_between(param_values,
                        np.array(val_mean) - np.array(val_std),
                        np.array(val_mean) + np.array(val_std),
                        alpha=0.1, color='red')
        
        plt.xlabel(param_name)
        plt.ylabel('F1 Score')
        plt.title(f'Validation Curve - {param_name} - {model_name}')
        plt.legend(loc='best')
        plt.grid(True)
        
        if self.config.save_plots:
            plt.savefig(f"{self.config.plot_dir}/validation_curve_{param_name}_{model_name}.png", 
                       dpi=300, bbox_inches='tight')
        plt.close()
    
    def _calculate_confidence_interval(self, scores: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for scores."""
        from scipy import stats
        
        mean = scores.mean()
        sem = stats.sem(scores)
        h = sem * stats.t.ppf((1 + confidence) / 2., len(scores) - 1)
        
        return (mean - h, mean + h)
    
    def compare_models(self, model_results: List[Dict]) -> Dict:
        """Compare multiple model evaluation results."""
        logger.info(f"🔍 Comparing {len(model_results)} models...")
        
        comparison = {
            'model_names': [result['model_name'] for result in model_results],
            'performance_comparison': {},
            'statistical_comparison': {},
            'efficiency_comparison': {}
        }
        
        # Performance comparison
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        for metric in metrics:
            comparison['performance_comparison'][metric] = {
                result['model_name']: result['basic_metrics'][metric]
                for result in model_results
            }
        
        # Cross-validation comparison
        if all('cross_validation' in result for result in model_results):
            comparison['cv_comparison'] = {}
            for metric in self.config.cv_scoring:
                comparison['cv_comparison'][metric] = {
                    result['model_name']: result['cross_validation'][metric]['mean']
                    for result in model_results
                    if result['cross_validation'][metric] is not None
                }
        
        # Efficiency comparison
        if all('efficiency_metrics' in result for result in model_results):
            efficiency_metrics = ['training_time', 'prediction_time_per_sample', 'memory_usage_mb']
            for metric in efficiency_metrics:
                if all(metric in result['efficiency_metrics'] for result in model_results):
                    comparison['efficiency_comparison'][metric] = {
                        result['model_name']: result['efficiency_metrics'][metric]
                        for result in model_results
                    }
        
        # Find best model for each metric
        comparison['best_models'] = {}
        for metric in metrics:
            best_model = max(
                model_results,
                key=lambda x: x['basic_metrics'][metric]
            )
            comparison['best_models'][metric] = {
                'model_name': best_model['model_name'],
                'score': best_model['basic_metrics'][metric]
            }
        
        return comparison
    
    def generate_evaluation_report(self, model_results: List[Dict], output_path: str = None) -> str:
        """Generate comprehensive evaluation report."""
        logger.info("📄 Generating evaluation report...")
        
        report_lines = []
        
        # Header
        report_lines.append("=" * 80)
        report_lines.append("COMPREHENSIVE MODEL EVALUATION REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Model comparison
        if len(model_results) > 1:
            comparison = self.compare_models(model_results)
            
            report_lines.append("MODEL PERFORMANCE COMPARISON")
            report_lines.append("-" * 40)
            
            for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']:
                report_lines.append(f"\n{metric.upper()}:")
                for model_name, score in comparison['performance_comparison'][metric].items():
                    report_lines.append(f"  {model_name}: {score:.4f}")
                
                best = comparison['best_models'][metric]
                report_lines.append(f"  → Best: {best['model_name']} ({best['score']:.4f})")
            
            report_lines.append("")
        
        # Detailed results for each model
        for result in model_results:
            model_name = result['model_name']
            
            report_lines.append(f"DETAILED RESULTS - {model_name.upper()}")
            report_lines.append("-" * 40)
            
            # Basic metrics
            basic = result['basic_metrics']
            report_lines.append("Basic Performance Metrics:")
            report_lines.append(f"  Accuracy:  {basic['accuracy']:.4f}")
            report_lines.append(f"  Precision: {basic['precision']:.4f}")
            report_lines.append(f"  Recall:    {basic['recall']:.4f}")
            report_lines.append(f"  F1-Score:  {basic['f1_score']:.4f}")
            report_lines.append(f"  ROC-AUC:   {basic['roc_auc']:.4f}")
            
            # Cross-validation results
            if 'cross_validation' in result:
                report_lines.append("\nCross-Validation Results:")
                cv = result['cross_validation']
                for metric, scores in cv.items():
                    if scores:
                        report_lines.append(f"  {metric}: {scores['mean']:.4f} ± {scores['std']:.4f}")
            
            # Statistical analysis
            if 'statistical_analysis' in result:
                stats = result['statistical_analysis']
                report_lines.append("\nStatistical Analysis:")
                report_lines.append(f"  Mean CV Score: {stats['mean_cv_score']:.4f}")
                report_lines.append(f"  95% CI: {stats['confidence_interval_95']}")
                report_lines.append(f"  Significantly better than random: {stats['significantly_better_than_random']}")
            
            # Efficiency metrics
            if 'efficiency_metrics' in result:
                eff = result['efficiency_metrics']
                report_lines.append("\nEfficiency Metrics:")
                if 'training_time' in eff:
                    report_lines.append(f"  Training Time: {eff['training_time']:.2f}s")
                report_lines.append(f"  Prediction Time: {eff['prediction_time_per_sample']*1000:.2f}ms per sample")
                if 'memory_usage_mb' in eff:
                    report_lines.append(f"  Memory Usage: {eff['memory_usage_mb']:.1f} MB")
            
            report_lines.append("")
        
        report_text = "\n".join(report_lines)
        
        # Save report if path provided
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
            logger.info(f"Evaluation report saved to {output_path}")
        
        return report_text


def main():
    """Example usage of ModelEvaluator."""
    # Create sample data
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_clusters_per_class=1,
        weights=[0.6, 0.4],
        random_state=42
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    
    # Train models
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb_model.fit(X_train, y_train)
    
    # Configure evaluation
    config = EvaluationConfig(
        cv_folds=5,
        generate_learning_curves=True,
        generate_validation_curves=True,
        enable_benchmarking=True,
        generate_plots=True,
        save_plots=True
    )
    
    # Initialize evaluator
    evaluator = ModelEvaluator(config)
    
    # Evaluate models
    rf_results = evaluator.evaluate_model_comprehensive(
        rf_model, X_train, y_train, X_test, y_test, "RandomForest"
    )
    
    gb_results = evaluator.evaluate_model_comprehensive(
        gb_model, X_train, y_train, X_test, y_test, "GradientBoosting"
    )
    
    # Compare models
    comparison = evaluator.compare_models([rf_results, gb_results])
    
    print("🎯 Model Evaluation Results:")
    print(f"RandomForest F1-Score: {rf_results['basic_metrics']['f1_score']:.3f}")
    print(f"GradientBoosting F1-Score: {gb_results['basic_metrics']['f1_score']:.3f}")
    
    print(f"\n🏆 Best Models by Metric:")
    for metric, best in comparison['best_models'].items():
        print(f"  {metric}: {best['model_name']} ({best['score']:.3f})")
    
    # Generate comprehensive report
    report = evaluator.generate_evaluation_report(
        [rf_results, gb_results], 
        "evaluation_report.txt"
    )
    
    print(f"\n📄 Evaluation report generated!")
    print(f"📊 Plots saved to: {config.plot_dir}/")


if __name__ == "__main__":
    main()