"""
Visualization Module

Generates charts and visualizations for fraud detection analysis.
"""

import logging
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve, auc

from .config import FIGURE_DPI, FIGURE_SIZE, COLORMAP

logger = logging.getLogger(__name__)


class VisualizationEngine:
    """
    Generates visualizations for fraud detection model evaluation.
    
    Creates confusion matrices, ROC curves, feature importance plots, and more.
    """

    def __init__(self, dpi: int = FIGURE_DPI, figsize: tuple = FIGURE_SIZE):
        """
        Initialize the visualization engine.
        
        Args:
            dpi: DPI for saved figures
            figsize: Default figure size (width, height)
        """
        self.dpi = dpi
        self.figsize = figsize
        self.figures = {}
        sns.set_style("whitegrid")

    def plot_confusion_matrix(self, cm: np.ndarray, labels: Optional[List[str]] = None) -> plt.Figure:
        """
        Plot confusion matrix as a heatmap.
        
        Args:
            cm: Confusion matrix
            labels: Class labels
            
        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        if labels is None:
            labels = ['Legitimate', 'Fraudulent']
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=labels, yticklabels=labels, cbar_kws={'label': 'Count'})
        
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        self.figures['confusion_matrix'] = fig
        logger.info("Generated confusion matrix plot")
        
        return fig

    def plot_roc_curve(self, y_true: np.ndarray, y_proba: np.ndarray) -> plt.Figure:
        """
        Plot ROC curve.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            
        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Handle multi-class case
        if y_proba.ndim > 1 and y_proba.shape[1] > 1:
            y_proba = y_proba[:, 1]
        
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self.figures['roc_curve'] = fig
        logger.info("Generated ROC curve plot")
        
        return fig

    def plot_feature_importance(self, feature_names: List[str], importances: np.ndarray,
                               top_n: int = 15) -> plt.Figure:
        """
        Plot feature importance.
        
        Args:
            feature_names: List of feature names
            importances: Feature importance scores
            top_n: Number of top features to display
            
        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=(self.figsize[0], max(8, top_n * 0.3)), dpi=self.dpi)
        
        # Sort features by importance
        indices = np.argsort(importances)[::-1][:top_n]
        top_features = [feature_names[i] for i in indices]
        top_importances = importances[indices]
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
        ax.barh(range(len(top_features)), top_importances, color=colors)
        
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features)
        ax.set_xlabel('Importance Score', fontsize=12)
        ax.set_title(f'Top {top_n} Feature Importance', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        
        plt.tight_layout()
        self.figures['feature_importance'] = fig
        logger.info(f"Generated feature importance plot (top {top_n})")
        
        return fig

    def plot_metrics_comparison(self, metrics: Dict[str, float]) -> plt.Figure:
        """
        Plot metrics comparison as a bar chart.
        
        Args:
            metrics: Dictionary of metric names and values
            
        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Filter out non-numeric metrics
        metric_names = []
        metric_values = []
        for name, value in metrics.items():
            if isinstance(value, (int, float)) and name != 'latency':
                metric_names.append(name.replace('_', ' ').title())
                metric_values.append(value)
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(metric_names)))
        bars = ax.bar(metric_names, metric_values, color=colors, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=10)
        
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Model Performance Metrics', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        self.figures['metrics_comparison'] = fig
        logger.info("Generated metrics comparison plot")
        
        return fig

    def plot_latency_distribution(self, latencies: List[float]) -> plt.Figure:
        """
        Plot latency distribution as a histogram.
        
        Args:
            latencies: List of latency measurements in milliseconds
            
        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        ax.hist(latencies, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
        
        mean_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        
        ax.axvline(mean_latency, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_latency:.2f}ms')
        ax.axvline(p95_latency, color='orange', linestyle='--', linewidth=2, label=f'P95: {p95_latency:.2f}ms')
        ax.axvline(p99_latency, color='green', linestyle='--', linewidth=2, label=f'P99: {p99_latency:.2f}ms')
        
        ax.set_xlabel('Latency (ms)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Prediction Latency Distribution', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        self.figures['latency_distribution'] = fig
        logger.info("Generated latency distribution plot")
        
        return fig

    def save_figure(self, figure_name: str, filepath: str) -> None:
        """
        Save a figure to disk.
        
        Args:
            figure_name: Name of the figure to save
            filepath: Path to save the figure
        """
        if figure_name not in self.figures:
            raise ValueError(f"Figure '{figure_name}' not found. Available: {list(self.figures.keys())}")
        
        fig = self.figures[figure_name]
        fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        logger.info(f"Saved figure '{figure_name}' to {filepath}")

    def save_all_figures(self, directory: str) -> Dict[str, str]:
        """
        Save all generated figures to a directory.
        
        Args:
            directory: Directory to save figures
            
        Returns:
            Dictionary mapping figure names to file paths
        """
        import os
        os.makedirs(directory, exist_ok=True)
        
        saved_files = {}
        for figure_name, fig in self.figures.items():
            filepath = os.path.join(directory, f"{figure_name}.png")
            fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            saved_files[figure_name] = filepath
            logger.info(f"Saved {figure_name} to {filepath}")
        
        return saved_files

    def close_all_figures(self) -> None:
        """Close all generated figures to free memory."""
        for fig in self.figures.values():
            plt.close(fig)
        self.figures.clear()
        logger.info("Closed all figures")

    def get_figure(self, figure_name: str) -> Optional[plt.Figure]:
        """
        Get a specific figure.
        
        Args:
            figure_name: Name of the figure
            
        Returns:
            Matplotlib figure object or None if not found
        """
        return self.figures.get(figure_name)

    def list_figures(self) -> List[str]:
        """
        List all available figures.
        
        Returns:
            List of figure names
        """
        return list(self.figures.keys())
