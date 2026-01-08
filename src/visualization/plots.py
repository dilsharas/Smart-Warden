"""
Advanced plotting utilities for Smart Contract AI Analyzer

This module provides functions to create complex plots and visualizations
for analysis trends, model performance, and comparative analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

# Set style for matplotlib plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def plot_analysis_trends(analysis_history: List[Dict[str, Any]], 
                        save_path: Optional[str] = None,
                        show_plot: bool = True) -> Optional[plt.Figure]:
    """
    Create comprehensive analysis trends visualization.
    
    Args:
        analysis_history: List of analysis results with timestamps
        save_path: Path to save the plot
        show_plot: Whether to display the plot
        
    Returns:
        Matplotlib figure object or None
    """
    try:
        if not analysis_history:
            logger.warning("No analysis history provided")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(analysis_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date'] = df['timestamp'].dt.date
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Smart Contract Analysis Trends', fontsize=16, fontweight='bold')
        
        # 1. Daily analysis count
        daily_counts = df.groupby('date').size()
        axes[0, 0].plot(daily_counts.index, daily_counts.values, marker='o', linewidth=2)
        axes[0, 0].set_title('Daily Analysis Count')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Number of Analyses')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Vulnerability detection rate
        if 'is_vulnerable' in df.columns:
            daily_vuln_rate = df.groupby('date')['is_vulnerable'].mean() * 100
            axes[0, 1].plot(daily_vuln_rate.index, daily_vuln_rate.values, 
                           marker='s', color='red', linewidth=2)
            axes[0, 1].set_title('Daily Vulnerability Detection Rate')
            axes[0, 1].set_xlabel('Date')
            axes[0, 1].set_ylabel('Vulnerability Rate (%)')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Risk score distribution
        if 'risk_score' in df.columns:
            axes[1, 0].hist(df['risk_score'], bins=20, alpha=0.7, color='orange', edgecolor='black')
            axes[1, 0].set_title('Risk Score Distribution')
            axes[1, 0].set_xlabel('Risk Score')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Analysis time trends
        if 'analysis_time' in df.columns:
            daily_avg_time = df.groupby('date')['analysis_time'].mean()
            axes[1, 1].plot(daily_avg_time.index, daily_avg_time.values, 
                           marker='^', color='green', linewidth=2)
            axes[1, 1].set_title('Average Analysis Time')
            axes[1, 1].set_xlabel('Date')
            axes[1, 1].set_ylabel('Time (seconds)')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Analysis trends plot saved to {save_path}")
        
        if show_plot:
            plt.show()
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating analysis trends plot: {e}")
        return None

def
 plot_model_performance(performance_data: Dict[str, Dict[str, float]], 
                          save_path: Optional[str] = None,
                          show_plot: bool = True) -> Optional[plt.Figure]:
    """
    Create model performance comparison visualization.
    
    Args:
        performance_data: Dictionary of model names and their performance metrics
        save_path: Path to save the plot
        show_plot: Whether to display the plot
        
    Returns:
        Matplotlib figure object or None
    """
    try:
        if not performance_data:
            logger.warning("No performance data provided")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(performance_data).T
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        # 1. Performance metrics bar chart
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        available_metrics = [m for m in metrics if m in df.columns]
        
        if available_metrics:
            df[available_metrics].plot(kind='bar', ax=axes[0, 0], width=0.8)
            axes[0, 0].set_title('Performance Metrics Comparison')
            axes[0, 0].set_xlabel('Models')
            axes[0, 0].set_ylabel('Score')
            axes[0, 0].legend(title='Metrics')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].set_xticklabels(df.index, rotation=45)
        
        # 2. ROC curves (if available)
        if 'fpr' in df.columns and 'tpr' in df.columns:
            for model_name in df.index:
                fpr = df.loc[model_name, 'fpr']
                tpr = df.loc[model_name, 'tpr']
                if isinstance(fpr, (list, np.ndarray)) and isinstance(tpr, (list, np.ndarray)):
                    axes[0, 1].plot(fpr, tpr, label=f'{model_name}', linewidth=2)
            
            axes[0, 1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
            axes[0, 1].set_title('ROC Curves')
            axes[0, 1].set_xlabel('False Positive Rate')
            axes[0, 1].set_ylabel('True Positive Rate')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Training time comparison
        if 'training_time' in df.columns:
            df['training_time'].plot(kind='bar', ax=axes[1, 0], color='skyblue')
            axes[1, 0].set_title('Training Time Comparison')
            axes[1, 0].set_xlabel('Models')
            axes[1, 0].set_ylabel('Time (seconds)')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].set_xticklabels(df.index, rotation=45)
        
        # 4. Model complexity vs performance
        if 'complexity' in df.columns and 'accuracy' in df.columns:
            scatter = axes[1, 1].scatter(df['complexity'], df['accuracy'], 
                                       s=100, alpha=0.7, c=range(len(df)), cmap='viridis')
            
            for i, model in enumerate(df.index):
                axes[1, 1].annotate(model, (df.loc[model, 'complexity'], df.loc[model, 'accuracy']),
                                  xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            axes[1, 1].set_title('Model Complexity vs Accuracy')
            axes[1, 1].set_xlabel('Model Complexity')
            axes[1, 1].set_ylabel('Accuracy')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Model performance plot saved to {save_path}")
        
        if show_plot:
            plt.show()
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating model performance plot: {e}")
        return None


def plot_tool_performance_comparison(tool_results: Dict[str, Dict[str, Any]], 
                                   save_path: Optional[str] = None,
                                   show_plot: bool = True) -> Optional[plt.Figure]:
    """
    Create tool performance comparison visualization.
    
    Args:
        tool_results: Dictionary of tool names and their results
        save_path: Path to save the plot
        show_path: Whether to display the plot
        
    Returns:
        Matplotlib figure object or None
    """
    try:
        if not tool_results:
            logger.warning("No tool results provided")
            return None
        
        # Prepare data
        tools = list(tool_results.keys())
        execution_times = []
        success_rates = []
        findings_counts = []
        
        for tool in tools:
            result = tool_results[tool]
            execution_times.append(result.get('execution_time', 0))
            success_rates.append(100 if result.get('success', False) else 0)
            findings_counts.append(len(result.get('vulnerabilities_found', [])))
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Tool Performance Comparison', fontsize=16, fontweight='bold')
        
        # 1. Execution time comparison
        bars1 = axes[0, 0].bar(tools, execution_times, color='lightblue', edgecolor='navy')
        axes[0, 0].set_title('Execution Time Comparison')
        axes[0, 0].set_xlabel('Tools')
        axes[0, 0].set_ylabel('Time (seconds)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, time in zip(bars1, execution_times):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                           f'{time:.1f}s', ha='center', va='bottom')
        
        # 2. Success rate comparison
        bars2 = axes[0, 1].bar(tools, success_rates, color='lightgreen', edgecolor='darkgreen')
        axes[0, 1].set_title('Success Rate Comparison')
        axes[0, 1].set_xlabel('Tools')
        axes[0, 1].set_ylabel('Success Rate (%)')
        axes[0, 1].set_ylim(0, 105)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, rate in zip(bars2, success_rates):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                           f'{rate:.0f}%', ha='center', va='bottom')
        
        # 3. Findings count comparison
        bars3 = axes[1, 0].bar(tools, findings_counts, color='lightcoral', edgecolor='darkred')
        axes[1, 0].set_title('Vulnerabilities Found')
        axes[1, 0].set_xlabel('Tools')
        axes[1, 0].set_ylabel('Number of Vulnerabilities')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, count in zip(bars3, findings_counts):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                           f'{count}', ha='center', va='bottom')
        
        # 4. Performance radar chart
        if len(tools) > 0:
            # Normalize metrics for radar chart
            max_time = max(execution_times) if max(execution_times) > 0 else 1
            normalized_times = [1 - (t / max_time) for t in execution_times]  # Invert so faster = better
            normalized_success = [r / 100 for r in success_rates]
            max_findings = max(findings_counts) if max(findings_counts) > 0 else 1
            normalized_findings = [f / max_findings for f in findings_counts]
            
            # Create radar chart data
            angles = np.linspace(0, 2 * np.pi, 3, endpoint=False).tolist()
            angles += angles[:1]  # Complete the circle
            
            ax = axes[1, 1]
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)
            ax.set_thetagrids(np.degrees(angles[:-1]), ['Speed', 'Success Rate', 'Findings'])
            
            for i, tool in enumerate(tools):
                values = [normalized_times[i], normalized_success[i], normalized_findings[i]]
                values += values[:1]  # Complete the circle
                
                ax.plot(angles, values, 'o-', linewidth=2, label=tool)
                ax.fill(angles, values, alpha=0.25)
            
            ax.set_ylim(0, 1)
            ax.set_title('Performance Radar Chart')
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Tool performance plot saved to {save_path}")
        
        if show_plot:
            plt.show()
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating tool performance plot: {e}")
        return None


def plot_vulnerability_timeline(vulnerabilities: List[Dict[str, Any]], 
                              save_path: Optional[str] = None,
                              show_plot: bool = True) -> Optional[plt.Figure]:
    """
    Create vulnerability detection timeline visualization.
    
    Args:
        vulnerabilities: List of vulnerability data with timestamps
        save_path: Path to save the plot
        show_plot: Whether to display the plot
        
    Returns:
        Matplotlib figure object or None
    """
    try:
        if not vulnerabilities:
            logger.warning("No vulnerability data provided")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(vulnerabilities)
        
        if 'timestamp' not in df.columns:
            logger.warning("No timestamp data in vulnerabilities")
            return None
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date'] = df['timestamp'].dt.date
        
        # Create figure
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        fig.suptitle('Vulnerability Detection Timeline', fontsize=16, fontweight='bold')
        
        # 1. Daily vulnerability count by type
        if 'type' in df.columns:
            vuln_by_date_type = df.groupby(['date', 'type']).size().unstack(fill_value=0)
            vuln_by_date_type.plot(kind='bar', stacked=True, ax=axes[0], 
                                 colormap='Set3', width=0.8)
            axes[0].set_title('Daily Vulnerabilities by Type')
            axes[0].set_xlabel('Date')
            axes[0].set_ylabel('Number of Vulnerabilities')
            axes[0].legend(title='Vulnerability Type', bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[0].grid(True, alpha=0.3)
        
        # 2. Severity timeline
        if 'severity' in df.columns:
            severity_by_date = df.groupby(['date', 'severity']).size().unstack(fill_value=0)
            
            # Define severity colors
            severity_colors = {'Critical': 'red', 'High': 'orange', 'Medium': 'yellow', 
                             'Low': 'lightgreen', 'Informational': 'blue'}
            
            severity_by_date.plot(kind='area', ax=axes[1], 
                                color=[severity_colors.get(col, 'gray') for col in severity_by_date.columns],
                                alpha=0.7)
            axes[1].set_title('Vulnerability Severity Timeline')
            axes[1].set_xlabel('Date')
            axes[1].set_ylabel('Number of Vulnerabilities')
            axes[1].legend(title='Severity', bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Vulnerability timeline plot saved to {save_path}")
        
        if show_plot:
            plt.show()
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating vulnerability timeline plot: {e}")
        return None


def plot_risk_distribution(risk_scores: List[float], 
                          labels: Optional[List[str]] = None,
                          save_path: Optional[str] = None,
                          show_plot: bool = True) -> Optional[plt.Figure]:
    """
    Create risk score distribution visualization.
    
    Args:
        risk_scores: List of risk scores
        labels: Optional labels for each risk score
        save_path: Path to save the plot
        show_plot: Whether to display the plot
        
    Returns:
        Matplotlib figure object or None
    """
    try:
        if not risk_scores:
            logger.warning("No risk scores provided")
            return None
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Risk Score Distribution Analysis', fontsize=16, fontweight='bold')
        
        # 1. Histogram
        axes[0, 0].hist(risk_scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].axvline(np.mean(risk_scores), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(risk_scores):.2f}')
        axes[0, 0].axvline(np.median(risk_scores), color='green', linestyle='--', 
                          label=f'Median: {np.median(risk_scores):.2f}')
        axes[0, 0].set_title('Risk Score Histogram')
        axes[0, 0].set_xlabel('Risk Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Box plot
        axes[0, 1].boxplot(risk_scores, vert=True, patch_artist=True,
                          boxprops=dict(facecolor='lightblue', alpha=0.7))
        axes[0, 1].set_title('Risk Score Box Plot')
        axes[0, 1].set_ylabel('Risk Score')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Risk categories pie chart
        risk_categories = {'Low (0-25)': 0, 'Medium (25-50)': 0, 'High (50-75)': 0, 'Critical (75-100)': 0}
        
        for score in risk_scores:
            if score <= 25:
                risk_categories['Low (0-25)'] += 1
            elif score <= 50:
                risk_categories['Medium (25-50)'] += 1
            elif score <= 75:
                risk_categories['High (50-75)'] += 1
            else:
                risk_categories['Critical (75-100)'] += 1
        
        # Filter out zero categories
        risk_categories = {k: v for k, v in risk_categories.items() if v > 0}
        
        if risk_categories:
            colors = ['green', 'yellow', 'orange', 'red'][:len(risk_categories)]
            axes[1, 0].pie(risk_categories.values(), labels=risk_categories.keys(), 
                          autopct='%1.1f%%', colors=colors, startangle=90)
            axes[1, 0].set_title('Risk Category Distribution')
        
        # 4. Cumulative distribution
        sorted_scores = np.sort(risk_scores)
        cumulative = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
        
        axes[1, 1].plot(sorted_scores, cumulative, marker='o', markersize=3, linewidth=2)
        axes[1, 1].set_title('Cumulative Risk Score Distribution')
        axes[1, 1].set_xlabel('Risk Score')
        axes[1, 1].set_ylabel('Cumulative Probability')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Risk distribution plot saved to {save_path}")
        
        if show_plot:
            plt.show()
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating risk distribution plot: {e}")
        return None