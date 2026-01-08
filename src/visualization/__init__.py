"""
Visualization utilities for Smart Contract AI Analyzer

This module provides comprehensive visualization capabilities for analysis results,
including charts, graphs, and interactive visualizations.
"""

from .charts import (
    create_vulnerability_pie_chart,
    create_severity_bar_chart,
    create_feature_importance_chart,
    create_tool_comparison_chart,
    create_risk_score_gauge,
    create_confusion_matrix_heatmap
)

from .plots import (
    plot_analysis_trends,
    plot_model_performance,
    plot_tool_performance_comparison,
    plot_vulnerability_timeline,
    plot_risk_distribution
)

from .interactive import (
    create_interactive_dashboard,
    create_vulnerability_explorer,
    create_comparison_matrix,
    create_performance_monitor
)

__all__ = [
    # Charts
    'create_vulnerability_pie_chart',
    'create_severity_bar_chart', 
    'create_feature_importance_chart',
    'create_tool_comparison_chart',
    'create_risk_score_gauge',
    'create_confusion_matrix_heatmap',
    
    # Plots
    'plot_analysis_trends',
    'plot_model_performance',
    'plot_tool_performance_comparison',
    'plot_vulnerability_timeline',
    'plot_risk_distribution',
    
    # Interactive
    'create_interactive_dashboard',
    'create_vulnerability_explorer',
    'create_comparison_matrix',
    'create_performance_monitor'
]