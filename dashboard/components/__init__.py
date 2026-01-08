"""
Dashboard components for the Smart Contract Security Analyzer.
"""

from .upload_widget import (
    show_upload_widget,
    show_drag_drop_area,
    validate_solidity_file,
    show_file_preview,
    show_upload_stats,
    show_sample_contracts as upload_samples
)

from .code_editor import (
    show_code_editor, 
    show_monaco_editor, 
    show_code_with_highlighting,
    show_sample_contracts,
    create_code_input_section
)

from .vulnerability_card import (
    show_vulnerability_card,
    show_vulnerability_summary,
    show_vulnerability_timeline,
    show_vulnerability_heatmap,
    show_vulnerability_comparison,
    create_vulnerability_filter
)

from .metric_display import (
    show_risk_score_gauge,
    show_analysis_metrics,
    show_feature_importance_chart,
    show_comparison_metrics,
    show_progress_indicator,
    show_security_score_breakdown,
    show_trend_chart,
    show_distribution_chart,
    show_summary_cards,
    create_metric_dashboard
)

__all__ = [
    # Upload widget components
    'show_upload_widget',
    'show_drag_drop_area',
    'validate_solidity_file',
    'show_file_preview',
    'show_upload_stats',
    'upload_samples',
    
    # Code editor components
    'show_code_editor',
    'show_monaco_editor', 
    'show_code_with_highlighting',
    'show_sample_contracts',
    'create_code_input_section',
    
    # Vulnerability display components
    'show_vulnerability_card',
    'show_vulnerability_summary',
    'show_vulnerability_timeline',
    'show_vulnerability_heatmap',
    'show_vulnerability_comparison',
    'create_vulnerability_filter',
    
    # Metric display components
    'show_risk_score_gauge',
    'show_analysis_metrics',
    'show_feature_importance_chart',
    'show_comparison_metrics',
    'show_progress_indicator',
    'show_security_score_breakdown',
    'show_trend_chart',
    'show_distribution_chart',
    'show_summary_cards',
    'create_metric_dashboard'
]