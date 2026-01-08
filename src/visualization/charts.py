"""
Chart generation utilities for Smart Contract AI Analyzer

This module provides functions to create various types of charts and visualizations
for analysis results, including pie charts, bar charts, gauges, and heatmaps.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def create_vulnerability_pie_chart(vulnerabilities: List[Dict[str, Any]], 
                                 title: str = "Vulnerability Distribution") -> go.Figure:
    """
    Create a pie chart showing vulnerability type distribution.
    
    Args:
        vulnerabilities: List of vulnerability dictionaries
        title: Chart title
        
    Returns:
        Plotly figure object
    """
    try:
        if not vulnerabilities:
            # Create empty chart
            fig = go.Figure()
            fig.add_annotation(
                text="No vulnerabilities detected",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font_size=16
            )
            fig.update_layout(title=title)
            return fig
        
        # Count vulnerability types
        vuln_counts = {}
        for vuln in vulnerabilities:
            vuln_type = vuln.get('type', vuln.get('vulnerability_type', 'Unknown'))
            vuln_counts[vuln_type] = vuln_counts.get(vuln_type, 0) + 1
        
        # Create pie chart
        fig = px.pie(
            values=list(vuln_counts.values()),
            names=[name.replace('_', ' ').title() for name in vuln_counts.keys()],
            title=title,
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
        )
        
        fig.update_layout(
            showlegend=True,
            legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.01)
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating vulnerability pie chart: {e}")
        return _create_error_chart(f"Error creating chart: {e}")


def create_severity_bar_chart(vulnerabilities: List[Dict[str, Any]], 
                             title: str = "Vulnerabilities by Severity") -> go.Figure:
    """
    Create a bar chart showing vulnerability severity distribution.
    
    Args:
        vulnerabilities: List of vulnerability dictionaries
        title: Chart title
        
    Returns:
        Plotly figure object
    """
    try:
        if not vulnerabilities:
            return _create_empty_chart("No vulnerabilities to display", title)
        
        # Count severities
        severity_counts = {}
        for vuln in vulnerabilities:
            severity = vuln.get('severity', 'Unknown')
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # Define severity order and colors
        severity_order = ['Critical', 'High', 'Medium', 'Low', 'Informational', 'Unknown']
        severity_colors = {
            'Critical': '#FF0000',
            'High': '#FF6600', 
            'Medium': '#FFAA00',
            'Low': '#FFDD00',
            'Informational': '#00AA00',
            'Unknown': '#888888'
        }
        
        # Prepare data
        severities = []
        counts = []
        colors = []
        
        for severity in severity_order:
            if severity in severity_counts:
                severities.append(severity)
                counts.append(severity_counts[severity])
                colors.append(severity_colors[severity])
        
        # Create bar chart
        fig = go.Figure(data=[
            go.Bar(
                x=severities,
                y=counts,
                marker_color=colors,
                text=counts,
                textposition='auto',
                hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title=title,
            xaxis_title="Severity Level",
            yaxis_title="Number of Vulnerabilities",
            showlegend=False
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating severity bar chart: {e}")
        return _create_error_chart(f"Error creating chart: {e}")


def create_feature_importance_chart(feature_importance: Dict[str, float], 
                                  top_n: int = 15,
                                  title: str = "Feature Importance") -> go.Figure:
    """
    Create a horizontal bar chart showing feature importance.
    
    Args:
        feature_importance: Dictionary of feature names and importance scores
        top_n: Number of top features to display
        title: Chart title
        
    Returns:
        Plotly figure object
    """
    try:
        if not feature_importance:
            return _create_empty_chart("No feature importance data available", title)
        
        # Sort features by importance and take top N
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        features = [item[0].replace('_', ' ').title() for item in sorted_features]
        importance_scores = [item[1] for item in sorted_features]
        
        # Create horizontal bar chart
        fig = go.Figure(data=[
            go.Bar(
                y=features,
                x=importance_scores,
                orientation='h',
                marker=dict(
                    color=importance_scores,
                    colorscale='viridis',
                    showscale=True,
                    colorbar=dict(title="Importance Score")
                ),
                text=[f"{score:.3f}" for score in importance_scores],
                textposition='auto',
                hovertemplate='<b>%{y}</b><br>Importance: %{x:.3f}<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title=title,
            xaxis_title="Importance Score",
            yaxis_title="Features",
            height=max(400, len(features) * 25),
            yaxis=dict(autorange="reversed")  # Show highest importance at top
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating feature importance chart: {e}")
        return _create_error_chart(f"Error creating chart: {e}")


def create_tool_comparison_chart(tool_results: Dict[str, Dict[str, Any]], 
                               title: str = "Tool Performance Comparison") -> go.Figure:
    """
    Create a grouped bar chart comparing tool performance metrics.
    
    Args:
        tool_results: Dictionary of tool names and their performance metrics
        title: Chart title
        
    Returns:
        Plotly figure object
    """
    try:
        if not tool_results:
            return _create_empty_chart("No tool comparison data available", title)
        
        # Prepare data
        tools = list(tool_results.keys())
        metrics = ['execution_time', 'findings_count', 'success_rate']
        
        data = []
        for metric in metrics:
            values = []
            for tool in tools:
                tool_data = tool_results[tool]
                if metric == 'execution_time':
                    values.append(tool_data.get('execution_time', 0))
                elif metric == 'findings_count':
                    values.append(len(tool_data.get('vulnerabilities_found', [])))
                elif metric == 'success_rate':
                    values.append(100 if tool_data.get('success', False) else 0)
            
            data.append(go.Bar(
                name=metric.replace('_', ' ').title(),
                x=tools,
                y=values,
                hovertemplate=f'<b>%{{x}}</b><br>{metric.replace("_", " ").title()}: %{{y}}<extra></extra>'
            ))
        
        fig = go.Figure(data=data)
        
        fig.update_layout(
            title=title,
            xaxis_title="Tools",
            yaxis_title="Value",
            barmode='group',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating tool comparison chart: {e}")
        return _create_error_chart(f"Error creating chart: {e}")


def create_risk_score_gauge(risk_score: float, 
                          title: str = "Risk Assessment") -> go.Figure:
    """
    Create a gauge chart showing risk score.
    
    Args:
        risk_score: Risk score (0-100)
        title: Chart title
        
    Returns:
        Plotly figure object
    """
    try:
        # Determine risk level and color
        if risk_score >= 80:
            risk_level = "CRITICAL"
            color = "red"
        elif risk_score >= 60:
            risk_level = "HIGH"
            color = "orange"
        elif risk_score >= 40:
            risk_level = "MEDIUM"
            color = "yellow"
        elif risk_score >= 20:
            risk_level = "LOW"
            color = "lightgreen"
        else:
            risk_level = "MINIMAL"
            color = "green"
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = risk_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': f"{title}<br><span style='font-size:0.8em;color:gray'>Risk Level: {risk_level}</span>"},
            delta = {'reference': 50},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 20], 'color': "lightgray"},
                    {'range': [20, 40], 'color': "lightblue"},
                    {'range': [40, 60], 'color': "lightyellow"},
                    {'range': [60, 80], 'color': "lightcoral"},
                    {'range': [80, 100], 'color': "lightpink"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(height=400)
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating risk score gauge: {e}")
        return _create_error_chart(f"Error creating chart: {e}")


def create_confusion_matrix_heatmap(confusion_matrix: np.ndarray, 
                                  labels: List[str],
                                  title: str = "Confusion Matrix") -> go.Figure:
    """
    Create a heatmap visualization of a confusion matrix.
    
    Args:
        confusion_matrix: 2D numpy array representing the confusion matrix
        labels: List of class labels
        title: Chart title
        
    Returns:
        Plotly figure object
    """
    try:
        # Normalize confusion matrix for better visualization
        cm_normalized = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        
        # Create annotations for the heatmap
        annotations = []
        for i in range(len(labels)):
            for j in range(len(labels)):
                annotations.append(
                    dict(
                        x=j, y=i,
                        text=f"{confusion_matrix[i, j]}<br>({cm_normalized[i, j]:.2%})",
                        showarrow=False,
                        font=dict(color="white" if cm_normalized[i, j] > 0.5 else "black")
                    )
                )
        
        fig = go.Figure(data=go.Heatmap(
            z=cm_normalized,
            x=labels,
            y=labels,
            colorscale='Blues',
            showscale=True,
            colorbar=dict(title="Normalized Count")
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Predicted",
            yaxis_title="Actual",
            annotations=annotations,
            width=500,
            height=500
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating confusion matrix heatmap: {e}")
        return _create_error_chart(f"Error creating chart: {e}")


def create_vulnerability_timeline(analysis_history: List[Dict[str, Any]], 
                                title: str = "Vulnerability Detection Timeline") -> go.Figure:
    """
    Create a timeline chart showing vulnerability detection over time.
    
    Args:
        analysis_history: List of analysis results with timestamps
        title: Chart title
        
    Returns:
        Plotly figure object
    """
    try:
        if not analysis_history:
            return _create_empty_chart("No analysis history available", title)
        
        # Convert to DataFrame for easier processing
        df = pd.DataFrame(analysis_history)
        
        if 'timestamp' not in df.columns:
            return _create_empty_chart("No timestamp data available", title)
        
        # Convert timestamps and group by date
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date'] = df['timestamp'].dt.date
        
        # Count vulnerabilities per day
        daily_stats = df.groupby('date').agg({
            'is_vulnerable': ['count', 'sum']
        }).reset_index()
        
        daily_stats.columns = ['date', 'total_analyses', 'vulnerable_count']
        daily_stats['safe_count'] = daily_stats['total_analyses'] - daily_stats['vulnerable_count']
        
        # Create stacked bar chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Safe Contracts',
            x=daily_stats['date'],
            y=daily_stats['safe_count'],
            marker_color='green',
            hovertemplate='<b>%{x}</b><br>Safe: %{y}<extra></extra>'
        ))
        
        fig.add_trace(go.Bar(
            name='Vulnerable Contracts',
            x=daily_stats['date'],
            y=daily_stats['vulnerable_count'],
            marker_color='red',
            hovertemplate='<b>%{x}</b><br>Vulnerable: %{y}<extra></extra>'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Number of Contracts",
            barmode='stack',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating vulnerability timeline: {e}")
        return _create_error_chart(f"Error creating chart: {e}")


def create_model_performance_radar(performance_metrics: Dict[str, float], 
                                 title: str = "Model Performance Radar") -> go.Figure:
    """
    Create a radar chart showing model performance across multiple metrics.
    
    Args:
        performance_metrics: Dictionary of metric names and values
        title: Chart title
        
    Returns:
        Plotly figure object
    """
    try:
        if not performance_metrics:
            return _create_empty_chart("No performance metrics available", title)
        
        metrics = list(performance_metrics.keys())
        values = list(performance_metrics.values())
        
        # Close the radar chart by repeating the first value
        metrics.append(metrics[0])
        values.append(values[0])
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=metrics,
            fill='toself',
            name='Performance',
            line_color='blue',
            fillcolor='rgba(0, 0, 255, 0.1)'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=False,
            title=title
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating radar chart: {e}")
        return _create_error_chart(f"Error creating chart: {e}")


def _create_empty_chart(message: str, title: str) -> go.Figure:
    """Create an empty chart with a message."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5, xanchor='center', yanchor='middle',
        showarrow=False, font_size=16
    )
    fig.update_layout(title=title)
    return fig


def _create_error_chart(error_message: str) -> go.Figure:
    """Create an error chart with error message."""
    fig = go.Figure()
    fig.add_annotation(
        text=f"Error: {error_message}",
        xref="paper", yref="paper",
        x=0.5, y=0.5, xanchor='center', yanchor='middle',
        showarrow=False, font_size=14, font_color="red"
    )
    fig.update_layout(title="Chart Error")
    return fig


# Utility functions for chart styling
def get_vulnerability_colors() -> Dict[str, str]:
    """Get standard colors for vulnerability types."""
    return {
        'reentrancy': '#FF4444',
        'access_control': '#FF8800',
        'arithmetic': '#FFAA00',
        'unchecked_calls': '#44AA44',
        'dos': '#8844AA',
        'bad_randomness': '#AA4488',
        'safe': '#00AA00'
    }


def get_severity_colors() -> Dict[str, str]:
    """Get standard colors for severity levels."""
    return {
        'Critical': '#FF0000',
        'High': '#FF6600',
        'Medium': '#FFAA00', 
        'Low': '#FFDD00',
        'Informational': '#00AA00',
        'Unknown': '#888888'
    }


def apply_standard_layout(fig: go.Figure, 
                        title: str = None,
                        width: int = None,
                        height: int = None) -> go.Figure:
    """Apply standard layout styling to a figure."""
    layout_updates = {
        'font': dict(family="Arial, sans-serif", size=12),
        'plot_bgcolor': 'white',
        'paper_bgcolor': 'white',
        'margin': dict(l=50, r=50, t=80, b=50)
    }
    
    if title:
        layout_updates['title'] = dict(
            text=title,
            x=0.5,
            xanchor='center',
            font=dict(size=16, color='#2E86AB')
        )
    
    if width:
        layout_updates['width'] = width
    
    if height:
        layout_updates['height'] = height
    
    fig.update_layout(**layout_updates)
    return fig