"""
Interactive visualization utilities for Smart Contract AI Analyzer

This module provides functions to create interactive dashboards and visualizations
using Plotly and other interactive libraries.
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
import streamlit as st

logger = logging.getLogger(__name__)


def create_interactive_dashboard(analysis_results: Dict[str, Any]) -> Dict[str, go.Figure]:
    """
    Create a comprehensive interactive dashboard with multiple visualizations.
    
    Args:
        analysis_results: Analysis results dictionary
        
    Returns:
        Dictionary of figure names and Plotly figures
    """
    try:
        figures = {}
        
        # 1. Risk Assessment Gauge
        if 'risk_score' in analysis_results:
            figures['risk_gauge'] = create_risk_gauge(analysis_results['risk_score'])
        
        # 2. Vulnerability Distribution
        if 'vulnerabilities' in analysis_results:
            figures['vuln_pie'] = create_vulnerability_pie(analysis_results['vulnerabilities'])
            figures['severity_bar'] = create_severity_bar(analysis_results['vulnerabilities'])
        
        # 3. Feature Importance
        if 'feature_importance' in analysis_results:
            figures['feature_importance'] = create_feature_importance_bar(
                analysis_results['feature_importance']
            )
        
        # 4. Tool Comparison
        if 'tool_comparison' in analysis_results:
            figures['tool_comparison'] = create_tool_comparison_radar(
                analysis_results['tool_comparison']
            )
        
        return figures
        
    except Exception as e:
        logger.error(f"Error creating interactive dashboard: {e}")
        return {}


def create_vulnerability_explorer(vulnerabilities: List[Dict[str, Any]]) -> go.Figure:
    """
    Create an interactive vulnerability explorer with filtering and drill-down capabilities.
    
    Args:
        vulnerabilities: List of vulnerability dictionaries
        
    Returns:
        Plotly figure object
    """
    try:
        if not vulnerabilities:
            return _create_empty_interactive_chart("No vulnerabilities to explore")
        
        # Convert to DataFrame
        df = pd.DataFrame(vulnerabilities)
        
        # Create sunburst chart for hierarchical exploration
        fig = go.Figure(go.Sunburst(
            labels=df['type'].tolist() + df['severity'].tolist(),
            parents=[''] * len(df['type']) + df['type'].tolist(),
            values=[1] * len(df) + [1] * len(df),
            branchvalues="total",
            hovertemplate='<b>%{label}</b><br>Count: %{value}<extra></extra>',
            maxdepth=2
        ))
        
        fig.update_layout(
            title="Interactive Vulnerability Explorer",
            font_size=12,
            height=600
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating vulnerability explorer: {e}")
        return _create_empty_interactive_chart(f"Error: {e}")


def create_comparison_matrix(tool_results: Dict[str, Dict[str, Any]]) -> go.Figure:
    """
    Create an interactive comparison matrix for tool results.
    
    Args:
        tool_results: Dictionary of tool names and their results
        
    Returns:
        Plotly figure object
    """
    try:
        if not tool_results:
            return _create_empty_interactive_chart("No tool results to compare")
        
        # Prepare data for heatmap
        tools = list(tool_results.keys())
        metrics = ['execution_time', 'success_rate', 'findings_count', 'accuracy']
        
        # Create matrix data
        matrix_data = []
        for metric in metrics:
            row = []
            for tool in tools:
                result = tool_results[tool]
                
                if metric == 'execution_time':
                    value = result.get('execution_time', 0)
                elif metric == 'success_rate':
                    value = 100 if result.get('success', False) else 0
                elif metric == 'findings_count':
                    value = len(result.get('vulnerabilities_found', []))
                elif metric == 'accuracy':
                    value = result.get('accuracy', 0) * 100
                else:
                    value = 0
                
                row.append(value)
            matrix_data.append(row)
        
        # Normalize data for better visualization
        normalized_data = []
        for i, row in enumerate(matrix_data):
            if max(row) > 0:
                normalized_row = [val / max(row) for val in row]
            else:
                normalized_row = row
            normalized_data.append(normalized_row)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=normalized_data,
            x=tools,
            y=[metric.replace('_', ' ').title() for metric in metrics],
            colorscale='RdYlGn',
            hovertemplate='<b>%{y}</b><br>Tool: %{x}<br>Normalized Score: %{z:.2f}<extra></extra>',
            showscale=True
        ))
        
        # Add annotations with actual values
        for i, metric in enumerate(metrics):
            for j, tool in enumerate(tools):
                actual_value = matrix_data[i][j]
                fig.add_annotation(
                    x=j, y=i,
                    text=f"{actual_value:.1f}",
                    showarrow=False,
                    font=dict(color="white" if normalized_data[i][j] < 0.5 else "black")
                )
        
        fig.update_layout(
            title="Interactive Tool Comparison Matrix",
            xaxis_title="Tools",
            yaxis_title="Metrics",
            height=500
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating comparison matrix: {e}")
        return _create_empty_interactive_chart(f"Error: {e}")


def create_performance_monitor(performance_history: List[Dict[str, Any]]) -> go.Figure:
    """
    Create an interactive performance monitoring dashboard.
    
    Args:
        performance_history: List of performance data over time
        
    Returns:
        Plotly figure object
    """
    try:
        if not performance_history:
            return _create_empty_interactive_chart("No performance history available")
        
        # Convert to DataFrame
        df = pd.DataFrame(performance_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Analysis Count', 'Success Rate', 'Average Response Time', 'Error Rate'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. Analysis count over time
        if 'analysis_count' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['analysis_count'],
                    mode='lines+markers',
                    name='Analysis Count',
                    line=dict(color='blue', width=2),
                    hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>'
                ),
                row=1, col=1
            )
        
        # 2. Success rate over time
        if 'success_rate' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['success_rate'],
                    mode='lines+markers',
                    name='Success Rate',
                    line=dict(color='green', width=2),
                    hovertemplate='<b>%{x}</b><br>Success Rate: %{y:.1f}%<extra></extra>'
                ),
                row=1, col=2
            )
        
        # 3. Average response time
        if 'avg_response_time' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['avg_response_time'],
                    mode='lines+markers',
                    name='Avg Response Time',
                    line=dict(color='orange', width=2),
                    hovertemplate='<b>%{x}</b><br>Response Time: %{y:.2f}s<extra></extra>'
                ),
                row=2, col=1
            )
        
        # 4. Error rate
        if 'error_rate' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['error_rate'],
                    mode='lines+markers',
                    name='Error Rate',
                    line=dict(color='red', width=2),
                    hovertemplate='<b>%{x}</b><br>Error Rate: %{y:.1f}%<extra></extra>'
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title="Interactive Performance Monitor",
            height=600,
            showlegend=False
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating performance monitor: {e}")
        return _create_empty_interactive_chart(f"Error: {e}")


def create_risk_gauge(risk_score: float, title: str = "Risk Assessment") -> go.Figure:
    """
    Create an interactive risk assessment gauge.
    
    Args:
        risk_score: Risk score (0-100)
        title: Gauge title
        
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
            mode="gauge+number+delta",
            value=risk_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"{title}<br><span style='font-size:0.8em;color:gray'>Risk Level: {risk_level}</span>"},
            delta={'reference': 50},
            gauge={
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
        logger.error(f"Error creating risk gauge: {e}")
        return _create_empty_interactive_chart(f"Error: {e}")


def create_vulnerability_pie(vulnerabilities: List[Dict[str, Any]]) -> go.Figure:
    """
    Create an interactive vulnerability distribution pie chart.
    
    Args:
        vulnerabilities: List of vulnerability dictionaries
        
    Returns:
        Plotly figure object
    """
    try:
        if not vulnerabilities:
            return _create_empty_interactive_chart("No vulnerabilities detected")
        
        # Count vulnerability types
        vuln_counts = {}
        for vuln in vulnerabilities:
            vuln_type = vuln.get('type', vuln.get('vulnerability_type', 'Unknown'))
            vuln_counts[vuln_type] = vuln_counts.get(vuln_type, 0) + 1
        
        # Create pie chart
        fig = go.Figure(data=[go.Pie(
            labels=[name.replace('_', ' ').title() for name in vuln_counts.keys()],
            values=list(vuln_counts.values()),
            hole=0.3,
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
        )])
        
        fig.update_layout(
            title="Vulnerability Distribution",
            annotations=[dict(text='Vulnerabilities', x=0.5, y=0.5, font_size=20, showarrow=False)]
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating vulnerability pie chart: {e}")
        return _create_empty_interactive_chart(f"Error: {e}")


def create_severity_bar(vulnerabilities: List[Dict[str, Any]]) -> go.Figure:
    """
    Create an interactive severity distribution bar chart.
    
    Args:
        vulnerabilities: List of vulnerability dictionaries
        
    Returns:
        Plotly figure object
    """
    try:
        if not vulnerabilities:
            return _create_empty_interactive_chart("No vulnerabilities detected")
        
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
            title="Vulnerabilities by Severity",
            xaxis_title="Severity Level",
            yaxis_title="Number of Vulnerabilities",
            showlegend=False
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating severity bar chart: {e}")
        return _create_empty_interactive_chart(f"Error: {e}")


def create_feature_importance_bar(feature_importance: Dict[str, float], top_n: int = 15) -> go.Figure:
    """
    Create an interactive feature importance bar chart.
    
    Args:
        feature_importance: Dictionary of feature names and importance scores
        top_n: Number of top features to display
        
    Returns:
        Plotly figure object
    """
    try:
        if not feature_importance:
            return _create_empty_interactive_chart("No feature importance data available")
        
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
            title="Feature Importance",
            xaxis_title="Importance Score",
            yaxis_title="Features",
            height=max(400, len(features) * 25),
            yaxis=dict(autorange="reversed")  # Show highest importance at top
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating feature importance chart: {e}")
        return _create_empty_interactive_chart(f"Error: {e}")


def create_tool_comparison_radar(tool_comparison: Dict[str, Any]) -> go.Figure:
    """
    Create an interactive radar chart for tool comparison.
    
    Args:
        tool_comparison: Tool comparison data
        
    Returns:
        Plotly figure object
    """
    try:
        if not tool_comparison or 'tool_performances' not in tool_comparison:
            return _create_empty_interactive_chart("No tool comparison data available")
        
        performances = tool_comparison['tool_performances']
        
        # Define metrics
        metrics = ['Speed', 'Accuracy', 'Findings', 'Success Rate']
        
        fig = go.Figure()
        
        for tool_name, perf in performances.items():
            # Normalize metrics (0-1 scale)
            speed = 1 - min(perf.get('execution_time', 0) / 60, 1)  # Invert and cap at 60s
            accuracy = perf.get('accuracy', 0)
            findings = min(len(perf.get('vulnerabilities_found', [])) / 10, 1)  # Cap at 10 findings
            success_rate = 1 if perf.get('success', False) else 0
            
            values = [speed, accuracy, findings, success_rate]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metrics,
                fill='toself',
                name=tool_name,
                hovertemplate='<b>%{theta}</b><br>%{fullData.name}: %{r:.2f}<extra></extra>'
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Tool Performance Comparison"
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating tool comparison radar: {e}")
        return _create_empty_interactive_chart(f"Error: {e}")


def create_timeline_chart(analysis_history: List[Dict[str, Any]]) -> go.Figure:
    """
    Create an interactive timeline chart for analysis history.
    
    Args:
        analysis_history: List of analysis results with timestamps
        
    Returns:
        Plotly figure object
    """
    try:
        if not analysis_history:
            return _create_empty_interactive_chart("No analysis history available")
        
        # Convert to DataFrame
        df = pd.DataFrame(analysis_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Create timeline
        fig = go.Figure()
        
        # Add vulnerable contracts
        if 'is_vulnerable' in df.columns:
            vulnerable_df = df[df['is_vulnerable'] == True]
            if not vulnerable_df.empty:
                fig.add_trace(go.Scatter(
                    x=vulnerable_df['timestamp'],
                    y=vulnerable_df.get('risk_score', [50] * len(vulnerable_df)),
                    mode='markers',
                    name='Vulnerable Contracts',
                    marker=dict(color='red', size=10, symbol='x'),
                    hovertemplate='<b>Vulnerable Contract</b><br>Time: %{x}<br>Risk Score: %{y}<extra></extra>'
                ))
            
            # Add safe contracts
            safe_df = df[df['is_vulnerable'] == False]
            if not safe_df.empty:
                fig.add_trace(go.Scatter(
                    x=safe_df['timestamp'],
                    y=safe_df.get('risk_score', [25] * len(safe_df)),
                    mode='markers',
                    name='Safe Contracts',
                    marker=dict(color='green', size=8, symbol='circle'),
                    hovertemplate='<b>Safe Contract</b><br>Time: %{x}<br>Risk Score: %{y}<extra></extra>'
                ))
        
        fig.update_layout(
            title="Analysis Timeline",
            xaxis_title="Time",
            yaxis_title="Risk Score",
            hovermode='closest'
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating timeline chart: {e}")
        return _create_empty_interactive_chart(f"Error: {e}")


def _create_empty_interactive_chart(message: str) -> go.Figure:
    """Create an empty interactive chart with a message."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5, xanchor='center', yanchor='middle',
        showarrow=False, font_size=16
    )
    fig.update_layout(
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False)
    )
    return fig


# Streamlit integration functions
def display_interactive_charts(figures: Dict[str, go.Figure]):
    """
    Display interactive charts in Streamlit.
    
    Args:
        figures: Dictionary of figure names and Plotly figures
    """
    try:
        for name, fig in figures.items():
            st.plotly_chart(fig, use_container_width=True, key=f"chart_{name}")
    except Exception as e:
        logger.error(f"Error displaying interactive charts: {e}")
        st.error(f"Error displaying charts: {e}")


def create_streamlit_dashboard(analysis_results: Dict[str, Any]):
    """
    Create a complete Streamlit dashboard with interactive visualizations.
    
    Args:
        analysis_results: Analysis results dictionary
    """
    try:
        st.title("🔍 Smart Contract Analysis Dashboard")
        
        # Create tabs for different visualizations
        tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Vulnerabilities", "Performance", "Comparison"])
        
        with tab1:
            st.header("Analysis Overview")
            
            # Risk gauge
            if 'risk_score' in analysis_results:
                risk_fig = create_risk_gauge(analysis_results['risk_score'])
                st.plotly_chart(risk_fig, use_container_width=True)
            
            # Key metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Risk Score", f"{analysis_results.get('risk_score', 0):.1f}/100")
            with col2:
                vuln_count = len(analysis_results.get('vulnerabilities', []))
                st.metric("Vulnerabilities", vuln_count)
            with col3:
                confidence = analysis_results.get('confidence_level', 0)
                st.metric("Confidence", f"{confidence:.1%}")
        
        with tab2:
            st.header("Vulnerability Analysis")
            
            vulnerabilities = analysis_results.get('vulnerabilities', [])
            if vulnerabilities:
                col1, col2 = st.columns(2)
                
                with col1:
                    pie_fig = create_vulnerability_pie(vulnerabilities)
                    st.plotly_chart(pie_fig, use_container_width=True)
                
                with col2:
                    bar_fig = create_severity_bar(vulnerabilities)
                    st.plotly_chart(bar_fig, use_container_width=True)
                
                # Vulnerability explorer
                explorer_fig = create_vulnerability_explorer(vulnerabilities)
                st.plotly_chart(explorer_fig, use_container_width=True)
            else:
                st.success("No vulnerabilities detected!")
        
        with tab3:
            st.header("Performance Analysis")
            
            # Feature importance
            if 'feature_importance' in analysis_results:
                importance_fig = create_feature_importance_bar(analysis_results['feature_importance'])
                st.plotly_chart(importance_fig, use_container_width=True)
            
            # Performance history (if available)
            if 'performance_history' in analysis_results:
                monitor_fig = create_performance_monitor(analysis_results['performance_history'])
                st.plotly_chart(monitor_fig, use_container_width=True)
        
        with tab4:
            st.header("Tool Comparison")
            
            if 'tool_comparison' in analysis_results:
                # Comparison radar
                radar_fig = create_tool_comparison_radar(analysis_results['tool_comparison'])
                st.plotly_chart(radar_fig, use_container_width=True)
                
                # Comparison matrix
                if 'tool_performances' in analysis_results['tool_comparison']:
                    matrix_fig = create_comparison_matrix(
                        analysis_results['tool_comparison']['tool_performances']
                    )
                    st.plotly_chart(matrix_fig, use_container_width=True)
            else:
                st.info("No tool comparison data available.")
        
    except Exception as e:
        logger.error(f"Error creating Streamlit dashboard: {e}")
        st.error(f"Error creating dashboard: {e}")