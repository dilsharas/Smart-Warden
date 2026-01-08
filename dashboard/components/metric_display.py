"""
Metric display components for the dashboard.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Any, Optional
import pandas as pd


def show_risk_score_gauge(risk_score: float, confidence: float = None) -> None:
    """
    Display risk score as a gauge chart.
    
    Args:
        risk_score: Risk score from 0-100
        confidence: Confidence level from 0-1 (optional)
    """
    # Determine risk level and color
    if risk_score >= 80:
        risk_level = "Critical"
        color = "#ff4444"
    elif risk_score >= 60:
        risk_level = "High"
        color = "#ff8800"
    elif risk_score >= 40:
        risk_level = "Medium"
        color = "#ffaa00"
    elif risk_score >= 20:
        risk_level = "Low"
        color = "#ffdd00"
    else:
        risk_level = "Minimal"
        color = "#4CAF50"
    
    # Create gauge chart
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = risk_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"Risk Score: {risk_level}"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 20], 'color': "#e8f5e8"},
                {'range': [20, 40], 'color': "#fff3cd"},
                {'range': [40, 60], 'color': "#ffeaa7"},
                {'range': [60, 80], 'color': "#fdcb6e"},
                {'range': [80, 100], 'color': "#e17055"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)
    
    # Show confidence if provided
    if confidence is not None:
        confidence_pct = confidence * 100
        st.metric("Confidence Level", f"{confidence_pct:.1f}%")


def show_analysis_metrics(metrics: Dict[str, Any]) -> None:
    """
    Display analysis metrics in a grid layout.
    
    Args:
        metrics: Dictionary containing various metrics
    """
    st.markdown("### 📊 Analysis Metrics")
    
    # Create columns for metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if 'lines_of_code' in metrics:
            st.metric("Lines of Code", f"{metrics['lines_of_code']:,}")
        if 'function_count' in metrics:
            st.metric("Functions", metrics['function_count'])
    
    with col2:
        if 'external_calls' in metrics:
            st.metric("External Calls", metrics['external_calls'])
        if 'modifiers' in metrics:
            st.metric("Modifiers", metrics['modifiers'])
    
    with col3:
        if 'complexity' in metrics:
            st.metric("Complexity", metrics['complexity'])
        if 'comment_ratio' in metrics:
            st.metric("Comment Ratio", f"{metrics['comment_ratio']:.1%}")
    
    with col4:
        if 'analysis_time' in metrics:
            st.metric("Analysis Time", f"{metrics['analysis_time']:.2f}s")
        if 'confidence' in metrics:
            st.metric("Confidence", f"{metrics['confidence']:.1%}")


def show_feature_importance_chart(feature_importance: Dict[str, float], top_n: int = 10) -> None:
    """
    Display feature importance as a horizontal bar chart.
    
    Args:
        feature_importance: Dictionary of feature names and importance scores
        top_n: Number of top features to display
    """
    if not feature_importance:
        st.info("No feature importance data available")
        return
    
    st.markdown("### 🎯 Feature Importance")
    
    # Sort and get top N features
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    if not sorted_features:
        st.info("No feature importance data to display")
        return
    
    # Create horizontal bar chart
    features, importances = zip(*sorted_features)
    
    fig = go.Figure(go.Bar(
        x=importances,
        y=features,
        orientation='h',
        marker_color='lightblue',
        text=[f'{imp:.3f}' for imp in importances],
        textposition='auto'
    ))
    
    fig.update_layout(
        title="Top Features Contributing to Analysis",
        xaxis_title="Importance Score",
        yaxis_title="Features",
        height=max(300, len(features) * 30),
        yaxis={'categoryorder': 'total ascending'}
    )
    
    st.plotly_chart(fig, use_container_width=True)


def show_comparison_metrics(tool_results: Dict[str, Dict]) -> None:
    """
    Display comparison metrics between different analysis tools.
    
    Args:
        tool_results: Dictionary with tool names as keys and results as values
    """
    if not tool_results:
        st.info("No comparison data available")
        return
    
    st.markdown("### ⚖️ Tool Performance Comparison")
    
    # Extract metrics for comparison
    tools = list(tool_results.keys())
    metrics_data = []
    
    for tool, results in tool_results.items():
        metrics_data.append({
            'Tool': tool,
            'Vulnerabilities Found': len(results.get('vulnerabilities', [])),
            'Analysis Time (s)': results.get('analysis_time', 0),
            'Success': results.get('success', False)
        })
    
    df = pd.DataFrame(metrics_data)
    
    # Display metrics table
    st.dataframe(df, use_container_width=True)
    
    # Create comparison charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Vulnerabilities found comparison
        fig1 = px.bar(
            df, 
            x='Tool', 
            y='Vulnerabilities Found',
            title="Vulnerabilities Detected by Tool",
            color='Tool'
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Analysis time comparison
        fig2 = px.bar(
            df, 
            x='Tool', 
            y='Analysis Time (s)',
            title="Analysis Time by Tool",
            color='Tool'
        )
        st.plotly_chart(fig2, use_container_width=True)


def show_progress_indicator(current_step: int, total_steps: int, step_name: str) -> None:
    """
    Display a progress indicator for multi-step analysis.
    
    Args:
        current_step: Current step number (1-based)
        total_steps: Total number of steps
        step_name: Name of the current step
    """
    progress = current_step / total_steps
    
    st.markdown(f"### 🔄 Analysis Progress: {step_name}")
    st.progress(progress)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Current Step", f"{current_step}/{total_steps}")
    with col2:
        st.metric("Progress", f"{progress:.0%}")
    with col3:
        st.metric("Status", step_name)


def show_security_score_breakdown(scores: Dict[str, float]) -> None:
    """
    Display security score breakdown by category.
    
    Args:
        scores: Dictionary of security category scores
    """
    st.markdown("### 🛡️ Security Score Breakdown")
    
    if not scores:
        st.info("No security scores available")
        return
    
    # Create radar chart
    categories = list(scores.keys())
    values = list(scores.values())
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Security Scores',
        line_color='rgb(32, 201, 151)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=False,
        title="Security Assessment by Category",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show detailed scores
    for category, score in scores.items():
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"**{category.replace('_', ' ').title()}**")
        with col2:
            if score >= 80:
                st.success(f"{score:.1f}")
            elif score >= 60:
                st.warning(f"{score:.1f}")
            else:
                st.error(f"{score:.1f}")


def show_trend_chart(data: List[Dict], x_field: str, y_field: str, title: str) -> None:
    """
    Display a trend chart for time-series data.
    
    Args:
        data: List of data points
        x_field: Field name for x-axis
        y_field: Field name for y-axis
        title: Chart title
    """
    if not data:
        st.info(f"No data available for {title}")
        return
    
    df = pd.DataFrame(data)
    
    fig = px.line(
        df, 
        x=x_field, 
        y=y_field,
        title=title,
        markers=True
    )
    
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)


def show_distribution_chart(data: List[Any], title: str, bins: int = 20) -> None:
    """
    Display a distribution histogram.
    
    Args:
        data: List of numerical data
        title: Chart title
        bins: Number of histogram bins
    """
    if not data:
        st.info(f"No data available for {title}")
        return
    
    fig = px.histogram(
        x=data,
        nbins=bins,
        title=title
    )
    
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)


def show_summary_cards(summary_data: Dict[str, Any]) -> None:
    """
    Display summary information as cards.
    
    Args:
        summary_data: Dictionary containing summary information
    """
    st.markdown("### 📋 Analysis Summary")
    
    # Create cards in columns
    num_cards = len(summary_data)
    cols = st.columns(min(num_cards, 4))
    
    for i, (key, value) in enumerate(summary_data.items()):
        with cols[i % 4]:
            # Format the key for display
            display_key = key.replace('_', ' ').title()
            
            # Determine card color based on key
            if 'error' in key.lower() or 'fail' in key.lower():
                st.error(f"**{display_key}**\n\n{value}")
            elif 'warning' in key.lower() or 'medium' in key.lower():
                st.warning(f"**{display_key}**\n\n{value}")
            elif 'success' in key.lower() or 'safe' in key.lower():
                st.success(f"**{display_key}**\n\n{value}")
            else:
                st.info(f"**{display_key}**\n\n{value}")


def create_metric_dashboard(analysis_results: Dict[str, Any]) -> None:
    """
    Create a comprehensive metric dashboard from analysis results.
    
    Args:
        analysis_results: Complete analysis results dictionary
    """
    # Risk Score Gauge
    if 'overall_risk_score' in analysis_results:
        show_risk_score_gauge(
            analysis_results['overall_risk_score'],
            analysis_results.get('confidence_level')
        )
    
    # Analysis Metrics
    metrics = {
        'lines_of_code': analysis_results.get('contract_metrics', {}).get('lines_of_code'),
        'function_count': analysis_results.get('contract_metrics', {}).get('function_count'),
        'external_calls': analysis_results.get('contract_metrics', {}).get('external_calls'),
        'analysis_time': analysis_results.get('analysis_time'),
        'confidence': analysis_results.get('confidence_level')
    }
    
    # Filter out None values
    metrics = {k: v for k, v in metrics.items() if v is not None}
    
    if metrics:
        show_analysis_metrics(metrics)
    
    # Feature Importance
    if 'feature_importance' in analysis_results:
        show_feature_importance_chart(analysis_results['feature_importance'])
    
    # Tool Comparison
    if 'tool_comparison' in analysis_results:
        show_comparison_metrics(analysis_results['tool_comparison'])
    
    # Security Score Breakdown
    if 'security_scores' in analysis_results:
        show_security_score_breakdown(analysis_results['security_scores'])