"""
Performance metrics page for the Streamlit dashboard.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def show_page():
    """Display the performance metrics page."""
    st.header("📈 Performance Metrics")
    
    # System overview
    display_system_overview()
    
    # Model performance metrics
    display_model_metrics()
    
    # Analysis history and trends
    display_analysis_trends()
    
    # Tool availability and status
    display_tool_status()

def display_system_overview():
    """Display system overview metrics."""
    st.subheader("🖥️ System Overview")
    
    # Mock system data - in real implementation, this would come from actual system monitoring
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_analyses = len(st.session_state.analysis_history) if st.session_state.analysis_history else 0
        st.metric("Total Analyses", total_analyses)
    
    with col2:
        if st.session_state.analysis_history:
            vulnerable_count = sum(1 for result in st.session_state.analysis_history if result.get('is_vulnerable', False))
            st.metric("Vulnerabilities Detected", vulnerable_count)
        else:
            st.metric("Vulnerabilities Detected", 0)
    
    with col3:
        # Mock uptime
        st.metric("System Uptime", "99.9%")
    
    with col4:
        # Mock average analysis time
        st.metric("Avg Analysis Time", "2.3s")
    
    # System health indicators
    st.markdown("#### 🔋 System Health")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # CPU usage (mock)
        cpu_usage = 45
        fig_cpu = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = cpu_usage,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "CPU Usage (%)"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "yellow"},
                    {'range': [80, 100], 'color': "red"}
                ]
            }
        ))
        fig_cpu.update_layout(height=250)
        st.plotly_chart(fig_cpu, use_container_width=True)
    
    with col2:
        # Memory usage (mock)
        memory_usage = 62
        fig_memory = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = memory_usage,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Memory Usage (%)"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkgreen"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "yellow"},
                    {'range': [80, 100], 'color': "red"}
                ]
            }
        ))
        fig_memory.update_layout(height=250)
        st.plotly_chart(fig_memory, use_container_width=True)
    
    with col3:
        # Cache hit rate (mock)
        cache_hit_rate = 78
        fig_cache = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = cache_hit_rate,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Cache Hit Rate (%)"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "purple"},
                'steps': [
                    {'range': [0, 30], 'color': "red"},
                    {'range': [30, 60], 'color': "yellow"},
                    {'range': [60, 100], 'color': "lightgreen"}
                ]
            }
        ))
        fig_cache.update_layout(height=250)
        st.plotly_chart(fig_cache, use_container_width=True)

def display_model_metrics():
    """Display ML model performance metrics."""
    st.subheader("🤖 Model Performance")
    
    # Mock model performance data
    model_data = {
        'Model': ['Binary Classifier', 'Multi-class Classifier'],
        'Accuracy': [0.87, 0.84],
        'Precision': [0.85, 0.83],
        'Recall': [0.89, 0.86],
        'F1-Score': [0.87, 0.84],
        'Training Date': ['2024-01-15', '2024-01-15'],
        'Status': ['✅ Active', '✅ Active']
    }
    
    df_models = pd.DataFrame(model_data)
    
    # Model performance table
    st.dataframe(df_models, use_container_width=True)
    
    # Performance visualization
    col1, col2 = st.columns(2)
    
    with col1:
        # Performance metrics comparison
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        fig_performance = px.bar(
            df_models,
            x='Model',
            y=metrics,
            title="Model Performance Comparison",
            barmode='group'
        )
        st.plotly_chart(fig_performance, use_container_width=True)
    
    with col2:
        # Model accuracy over time (mock data)
        dates = pd.date_range(start='2024-01-01', end='2024-01-30', freq='D')
        accuracy_trend = 0.85 + 0.02 * np.sin(np.arange(len(dates)) * 0.2) + np.random.normal(0, 0.01, len(dates))
        
        fig_trend = px.line(
            x=dates,
            y=accuracy_trend,
            title="Model Accuracy Trend",
            labels={'x': 'Date', 'y': 'Accuracy'}
        )
        fig_trend.update_layout(yaxis_range=[0.8, 0.9])
        st.plotly_chart(fig_trend, use_container_width=True)
    
    # Confusion matrix (mock data)
    with st.expander("📊 Detailed Model Metrics"):
        st.markdown("#### Binary Classifier Confusion Matrix")
        
        # Mock confusion matrix
        confusion_matrix = np.array([[85, 12], [8, 95]])
        
        fig_cm = px.imshow(
            confusion_matrix,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=['Safe', 'Vulnerable'],
            y=['Safe', 'Vulnerable'],
            color_continuous_scale='Blues',
            text_auto=True
        )
        fig_cm.update_layout(title="Confusion Matrix - Binary Classifier")
        st.plotly_chart(fig_cm, use_container_width=True)
        
        # Feature importance (mock data)
        st.markdown("#### Top Feature Importance")
        
        features = [
            'external_call_count', 'state_change_after_call', 'uses_block_timestamp',
            'public_function_count', 'payable_function_count', 'dangerous_function_count',
            'modifier_count', 'require_count', 'loop_count', 'cyclomatic_complexity'
        ]
        importance_scores = np.random.uniform(0.05, 0.25, len(features))
        importance_scores = sorted(importance_scores, reverse=True)
        
        df_importance = pd.DataFrame({
            'Feature': features,
            'Importance': importance_scores
        })
        
        fig_importance = px.bar(
            df_importance,
            x='Importance',
            y='Feature',
            orientation='h',
            title="Feature Importance Scores"
        )
        st.plotly_chart(fig_importance, use_container_width=True)

def display_analysis_trends():
    """Display analysis history and trends."""
    st.subheader("📊 Analysis Trends")
    
    if not st.session_state.analysis_history:
        st.info("No analysis history available. Perform some analyses to see trends.")
        return
    
    # Convert analysis history to DataFrame
    df_history = pd.DataFrame(st.session_state.analysis_history)
    
    if len(df_history) == 0:
        st.info("No analysis data available.")
        return
    
    # Analysis over time
    if 'timestamp' in df_history.columns:
        df_history['date'] = pd.to_datetime(df_history['timestamp']).dt.date
        
        daily_counts = df_history.groupby('date').size().reset_index(name='count')
        
        fig_daily = px.line(
            daily_counts,
            x='date',
            y='count',
            title="Daily Analysis Count",
            markers=True
        )
        st.plotly_chart(fig_daily, use_container_width=True)
    
    # Vulnerability distribution
    if 'is_vulnerable' in df_history.columns:
        vuln_counts = df_history['is_vulnerable'].value_counts()
        
        fig_vuln_dist = px.pie(
            values=vuln_counts.values,
            names=['Safe' if not x else 'Vulnerable' for x in vuln_counts.index],
            title="Contract Safety Distribution"
        )
        st.plotly_chart(fig_vuln_dist, use_container_width=True)
    
    # Risk score distribution
    if 'risk_score' in df_history.columns:
        fig_risk_hist = px.histogram(
            df_history,
            x='risk_score',
            title="Risk Score Distribution",
            nbins=20,
            labels={'risk_score': 'Risk Score', 'count': 'Number of Contracts'}
        )
        st.plotly_chart(fig_risk_hist, use_container_width=True)
    
    # Recent analyses table
    st.markdown("#### 📋 Recent Analyses")
    
    # Show last 10 analyses
    recent_analyses = df_history.tail(10).copy()
    if 'timestamp' in recent_analyses.columns:
        recent_analyses['timestamp'] = pd.to_datetime(recent_analyses['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
    
    st.dataframe(recent_analyses, use_container_width=True)

def display_tool_status():
    """Display external tool availability and status."""
    st.subheader("🛠️ Tool Status")
    
    # Mock tool status data
    tools_status = [
        {'Tool': 'Slither', 'Status': '✅ Available', 'Version': '0.9.6', 'Last Check': '2024-01-20 10:30:00'},
        {'Tool': 'Mythril', 'Status': '⚠️ Slow', 'Version': '0.23.25', 'Last Check': '2024-01-20 10:30:00'},
        {'Tool': 'AI Binary Model', 'Status': '✅ Loaded', 'Version': '1.0.0', 'Last Check': '2024-01-20 10:30:00'},
        {'Tool': 'AI Multi-class Model', 'Status': '✅ Loaded', 'Version': '1.0.0', 'Last Check': '2024-01-20 10:30:00'}
    ]
    
    df_tools = pd.DataFrame(tools_status)
    st.dataframe(df_tools, use_container_width=True)
    
    # Tool performance comparison
    st.markdown("#### ⚡ Tool Performance Comparison")
    
    performance_data = {
        'Tool': ['AI Binary', 'AI Multi-class', 'Slither', 'Mythril'],
        'Avg Response Time (s)': [2.1, 2.3, 8.5, 45.2],
        'Success Rate (%)': [99.5, 99.2, 95.8, 87.3],
        'Accuracy (%)': [87.0, 84.0, 79.0, 82.0]
    }
    
    df_perf = pd.DataFrame(performance_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_response = px.bar(
            df_perf,
            x='Tool',
            y='Avg Response Time (s)',
            title="Average Response Time",
            color='Avg Response Time (s)',
            color_continuous_scale='viridis'
        )
        st.plotly_chart(fig_response, use_container_width=True)
    
    with col2:
        fig_success = px.bar(
            df_perf,
            x='Tool',
            y='Success Rate (%)',
            title="Success Rate",
            color='Success Rate (%)',
            color_continuous_scale='greens'
        )
        st.plotly_chart(fig_success, use_container_width=True)
    
    # System recommendations
    st.markdown("#### 💡 System Recommendations")
    
    recommendations = [
        "✅ AI models are performing well with high accuracy",
        "⚠️ Consider optimizing Mythril timeout settings for better performance",
        "📈 System load is within normal parameters",
        "🔄 Regular model retraining recommended every 3 months"
    ]
    
    for rec in recommendations:
        st.write(rec)
    
    # Export metrics
    with st.expander("📥 Export Metrics"):
        if st.button("Download Performance Report"):
            # In a real implementation, this would generate and download a comprehensive report
            st.info("Performance report download would be implemented here.")
        
        if st.button("Export System Logs"):
            # In a real implementation, this would export system logs
            st.info("System log export would be implemented here.")