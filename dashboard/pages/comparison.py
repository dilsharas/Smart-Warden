"""
Tool comparison page for the Streamlit dashboard.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

def show_page():
    """Display the tool comparison page."""
    st.header("⚖️ Tool Comparison Analysis")
    
    # Check if we have comparison results
    if not st.session_state.analysis_results or not st.session_state.analysis_results.tool_comparison:
        st.warning("No tool comparison results available. Please run an analysis with tool comparison enabled.")
        
        if st.button("🔍 Go to Analysis Page"):
            st.session_state.current_page = "🔍 Analyze Contract"
            st.rerun()
        
        return
    
    result = st.session_state.analysis_results
    tool_comparison = result.tool_comparison
    
    # Overview metrics
    display_comparison_overview(tool_comparison)
    
    # Detailed tool performance
    display_tool_performance(tool_comparison)
    
    # Agreement analysis
    display_agreement_analysis(tool_comparison)
    
    # Benchmark comparison (if available)
    display_benchmark_comparison()

def display_comparison_overview(tool_comparison):
    """Display comparison overview metrics."""
    st.subheader("📊 Comparison Overview")
    
    # Extract key metrics
    performances = tool_comparison.get('tool_performances', {})
    consensus_findings = tool_comparison.get('consensus_findings', [])
    agreement_score = tool_comparison.get('agreement_score', 0.0)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Tools Analyzed", len(performances))
    
    with col2:
        successful_tools = sum(1 for p in performances.values() if p.get('success', False))
        st.metric("Successful Runs", successful_tools)
    
    with col3:
        st.metric("Consensus Findings", len(consensus_findings))
    
    with col4:
        st.metric("Agreement Score", f"{agreement_score:.1%}")
    
    # Agreement score visualization
    if agreement_score > 0:
        fig_agreement = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = agreement_score * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Inter-Tool Agreement"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 30], 'color': "red"},
                    {'range': [30, 60], 'color': "yellow"},
                    {'range': [60, 80], 'color': "lightgreen"},
                    {'range': [80, 100], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        fig_agreement.update_layout(height=300)
        st.plotly_chart(fig_agreement, use_container_width=True)

def display_tool_performance(tool_comparison):
    """Display detailed tool performance analysis."""
    st.subheader("🛠️ Tool Performance Analysis")
    
    performances = tool_comparison.get('tool_performances', {})
    
    if not performances:
        st.info("No tool performance data available.")
        return
    
    # Performance comparison table
    st.markdown("#### 📋 Performance Summary")
    
    performance_data = []
    for tool_name, perf in performances.items():
        performance_data.append({
            'Tool': tool_name,
            'Status': '✅ Success' if perf.get('success', False) else '❌ Failed',
            'Execution Time (s)': f"{perf.get('execution_time', 0):.2f}",
            'Findings Count': perf.get('findings_count', 0),
            'Vulnerabilities Found': len(perf.get('vulnerabilities_found', [])),
            'Error Message': perf.get('error_message', 'None')
        })
    
    df_performance = pd.DataFrame(performance_data)
    st.dataframe(df_performance, use_container_width=True)
    
    # Execution time comparison
    successful_tools = {name: perf for name, perf in performances.items() if perf.get('success', False)}
    
    if len(successful_tools) > 1:
        st.markdown("#### ⏱️ Execution Time Comparison")
        
        tool_names = list(successful_tools.keys())
        exec_times = [perf['execution_time'] for perf in successful_tools.values()]
        
        fig_time = px.bar(
            x=tool_names,
            y=exec_times,
            title="Execution Time by Tool",
            labels={'x': 'Tool', 'y': 'Execution Time (seconds)'},
            color=exec_times,
            color_continuous_scale='viridis'
        )
        st.plotly_chart(fig_time, use_container_width=True)
    
    # Findings comparison
    if successful_tools:
        st.markdown("#### 🔍 Findings Comparison")
        
        findings_data = []
        for tool_name, perf in successful_tools.items():
            findings_data.append({
                'Tool': tool_name,
                'Total Findings': perf.get('findings_count', 0),
                'Unique Vulnerabilities': len(set(perf.get('vulnerabilities_found', [])))
            })
        
        df_findings = pd.DataFrame(findings_data)
        
        fig_findings = px.bar(
            df_findings,
            x='Tool',
            y=['Total Findings', 'Unique Vulnerabilities'],
            title="Findings Count by Tool",
            barmode='group'
        )
        st.plotly_chart(fig_findings, use_container_width=True)

def display_agreement_analysis(tool_comparison):
    """Display agreement analysis between tools."""
    st.subheader("🤝 Agreement Analysis")
    
    performances = tool_comparison.get('tool_performances', {})
    consensus_findings = tool_comparison.get('consensus_findings', [])
    unique_findings = tool_comparison.get('unique_findings', {})
    
    # Consensus findings
    if consensus_findings:
        st.markdown("#### ✅ Consensus Findings")
        st.success(f"All tools agree on these vulnerabilities: **{', '.join(consensus_findings)}**")
    else:
        st.markdown("#### ✅ Consensus Findings")
        st.info("No vulnerabilities were detected by multiple tools.")
    
    # Unique findings per tool
    if unique_findings:
        st.markdown("#### 🔍 Unique Findings per Tool")
        
        for tool_name, unique_vulns in unique_findings.items():
            if unique_vulns:
                st.write(f"**{tool_name}** found uniquely:")
                for vuln in unique_vulns:
                    st.write(f"  • {vuln.replace('_', ' ').title()}")
            else:
                st.write(f"**{tool_name}**: No unique findings")
    
    # Vulnerability detection matrix
    successful_tools = {name: perf for name, perf in performances.items() if perf.get('success', False)}
    
    if len(successful_tools) > 1:
        st.markdown("#### 🎯 Vulnerability Detection Matrix")
        
        # Get all unique vulnerabilities
        all_vulns = set()
        for perf in successful_tools.values():
            all_vulns.update(perf.get('vulnerabilities_found', []))
        
        if all_vulns:
            # Create detection matrix
            matrix_data = []
            for vuln in sorted(all_vulns):
                row = {'Vulnerability': vuln.replace('_', ' ').title()}
                for tool_name, perf in successful_tools.items():
                    detected = vuln in perf.get('vulnerabilities_found', [])
                    row[tool_name] = '✅' if detected else '❌'
                matrix_data.append(row)
            
            df_matrix = pd.DataFrame(matrix_data)
            st.dataframe(df_matrix, use_container_width=True)
            
            # Heatmap visualization
            numeric_matrix = []
            for vuln in sorted(all_vulns):
                row = []
                for tool_name in successful_tools.keys():
                    perf = successful_tools[tool_name]
                    detected = 1 if vuln in perf.get('vulnerabilities_found', []) else 0
                    row.append(detected)
                numeric_matrix.append(row)
            
            if numeric_matrix:
                fig_heatmap = px.imshow(
                    numeric_matrix,
                    labels=dict(x="Tool", y="Vulnerability", color="Detected"),
                    x=list(successful_tools.keys()),
                    y=[vuln.replace('_', ' ').title() for vuln in sorted(all_vulns)],
                    color_continuous_scale='RdYlGn',
                    title="Vulnerability Detection Heatmap"
                )
                st.plotly_chart(fig_heatmap, use_container_width=True)

def display_benchmark_comparison():
    """Display benchmark comparison if available."""
    st.subheader("📈 Benchmark Comparison")
    
    # This would typically show historical benchmark data
    # For now, we'll show a placeholder with mock data
    
    st.info("Historical benchmark data would be displayed here in a full implementation.")
    
    # Mock benchmark data for demonstration
    mock_benchmark_data = {
        'Tool': ['AI Binary Classifier', 'AI Multi-class', 'Slither', 'Mythril'],
        'Accuracy': [0.87, 0.84, 0.79, 0.82],
        'Precision': [0.85, 0.83, 0.76, 0.80],
        'Recall': [0.89, 0.86, 0.82, 0.84],
        'F1-Score': [0.87, 0.84, 0.79, 0.82],
        'Avg Execution Time (s)': [2.1, 2.3, 8.5, 45.2]
    }
    
    df_benchmark = pd.DataFrame(mock_benchmark_data)
    
    # Performance metrics comparison
    col1, col2 = st.columns(2)
    
    with col1:
        fig_metrics = px.bar(
            df_benchmark,
            x='Tool',
            y=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            title="Performance Metrics Comparison",
            barmode='group'
        )
        fig_metrics.update_xaxes(tickangle=45)
        st.plotly_chart(fig_metrics, use_container_width=True)
    
    with col2:
        fig_time = px.bar(
            df_benchmark,
            x='Tool',
            y='Avg Execution Time (s)',
            title="Average Execution Time",
            color='Avg Execution Time (s)',
            color_continuous_scale='viridis'
        )
        fig_time.update_xaxes(tickangle=45)
        st.plotly_chart(fig_time, use_container_width=True)
    
    # Detailed benchmark table
    with st.expander("📊 Detailed Benchmark Results"):
        st.dataframe(df_benchmark, use_container_width=True)
        
        st.markdown("""
        **Note:** These are example benchmark results. In a production system, 
        these would be based on actual performance testing across standardized datasets.
        
        **Metrics Explanation:**
        - **Accuracy**: Overall correctness of predictions
        - **Precision**: Proportion of positive predictions that were correct
        - **Recall**: Proportion of actual positives that were correctly identified
        - **F1-Score**: Harmonic mean of precision and recall
        - **Execution Time**: Average time to analyze a contract
        """)