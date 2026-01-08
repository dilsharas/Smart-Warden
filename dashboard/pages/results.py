"""
Results visualization page for the Streamlit dashboard.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
import json
from io import BytesIO
import base64

def show_page():
    """Display the results visualization page."""
    st.header("📊 Analysis Results")
    
    # Check if we have analysis results
    if not st.session_state.analysis_results:
        st.warning("No analysis results available. Please analyze a contract first.")
        
        if st.button("🔍 Go to Analysis Page"):
            st.session_state.current_page = "🔍 Analyze Contract"
            st.rerun()
        
        return
    
    result = st.session_state.analysis_results
    
    # Results overview
    display_results_overview(result)
    
    # Detailed vulnerability analysis
    if result.vulnerabilities:
        display_vulnerability_details(result)
    
    # Feature importance analysis
    if result.feature_importance:
        display_feature_importance(result)
    
    # Tool comparison results
    if result.tool_comparison:
        display_tool_comparison(result)
    
    # Export options
    display_export_options(result)

def display_results_overview(result):
    """Display the results overview section."""
    st.subheader("🎯 Analysis Overview")
    
    # Main metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        risk_color = "🔴" if result.overall_risk_score > 70 else "🟡" if result.overall_risk_score > 40 else "🟢"
        st.metric(
            "Risk Score",
            f"{result.overall_risk_score}/100",
            help="Overall security risk assessment (0=safe, 100=critical)"
        )
        st.markdown(f"{risk_color} Risk Level")
    
    with col2:
        status_icon = "🚨" if result.is_vulnerable else "✅"
        status_text = "Vulnerable" if result.is_vulnerable else "Safe"
        st.metric("Security Status", status_text)
        st.markdown(f"{status_icon} {status_text}")
    
    with col3:
        st.metric(
            "Vulnerabilities Found",
            len(result.vulnerabilities),
            help="Total number of vulnerabilities detected"
        )
    
    with col4:
        st.metric(
            "Analysis Confidence",
            f"{result.confidence_level:.1%}",
            help="Confidence level of the analysis results"
        )
    
    # Analysis metadata
    with st.expander("📋 Analysis Details"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Analysis ID:** {result.analysis_id}")
            st.write(f"**Contract Hash:** {result.contract_hash[:16]}...")
            st.write(f"**Analysis Time:** {result.analysis_time:.2f} seconds")
        
        with col2:
            st.write(f"**Timestamp:** {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            st.write(f"**Success:** {'✅ Yes' if result.success else '❌ No'}")
            if hasattr(result, 'error_message') and result.error_message:
                st.write(f"**Error:** {result.error_message}")

def display_vulnerability_details(result):
    """Display detailed vulnerability information."""
    st.subheader("🚨 Vulnerability Analysis")
    
    vulnerabilities = result.vulnerabilities
    
    # Vulnerability summary chart
    if vulnerabilities:
        # Severity distribution
        severity_counts = {}
        vuln_type_counts = {}
        
        for vuln in vulnerabilities:
            severity_counts[vuln.severity] = severity_counts.get(vuln.severity, 0) + 1
            vuln_type_counts[vuln.vulnerability_type] = vuln_type_counts.get(vuln.vulnerability_type, 0) + 1
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Severity pie chart
            if severity_counts:
                fig_severity = px.pie(
                    values=list(severity_counts.values()),
                    names=list(severity_counts.keys()),
                    title="Vulnerabilities by Severity",
                    color_discrete_map={
                        'Critical': '#ff4444',
                        'High': '#ff8800',
                        'Medium': '#ffaa00',
                        'Low': '#44aa44'
                    }
                )
                st.plotly_chart(fig_severity, use_container_width=True)
        
        with col2:
            # Vulnerability type bar chart
            if vuln_type_counts:
                fig_types = px.bar(
                    x=list(vuln_type_counts.keys()),
                    y=list(vuln_type_counts.values()),
                    title="Vulnerabilities by Type",
                    labels={'x': 'Vulnerability Type', 'y': 'Count'}
                )
                fig_types.update_xaxes(tickangle=45)
                st.plotly_chart(fig_types, use_container_width=True)
    
    # Detailed vulnerability cards
    st.markdown("#### 📋 Detailed Findings")
    
    for i, vuln in enumerate(vulnerabilities, 1):
        severity_class = f"vulnerability-{vuln.severity.lower()}"
        
        with st.expander(f"{i}. {vuln.vulnerability_type.replace('_', ' ').title()} - {vuln.severity}"):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"**Description:** {vuln.description}")
                st.markdown(f"**Recommendation:** {vuln.recommendation}")
                
                if vuln.code_snippet:
                    st.markdown("**Code Snippet:**")
                    st.code(vuln.code_snippet, language='solidity')
            
            with col2:
                st.metric("Confidence", f"{vuln.confidence:.1%}")
                st.metric("Line Number", vuln.line_number)
                st.write(f"**Tool Source:** {vuln.tool_source}")
                
                # Severity indicator
                severity_colors = {
                    'Critical': '🔴',
                    'High': '🟠', 
                    'Medium': '🟡',
                    'Low': '🟢'
                }
                st.markdown(f"**Severity:** {severity_colors.get(vuln.severity, '⚪')} {vuln.severity}")

def display_feature_importance(result):
    """Display feature importance analysis."""
    st.subheader("🎯 Feature Importance Analysis")
    
    feature_importance = result.feature_importance
    
    if not feature_importance:
        st.info("No feature importance data available.")
        return
    
    # Convert to DataFrame for easier plotting
    df_features = pd.DataFrame([
        {'feature': feature, 'importance': importance}
        for feature, importance in feature_importance.items()
    ]).sort_values('importance', ascending=True)
    
    # Feature importance bar chart
    fig_features = px.bar(
        df_features,
        x='importance',
        y='feature',
        orientation='h',
        title="Feature Importance in Vulnerability Detection",
        labels={'importance': 'Importance Score', 'feature': 'Feature'},
        color='importance',
        color_continuous_scale='viridis'
    )
    
    fig_features.update_layout(height=max(400, len(df_features) * 25))
    st.plotly_chart(fig_features, use_container_width=True)
    
    # Feature explanations
    with st.expander("📖 Feature Explanations"):
        feature_explanations = {
            'external_call_count': 'Number of external contract calls (higher = more risky)',
            'state_change_after_call': 'State changes after external calls (reentrancy indicator)',
            'uses_block_timestamp': 'Usage of block.timestamp (randomness vulnerability)',
            'public_function_count': 'Number of public functions (attack surface)',
            'payable_function_count': 'Number of payable functions (financial risk)',
            'dangerous_function_count': 'Usage of dangerous functions (selfdestruct, delegatecall)',
            'modifier_count': 'Number of access control modifiers',
            'require_count': 'Number of require statements (input validation)',
            'loop_count': 'Number of loops (DoS potential)',
            'cyclomatic_complexity': 'Code complexity measure'
        }
        
        for feature, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
            explanation = feature_explanations.get(feature, 'Custom security feature')
            st.write(f"**{feature}** (importance: {importance:.3f}): {explanation}")

def display_tool_comparison(result):
    """Display tool comparison results."""
    st.subheader("⚖️ Tool Comparison")
    
    tool_comparison = result.tool_comparison
    
    if not tool_comparison:
        st.info("No tool comparison data available.")
        return
    
    # Agreement score
    if 'agreement_score' in tool_comparison:
        agreement_score = tool_comparison['agreement_score']
        
        col1, col2, col3 = st.columns(3)
        with col2:
            # Create a gauge chart for agreement score
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = agreement_score * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Tool Agreement Score"},
                delta = {'reference': 80},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            fig_gauge.update_layout(height=300)
            st.plotly_chart(fig_gauge, use_container_width=True)
    
    # Consensus findings
    if 'consensus_findings' in tool_comparison:
        consensus = tool_comparison['consensus_findings']
        
        if consensus:
            st.markdown("#### 🤝 Consensus Findings")
            st.success(f"Multiple tools agree on: {', '.join(consensus)}")
        else:
            st.markdown("#### 🤝 Consensus Findings")
            st.info("No consensus findings across tools.")
    
    # Tool performance comparison
    if 'tool_performances' in tool_comparison:
        performances = tool_comparison['tool_performances']
        
        st.markdown("#### 🛠️ Individual Tool Results")
        
        # Create comparison table
        tool_data = []
        for tool_name, performance in performances.items():
            tool_data.append({
                'Tool': tool_name,
                'Success': '✅' if performance.get('success', False) else '❌',
                'Execution Time': f"{performance.get('execution_time', 0):.2f}s",
                'Findings': performance.get('findings_count', 0),
                'Vulnerabilities': ', '.join(performance.get('vulnerabilities_found', []))
            })
        
        if tool_data:
            df_tools = pd.DataFrame(tool_data)
            st.dataframe(df_tools, use_container_width=True)

def generate_html_report(result):
    """Generate HTML report that can be converted to PDF."""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Smart Contract Security Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .header {{ color: #1f77b4; text-align: center; margin-bottom: 30px; }}
            .section {{ margin-bottom: 30px; }}
            .vulnerability {{ background-color: #f8f9fa; padding: 15px; margin: 10px 0; border-left: 4px solid #dc3545; }}
            .safe {{ background-color: #d4edda; padding: 15px; margin: 10px 0; border-left: 4px solid #28a745; }}
            table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .risk-high {{ color: #dc3545; font-weight: bold; }}
            .risk-medium {{ color: #ffc107; font-weight: bold; }}
            .risk-low {{ color: #28a745; font-weight: bold; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🔒 Smart Contract Security Analysis Report</h1>
            <p>Generated on {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="section">
            <h2>📊 Analysis Overview</h2>
            <table>
                <tr><th>Analysis ID</th><td>{result.analysis_id}</td></tr>
                <tr><th>Risk Score</th><td class="{'risk-high' if result.overall_risk_score > 70 else 'risk-medium' if result.overall_risk_score > 40 else 'risk-low'}">{result.overall_risk_score}/100</td></tr>
                <tr><th>Status</th><td>{'🚨 Vulnerable' if result.is_vulnerable else '✅ Safe'}</td></tr>
                <tr><th>Confidence</th><td>{result.confidence_level:.1%}</td></tr>
                <tr><th>Analysis Time</th><td>{result.analysis_time:.2f} seconds</td></tr>
                <tr><th>Vulnerabilities Found</th><td>{len(result.vulnerabilities)}</td></tr>
            </table>
        </div>
    """
    
    if result.vulnerabilities:
        html_content += """
        <div class="section">
            <h2>🚨 Detected Vulnerabilities</h2>
        """
        
        for i, vuln in enumerate(result.vulnerabilities, 1):
            html_content += f"""
            <div class="vulnerability">
                <h3>{i}. {vuln.vulnerability_type.replace('_', ' ').title()}</h3>
                <p><strong>Severity:</strong> {vuln.severity}</p>
                <p><strong>Confidence:</strong> {vuln.confidence:.1%}</p>
                <p><strong>Line:</strong> {vuln.line_number}</p>
                <p><strong>Description:</strong> {vuln.description}</p>
                <p><strong>Recommendation:</strong> {vuln.recommendation}</p>
                {f'<p><strong>Code:</strong> <code>{vuln.code_snippet}</code></p>' if vuln.code_snippet else ''}
            </div>
            """
        
        html_content += "</div>"
    else:
        html_content += """
        <div class="section">
            <div class="safe">
                <h2>✅ No Vulnerabilities Detected</h2>
                <p>The smart contract appears to be secure based on the analysis performed.</p>
            </div>
        </div>
        """
    
    if result.feature_importance:
        html_content += """
        <div class="section">
            <h2>🎯 Feature Importance</h2>
            <table>
                <tr><th>Feature</th><th>Importance Score</th></tr>
        """
        
        for feature, importance in sorted(result.feature_importance.items(), key=lambda x: x[1], reverse=True):
            html_content += f"<tr><td>{feature.replace('_', ' ').title()}</td><td>{importance:.3f}</td></tr>"
        
        html_content += "</table></div>"
    
    html_content += """
        <div class="section">
            <h2>ℹ️ Disclaimer</h2>
            <p><em>This report is generated by an AI-powered analysis tool and should be used as a starting point for security review. 
            Always perform comprehensive manual security audits before deploying smart contracts to production.</em></p>
        </div>
    </body>
    </html>
    """
    
    return html_content

def generate_pdf_report(result):
    """Generate PDF report from analysis results."""
    try:
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib import colors
        
        # Create PDF buffer
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.darkblue
        )
        story.append(Paragraph("Smart Contract Security Analysis Report", title_style))
        story.append(Spacer(1, 20))
        
        # Analysis Overview
        story.append(Paragraph("Analysis Overview", styles['Heading2']))
        overview_data = [
            ['Analysis ID', result.analysis_id],
            ['Timestamp', result.timestamp.strftime('%Y-%m-%d %H:%M:%S')],
            ['Risk Score', f"{result.overall_risk_score}/100"],
            ['Status', 'Vulnerable' if result.is_vulnerable else 'Safe'],
            ['Confidence', f"{result.confidence_level:.1%}"],
            ['Analysis Time', f"{result.analysis_time:.2f} seconds"]
        ]
        
        overview_table = Table(overview_data, colWidths=[2*inch, 3*inch])
        overview_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(overview_table)
        story.append(Spacer(1, 20))
        
        # Vulnerabilities Section
        if result.vulnerabilities:
            story.append(Paragraph("Detected Vulnerabilities", styles['Heading2']))
            
            for i, vuln in enumerate(result.vulnerabilities, 1):
                story.append(Paragraph(f"{i}. {vuln.vulnerability_type.replace('_', ' ').title()}", styles['Heading3']))
                
                vuln_data = [
                    ['Severity', vuln.severity],
                    ['Confidence', f"{vuln.confidence:.1%}"],
                    ['Line Number', str(vuln.line_number)],
                    ['Description', vuln.description],
                    ['Recommendation', vuln.recommendation]
                ]
                
                vuln_table = Table(vuln_data, colWidths=[1.5*inch, 3.5*inch])
                vuln_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                story.append(vuln_table)
                story.append(Spacer(1, 15))
        else:
            story.append(Paragraph("No vulnerabilities detected.", styles['Normal']))
            story.append(Spacer(1, 20))
        
        # Feature Importance
        if result.feature_importance:
            story.append(Paragraph("Feature Importance", styles['Heading2']))
            feature_data = [['Feature', 'Importance Score']]
            for feature, importance in sorted(result.feature_importance.items(), key=lambda x: x[1], reverse=True):
                feature_data.append([feature.replace('_', ' ').title(), f"{importance:.3f}"])
            
            feature_table = Table(feature_data, colWidths=[3*inch, 2*inch])
            feature_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(feature_table)
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()
        
    except ImportError:
        return None
    except Exception as e:
        st.error(f"Error generating PDF: {str(e)}")
        return None

def display_export_options(result):
    """Display export and download options."""
    st.subheader("📥 Export Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # JSON export
        json_data = result.to_dict() if hasattr(result, 'to_dict') else result.__dict__
        json_str = json.dumps(json_data, indent=2, default=str)
        
        st.download_button(
            label="📄 Download JSON Report",
            data=json_str,
            file_name=f"analysis_report_{result.analysis_id}.json",
            mime="application/json"
        )
    
    with col2:
        # CSV export for vulnerabilities
        if result.vulnerabilities:
            vuln_data = []
            for vuln in result.vulnerabilities:
                vuln_dict = vuln.to_dict() if hasattr(vuln, 'to_dict') else vuln.__dict__
                vuln_data.append(vuln_dict)
            
            df_vulns = pd.DataFrame(vuln_data)
            csv_data = df_vulns.to_csv(index=False)
            
            st.download_button(
                label="📊 Download CSV Report",
                data=csv_data,
                file_name=f"vulnerabilities_{result.analysis_id}.csv",
                mime="text/csv"
            )
        else:
            st.info("No vulnerabilities to export as CSV")
    
    with col3:
        # PDF export options
        pdf_method = st.selectbox("PDF Format:", ["HTML Report", "Professional PDF"], key="pdf_format")
        
        if st.button("📑 Generate Report"):
            with st.spinner("Generating report..."):
                if pdf_method == "Professional PDF":
                    pdf_data = generate_pdf_report(result)
                    
                    if pdf_data:
                        st.download_button(
                            label="📑 Download PDF Report",
                            data=pdf_data,
                            file_name=f"security_analysis_{result.analysis_id}.pdf",
                            mime="application/pdf"
                        )
                        st.success("PDF report generated successfully!")
                    else:
                        st.error("Failed to generate PDF. ReportLab library may not be installed.")
                        st.info("To enable PDF generation, install: `pip install reportlab`")
                        st.info("Alternatively, use 'HTML Report' option below.")
                
                else:  # HTML Report
                    html_content = generate_html_report(result)
                    
                    st.download_button(
                        label="📄 Download HTML Report",
                        data=html_content,
                        file_name=f"security_analysis_{result.analysis_id}.html",
                        mime="text/html"
                    )
                    st.success("HTML report generated! You can open it in a browser and print to PDF.")
                    
                    # Show preview
                    with st.expander("📋 Preview HTML Report"):
                        st.components.v1.html(html_content, height=600, scrolling=True)
    
    # Raw data viewer
    with st.expander("🔍 View Raw Analysis Data"):
        if hasattr(result, 'to_dict'):
            st.json(result.to_dict())
        else:
            # Convert to JSON-serializable format first
            try:
                json_data = json.loads(json.dumps(result.__dict__, default=str))
                st.json(json_data)
            except Exception as e:
                st.code(str(result.__dict__))