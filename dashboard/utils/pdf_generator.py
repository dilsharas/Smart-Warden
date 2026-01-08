"""
PDF Report Generator for Smart Contract Analysis Results

This module provides functionality to generate professional PDF reports
containing vulnerability analysis results, tool comparisons, and recommendations.
"""

import io
import base64
from datetime import datetime
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image, KeepTogether
)
from reportlab.platypus.flowables import HRFlowable
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
import pandas as pd


class PDFReportGenerator:
    """Generate comprehensive PDF reports for smart contract analysis results.""" 
   
    def __init__(self):
        """Initialize the PDF report generator."""
        self.styles = getSampleStyleSheet()
        self.setup_custom_styles()
    
    def setup_custom_styles(self):
        """Setup custom paragraph styles for the report."""
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Title'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.darkblue
        ))
        
        # Heading style
        self.styles.add(ParagraphStyle(
            name='CustomHeading',
            parent=self.styles['Heading1'],
            fontSize=16,
            spaceAfter=12,
            spaceBefore=20,
            textColor=colors.darkblue
        ))
        
        # Subheading style
        self.styles.add(ParagraphStyle(
            name='CustomSubheading',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=10,
            spaceBefore=15,
            textColor=colors.darkgreen
        ))
        
        # Critical vulnerability style
        self.styles.add(ParagraphStyle(
            name='CriticalVuln',
            parent=self.styles['Normal'],
            fontSize=12,
            spaceAfter=8,
            leftIndent=20,
            textColor=colors.red,
            backColor=colors.mistyrose
        ))
        
        # High vulnerability style
        self.styles.add(ParagraphStyle(
            name='HighVuln',
            parent=self.styles['Normal'],
            fontSize=12,
            spaceAfter=8,
            leftIndent=20,
            textColor=colors.darkorange,
            backColor=colors.lightyellow
        ))
        
        # Medium vulnerability style
        self.styles.add(ParagraphStyle(
            name='MediumVuln',
            parent=self.styles['Normal'],
            fontSize=12,
            spaceAfter=8,
            leftIndent=20,
            textColor=colors.purple,
            backColor=colors.lavender
        ))
        
        # Low vulnerability style
        self.styles.add(ParagraphStyle(
            name='LowVuln',
            parent=self.styles['Normal'],
            fontSize=12,
            spaceAfter=8,
            leftIndent=20,
            textColor=colors.darkgreen,
            backColor=colors.lightgreen
        ))
    
    def generate_report(self, analysis_result: Dict[str, Any], output_path: str) -> str:
        """
        Generate a comprehensive PDF report from analysis results.
        
        Args:
            analysis_result: Analysis results dictionary
            output_path: Path to save the PDF report
            
        Returns:
            Path to the generated PDF file
        """
        # Create PDF document
        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18
        )
        
        # Build report content
        story = []
        
        # Title page
        story.extend(self._create_title_page(analysis_result))
        story.append(PageBreak())
        
        # Executive summary
        story.extend(self._create_executive_summary(analysis_result))
        story.append(PageBreak())
        
        # Vulnerability details
        if analysis_result.get('vulnerabilities'):
            story.extend(self._create_vulnerability_section(analysis_result))
            story.append(PageBreak())
        
        # Tool comparison
        if analysis_result.get('tool_comparison'):
            story.extend(self._create_tool_comparison_section(analysis_result))
            story.append(PageBreak())
        
        # Feature importance
        if analysis_result.get('feature_importance'):
            story.extend(self._create_feature_importance_section(analysis_result))
            story.append(PageBreak())
        
        # Recommendations
        story.extend(self._create_recommendations_section(analysis_result))
        
        # Build PDF
        doc.build(story)
        return output_path
    
    def _create_title_page(self, analysis_result: Dict[str, Any]) -> List:
        """Create the title page of the report."""
        story = []
        
        # Main title
        story.append(Paragraph("Smart Contract Security Analysis Report", self.styles['CustomTitle']))
        story.append(Spacer(1, 0.5*inch))
        
        # Contract information
        contract_info = [
            ['Analysis ID:', analysis_result.get('analysis_id', 'N/A')],
            ['Contract Hash:', analysis_result.get('contract_hash', 'N/A')[:16] + '...'],
            ['Analysis Date:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            ['Overall Risk Score:', f"{analysis_result.get('overall_risk_score', 0)}/100"],
            ['Vulnerability Status:', 'Vulnerable' if analysis_result.get('is_vulnerable', False) else 'Safe'],
            ['Confidence Level:', f"{analysis_result.get('confidence_level', 0):.1%}"]
        ]
        
        table = Table(contract_info, colWidths=[2*inch, 3*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('BACKGROUND', (0, 0), (0, -1), colors.grey),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.whitesmoke),
        ]))
        
        story.append(table)
        story.append(Spacer(1, 1*inch))
        
        # Risk level indicator
        risk_score = analysis_result.get('overall_risk_score', 0)
        if risk_score > 70:
            risk_level = "HIGH RISK"
            risk_color = colors.red
        elif risk_score > 40:
            risk_level = "MEDIUM RISK"
            risk_color = colors.orange
        else:
            risk_level = "LOW RISK"
            risk_color = colors.green
        
        risk_style = ParagraphStyle(
            name='RiskLevel',
            parent=self.styles['Normal'],
            fontSize=18,
            alignment=TA_CENTER,
            textColor=risk_color,
            spaceAfter=20
        )
        
        story.append(Paragraph(f"<b>{risk_level}</b>", risk_style))
        
        return story
    
    def _create_executive_summary(self, analysis_result: Dict[str, Any]) -> List:
        """Create the executive summary section."""
        story = []
        
        story.append(Paragraph("Executive Summary", self.styles['CustomHeading']))
        story.append(HRFlowable(width="100%", thickness=1, lineCap='round', color=colors.darkblue))
        story.append(Spacer(1, 0.2*inch))
        
        # Summary text
        vulnerabilities = analysis_result.get('vulnerabilities', [])
        vuln_count = len(vulnerabilities)
        risk_score = analysis_result.get('overall_risk_score', 0)
        
        if vuln_count == 0:
            summary_text = """
            This smart contract analysis found no critical security vulnerabilities. 
            The contract appears to follow security best practices and has a low risk profile.
            However, it is recommended to conduct regular security audits and stay updated 
            with the latest security practices.
            """
        else:
            severity_counts = {}
            for vuln in vulnerabilities:
                severity = vuln.get('severity', 'Unknown')
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            severity_text = ", ".join([f"{count} {severity}" for severity, count in severity_counts.items()])
            
            summary_text = f"""
            This smart contract analysis identified {vuln_count} security vulnerabilities 
            with an overall risk score of {risk_score}/100. The vulnerabilities include: {severity_text}.
            
            Immediate attention is required to address these security issues before deployment 
            or continued use of this contract. Each vulnerability has been analyzed and 
            specific recommendations are provided in this report.
            """
        
        story.append(Paragraph(summary_text, self.styles['Normal']))
        story.append(Spacer(1, 0.3*inch))
        
        # Key metrics table
        metrics_data = [
            ['Metric', 'Value'],
            ['Total Vulnerabilities', str(vuln_count)],
            ['Risk Score', f"{risk_score}/100"],
            ['Analysis Confidence', f"{analysis_result.get('confidence_level', 0):.1%}"],
            ['Analysis Time', f"{analysis_result.get('analysis_time', 0):.2f} seconds"],
        ]
        
        # Add tool results if available
        tool_comparison = analysis_result.get('tool_comparison', {})
        if tool_comparison:
            consensus_count = len(tool_comparison.get('consensus_findings', []))
            agreement_score = tool_comparison.get('agreement_score', 0)
            metrics_data.extend([
                ['Consensus Findings', str(consensus_count)],
                ['Tool Agreement', f"{agreement_score:.1%}"]
            ])
        
        metrics_table = Table(metrics_data, colWidths=[2.5*inch, 2*inch])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(metrics_table)
        
        return story
    
    def _create_vulnerability_section(self, analysis_result: Dict[str, Any]) -> List:
        """Create the detailed vulnerability analysis section."""
        story = []
        
        story.append(Paragraph("Vulnerability Analysis", self.styles['CustomHeading']))
        story.append(HRFlowable(width="100%", thickness=1, lineCap='round', color=colors.darkblue))
        story.append(Spacer(1, 0.2*inch))
        
        vulnerabilities = analysis_result.get('vulnerabilities', [])
        
        for i, vuln in enumerate(vulnerabilities, 1):
            # Vulnerability header
            vuln_type = vuln.get('vulnerability_type', 'Unknown').replace('_', ' ').title()
            severity = vuln.get('severity', 'Unknown')
            
            header_text = f"{i}. {vuln_type} - {severity} Severity"
            story.append(Paragraph(header_text, self.styles['CustomSubheading']))
            
            # Vulnerability details table
            vuln_data = [
                ['Property', 'Value'],
                ['Vulnerability Type', vuln_type],
                ['Severity Level', severity],
                ['Confidence Score', f"{vuln.get('confidence', 0):.1%}"],
                ['Line Number', str(vuln.get('line_number', 'N/A'))],
                ['Tool Source', vuln.get('tool_source', 'N/A')]
            ]
            
            vuln_table = Table(vuln_data, colWidths=[1.5*inch, 3*inch])
            vuln_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(vuln_table)
            story.append(Spacer(1, 0.1*inch))
            
            # Description
            description = vuln.get('description', 'No description available.')
            story.append(Paragraph(f"<b>Description:</b> {description}", self.styles['Normal']))
            story.append(Spacer(1, 0.1*inch))
            
            # Code snippet if available
            code_snippet = vuln.get('code_snippet', '')
            if code_snippet:
                story.append(Paragraph("<b>Code Snippet:</b>", self.styles['Normal']))
                code_style = ParagraphStyle(
                    name='CodeSnippet',
                    parent=self.styles['Normal'],
                    fontSize=9,
                    fontName='Courier',
                    leftIndent=20,
                    backColor=colors.lightgrey,
                    borderColor=colors.black,
                    borderWidth=1
                )
                story.append(Paragraph(f"<pre>{code_snippet}</pre>", code_style))
                story.append(Spacer(1, 0.1*inch))
            
            # Recommendation
            recommendation = vuln.get('recommendation', 'No specific recommendation available.')
            story.append(Paragraph(f"<b>Recommendation:</b> {recommendation}", self.styles['Normal']))
            
            story.append(Spacer(1, 0.3*inch))
        
        return story
    
    def _create_tool_comparison_section(self, analysis_result: Dict[str, Any]) -> List:
        """Create the tool comparison section."""
        story = []
        
        story.append(Paragraph("Tool Comparison Analysis", self.styles['CustomHeading']))
        story.append(HRFlowable(width="100%", thickness=1, lineCap='round', color=colors.darkblue))
        story.append(Spacer(1, 0.2*inch))
        
        tool_comparison = analysis_result.get('tool_comparison', {})
        
        # Agreement score
        agreement_score = tool_comparison.get('agreement_score', 0)
        story.append(Paragraph(f"<b>Tool Agreement Score:</b> {agreement_score:.1%}", self.styles['Normal']))
        story.append(Spacer(1, 0.1*inch))
        
        # Consensus findings
        consensus_findings = tool_comparison.get('consensus_findings', [])
        if consensus_findings:
            consensus_text = ", ".join(consensus_findings)
            story.append(Paragraph(f"<b>Consensus Findings:</b> {consensus_text}", self.styles['Normal']))
        else:
            story.append(Paragraph("<b>Consensus Findings:</b> No consensus across tools", self.styles['Normal']))
        
        story.append(Spacer(1, 0.2*inch))
        
        # Tool performance table
        tool_performances = tool_comparison.get('tool_performances', {})
        if tool_performances:
            perf_data = [['Tool', 'Success', 'Execution Time', 'Findings', 'Vulnerabilities Found']]
            
            for tool_name, performance in tool_performances.items():
                success = '✓' if performance.get('success', False) else '✗'
                exec_time = f"{performance.get('execution_time', 0):.2f}s"
                findings = str(performance.get('findings_count', 0))
                vulns = ', '.join(performance.get('vulnerabilities_found', []))
                if not vulns:
                    vulns = 'None'
                
                perf_data.append([tool_name, success, exec_time, findings, vulns])
            
            perf_table = Table(perf_data, colWidths=[1*inch, 0.7*inch, 1*inch, 0.8*inch, 2*inch])
            perf_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(perf_table)
        
        return story
    
    def _create_feature_importance_section(self, analysis_result: Dict[str, Any]) -> List:
        """Create the feature importance section."""
        story = []
        
        story.append(Paragraph("Feature Importance Analysis", self.styles['CustomHeading']))
        story.append(HRFlowable(width="100%", thickness=1, lineCap='round', color=colors.darkblue))
        story.append(Spacer(1, 0.2*inch))
        
        feature_importance = analysis_result.get('feature_importance', {})
        
        # Sort features by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        # Create feature importance table
        feature_data = [['Feature', 'Importance Score', 'Description']]
        
        # Feature descriptions
        feature_descriptions = {
            'external_call_count': 'Number of external contract calls',
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
        
        for feature, importance in sorted_features[:10]:  # Top 10 features
            description = feature_descriptions.get(feature, 'Security-related feature')
            feature_data.append([feature, f"{importance:.3f}", description])
        
        feature_table = Table(feature_data, colWidths=[2*inch, 1*inch, 2.5*inch])
        feature_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(feature_table)
        
        return story
    
    def _create_recommendations_section(self, analysis_result: Dict[str, Any]) -> List:
        """Create the recommendations section."""
        story = []
        
        story.append(Paragraph("Security Recommendations", self.styles['CustomHeading']))
        story.append(HRFlowable(width="100%", thickness=1, lineCap='round', color=colors.darkblue))
        story.append(Spacer(1, 0.2*inch))
        
        vulnerabilities = analysis_result.get('vulnerabilities', [])
        
        if not vulnerabilities:
            # Safe contract recommendations
            recommendations = [
                "Continue following security best practices in smart contract development.",
                "Implement comprehensive unit and integration tests for all contract functions.",
                "Consider formal verification for critical contract logic.",
                "Regularly update dependencies and development tools.",
                "Conduct periodic security audits, especially before major updates.",
                "Monitor the contract after deployment for unusual activity.",
                "Stay informed about new security vulnerabilities and best practices."
            ]
        else:
            # Vulnerability-specific recommendations
            recommendations = []
            
            # Collect unique recommendations
            unique_recommendations = set()
            for vuln in vulnerabilities:
                recommendation = vuln.get('recommendation', '')
                if recommendation:
                    unique_recommendations.add(recommendation)
            
            recommendations.extend(list(unique_recommendations))
            
            # Add general security recommendations
            recommendations.extend([
                "Conduct a thorough security audit before deployment.",
                "Implement comprehensive testing including edge cases.",
                "Consider using established security libraries and patterns.",
                "Set up monitoring and alerting for contract interactions.",
                "Prepare an incident response plan for security issues."
            ])
        
        # Create numbered list of recommendations
        for i, rec in enumerate(recommendations, 1):
            story.append(Paragraph(f"{i}. {rec}", self.styles['Normal']))
            story.append(Spacer(1, 0.1*inch))
        
        story.append(Spacer(1, 0.2*inch))
        
        # Additional resources
        story.append(Paragraph("Additional Resources", self.styles['CustomSubheading']))
        resources = [
            "• Consensys Smart Contract Security Best Practices",
            "• OpenZeppelin Security Guidelines",
            "• Ethereum Smart Contract Security Audit Checklist",
            "• OWASP Smart Contract Top 10",
            "• Trail of Bits Security Guidelines"
        ]
        
        for resource in resources:
            story.append(Paragraph(resource, self.styles['Normal']))
            story.append(Spacer(1, 0.05*inch))
        
        return story
    
    def create_chart_image(self, chart_data: Dict[str, Any], chart_type: str = 'bar') -> str:
        """
        Create a chart image and return it as base64 string.
        
        Args:
            chart_data: Data for the chart
            chart_type: Type of chart ('bar', 'pie', etc.)
            
        Returns:
            Base64 encoded image string
        """
        plt.figure(figsize=(8, 6))
        
        if chart_type == 'bar':
            plt.bar(chart_data.keys(), chart_data.values())
            plt.title('Vulnerability Distribution')
            plt.xlabel('Vulnerability Type')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
        
        elif chart_type == 'pie':
            plt.pie(chart_data.values(), labels=chart_data.keys(), autopct='%1.1f%%')
            plt.title('Vulnerability Distribution')
        
        # Save to bytes
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=150)
        img_buffer.seek(0)
        
        # Convert to base64
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        
        plt.close()
        return img_base64


def generate_analysis_report(analysis_result: Dict[str, Any], output_path: str) -> str:
    """
    Convenience function to generate a PDF report from analysis results.
    
    Args:
        analysis_result: Analysis results dictionary
        output_path: Path to save the PDF report
        
    Returns:
        Path to the generated PDF file
    """
    generator = PDFReportGenerator()
    return generator.generate_report(analysis_result, output_path)


# Example usage
if __name__ == "__main__":
    # Sample analysis result for testing
    sample_result = {
        'analysis_id': 'analysis_1642248600',
        'contract_hash': 'abc123def456789',
        'overall_risk_score': 75,
        'is_vulnerable': True,
        'confidence_level': 0.87,
        'analysis_time': 2.5,
        'vulnerabilities': [
            {
                'vulnerability_type': 'reentrancy',
                'severity': 'Critical',
                'confidence': 0.92,
                'line_number': 15,
                'description': 'Potential reentrancy vulnerability detected in external call',
                'recommendation': 'Use the checks-effects-interactions pattern and consider reentrancy guards',
                'code_snippet': 'msg.sender.call{value: amount}("")',
                'tool_source': 'AI Classifier'
            },
            {
                'vulnerability_type': 'bad_randomness',
                'severity': 'Medium',
                'confidence': 0.75,
                'line_number': 25,
                'description': 'Use of block.timestamp for randomness detected',
                'recommendation': 'Avoid using block.timestamp for randomness. Use secure random number generators',
                'code_snippet': 'block.timestamp',
                'tool_source': 'Pattern Matcher'
            }
        ],
        'feature_importance': {
            'external_call_count': 0.25,
            'state_change_after_call': 0.20,
            'uses_block_timestamp': 0.15,
            'public_function_count': 0.10,
            'payable_function_count': 0.08
        },
        'tool_comparison': {
            'consensus_findings': ['reentrancy'],
            'agreement_score': 0.85,
            'tool_performances': {
                'ai_binary': {
                    'success': True,
                    'execution_time': 0.5,
                    'findings_count': 1,
                    'vulnerabilities_found': ['reentrancy']
                },
                'slither': {
                    'success': True,
                    'execution_time': 1.2,
                    'findings_count': 2,
                    'vulnerabilities_found': ['reentrancy', 'bad_randomness']
                }
            }
        }
    }
    
    # Generate report
    output_file = "sample_analysis_report.pdf"
    generate_analysis_report(sample_result, output_file)
    print(f"Sample report generated: {output_file}")