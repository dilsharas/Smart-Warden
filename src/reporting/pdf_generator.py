"""
PDF Report Generator for Smart Contract Analysis Results

This module provides functionality to generate comprehensive PDF reports
from smart contract analysis results, including AI predictions, external
tool findings, and visual charts.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import logging

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        PageBreak, Image, KeepTogether
    )
    from reportlab.graphics.shapes import Drawing
    from reportlab.graphics.charts.piecharts import Pie
    from reportlab.graphics.charts.barcharts import VerticalBarChart
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    logging.warning("ReportLab not available. PDF generation will be disabled.")

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from io import BytesIO
    import base64
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logging.warning("Matplotlib not available. Chart generation will be limited.")


class PDFReportGenerator:
    """Generate comprehensive PDF reports from analysis results"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        if not REPORTLAB_AVAILABLE:
            raise ImportError(
                "ReportLab is required for PDF generation. "
                "Install with: pip install reportlab"
            )
        
        # Initialize styles
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
        
        # Color scheme
        self.colors = {
            'primary': colors.HexColor('#2E86AB'),
            'secondary': colors.HexColor('#A23B72'),
            'success': colors.HexColor('#F18F01'),
            'warning': colors.HexColor('#C73E1D'),
            'danger': colors.HexColor('#8B0000'),
            'light_gray': colors.HexColor('#F5F5F5'),
            'dark_gray': colors.HexColor('#333333')
        }
    
    def _setup_custom_styles(self):
        """Setup custom paragraph styles"""
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Title'],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.HexColor('#2E86AB'),
            alignment=TA_CENTER
        ))
        
        # Subtitle style
        self.styles.add(ParagraphStyle(
            name='CustomSubtitle',
            parent=self.styles['Heading1'],
            fontSize=18,
            spaceAfter=20,
            textColor=colors.HexColor('#A23B72'),
            borderWidth=1,
            borderColor=colors.HexColor('#A23B72'),
            borderPadding=5
        ))
        
        # Section header style
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=12,
            textColor=colors.HexColor('#2E86AB'),
            borderWidth=0,
            borderColor=colors.HexColor('#2E86AB'),
            leftIndent=0
        ))
        
        # Vulnerability style
        self.styles.add(ParagraphStyle(
            name='VulnerabilityHigh',
            parent=self.styles['Normal'],
            fontSize=11,
            textColor=colors.HexColor('#8B0000'),
            backColor=colors.HexColor('#FFE6E6'),
            borderWidth=1,
            borderColor=colors.HexColor('#8B0000'),
            borderPadding=5
        ))
        
        # Safe contract style
        self.styles.add(ParagraphStyle(
            name='SafeContract',
            parent=self.styles['Normal'],
            fontSize=11,
            textColor=colors.HexColor('#006400'),
            backColor=colors.HexColor('#E6FFE6'),
            borderWidth=1,
            borderColor=colors.HexColor('#006400'),
            borderPadding=5
        ))
    
    def generate_report(self, 
                       analysis_results: Dict[str, Any], 
                       output_path: str,
                       include_charts: bool = True,
                       include_code_snippets: bool = False) -> bool:
        """
        Generate a comprehensive PDF report from analysis results
        
        Args:
            analysis_results: Analysis results dictionary
            output_path: Path to save the PDF report
            include_charts: Whether to include charts and visualizations
            include_code_snippets: Whether to include code snippets in report
            
        Returns:
            bool: True if report generated successfully, False otherwise
        """
        try:
            # Create output directory if it doesn't exist
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
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
            
            # Add title page
            story.extend(self._create_title_page(analysis_results))
            story.append(PageBreak())
            
            # Add executive summary
            story.extend(self._create_executive_summary(analysis_results))
            story.append(PageBreak())
            
            # Add detailed analysis
            if 'batch_analysis_summary' in analysis_results:
                # Batch analysis report
                story.extend(self._create_batch_analysis_section(analysis_results))
                
                if include_charts:
                    story.extend(self._create_charts_section(analysis_results))
                    story.append(PageBreak())
                
                story.extend(self._create_contract_details_section(
                    analysis_results, include_code_snippets
                ))
            else:
                # Single contract analysis report
                story.extend(self._create_single_contract_section(
                    analysis_results, include_code_snippets
                ))
            
            # Add recommendations
            story.append(PageBreak())
            story.extend(self._create_recommendations_section(analysis_results))
            
            # Add appendix
            story.append(PageBreak())
            story.extend(self._create_appendix_section(analysis_results))
            
            # Build PDF
            doc.build(story)
            
            self.logger.info(f"PDF report generated successfully: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error generating PDF report: {e}")
            return False
    
    def _create_title_page(self, analysis_results: Dict[str, Any]) -> List:
        """Create title page content"""
        story = []
        
        # Main title
        story.append(Spacer(1, 2*inch))
        story.append(Paragraph(
            "Smart Contract Security Analysis Report",
            self.styles['CustomTitle']
        ))
        
        story.append(Spacer(1, 0.5*inch))
        
        # Subtitle based on analysis type
        if 'batch_analysis_summary' in analysis_results:
            summary = analysis_results['batch_analysis_summary']
            subtitle = f"Batch Analysis of {summary['total_contracts']} Contracts"
        else:
            contract_path = analysis_results.get('contract_path', 'Unknown Contract')
            subtitle = f"Analysis of {Path(contract_path).name}"
        
        story.append(Paragraph(subtitle, self.styles['CustomSubtitle']))
        
        story.append(Spacer(1, 1*inch))
        
        # Analysis details table
        analysis_info = [
            ['Analysis Date:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            ['Report Generated:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        ]
        
        if 'batch_analysis_summary' in analysis_results:
            summary = analysis_results['batch_analysis_summary']
            analysis_info.extend([
                ['Total Contracts:', str(summary['total_contracts'])],
                ['Successful Analyses:', str(summary['successful_analyses'])],
                ['Failed Analyses:', str(summary['failed_analyses'])]
            ])
        else:
            analysis_info.extend([
                ['Contract Path:', analysis_results.get('contract_path', 'Unknown')],
                ['Analysis Timestamp:', analysis_results.get('timestamp', 'Unknown')]
            ])
        
        # Add tools used
        tools_used = self._get_tools_used(analysis_results)
        if tools_used:
            analysis_info.append(['Tools Used:', ', '.join(tools_used)])
        
        info_table = Table(analysis_info, colWidths=[2*inch, 3*inch])
        info_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('BACKGROUND', (0, 0), (0, -1), self.colors['light_gray'])
        ]))
        
        story.append(info_table)
        
        story.append(Spacer(1, 1*inch))
        
        # Disclaimer
        disclaimer = """
        <b>DISCLAIMER:</b> This report is generated by automated analysis tools and AI models. 
        While these tools are designed to identify potential security vulnerabilities, they may 
        produce false positives or miss certain types of vulnerabilities. This report should be 
        used as part of a comprehensive security review process and should not be considered 
        a substitute for manual security auditing by qualified professionals.
        """
        
        story.append(Paragraph(disclaimer, self.styles['Normal']))
        
        return story
    
    def _create_executive_summary(self, analysis_results: Dict[str, Any]) -> List:
        """Create executive summary section"""
        story = []
        
        story.append(Paragraph("Executive Summary", self.styles['CustomSubtitle']))
        story.append(Spacer(1, 12))
        
        if 'batch_analysis_summary' in analysis_results:
            # Batch analysis summary
            summary = analysis_results['batch_analysis_summary']
            results = analysis_results.get('results', [])
            
            # Calculate statistics
            vulnerable_count = 0
            safe_count = 0
            vulnerability_types = {}
            
            for result in results:
                if 'results' in result and 'ai_analysis' in result['results']:
                    ai_result = result['results']['ai_analysis']
                    if 'binary_classification' in ai_result:
                        if ai_result['binary_classification']['prediction'] == 'vulnerable':
                            vulnerable_count += 1
                            
                            # Count vulnerability types
                            if 'multiclass_classification' in ai_result:
                                vuln_type = ai_result['multiclass_classification']['predicted_type']
                                vulnerability_types[vuln_type] = vulnerability_types.get(vuln_type, 0) + 1
                        else:
                            safe_count += 1
            
            # Summary text
            total_contracts = summary['total_contracts']
            success_rate = (summary['successful_analyses'] / total_contracts * 100) if total_contracts > 0 else 0
            
            summary_text = f"""
            This report presents the results of a comprehensive security analysis of {total_contracts} 
            smart contracts. The analysis was completed with a {success_rate:.1f}% success rate, 
            analyzing {summary['successful_analyses']} contracts successfully.
            
            <b>Key Findings:</b>
            • {vulnerable_count} contracts identified as potentially vulnerable
            • {safe_count} contracts identified as safe
            • {len(vulnerability_types)} different vulnerability types detected
            """
            
            if vulnerability_types:
                summary_text += "\n\n<b>Most Common Vulnerability Types:</b>\n"
                sorted_vulns = sorted(vulnerability_types.items(), key=lambda x: x[1], reverse=True)
                for vuln_type, count in sorted_vulns[:3]:
                    summary_text += f"• {vuln_type.replace('_', ' ').title()}: {count} contracts\n"
            
        else:
            # Single contract summary
            contract_path = analysis_results.get('contract_path', 'Unknown')
            
            summary_text = f"""
            This report presents the security analysis results for the smart contract: 
            <b>{Path(contract_path).name}</b>
            
            The analysis was performed using multiple security analysis tools and AI models 
            to identify potential vulnerabilities and security issues.
            """
            
            # Add specific findings
            if 'results' in analysis_results:
                results = analysis_results['results']
                
                if 'ai_analysis' in results:
                    ai_result = results['ai_analysis']
                    if 'binary_classification' in ai_result:
                        prediction = ai_result['binary_classification']['prediction']
                        confidence = ai_result['binary_classification']['confidence']
                        
                        summary_text += f"""
                        
                        <b>AI Analysis Result:</b>
                        • Contract Classification: {prediction.upper()}
                        • Confidence Level: {confidence:.1%}
                        """
                        
                        if 'multiclass_classification' in ai_result:
                            predicted_type = ai_result['multiclass_classification']['predicted_type']
                            summary_text += f"\n• Predicted Vulnerability Type: {predicted_type.replace('_', ' ').title()}"
        
        story.append(Paragraph(summary_text, self.styles['Normal']))
        
        return story
    
    def _create_batch_analysis_section(self, analysis_results: Dict[str, Any]) -> List:
        """Create batch analysis section"""
        story = []
        
        story.append(Paragraph("Batch Analysis Overview", self.styles['CustomSubtitle']))
        story.append(Spacer(1, 12))
        
        summary = analysis_results['batch_analysis_summary']
        results = analysis_results.get('results', [])
        
        # Create summary statistics table
        stats_data = [
            ['Metric', 'Value', 'Percentage'],
            ['Total Contracts', str(summary['total_contracts']), '100%'],
            ['Successful Analyses', str(summary['successful_analyses']), 
             f"{summary['successful_analyses']/summary['total_contracts']*100:.1f}%"],
            ['Failed Analyses', str(summary['failed_analyses']),
             f"{summary['failed_analyses']/summary['total_contracts']*100:.1f}%"]
        ]
        
        # Calculate vulnerability statistics
        vulnerable_count = 0
        safe_count = 0
        
        for result in results:
            if 'results' in result and 'ai_analysis' in result['results']:
                ai_result = result['results']['ai_analysis']
                if 'binary_classification' in ai_result:
                    if ai_result['binary_classification']['prediction'] == 'vulnerable':
                        vulnerable_count += 1
                    else:
                        safe_count += 1
        
        analyzed_count = vulnerable_count + safe_count
        if analyzed_count > 0:
            stats_data.extend([
                ['Vulnerable Contracts', str(vulnerable_count), 
                 f"{vulnerable_count/analyzed_count*100:.1f}%"],
                ['Safe Contracts', str(safe_count),
                 f"{safe_count/analyzed_count*100:.1f}%"]
            ])
        
        stats_table = Table(stats_data, colWidths=[2.5*inch, 1*inch, 1*inch])
        stats_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), self.colors['primary']),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(stats_table)
        story.append(Spacer(1, 20))
        
        return story
    
    def _create_charts_section(self, analysis_results: Dict[str, Any]) -> List:
        """Create charts and visualizations section"""
        story = []
        
        if not MATPLOTLIB_AVAILABLE:
            story.append(Paragraph(
                "Charts section skipped - matplotlib not available",
                self.styles['Normal']
            ))
            return story
        
        story.append(Paragraph("Analysis Visualizations", self.styles['CustomSubtitle']))
        story.append(Spacer(1, 12))
        
        results = analysis_results.get('results', [])
        
        # Create vulnerability distribution chart
        vulnerability_types = {}
        safe_count = 0
        
        for result in results:
            if 'results' in result and 'ai_analysis' in result['results']:
                ai_result = result['results']['ai_analysis']
                if 'binary_classification' in ai_result:
                    if ai_result['binary_classification']['prediction'] == 'vulnerable':
                        if 'multiclass_classification' in ai_result:
                            vuln_type = ai_result['multiclass_classification']['predicted_type']
                            vulnerability_types[vuln_type] = vulnerability_types.get(vuln_type, 0) + 1
                    else:
                        safe_count += 1
        
        if vulnerability_types or safe_count > 0:
            # Create pie chart
            chart_data = list(vulnerability_types.items())
            if safe_count > 0:
                chart_data.append(('safe', safe_count))
            
            chart_image = self._create_pie_chart(
                chart_data,
                "Vulnerability Distribution",
                figsize=(8, 6)
            )
            
            if chart_image:
                story.append(chart_image)
                story.append(Spacer(1, 20))
        
        return story
    
    def _create_pie_chart(self, data: List[Tuple[str, int]], title: str, figsize: Tuple[int, int] = (8, 6)):
        """Create a pie chart using matplotlib"""
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        try:
            fig, ax = plt.subplots(figsize=figsize)
            
            labels = [item[0].replace('_', ' ').title() for item in data]
            sizes = [item[1] for item in data]
            
            # Color scheme
            colors_list = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8']
            
            wedges, texts, autotexts = ax.pie(
                sizes, 
                labels=labels, 
                autopct='%1.1f%%',
                startangle=90,
                colors=colors_list[:len(data)]
            )
            
            ax.set_title(title, fontsize=14, fontweight='bold')
            
            # Save to bytes
            img_buffer = BytesIO()
            plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
            img_buffer.seek(0)
            
            # Create ReportLab image
            img = Image(img_buffer, width=6*inch, height=4*inch)
            plt.close(fig)
            
            return img
            
        except Exception as e:
            self.logger.error(f"Error creating pie chart: {e}")
            return None
    
    def _create_single_contract_section(self, analysis_results: Dict[str, Any], include_code: bool = False) -> List:
        """Create single contract analysis section"""
        story = []
        
        contract_path = analysis_results.get('contract_path', 'Unknown')
        story.append(Paragraph(f"Analysis Results for {Path(contract_path).name}", self.styles['CustomSubtitle']))
        story.append(Spacer(1, 12))
        
        if 'results' not in analysis_results:
            story.append(Paragraph("No analysis results available.", self.styles['Normal']))
            return story
        
        results = analysis_results['results']
        
        # AI Analysis Results
        if 'ai_analysis' in results:
            story.extend(self._create_ai_analysis_section(results['ai_analysis']))
        
        # External Tool Results
        for tool_name in ['slither_analysis', 'mythril_analysis']:
            if tool_name in results:
                tool_display_name = tool_name.replace('_analysis', '').title()
                story.extend(self._create_tool_analysis_section(
                    results[tool_name], tool_display_name
                ))
        
        # Tool Comparison
        if 'tool_comparison' in results:
            story.extend(self._create_tool_comparison_section(results['tool_comparison']))
        
        return story
    
    def _create_ai_analysis_section(self, ai_results: Dict[str, Any]) -> List:
        """Create AI analysis results section"""
        story = []
        
        story.append(Paragraph("AI Analysis Results", self.styles['SectionHeader']))
        story.append(Spacer(1, 8))
        
        # Binary Classification
        if 'binary_classification' in ai_results:
            binary = ai_results['binary_classification']
            prediction = binary['prediction']
            confidence = binary['confidence']
            
            # Style based on prediction
            if prediction == 'vulnerable':
                style = self.styles['VulnerabilityHigh']
                status_text = f"🚨 VULNERABLE (Confidence: {confidence:.1%})"
            else:
                style = self.styles['SafeContract']
                status_text = f"✅ SAFE (Confidence: {confidence:.1%})"
            
            story.append(Paragraph(f"<b>Binary Classification:</b> {status_text}", style))
            story.append(Spacer(1, 8))
        
        # Multi-class Classification
        if 'multiclass_classification' in ai_results:
            multiclass = ai_results['multiclass_classification']
            predicted_type = multiclass['predicted_type']
            probabilities = multiclass['probabilities']
            
            story.append(Paragraph(
                f"<b>Predicted Vulnerability Type:</b> {predicted_type.replace('_', ' ').title()}",
                self.styles['Normal']
            ))
            story.append(Spacer(1, 8))
            
            # Probability table
            prob_data = [['Vulnerability Type', 'Probability']]
            sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
            
            for vuln_type, prob in sorted_probs:
                prob_data.append([
                    vuln_type.replace('_', ' ').title(),
                    f"{prob:.1%}"
                ])
            
            prob_table = Table(prob_data, colWidths=[3*inch, 1.5*inch])
            prob_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), self.colors['secondary']),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige)
            ]))
            
            story.append(prob_table)
        
        # Execution time
        if 'execution_time' in ai_results:
            story.append(Spacer(1, 8))
            story.append(Paragraph(
                f"<b>Analysis Time:</b> {ai_results['execution_time']:.2f} seconds",
                self.styles['Normal']
            ))
        
        story.append(Spacer(1, 16))
        return story
    
    def _create_tool_analysis_section(self, tool_results: Dict[str, Any], tool_name: str) -> List:
        """Create external tool analysis section"""
        story = []
        
        story.append(Paragraph(f"{tool_name} Analysis Results", self.styles['SectionHeader']))
        story.append(Spacer(1, 8))
        
        vulnerabilities = tool_results.get('vulnerabilities', [])
        
        if not vulnerabilities:
            story.append(Paragraph("No vulnerabilities detected.", self.styles['SafeContract']))
        else:
            story.append(Paragraph(
                f"<b>{len(vulnerabilities)} vulnerabilities detected:</b>",
                self.styles['Normal']
            ))
            story.append(Spacer(1, 8))
            
            # Vulnerabilities table
            vuln_data = [['Type', 'Severity', 'Description']]
            
            for vuln in vulnerabilities:
                vuln_type = vuln.get('type', 'Unknown')
                severity = vuln.get('severity', 'Unknown')
                description = vuln.get('description', 'No description available')
                
                # Truncate long descriptions
                if len(description) > 100:
                    description = description[:97] + "..."
                
                vuln_data.append([vuln_type, severity, description])
            
            vuln_table = Table(vuln_data, colWidths=[1.5*inch, 1*inch, 3*inch])
            vuln_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), self.colors['warning']),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('VALIGN', (0, 0), (-1, -1), 'TOP')
            ]))
            
            story.append(vuln_table)
        
        # Execution time
        if 'execution_time' in tool_results:
            story.append(Spacer(1, 8))
            story.append(Paragraph(
                f"<b>Analysis Time:</b> {tool_results['execution_time']:.2f} seconds",
                self.styles['Normal']
            ))
        
        story.append(Spacer(1, 16))
        return story
    
    def _create_tool_comparison_section(self, comparison_results: Dict[str, Any]) -> List:
        """Create tool comparison section"""
        story = []
        
        story.append(Paragraph("Tool Comparison", self.styles['SectionHeader']))
        story.append(Spacer(1, 8))
        
        consensus = comparison_results.get('consensus', 'unknown')
        agreement_score = comparison_results.get('agreement_score', 0)
        
        if consensus == 'vulnerable':
            style = self.styles['VulnerabilityHigh']
            consensus_text = f"🚨 CONSENSUS: VULNERABLE (Agreement: {agreement_score:.1%})"
        else:
            style = self.styles['SafeContract']
            consensus_text = f"✅ CONSENSUS: SAFE (Agreement: {agreement_score:.1%})"
        
        story.append(Paragraph(consensus_text, style))
        
        if comparison_results.get('conflicting_results', False):
            story.append(Spacer(1, 8))
            story.append(Paragraph(
                "⚠️ <b>Note:</b> Conflicting results detected between analysis tools. "
                "Manual review recommended.",
                self.styles['Normal']
            ))
        
        story.append(Spacer(1, 16))
        return story
    
    def _create_contract_details_section(self, analysis_results: Dict[str, Any], include_code: bool = False) -> List:
        """Create detailed contract analysis section for batch reports"""
        story = []
        
        story.append(Paragraph("Detailed Contract Analysis", self.styles['CustomSubtitle']))
        story.append(Spacer(1, 12))
        
        results = analysis_results.get('results', [])
        
        for i, result in enumerate(results, 1):
            contract_path = result.get('contract_path', f'Contract {i}')
            
            story.append(Paragraph(
                f"{i}. {Path(contract_path).name}",
                self.styles['SectionHeader']
            ))
            
            # Contract summary
            if 'results' in result:
                contract_results = result['results']
                
                # AI Analysis summary
                if 'ai_analysis' in contract_results:
                    ai_result = contract_results['ai_analysis']
                    if 'binary_classification' in ai_result:
                        prediction = ai_result['binary_classification']['prediction']
                        confidence = ai_result['binary_classification']['confidence']
                        
                        if prediction == 'vulnerable':
                            story.append(Paragraph(
                                f"🚨 AI Classification: VULNERABLE ({confidence:.1%} confidence)",
                                self.styles['VulnerabilityHigh']
                            ))
                        else:
                            story.append(Paragraph(
                                f"✅ AI Classification: SAFE ({confidence:.1%} confidence)",
                                self.styles['SafeContract']
                            ))
                
                # External tool summaries
                for tool_name in ['slither_analysis', 'mythril_analysis']:
                    if tool_name in contract_results:
                        tool_result = contract_results[tool_name]
                        vulnerabilities = tool_result.get('vulnerabilities', [])
                        tool_display = tool_name.replace('_analysis', '').title()
                        
                        if vulnerabilities:
                            story.append(Paragraph(
                                f"⚠️ {tool_display}: {len(vulnerabilities)} vulnerabilities found",
                                self.styles['Normal']
                            ))
                        else:
                            story.append(Paragraph(
                                f"✅ {tool_display}: No vulnerabilities detected",
                                self.styles['Normal']
                            ))
            
            story.append(Spacer(1, 12))
            
            # Add page break every 5 contracts to avoid overcrowding
            if i % 5 == 0 and i < len(results):
                story.append(PageBreak())
        
        return story
    
    def _create_recommendations_section(self, analysis_results: Dict[str, Any]) -> List:
        """Create recommendations section"""
        story = []
        
        story.append(Paragraph("Security Recommendations", self.styles['CustomSubtitle']))
        story.append(Spacer(1, 12))
        
        recommendations = self._generate_recommendations(analysis_results)
        
        for category, recs in recommendations.items():
            if recs:
                story.append(Paragraph(f"<b>{category}:</b>", self.styles['SectionHeader']))
                
                for rec in recs:
                    story.append(Paragraph(f"• {rec}", self.styles['Normal']))
                
                story.append(Spacer(1, 12))
        
        return story
    
    def _create_appendix_section(self, analysis_results: Dict[str, Any]) -> List:
        """Create appendix section"""
        story = []
        
        story.append(Paragraph("Appendix", self.styles['CustomSubtitle']))
        story.append(Spacer(1, 12))
        
        # Tool information
        story.append(Paragraph("Analysis Tools Used", self.styles['SectionHeader']))
        
        tools_info = {
            'AI Models': 'Machine learning models trained on smart contract vulnerability datasets',
            'Slither': 'Static analysis framework for Solidity smart contracts',
            'Mythril': 'Security analysis tool using symbolic execution and SMT solving'
        }
        
        for tool, description in tools_info.items():
            story.append(Paragraph(f"<b>{tool}:</b> {description}", self.styles['Normal']))
            story.append(Spacer(1, 6))
        
        story.append(Spacer(1, 12))
        
        # Vulnerability types glossary
        story.append(Paragraph("Vulnerability Types Glossary", self.styles['SectionHeader']))
        
        vuln_glossary = {
            'Reentrancy': 'Vulnerabilities where external calls can re-enter the contract before state changes are finalized',
            'Access Control': 'Issues with function visibility and permission management',
            'Arithmetic': 'Integer overflow, underflow, and other arithmetic-related vulnerabilities',
            'Unchecked Calls': 'External calls that do not properly handle return values or failures',
            'DoS': 'Denial of Service vulnerabilities that can make contracts unusable',
            'Bad Randomness': 'Use of predictable or manipulable sources of randomness'
        }
        
        for vuln_type, description in vuln_glossary.items():
            story.append(Paragraph(f"<b>{vuln_type}:</b> {description}", self.styles['Normal']))
            story.append(Spacer(1, 6))
        
        return story
    
    def _generate_recommendations(self, analysis_results: Dict[str, Any]) -> Dict[str, List[str]]:
        """Generate security recommendations based on analysis results"""
        recommendations = {
            'Immediate Actions': [],
            'Security Best Practices': [],
            'Development Recommendations': [],
            'Monitoring and Maintenance': []
        }
        
        # Analyze results to generate specific recommendations
        if 'batch_analysis_summary' in analysis_results:
            # Batch analysis recommendations
            results = analysis_results.get('results', [])
            vulnerable_count = 0
            vulnerability_types = set()
            
            for result in results:
                if 'results' in result and 'ai_analysis' in result['results']:
                    ai_result = result['results']['ai_analysis']
                    if 'binary_classification' in ai_result:
                        if ai_result['binary_classification']['prediction'] == 'vulnerable':
                            vulnerable_count += 1
                            
                            if 'multiclass_classification' in ai_result:
                                vuln_type = ai_result['multiclass_classification']['predicted_type']
                                vulnerability_types.add(vuln_type)
            
            if vulnerable_count > 0:
                recommendations['Immediate Actions'].extend([
                    f"Review and address vulnerabilities in {vulnerable_count} identified contracts",
                    "Prioritize contracts with high confidence vulnerability predictions",
                    "Conduct manual security audits for high-risk contracts"
                ])
            
            # Type-specific recommendations
            if 'reentrancy' in vulnerability_types:
                recommendations['Security Best Practices'].extend([
                    "Implement reentrancy guards using OpenZeppelin's ReentrancyGuard",
                    "Follow the checks-effects-interactions pattern"
                ])
            
            if 'access_control' in vulnerability_types:
                recommendations['Security Best Practices'].extend([
                    "Implement proper access control using OpenZeppelin's AccessControl",
                    "Review function visibility modifiers"
                ])
            
        else:
            # Single contract recommendations
            if 'results' in analysis_results:
                results = analysis_results['results']
                
                if 'ai_analysis' in results:
                    ai_result = results['ai_analysis']
                    if 'binary_classification' in ai_result:
                        if ai_result['binary_classification']['prediction'] == 'vulnerable':
                            recommendations['Immediate Actions'].append(
                                "Conduct thorough manual review of identified vulnerabilities"
                            )
        
        # General recommendations
        recommendations['Development Recommendations'].extend([
            "Use latest Solidity compiler version with security features enabled",
            "Implement comprehensive unit and integration tests",
            "Use established libraries like OpenZeppelin for common functionality",
            "Follow security best practices and coding standards"
        ])
        
        recommendations['Monitoring and Maintenance'].extend([
            "Regularly update dependencies and libraries",
            "Monitor for new vulnerability disclosures",
            "Implement continuous security testing in CI/CD pipeline",
            "Consider bug bounty programs for additional security review"
        ])
        
        return recommendations
    
    def _get_tools_used(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Extract list of analysis tools used"""
        tools = []
        
        if 'batch_analysis_summary' in analysis_results:
            # Check first result for tools used
            results = analysis_results.get('results', [])
            if results and 'results' in results[0]:
                sample_result = results[0]['results']
                if 'ai_analysis' in sample_result:
                    tools.append('AI Models')
                if 'slither_analysis' in sample_result:
                    tools.append('Slither')
                if 'mythril_analysis' in sample_result:
                    tools.append('Mythril')
        else:
            # Single contract analysis
            if 'results' in analysis_results:
                results = analysis_results['results']
                if 'ai_analysis' in results:
                    tools.append('AI Models')
                if 'slither_analysis' in results:
                    tools.append('Slither')
                if 'mythril_analysis' in results:
                    tools.append('Mythril')
        
        return tools


def generate_pdf_report(analysis_results: Dict[str, Any], 
                       output_path: str,
                       include_charts: bool = True,
                       include_code_snippets: bool = False) -> bool:
    """
    Convenience function to generate PDF report
    
    Args:
        analysis_results: Analysis results dictionary
        output_path: Path to save the PDF report
        include_charts: Whether to include charts and visualizations
        include_code_snippets: Whether to include code snippets in report
        
    Returns:
        bool: True if report generated successfully, False otherwise
    """
    try:
        generator = PDFReportGenerator()
        return generator.generate_report(
            analysis_results, 
            output_path, 
            include_charts, 
            include_code_snippets
        )
    except Exception as e:
        logging.error(f"Error generating PDF report: {e}")
        return False


# CLI interface for PDF generation
if __name__ == '__main__':
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate PDF report from analysis results')
    parser.add_argument('results_file', help='Path to analysis results JSON file')
    parser.add_argument('output_file', help='Path to output PDF file')
    parser.add_argument('--no-charts', action='store_true', help='Exclude charts from report')
    parser.add_argument('--include-code', action='store_true', help='Include code snippets in report')
    
    args = parser.parse_args()
    
    try:
        with open(args.results_file, 'r') as f:
            results = json.load(f)
        
        success = generate_pdf_report(
            results,
            args.output_file,
            include_charts=not args.no_charts,
            include_code_snippets=args.include_code
        )
        
        if success:
            print(f"PDF report generated successfully: {args.output_file}")
            sys.exit(0)
        else:
            print("Failed to generate PDF report")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)