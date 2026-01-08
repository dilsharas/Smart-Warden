"""
Report Generation Module

Exports analysis results in multiple formats (JSON, CSV, PDF).
"""

import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib import colors

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Generates analysis reports in multiple formats.
    
    Supports JSON, CSV, and PDF export with comprehensive metadata and visualizations.
    """

    def __init__(self):
        """Initialize the report generator."""
        self.results = {}
        self.timestamp = datetime.now().isoformat()

    def export_json(self, results: Dict, filepath: str) -> None:
        """
        Export results to JSON format.
        
        Args:
            results: Dictionary with analysis results
            filepath: Path to save JSON file
        """
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        
        export_data = {
            "timestamp": self.timestamp,
            "results": results,
            "metadata": {
                "format": "json",
                "version": "1.0",
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Exported results to JSON: {filepath}")

    def export_csv(self, results: Dict, filepath: str) -> None:
        """
        Export results to CSV format.
        
        Args:
            results: Dictionary with analysis results
            filepath: Path to save CSV file
        """
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        
        # Flatten results for CSV export
        csv_data = []
        
        # Add metrics
        if 'metrics' in results:
            for metric_name, metric_value in results['metrics'].items():
                if metric_name != 'latency':
                    csv_data.append({
                        'Category': 'Metrics',
                        'Name': metric_name,
                        'Value': metric_value,
                    })
        
        # Add latency stats
        if 'latency' in results.get('metrics', {}):
            latency = results['metrics']['latency']
            for stat_name, stat_value in latency.items():
                csv_data.append({
                    'Category': 'Latency',
                    'Name': stat_name,
                    'Value': stat_value,
                })
        
        # Add confusion matrix stats
        if 'confusion_matrix_stats' in results:
            for stat_name, stat_value in results['confusion_matrix_stats'].items():
                csv_data.append({
                    'Category': 'Confusion Matrix',
                    'Name': stat_name,
                    'Value': stat_value,
                })
        
        df = pd.DataFrame(csv_data)
        df.to_csv(filepath, index=False)
        
        logger.info(f"Exported results to CSV: {filepath}")

    def export_pdf(self, results: Dict, filepath: str, chart_paths: Optional[List[str]] = None) -> None:
        """
        Export results to PDF format with embedded charts.
        
        Args:
            results: Dictionary with analysis results
            filepath: Path to save PDF file
            chart_paths: List of paths to chart images to embed
        """
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        
        doc = SimpleDocTemplate(filepath, pagesize=letter)
        story = []
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1f77b4'),
            spaceAfter=30,
            alignment=1,  # Center
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#1f77b4'),
            spaceAfter=12,
            spaceBefore=12,
        )
        
        # Title
        story.append(Paragraph("Blockchain Fraud Detection Analysis Report", title_style))
        story.append(Spacer(1, 0.2*inch))
        
        # Timestamp
        story.append(Paragraph(f"Generated: {self.timestamp}", styles['Normal']))
        story.append(Spacer(1, 0.3*inch))
        
        # Metrics Section
        story.append(Paragraph("Performance Metrics", heading_style))
        
        if 'metrics' in results:
            metrics_data = [['Metric', 'Value']]
            for metric_name, metric_value in results['metrics'].items():
                if metric_name != 'latency':
                    if isinstance(metric_value, float):
                        metrics_data.append([metric_name.replace('_', ' ').title(), f"{metric_value:.4f}"])
                    else:
                        metrics_data.append([metric_name.replace('_', ' ').title(), str(metric_value)])
            
            metrics_table = Table(metrics_data, colWidths=[3*inch, 2*inch])
            metrics_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ]))
            story.append(metrics_table)
            story.append(Spacer(1, 0.3*inch))
        
        # Latency Section
        if 'latency' in results.get('metrics', {}):
            story.append(Paragraph("Latency Analysis", heading_style))
            latency = results['metrics']['latency']
            latency_data = [['Metric', 'Value (ms)']]
            for stat_name, stat_value in latency.items():
                if stat_name != 'iterations':
                    latency_data.append([stat_name.replace('_', ' ').title(), f"{stat_value:.2f}"])
            
            latency_table = Table(latency_data, colWidths=[3*inch, 2*inch])
            latency_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ]))
            story.append(latency_table)
            story.append(Spacer(1, 0.3*inch))
        
        # Charts Section
        if chart_paths:
            story.append(PageBreak())
            story.append(Paragraph("Visualizations", heading_style))
            story.append(Spacer(1, 0.2*inch))
            
            for chart_path in chart_paths:
                if os.path.exists(chart_path):
                    try:
                        img = Image(chart_path, width=6*inch, height=4*inch)
                        story.append(img)
                        story.append(Spacer(1, 0.3*inch))
                    except Exception as e:
                        logger.warning(f"Could not embed chart {chart_path}: {str(e)}")
        
        # Build PDF
        doc.build(story)
        logger.info(f"Exported results to PDF: {filepath}")

    def generate_summary_report(self, results: Dict) -> str:
        """
        Generate a text summary report.
        
        Args:
            results: Dictionary with analysis results
            
        Returns:
            Formatted text report
        """
        report = "=" * 70 + "\n"
        report += "BLOCKCHAIN FRAUD DETECTION ANALYSIS REPORT\n"
        report += "=" * 70 + "\n\n"
        
        report += f"Generated: {self.timestamp}\n\n"
        
        # Metrics Summary
        report += "PERFORMANCE METRICS:\n"
        report += "-" * 70 + "\n"
        if 'metrics' in results:
            for metric_name, metric_value in results['metrics'].items():
                if metric_name != 'latency':
                    if isinstance(metric_value, float):
                        report += f"  {metric_name.upper():20s}: {metric_value:.4f}\n"
                    else:
                        report += f"  {metric_name.upper():20s}: {metric_value}\n"
        
        # Latency Summary
        report += "\nLATENCY ANALYSIS (milliseconds):\n"
        report += "-" * 70 + "\n"
        if 'latency' in results.get('metrics', {}):
            latency = results['metrics']['latency']
            report += f"  {'Mean':20s}: {latency['mean_ms']:.2f} ms\n"
            report += f"  {'Std Dev':20s}: {latency['std_ms']:.2f} ms\n"
            report += f"  {'P50':20s}: {latency['p50_ms']:.2f} ms\n"
            report += f"  {'P95':20s}: {latency['p95_ms']:.2f} ms\n"
            report += f"  {'P99':20s}: {latency['p99_ms']:.2f} ms\n"
        
        # Confusion Matrix Summary
        report += "\nCONFUSION MATRIX:\n"
        report += "-" * 70 + "\n"
        if 'confusion_matrix_stats' in results:
            stats = results['confusion_matrix_stats']
            if 'true_positives' in stats:
                report += f"  True Positives:  {stats['true_positives']}\n"
                report += f"  True Negatives:  {stats['true_negatives']}\n"
                report += f"  False Positives: {stats['false_positives']}\n"
                report += f"  False Negatives: {stats['false_negatives']}\n"
        
        report += "\n" + "=" * 70 + "\n"
        
        return report

    def export_all_formats(self, results: Dict, output_dir: str, 
                          chart_paths: Optional[List[str]] = None) -> Dict[str, str]:
        """
        Export results in all supported formats.
        
        Args:
            results: Dictionary with analysis results
            output_dir: Directory to save files
            chart_paths: List of chart image paths for PDF
            
        Returns:
            Dictionary mapping format names to file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exported_files = {}
        
        # JSON export
        json_path = os.path.join(output_dir, f"fraud_detection_report_{timestamp}.json")
        self.export_json(results, json_path)
        exported_files['json'] = json_path
        
        # CSV export
        csv_path = os.path.join(output_dir, f"fraud_detection_report_{timestamp}.csv")
        self.export_csv(results, csv_path)
        exported_files['csv'] = csv_path
        
        # PDF export
        pdf_path = os.path.join(output_dir, f"fraud_detection_report_{timestamp}.pdf")
        self.export_pdf(results, pdf_path, chart_paths)
        exported_files['pdf'] = pdf_path
        
        # Text summary
        summary_path = os.path.join(output_dir, f"fraud_detection_summary_{timestamp}.txt")
        with open(summary_path, 'w') as f:
            f.write(self.generate_summary_report(results))
        exported_files['summary'] = summary_path
        
        logger.info(f"Exported all formats to {output_dir}")
        return exported_files
