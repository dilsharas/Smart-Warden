#!/usr/bin/env python3
"""
Data Validation and Quality Checking for SmartBugs Wild Dataset.
Implements comprehensive validation checks and quality metrics.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
import logging
import re
import hashlib
from dataclasses import dataclass, asdict
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path for imports
import sys
sys.path.insert(0, 'src')

from data.efficient_data_loader import ContractData

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Result of data validation."""
    is_valid: bool
    quality_score: float
    issues: Dict[str, List[str]]
    recommendations: List[str]
    statistics: Dict[str, any]

@dataclass
class QualityMetrics:
    """Quality metrics for dataset."""
    completeness_score: float
    consistency_score: float
    validity_score: float
    uniqueness_score: float
    accuracy_score: float
    overall_score: float

class DataValidator:
    """
    Comprehensive data validator for SmartBugs Wild dataset.
    Checks data quality, consistency, and validity.
    """
    
    def __init__(self, min_quality_score: float = 0.8):
        """
        Initialize the data validator.
        
        Args:
            min_quality_score: Minimum acceptable quality score
        """
        self.min_quality_score = min_quality_score
        
        # Solidity patterns for validation
        self.solidity_patterns = {
            'pragma': r'pragma\s+solidity\s+[\^~>=<]*\d+\.\d+\.\d+',
            'contract': r'contract\s+\w+',
            'function': r'function\s+\w+',
            'modifier': r'modifier\s+\w+',
            'event': r'event\s+\w+',
            'import': r'import\s+',
            'license': r'SPDX-License-Identifier'
        }
        
        # Vulnerability patterns
        self.vulnerability_patterns = {
            'reentrancy': [
                r'\.call\s*\(',
                r'\.send\s*\(',
                r'\.transfer\s*\(',
                r'external.*call'
            ],
            'access_control': [
                r'onlyOwner',
                r'require\s*\(\s*msg\.sender',
                r'modifier.*owner',
                r'access.*control'
            ],
            'arithmetic': [
                r'\+\+',
                r'--',
                r'\*',
                r'/',
                r'SafeMath',
                r'overflow',
                r'underflow'
            ],
            'unchecked_calls': [
                r'\.call\s*\(',
                r'\.delegatecall\s*\(',
                r'\.staticcall\s*\(',
                r'low.*level.*call'
            ],
            'bad_randomness': [
                r'block\.timestamp',
                r'block\.number',
                r'blockhash',
                r'now',
                r'random'
            ]
        }
    
    def validate_dataset(self, contracts: List[ContractData]) -> ValidationResult:
        """
        Perform comprehensive validation of the dataset.
        
        Args:
            contracts: List of contract data to validate
            
        Returns:
            ValidationResult with detailed findings
        """
        logger.info(f"🔍 Validating dataset with {len(contracts)} contracts...")
        
        issues = {
            'empty_contracts': [],
            'invalid_solidity': [],
            'duplicate_contracts': [],
            'encoding_issues': [],
            'missing_metadata': [],
            'suspicious_content': [],
            'label_inconsistencies': [],
            'size_anomalies': []
        }
        
        statistics = {}
        
        # Basic validation checks
        issues.update(self._check_basic_validity(contracts))
        
        # Content validation
        issues.update(self._check_content_validity(contracts))
        
        # Duplicate detection
        issues.update(self._check_duplicates(contracts))
        
        # Label consistency
        issues.update(self._check_label_consistency(contracts))
        
        # Size and structure validation
        issues.update(self._check_size_anomalies(contracts))
        
        # Calculate quality metrics
        quality_metrics = self._calculate_quality_metrics(contracts, issues)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(issues, quality_metrics)
        
        # Compile statistics
        statistics = self._compile_statistics(contracts, issues, quality_metrics)
        
        # Determine overall validity
        is_valid = quality_metrics.overall_score >= self.min_quality_score
        
        result = ValidationResult(
            is_valid=is_valid,
            quality_score=quality_metrics.overall_score,
            issues=issues,
            recommendations=recommendations,
            statistics=statistics
        )
        
        logger.info(f"✅ Validation complete. Quality score: {quality_metrics.overall_score:.3f}")
        
        return result
    
    def _check_basic_validity(self, contracts: List[ContractData]) -> Dict[str, List[str]]:
        """Check basic validity of contracts."""
        issues = {
            'empty_contracts': [],
            'invalid_solidity': [],
            'encoding_issues': []
        }
        
        for contract in contracts:
            # Check for empty contracts
            if not contract.code or len(contract.code.strip()) < 20:
                issues['empty_contracts'].append(contract.filename)
                continue
            
            # Check for basic Solidity structure
            if not self._is_valid_solidity(contract.code):
                issues['invalid_solidity'].append(contract.filename)
            
            # Check for encoding issues
            if self._has_encoding_issues(contract.code):
                issues['encoding_issues'].append(contract.filename)
        
        return issues
    
    def _check_content_validity(self, contracts: List[ContractData]) -> Dict[str, List[str]]:
        """Check content validity and suspicious patterns."""
        issues = {
            'suspicious_content': [],
            'missing_metadata': []
        }
        
        for contract in contracts:
            # Check for suspicious content
            if self._has_suspicious_content(contract.code):
                issues['suspicious_content'].append(contract.filename)
            
            # Check for missing metadata
            if not contract.metadata or len(contract.metadata) == 0:
                issues['missing_metadata'].append(contract.filename)
        
        return issues
    
    def _check_duplicates(self, contracts: List[ContractData]) -> Dict[str, List[str]]:
        """Check for duplicate contracts."""
        issues = {'duplicate_contracts': []}
        
        seen_hashes = set()
        seen_content = {}
        
        for contract in contracts:
            # Check hash duplicates
            if contract.hash in seen_hashes:
                issues['duplicate_contracts'].append(contract.filename)
            else:
                seen_hashes.add(contract.hash)
            
            # Check content similarity (normalized)
            normalized_content = self._normalize_code(contract.code)
            content_hash = hashlib.md5(normalized_content.encode()).hexdigest()
            
            if content_hash in seen_content:
                issues['duplicate_contracts'].append(
                    f"{contract.filename} (similar to {seen_content[content_hash]})"
                )
            else:
                seen_content[content_hash] = contract.filename
        
        return issues
    
    def _check_label_consistency(self, contracts: List[ContractData]) -> Dict[str, List[str]]:
        """Check consistency of vulnerability labels."""
        issues = {'label_inconsistencies': []}
        
        for contract in contracts:
            # Check if vulnerability labels match code patterns
            detected_vulns = self._detect_vulnerability_patterns(contract.code)
            labeled_vulns = set(contract.vulnerabilities)
            
            # Check for missing labels
            missing_labels = detected_vulns - labeled_vulns
            if missing_labels:
                issues['label_inconsistencies'].append(
                    f"{contract.filename}: Missing labels {missing_labels}"
                )
            
            # Check for extra labels
            extra_labels = labeled_vulns - detected_vulns
            if extra_labels and len(extra_labels) > 1:  # Allow some flexibility
                issues['label_inconsistencies'].append(
                    f"{contract.filename}: Questionable labels {extra_labels}"
                )
        
        return issues
    
    def _check_size_anomalies(self, contracts: List[ContractData]) -> Dict[str, List[str]]:
        """Check for size and structure anomalies."""
        issues = {'size_anomalies': []}
        
        # Calculate size statistics
        sizes = [contract.size for contract in contracts]
        mean_size = np.mean(sizes)
        std_size = np.std(sizes)
        
        # Define anomaly thresholds
        min_threshold = max(50, mean_size - 3 * std_size)
        max_threshold = mean_size + 3 * std_size
        
        for contract in contracts:
            # Check for extremely small contracts
            if contract.size < min_threshold:
                issues['size_anomalies'].append(
                    f"{contract.filename}: Too small ({contract.size} bytes)"
                )
            
            # Check for extremely large contracts
            elif contract.size > max_threshold:
                issues['size_anomalies'].append(
                    f"{contract.filename}: Too large ({contract.size} bytes)"
                )
        
        return issues
    
    def _is_valid_solidity(self, code: str) -> bool:
        """Check if code appears to be valid Solidity."""
        # Must have pragma or contract declaration
        has_pragma = bool(re.search(self.solidity_patterns['pragma'], code, re.IGNORECASE))
        has_contract = bool(re.search(self.solidity_patterns['contract'], code, re.IGNORECASE))
        
        return has_pragma or has_contract
    
    def _has_encoding_issues(self, code: str) -> bool:
        """Check for encoding issues in code."""
        # Look for common encoding artifacts
        encoding_artifacts = [
            '\ufffd',  # Replacement character
            '\x00',    # Null bytes
            '\\x',     # Escaped hex sequences
            '\\u',     # Escaped unicode
        ]
        
        return any(artifact in code for artifact in encoding_artifacts)
    
    def _has_suspicious_content(self, code: str) -> bool:
        """Check for suspicious content patterns."""
        suspicious_patterns = [
            r'test.*contract',
            r'mock.*contract',
            r'example.*contract',
            r'demo.*contract',
            r'hello.*world',
            r'foo.*bar',
            r'placeholder',
            r'todo',
            r'fixme'
        ]
        
        code_lower = code.lower()
        return any(re.search(pattern, code_lower) for pattern in suspicious_patterns)
    
    def _normalize_code(self, code: str) -> str:
        """Normalize code for similarity comparison."""
        # Remove comments
        code = re.sub(r'//.*$', '', code, flags=re.MULTILINE)
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        
        # Remove whitespace and normalize
        code = re.sub(r'\s+', ' ', code)
        code = code.strip().lower()
        
        return code
    
    def _detect_vulnerability_patterns(self, code: str) -> Set[str]:
        """Detect vulnerability patterns in code."""
        detected = set()
        code_lower = code.lower()
        
        for vuln_type, patterns in self.vulnerability_patterns.items():
            for pattern in patterns:
                if re.search(pattern, code_lower):
                    detected.add(vuln_type)
                    break
        
        return detected
    
    def _calculate_quality_metrics(self, 
                                 contracts: List[ContractData], 
                                 issues: Dict[str, List[str]]) -> QualityMetrics:
        """Calculate comprehensive quality metrics."""
        total_contracts = len(contracts)
        
        # Completeness: contracts with all required fields
        complete_contracts = total_contracts - len(issues.get('missing_metadata', []))
        completeness_score = complete_contracts / total_contracts if total_contracts > 0 else 0
        
        # Consistency: contracts without label inconsistencies
        consistent_contracts = total_contracts - len(issues.get('label_inconsistencies', []))
        consistency_score = consistent_contracts / total_contracts if total_contracts > 0 else 0
        
        # Validity: contracts without validity issues
        invalid_contracts = (
            len(issues.get('empty_contracts', [])) +
            len(issues.get('invalid_solidity', [])) +
            len(issues.get('encoding_issues', []))
        )
        valid_contracts = total_contracts - invalid_contracts
        validity_score = valid_contracts / total_contracts if total_contracts > 0 else 0
        
        # Uniqueness: contracts without duplicates
        unique_contracts = total_contracts - len(issues.get('duplicate_contracts', []))
        uniqueness_score = unique_contracts / total_contracts if total_contracts > 0 else 0
        
        # Accuracy: contracts without suspicious content or anomalies
        inaccurate_contracts = (
            len(issues.get('suspicious_content', [])) +
            len(issues.get('size_anomalies', []))
        )
        accurate_contracts = total_contracts - inaccurate_contracts
        accuracy_score = accurate_contracts / total_contracts if total_contracts > 0 else 0
        
        # Overall score (weighted average)
        overall_score = (
            completeness_score * 0.2 +
            consistency_score * 0.25 +
            validity_score * 0.25 +
            uniqueness_score * 0.15 +
            accuracy_score * 0.15
        )
        
        return QualityMetrics(
            completeness_score=completeness_score,
            consistency_score=consistency_score,
            validity_score=validity_score,
            uniqueness_score=uniqueness_score,
            accuracy_score=accuracy_score,
            overall_score=overall_score
        )
    
    def _generate_recommendations(self, 
                                issues: Dict[str, List[str]], 
                                quality_metrics: QualityMetrics) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        # Recommendations based on issues
        if issues.get('empty_contracts'):
            recommendations.append(
                f"Remove {len(issues['empty_contracts'])} empty contracts"
            )
        
        if issues.get('duplicate_contracts'):
            recommendations.append(
                f"Remove {len(issues['duplicate_contracts'])} duplicate contracts"
            )
        
        if issues.get('invalid_solidity'):
            recommendations.append(
                f"Review {len(issues['invalid_solidity'])} contracts with invalid Solidity"
            )
        
        if issues.get('label_inconsistencies'):
            recommendations.append(
                f"Review {len(issues['label_inconsistencies'])} contracts with label inconsistencies"
            )
        
        if issues.get('suspicious_content'):
            recommendations.append(
                f"Review {len(issues['suspicious_content'])} contracts with suspicious content"
            )
        
        # Recommendations based on quality scores
        if quality_metrics.completeness_score < 0.9:
            recommendations.append("Improve data completeness by adding missing metadata")
        
        if quality_metrics.consistency_score < 0.8:
            recommendations.append("Improve label consistency through manual review")
        
        if quality_metrics.validity_score < 0.9:
            recommendations.append("Remove or fix invalid contracts")
        
        if quality_metrics.uniqueness_score < 0.95:
            recommendations.append("Remove duplicate contracts to improve uniqueness")
        
        if quality_metrics.accuracy_score < 0.85:
            recommendations.append("Review and clean suspicious or anomalous contracts")
        
        return recommendations
    
    def _compile_statistics(self, 
                          contracts: List[ContractData], 
                          issues: Dict[str, List[str]], 
                          quality_metrics: QualityMetrics) -> Dict:
        """Compile comprehensive statistics."""
        # Basic statistics
        sizes = [contract.size for contract in contracts]
        vuln_counts = [len(contract.vulnerabilities) for contract in contracts]
        
        # Vulnerability distribution
        all_vulns = []
        for contract in contracts:
            all_vulns.extend(contract.vulnerabilities)
        vuln_distribution = Counter(all_vulns)
        
        # Issue statistics
        total_issues = sum(len(issue_list) for issue_list in issues.values())
        
        statistics = {
            'total_contracts': len(contracts),
            'total_issues': total_issues,
            'issue_rate': total_issues / len(contracts) if contracts else 0,
            'size_statistics': {
                'mean': np.mean(sizes),
                'median': np.median(sizes),
                'std': np.std(sizes),
                'min': np.min(sizes),
                'max': np.max(sizes)
            },
            'vulnerability_statistics': {
                'mean_vulns_per_contract': np.mean(vuln_counts),
                'vulnerable_contracts': sum(1 for c in contracts if c.is_vulnerable),
                'vulnerability_distribution': dict(vuln_distribution)
            },
            'quality_metrics': asdict(quality_metrics),
            'issue_breakdown': {k: len(v) for k, v in issues.items()}
        }
        
        return statistics
    
    def generate_validation_report(self, 
                                 validation_result: ValidationResult,
                                 output_path: Optional[str] = None) -> str:
        """Generate a comprehensive validation report."""
        report_lines = []
        
        # Header
        report_lines.append("=" * 80)
        report_lines.append("SMARTBUGS WILD DATASET VALIDATION REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Overall status
        status = "PASSED" if validation_result.is_valid else "FAILED"
        report_lines.append(f"Overall Status: {status}")
        report_lines.append(f"Quality Score: {validation_result.quality_score:.3f}")
        report_lines.append("")
        
        # Statistics
        stats = validation_result.statistics
        report_lines.append("DATASET STATISTICS")
        report_lines.append("-" * 40)
        report_lines.append(f"Total Contracts: {stats['total_contracts']}")
        report_lines.append(f"Total Issues: {stats['total_issues']}")
        report_lines.append(f"Issue Rate: {stats['issue_rate']:.3f}")
        report_lines.append("")
        
        # Quality metrics
        quality = stats['quality_metrics']
        report_lines.append("QUALITY METRICS")
        report_lines.append("-" * 40)
        for metric, score in quality.items():
            report_lines.append(f"{metric.replace('_', ' ').title()}: {score:.3f}")
        report_lines.append("")
        
        # Issues breakdown
        report_lines.append("ISSUES BREAKDOWN")
        report_lines.append("-" * 40)
        for issue_type, count in stats['issue_breakdown'].items():
            if count > 0:
                report_lines.append(f"{issue_type.replace('_', ' ').title()}: {count}")
        report_lines.append("")
        
        # Recommendations
        if validation_result.recommendations:
            report_lines.append("RECOMMENDATIONS")
            report_lines.append("-" * 40)
            for i, rec in enumerate(validation_result.recommendations, 1):
                report_lines.append(f"{i}. {rec}")
            report_lines.append("")
        
        # Detailed issues (first 10 of each type)
        report_lines.append("DETAILED ISSUES (Sample)")
        report_lines.append("-" * 40)
        for issue_type, issue_list in validation_result.issues.items():
            if issue_list:
                report_lines.append(f"\n{issue_type.replace('_', ' ').title()}:")
                for issue in issue_list[:10]:  # Show first 10
                    report_lines.append(f"  - {issue}")
                if len(issue_list) > 10:
                    report_lines.append(f"  ... and {len(issue_list) - 10} more")
        
        report_text = "\n".join(report_lines)
        
        # Save to file if requested
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
            logger.info(f"Validation report saved to {output_path}")
        
        return report_text
    
    def create_quality_visualization(self, 
                                   validation_result: ValidationResult,
                                   output_dir: str = "data/validation_plots"):
        """Create visualizations of data quality metrics."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Quality metrics radar chart
        metrics = validation_result.statistics['quality_metrics']
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Quality metrics bar chart
        ax1.bar(range(len(metric_names)), metric_values, color='skyblue')
        ax1.set_xticks(range(len(metric_names)))
        ax1.set_xticklabels([name.replace('_', '\n') for name in metric_names], rotation=45)
        ax1.set_ylabel('Score')
        ax1.set_title('Quality Metrics')
        ax1.set_ylim(0, 1)
        
        # Issues breakdown pie chart
        issue_counts = validation_result.statistics['issue_breakdown']
        non_zero_issues = {k: v for k, v in issue_counts.items() if v > 0}
        
        if non_zero_issues:
            ax2.pie(non_zero_issues.values(), labels=non_zero_issues.keys(), autopct='%1.1f%%')
            ax2.set_title('Issues Distribution')
        else:
            ax2.text(0.5, 0.5, 'No Issues Found', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Issues Distribution')
        
        # Vulnerability distribution
        vuln_dist = validation_result.statistics['vulnerability_statistics']['vulnerability_distribution']
        if vuln_dist:
            ax3.bar(vuln_dist.keys(), vuln_dist.values(), color='lightcoral')
            ax3.set_xlabel('Vulnerability Type')
            ax3.set_ylabel('Count')
            ax3.set_title('Vulnerability Distribution')
            ax3.tick_params(axis='x', rotation=45)
        
        # Size distribution histogram
        # This would need contract data to be passed separately
        ax4.text(0.5, 0.5, 'Size Distribution\n(Requires contract data)', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Contract Size Distribution')
        
        plt.tight_layout()
        plt.savefig(output_path / 'quality_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Quality visualization saved to {output_path}")


def main():
    """Example usage of DataValidator."""
    from data.efficient_data_loader import SmartBugsDataLoader
    
    # Load sample data
    loader = SmartBugsDataLoader("dataset/smartbugs-wild-master")
    contracts = loader.load_contracts_parallel(max_contracts=100)
    
    # Initialize validator
    validator = DataValidator(min_quality_score=0.8)
    
    # Validate dataset
    result = validator.validate_dataset(contracts)
    
    # Generate report
    report = validator.generate_validation_report(result, "data/validation_report.txt")
    print(report)
    
    # Create visualizations
    validator.create_quality_visualization(result)
    
    print(f"\n✅ Validation complete!")
    print(f"Quality Score: {result.quality_score:.3f}")
    print(f"Dataset Valid: {result.is_valid}")


if __name__ == "__main__":
    main()