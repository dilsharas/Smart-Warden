"""
Tool comparison system for Smart Warden.
Compares results from AI models, Slither, and Mythril.
"""

from typing import Dict, List, Any, Optional, Tuple
import logging
from collections import defaultdict
import statistics

logger = logging.getLogger(__name__)

class ToolComparator:
    """Compares and analyzes results from multiple security analysis tools."""
    
    def __init__(self):
        self.severity_weights = {
            'critical': 4,
            'high': 3,
            'medium': 2,
            'low': 1,
            'info': 0
        }
        
        self.confidence_weights = {
            'high': 3,
            'medium': 2,
            'low': 1
        }
    
    def compare_results(self, ai_result: Dict[str, Any], 
                       slither_result: Dict[str, Any], 
                       mythril_result: Dict[str, Any]) -> Dict[str, Any]:
        """Compare results from all three tools."""
        logger.info("Comparing results from AI, Slither, and Mythril")
        
        comparison = {
            'tools_used': [],
            'consensus': {},
            'discrepancies': [],
            'combined_score': 0,
            'recommendations': [],
            'tool_agreement': 0.0,
            'vulnerability_summary': {},
            'detailed_comparison': {}
        }
        
        # Collect available tools
        tools_data = {}
        
        if ai_result.get('available'):
            comparison['tools_used'].append('AI')
            tools_data['AI'] = ai_result
        
        if slither_result.get('available'):
            comparison['tools_used'].append('Slither')
            tools_data['Slither'] = slither_result
        
        if mythril_result.get('available'):
            comparison['tools_used'].append('Mythril')
            tools_data['Mythril'] = mythril_result
        
        if not tools_data:
            comparison['error'] = 'No tools available for comparison'
            return comparison
        
        # Analyze consensus
        comparison['consensus'] = self._analyze_consensus(tools_data)
        
        # Find discrepancies
        comparison['discrepancies'] = self._find_discrepancies(tools_data)
        
        # Calculate combined score
        comparison['combined_score'] = self._calculate_combined_score(tools_data)
        
        # Generate recommendations
        comparison['recommendations'] = self._generate_recommendations(tools_data)
        
        # Calculate tool agreement
        comparison['tool_agreement'] = self._calculate_agreement(tools_data)
        
        # Create vulnerability summary
        comparison['vulnerability_summary'] = self._create_vulnerability_summary(tools_data)
        
        # Detailed comparison
        comparison['detailed_comparison'] = self._create_detailed_comparison(tools_data)
        
        return comparison
    
    def _analyze_consensus(self, tools_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze consensus between tools."""
        consensus = {
            'is_vulnerable': None,
            'confidence': 0.0,
            'agreed_vulnerabilities': [],
            'agreement_level': 'none'
        }
        
        # Collect vulnerability assessments
        vulnerability_votes = []
        
        for tool_name, tool_data in tools_data.items():
            if tool_name == 'AI':
                # AI binary prediction
                if 'binary_prediction' in tool_data:
                    is_vuln = tool_data['binary_prediction'].get('is_vulnerable', False)
                    vulnerability_votes.append(is_vuln)
            else:
                # External tools
                vulns = tool_data.get('vulnerabilities', [])
                vulnerability_votes.append(len(vulns) > 0)
        
        # Determine consensus
        if vulnerability_votes:
            vuln_count = sum(vulnerability_votes)
            total_tools = len(vulnerability_votes)
            
            if vuln_count == total_tools:
                consensus['is_vulnerable'] = True
                consensus['agreement_level'] = 'unanimous'
                consensus['confidence'] = 0.9
            elif vuln_count > total_tools / 2:
                consensus['is_vulnerable'] = True
                consensus['agreement_level'] = 'majority'
                consensus['confidence'] = 0.7
            elif vuln_count == 0:
                consensus['is_vulnerable'] = False
                consensus['agreement_level'] = 'unanimous'
                consensus['confidence'] = 0.9
            else:
                consensus['is_vulnerable'] = False
                consensus['agreement_level'] = 'split'
                consensus['confidence'] = 0.5
        
        # Find agreed vulnerabilities
        consensus['agreed_vulnerabilities'] = self._find_agreed_vulnerabilities(tools_data)
        
        return consensus
    
    def _find_agreed_vulnerabilities(self, tools_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find vulnerabilities that multiple tools agree on."""
        agreed_vulns = []
        
        # Collect all vulnerability types
        vuln_types = defaultdict(list)
        
        for tool_name, tool_data in tools_data.items():
            if tool_name == 'AI' and 'multiclass_prediction' in tool_data:
                # AI multiclass prediction
                vuln_type = tool_data['multiclass_prediction'].get('vulnerability_type')
                if vuln_type and vuln_type != 'safe':
                    vuln_types[vuln_type].append({
                        'tool': tool_name,
                        'confidence': tool_data['multiclass_prediction'].get('confidence', 0.5)
                    })
            else:
                # External tools
                for vuln in tool_data.get('vulnerabilities', []):
                    vuln_type = vuln.get('type', 'unknown')
                    vuln_types[vuln_type].append({
                        'tool': tool_name,
                        'severity': vuln.get('severity', 'medium'),
                        'confidence': vuln.get('confidence', 'medium')
                    })
        
        # Find vulnerabilities detected by multiple tools
        for vuln_type, detections in vuln_types.items():
            if len(detections) > 1:
                agreed_vulns.append({
                    'type': vuln_type,
                    'detected_by': [d['tool'] for d in detections],
                    'detection_count': len(detections),
                    'average_confidence': self._calculate_average_confidence(detections)
                })
        
        return agreed_vulns
    
    def _find_discrepancies(self, tools_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find discrepancies between tool results."""
        discrepancies = []
        
        # Compare vulnerability assessments
        assessments = {}
        for tool_name, tool_data in tools_data.items():
            if tool_name == 'AI':
                if 'binary_prediction' in tool_data:
                    assessments[tool_name] = {
                        'is_vulnerable': tool_data['binary_prediction'].get('is_vulnerable', False),
                        'confidence': tool_data['binary_prediction'].get('confidence', 0.5)
                    }
            else:
                vulns = tool_data.get('vulnerabilities', [])
                assessments[tool_name] = {
                    'is_vulnerable': len(vulns) > 0,
                    'vulnerability_count': len(vulns)
                }
        
        # Find disagreements
        if len(assessments) > 1:
            vulnerability_opinions = [data['is_vulnerable'] for data in assessments.values()]
            if not all(op == vulnerability_opinions[0] for op in vulnerability_opinions):
                discrepancies.append({
                    'type': 'vulnerability_assessment',
                    'description': 'Tools disagree on whether contract is vulnerable',
                    'details': assessments
                })
        
        return discrepancies
    
    def _calculate_combined_score(self, tools_data: Dict[str, Any]) -> float:
        """Calculate combined risk score from all tools."""
        scores = []
        weights = []
        
        for tool_name, tool_data in tools_data.items():
            if tool_name == 'AI':
                if 'binary_prediction' in tool_data:
                    # AI score based on vulnerability probability
                    prob = tool_data['binary_prediction'].get('vulnerability_probability', 0.5)
                    confidence = tool_data['binary_prediction'].get('confidence', 0.5)
                    score = prob * 100
                    weight = confidence
                    scores.append(score)
                    weights.append(weight)
            else:
                # External tools score based on vulnerability count and severity
                vulns = tool_data.get('vulnerabilities', [])
                if vulns:
                    severity_scores = []
                    for vuln in vulns:
                        severity = vuln.get('severity', 'medium').lower()
                        severity_scores.append(self.severity_weights.get(severity, 2))
                    
                    # Average severity score, scaled to 0-100
                    avg_severity = statistics.mean(severity_scores)
                    score = min(avg_severity * 25, 100)  # Scale to 0-100
                    weight = 0.8  # External tools get high weight
                    scores.append(score)
                    weights.append(weight)
                else:
                    scores.append(0)
                    weights.append(0.5)
        
        # Calculate weighted average
        if scores and sum(weights) > 0:
            combined_score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
            return round(combined_score, 1)
        
        return 0.0
    
    def _generate_recommendations(self, tools_data: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on tool results."""
        recommendations = []
        
        # Collect all vulnerabilities
        all_vulns = []
        for tool_name, tool_data in tools_data.items():
            if tool_name == 'AI':
                if tool_data.get('binary_prediction', {}).get('is_vulnerable'):
                    vuln_type = tool_data.get('multiclass_prediction', {}).get('vulnerability_type', 'unknown')
                    all_vulns.append(vuln_type)
            else:
                for vuln in tool_data.get('vulnerabilities', []):
                    all_vulns.append(vuln.get('type', 'unknown'))
        
        # Generate specific recommendations
        unique_vulns = set(all_vulns)
        
        if 'reentrancy' in unique_vulns:
            recommendations.append("Implement checks-effects-interactions pattern to prevent reentrancy attacks")
        
        if 'access_control' in unique_vulns:
            recommendations.append("Add proper access control modifiers (onlyOwner, etc.)")
        
        if 'arithmetic' in unique_vulns:
            recommendations.append("Use Solidity 0.8+ for built-in overflow protection or SafeMath library")
        
        if 'unchecked_calls' in unique_vulns or 'external_call' in unique_vulns:
            recommendations.append("Always check return values of external calls")
        
        if 'bad_randomness' in unique_vulns:
            recommendations.append("Use secure randomness sources (Chainlink VRF, commit-reveal schemes)")
        
        if 'dos' in unique_vulns:
            recommendations.append("Avoid unbounded loops and implement gas-efficient patterns")
        
        # General recommendations
        if len(tools_data) > 1:
            recommendations.append("Consider running multiple analysis tools for comprehensive security review")
        
        if not recommendations:
            recommendations.append("Contract appears secure, but consider professional audit for production use")
        
        return recommendations
    
    def _calculate_agreement(self, tools_data: Dict[str, Any]) -> float:
        """Calculate agreement percentage between tools."""
        if len(tools_data) < 2:
            return 1.0
        
        # Compare vulnerability assessments
        vulnerability_opinions = []
        
        for tool_name, tool_data in tools_data.items():
            if tool_name == 'AI':
                is_vuln = tool_data.get('binary_prediction', {}).get('is_vulnerable', False)
                vulnerability_opinions.append(is_vuln)
            else:
                vulns = tool_data.get('vulnerabilities', [])
                vulnerability_opinions.append(len(vulns) > 0)
        
        # Calculate agreement
        if vulnerability_opinions:
            agreements = sum(1 for op in vulnerability_opinions if op == vulnerability_opinions[0])
            agreement_rate = agreements / len(vulnerability_opinions)
            return round(agreement_rate, 2)
        
        return 0.0
    
    def _create_vulnerability_summary(self, tools_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary of all detected vulnerabilities."""
        summary = {
            'total_vulnerabilities': 0,
            'by_severity': defaultdict(int),
            'by_type': defaultdict(int),
            'unique_types': set()
        }
        
        for tool_name, tool_data in tools_data.items():
            if tool_name == 'AI':
                if tool_data.get('binary_prediction', {}).get('is_vulnerable'):
                    summary['total_vulnerabilities'] += 1
                    vuln_type = tool_data.get('multiclass_prediction', {}).get('vulnerability_type', 'unknown')
                    summary['by_type'][vuln_type] += 1
                    summary['unique_types'].add(vuln_type)
                    summary['by_severity']['medium'] += 1  # Default AI severity
            else:
                vulns = tool_data.get('vulnerabilities', [])
                summary['total_vulnerabilities'] += len(vulns)
                
                for vuln in vulns:
                    vuln_type = vuln.get('type', 'unknown')
                    severity = vuln.get('severity', 'medium').lower()
                    
                    summary['by_type'][vuln_type] += 1
                    summary['by_severity'][severity] += 1
                    summary['unique_types'].add(vuln_type)
        
        # Convert to regular dict for JSON serialization
        summary['by_severity'] = dict(summary['by_severity'])
        summary['by_type'] = dict(summary['by_type'])
        summary['unique_types'] = list(summary['unique_types'])
        
        return summary
    
    def _create_detailed_comparison(self, tools_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create detailed comparison of tool results."""
        detailed = {
            'tool_results': {},
            'performance_comparison': {},
            'reliability_scores': {}
        }
        
        for tool_name, tool_data in tools_data.items():
            detailed['tool_results'][tool_name] = {
                'available': tool_data.get('available', False),
                'vulnerability_count': 0,
                'findings': []
            }
            
            if tool_name == 'AI':
                if 'binary_prediction' in tool_data:
                    binary = tool_data['binary_prediction']
                    detailed['tool_results'][tool_name].update({
                        'is_vulnerable': binary.get('is_vulnerable', False),
                        'confidence': binary.get('confidence', 0.5),
                        'vulnerability_count': 1 if binary.get('is_vulnerable') else 0
                    })
                
                if 'multiclass_prediction' in tool_data:
                    multiclass = tool_data['multiclass_prediction']
                    detailed['tool_results'][tool_name]['findings'].append({
                        'type': multiclass.get('vulnerability_type', 'unknown'),
                        'confidence': multiclass.get('confidence', 0.5)
                    })
            else:
                vulns = tool_data.get('vulnerabilities', [])
                detailed['tool_results'][tool_name]['vulnerability_count'] = len(vulns)
                detailed['tool_results'][tool_name]['findings'] = vulns
        
        return detailed
    
    def _calculate_average_confidence(self, detections: List[Dict[str, Any]]) -> float:
        """Calculate average confidence from multiple detections."""
        confidences = []
        
        for detection in detections:
            if 'confidence' in detection:
                conf = detection['confidence']
                if isinstance(conf, str):
                    # Convert string confidence to numeric
                    conf_map = {'high': 0.8, 'medium': 0.6, 'low': 0.4}
                    confidences.append(conf_map.get(conf.lower(), 0.5))
                else:
                    confidences.append(float(conf))
        
        return statistics.mean(confidences) if confidences else 0.5