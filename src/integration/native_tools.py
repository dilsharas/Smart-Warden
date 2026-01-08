"""
Native Python implementations of security analysis tools.
Provides Slither-like and Mythril-like analysis without external dependencies.
"""

import re
import ast
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class NativeSlitherAnalyzer:
    """Native Python implementation of Slither-like static analysis."""
    
    def __init__(self):
        self.vulnerability_patterns = {
            'reentrancy': [
                r'\.call\s*\{[^}]*value\s*:[^}]*\}\s*\([^)]*\)',
                r'\.call\s*\([^)]*\)',
                r'\.send\s*\([^)]*\)',
                r'\.transfer\s*\([^)]*\)'
            ],
            'access_control': [
                r'function\s+\w+\s*\([^)]*\)\s*public\s*(?!.*onlyOwner)',
                r'selfdestruct\s*\([^)]*\)',
                r'suicide\s*\([^)]*\)'
            ],
            'arithmetic': [
                r'\+\+|\-\-',
                r'\+=|\-=|\*=|\/=',
                r'[^a-zA-Z_](\w+)\s*\+\s*(\w+)',
                r'[^a-zA-Z_](\w+)\s*\*\s*(\w+)'
            ],
            'unchecked_calls': [
                r'\.call\s*\([^)]*\)\s*;',
                r'\.delegatecall\s*\([^)]*\)\s*;',
                r'\.staticcall\s*\([^)]*\)\s*;'
            ],
            'dos': [
                r'for\s*\([^)]*\)\s*\{[^}]*\.call',
                r'while\s*\([^)]*\)\s*\{[^}]*\.call',
                r'for\s*\([^)]*\)\s*\{[^}]*\.transfer'
            ],
            'bad_randomness': [
                r'block\.timestamp',
                r'block\.number',
                r'block\.difficulty',
                r'blockhash\s*\(',
                r'\bnow\b'
            ]
        }
        
        self.severity_mapping = {
            'reentrancy': 'high',
            'access_control': 'high',
            'arithmetic': 'medium',
            'unchecked_calls': 'medium',
            'dos': 'medium',
            'bad_randomness': 'low'
        }
    
    def analyze_contract(self, contract_code: str) -> Dict[str, Any]:
        """Analyze contract using native static analysis."""
        logger.info("Running native Slither-like analysis...")
        
        vulnerabilities = []
        
        # Analyze each vulnerability type
        for vuln_type, patterns in self.vulnerability_patterns.items():
            findings = self._detect_pattern_vulnerabilities(
                contract_code, vuln_type, patterns
            )
            vulnerabilities.extend(findings)
        
        # Additional context-aware analysis
        vulnerabilities.extend(self._analyze_reentrancy_context(contract_code))
        vulnerabilities.extend(self._analyze_access_control_context(contract_code))
        
        return {
            'tool': 'native_slither',
            'available': True,
            'vulnerabilities': vulnerabilities,
            'analysis_type': 'static_analysis',
            'confidence': 'medium'
        }
    
    def _detect_pattern_vulnerabilities(self, code: str, vuln_type: str, 
                                      patterns: List[str]) -> List[Dict[str, Any]]:
        """Detect vulnerabilities using regex patterns."""
        vulnerabilities = []
        lines = code.split('\n')
        
        for pattern in patterns:
            for line_num, line in enumerate(lines, 1):
                if re.search(pattern, line, re.IGNORECASE):
                    vulnerabilities.append({
                        'type': vuln_type,
                        'severity': self.severity_mapping.get(vuln_type, 'medium'),
                        'confidence': 'medium',
                        'line_number': line_num,
                        'description': self._get_vulnerability_description(vuln_type),
                        'recommendation': self._get_vulnerability_recommendation(vuln_type),
                        'code_snippet': line.strip(),
                        'pattern_matched': pattern
                    })
        
        return vulnerabilities
    
    def _analyze_reentrancy_context(self, code: str) -> List[Dict[str, Any]]:
        """Advanced reentrancy analysis with context."""
        vulnerabilities = []
        lines = code.split('\n')
        
        # Look for state changes after external calls
        for i, line in enumerate(lines):
            if re.search(r'\.call\s*\(|\.send\s*\(|\.transfer\s*\(', line):
                # Check next few lines for state changes
                for j in range(i + 1, min(i + 5, len(lines))):
                    next_line = lines[j]
                    if re.search(r'\w+\s*=\s*\w+|balances\[|mapping\[', next_line):
                        vulnerabilities.append({
                            'type': 'reentrancy',
                            'severity': 'high',
                            'confidence': 'high',
                            'line_number': i + 1,
                            'description': 'Potential reentrancy: state change after external call',
                            'recommendation': 'Use checks-effects-interactions pattern',
                            'code_snippet': f"{line.strip()} ... {next_line.strip()}",
                            'context': 'state_change_after_call'
                        })
                        break
        
        return vulnerabilities
    
    def _analyze_access_control_context(self, code: str) -> List[Dict[str, Any]]:
        """Advanced access control analysis."""
        vulnerabilities = []
        lines = code.split('\n')
        
        # Look for sensitive functions without access control
        sensitive_patterns = [
            r'selfdestruct\s*\(',
            r'suicide\s*\(',
            r'\.transfer\s*\(\s*address\s*\(\s*this\s*\)\.balance',
            r'owner\s*=\s*'
        ]
        
        for line_num, line in enumerate(lines, 1):
            for pattern in sensitive_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    # Check if function has access control
                    function_start = self._find_function_start(lines, line_num - 1)
                    if function_start and not self._has_access_control(lines, function_start):
                        vulnerabilities.append({
                            'type': 'access_control',
                            'severity': 'high',
                            'confidence': 'high',
                            'line_number': line_num,
                            'description': 'Sensitive function lacks access control',
                            'recommendation': 'Add onlyOwner or similar modifier',
                            'code_snippet': line.strip(),
                            'context': 'missing_access_control'
                        })
        
        return vulnerabilities
    
    def _find_function_start(self, lines: List[str], current_line: int) -> Optional[int]:
        """Find the start of the function containing the current line."""
        for i in range(current_line, -1, -1):
            if re.search(r'function\s+\w+', lines[i]):
                return i
        return None
    
    def _has_access_control(self, lines: List[str], function_start: int) -> bool:
        """Check if function has access control modifiers."""
        # Check function declaration and next few lines
        for i in range(function_start, min(function_start + 3, len(lines))):
            line = lines[i]
            if re.search(r'onlyOwner|onlyAdmin|require\s*\(\s*msg\.sender', line):
                return True
        return False
    
    def _get_vulnerability_description(self, vuln_type: str) -> str:
        """Get description for vulnerability type."""
        descriptions = {
            'reentrancy': 'Potential reentrancy vulnerability detected',
            'access_control': 'Missing or insufficient access control',
            'arithmetic': 'Potential arithmetic overflow/underflow',
            'unchecked_calls': 'External call return value not checked',
            'dos': 'Potential denial of service vulnerability',
            'bad_randomness': 'Weak randomness source detected'
        }
        return descriptions.get(vuln_type, 'Security vulnerability detected')
    
    def _get_vulnerability_recommendation(self, vuln_type: str) -> str:
        """Get recommendation for vulnerability type."""
        recommendations = {
            'reentrancy': 'Use checks-effects-interactions pattern or reentrancy guard',
            'access_control': 'Add proper access control modifiers (onlyOwner, etc.)',
            'arithmetic': 'Use SafeMath library or Solidity 0.8+ built-in protection',
            'unchecked_calls': 'Always check return values of external calls',
            'dos': 'Avoid unbounded loops and implement gas-efficient patterns',
            'bad_randomness': 'Use secure randomness sources (Chainlink VRF, etc.)'
        }
        return recommendations.get(vuln_type, 'Review and fix the identified issue')

class NativeMythrilAnalyzer:
    """Native Python implementation of Mythril-like symbolic analysis."""
    
    def __init__(self):
        self.symbolic_patterns = {
            'integer_overflow': [
                r'(\w+)\s*\+\s*(\w+)',
                r'(\w+)\s*\*\s*(\w+)',
                r'(\w+)\s*\*\*\s*(\w+)'
            ],
            'reentrancy': [
                r'\.call\.value\s*\(',
                r'\.call\s*\{[^}]*value[^}]*\}'
            ],
            'unchecked_call': [
                r'\.call\s*\([^)]*\)\s*;',
                r'\.send\s*\([^)]*\)\s*;'
            ],
            'suicide': [
                r'selfdestruct\s*\(',
                r'suicide\s*\('
            ],
            'delegatecall': [
                r'\.delegatecall\s*\('
            ]
        }
        
        self.swc_mapping = {
            'integer_overflow': 'SWC-101',
            'reentrancy': 'SWC-107',
            'unchecked_call': 'SWC-104',
            'suicide': 'SWC-106',
            'delegatecall': 'SWC-112'
        }
    
    def analyze_contract(self, contract_code: str) -> Dict[str, Any]:
        """Analyze contract using native symbolic analysis."""
        logger.info("Running native Mythril-like analysis...")
        
        vulnerabilities = []
        
        # Symbolic execution simulation
        vulnerabilities.extend(self._simulate_execution_paths(contract_code))
        
        # Pattern-based detection with symbolic context
        for vuln_type, patterns in self.symbolic_patterns.items():
            findings = self._detect_symbolic_vulnerabilities(
                contract_code, vuln_type, patterns
            )
            vulnerabilities.extend(findings)
        
        return {
            'tool': 'native_mythril',
            'available': True,
            'vulnerabilities': vulnerabilities,
            'analysis_type': 'symbolic_execution',
            'confidence': 'medium'
        }
    
    def _simulate_execution_paths(self, code: str) -> List[Dict[str, Any]]:
        """Simulate symbolic execution paths."""
        vulnerabilities = []
        lines = code.split('\n')
        
        # Look for complex control flow that might hide vulnerabilities
        for line_num, line in enumerate(lines, 1):
            # Check for complex conditions that might be exploitable
            if re.search(r'if\s*\([^)]*&&[^)]*\|\|[^)]*\)', line):
                vulnerabilities.append({
                    'type': 'complex_condition',
                    'severity': 'medium',
                    'confidence': 'low',
                    'line_number': line_num,
                    'description': 'Complex condition detected - potential logic vulnerability',
                    'recommendation': 'Simplify conditions and add explicit checks',
                    'code_snippet': line.strip(),
                    'swc_id': 'SWC-123'
                })
        
        return vulnerabilities
    
    def _detect_symbolic_vulnerabilities(self, code: str, vuln_type: str, 
                                       patterns: List[str]) -> List[Dict[str, Any]]:
        """Detect vulnerabilities with symbolic execution context."""
        vulnerabilities = []
        lines = code.split('\n')
        
        for pattern in patterns:
            for line_num, line in enumerate(lines, 1):
                if re.search(pattern, line, re.IGNORECASE):
                    # Add symbolic execution context
                    context = self._analyze_symbolic_context(lines, line_num - 1)
                    
                    vulnerabilities.append({
                        'type': vuln_type,
                        'severity': self._get_mythril_severity(vuln_type),
                        'confidence': 'medium',
                        'line_number': line_num,
                        'description': self._get_mythril_description(vuln_type),
                        'recommendation': self._get_mythril_recommendation(vuln_type),
                        'code_snippet': line.strip(),
                        'swc_id': self.swc_mapping.get(vuln_type, 'SWC-000'),
                        'symbolic_context': context
                    })
        
        return vulnerabilities
    
    def _analyze_symbolic_context(self, lines: List[str], line_num: int) -> Dict[str, Any]:
        """Analyze symbolic execution context around a line."""
        context = {
            'variables_modified': [],
            'external_calls': 0,
            'state_changes': 0
        }
        
        # Analyze surrounding lines for context
        start = max(0, line_num - 3)
        end = min(len(lines), line_num + 3)
        
        for i in range(start, end):
            line = lines[i]
            
            # Count variable modifications
            if re.search(r'\w+\s*=\s*', line):
                context['state_changes'] += 1
            
            # Count external calls
            if re.search(r'\.call\s*\(|\.send\s*\(|\.transfer\s*\(', line):
                context['external_calls'] += 1
        
        return context
    
    def _get_mythril_severity(self, vuln_type: str) -> str:
        """Get Mythril-style severity."""
        severity_map = {
            'integer_overflow': 'high',
            'reentrancy': 'high',
            'unchecked_call': 'medium',
            'suicide': 'medium',
            'delegatecall': 'high'
        }
        return severity_map.get(vuln_type, 'medium')
    
    def _get_mythril_description(self, vuln_type: str) -> str:
        """Get Mythril-style description."""
        descriptions = {
            'integer_overflow': 'Integer overflow/underflow vulnerability',
            'reentrancy': 'Reentrancy vulnerability in external call',
            'unchecked_call': 'Unchecked return value from external call',
            'suicide': 'Unprotected selfdestruct instruction',
            'delegatecall': 'Delegatecall to untrusted contract'
        }
        return descriptions.get(vuln_type, 'Security vulnerability detected')
    
    def _get_mythril_recommendation(self, vuln_type: str) -> str:
        """Get Mythril-style recommendation."""
        recommendations = {
            'integer_overflow': 'Use SafeMath or Solidity 0.8+ for overflow protection',
            'reentrancy': 'Use reentrancy guard or checks-effects-interactions pattern',
            'unchecked_call': 'Check return value and handle failures appropriately',
            'suicide': 'Add access control to selfdestruct function',
            'delegatecall': 'Validate target contract or avoid delegatecall'
        }
        return recommendations.get(vuln_type, 'Review and fix the identified issue')

# Global instances
native_slither = NativeSlitherAnalyzer()
native_mythril = NativeMythrilAnalyzer()

def analyze_with_native_slither(contract_code: str) -> Dict[str, Any]:
    """Analyze contract with native Slither implementation."""
    return native_slither.analyze_contract(contract_code)

def analyze_with_native_mythril(contract_code: str) -> Dict[str, Any]:
    """Analyze contract with native Mythril implementation."""
    return native_mythril.analyze_contract(contract_code)

def check_native_tools_availability() -> Dict[str, bool]:
    """Check availability of native tools."""
    return {
        'native_slither': True,
        'native_mythril': True,
        'docker': False  # Not using Docker
    }