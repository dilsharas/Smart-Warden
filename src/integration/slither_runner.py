"""
Slither static analysis tool integration for smart contract security analysis.
"""

import subprocess
import json
import tempfile
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging
from dataclasses import dataclass
from datetime import datetime
import time
import shutil

logger = logging.getLogger(__name__)


@dataclass
class SlitherFinding:
    """Represents a vulnerability finding from Slither analysis."""
    vulnerability_type: str
    severity: str
    confidence: str
    description: str
    line_number: int
    function_name: str
    contract_name: str
    code_snippet: str
    recommendation: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert finding to dictionary."""
        return {
            'vulnerability_type': self.vulnerability_type,
            'severity': self.severity,
            'confidence': self.confidence,
            'description': self.description,
            'line_number': self.line_number,
            'function_name': self.function_name,
            'contract_name': self.contract_name,
            'code_snippet': self.code_snippet,
            'recommendation': self.recommendation
        }


@dataclass
class SlitherResult:
    """Represents the complete result of Slither analysis."""
    success: bool
    execution_time: float
    findings: List[SlitherFinding]
    error_message: Optional[str]
    raw_output: str
    contract_path: str
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'success': self.success,
            'execution_time': self.execution_time,
            'findings': [finding.to_dict() for finding in self.findings],
            'error_message': self.error_message,
            'raw_output': self.raw_output,
            'contract_path': self.contract_path,
            'timestamp': self.timestamp.isoformat(),
            'total_findings': len(self.findings),
            'severity_distribution': self._get_severity_distribution()
        }
    
    def _get_severity_distribution(self) -> Dict[str, int]:
        """Get distribution of findings by severity."""
        distribution = {'High': 0, 'Medium': 0, 'Low': 0, 'Informational': 0}
        for finding in self.findings:
            severity = finding.severity
            if severity in distribution:
                distribution[severity] += 1
        return distribution


class SlitherAnalyzer:
    """
    Integrates with Slither static analysis tool for smart contract security analysis.
    
    Features:
    - Execute Slither analysis on Solidity contracts
    - Parse JSON output and normalize results
    - Handle execution timeouts and errors
    - Map Slither detectors to vulnerability types
    - Provide detailed vulnerability findings
    """
    
    # Mapping of Slither detector names to vulnerability types
    DETECTOR_MAPPING = {
        'reentrancy-eth': 'reentrancy',
        'reentrancy-no-eth': 'reentrancy',
        'reentrancy-benign': 'reentrancy',
        'reentrancy-events': 'reentrancy',
        'arbitrary-send': 'access_control',
        'suicidal': 'access_control',
        'unprotected-upgrade': 'access_control',
        'missing-zero-check': 'access_control',
        'tx-origin': 'access_control',
        'integer-overflow': 'arithmetic',
        'divide-before-multiply': 'arithmetic',
        'weak-prng': 'bad_randomness',
        'timestamp': 'bad_randomness',
        'block-timestamp': 'bad_randomness',
        'unchecked-lowlevel': 'unchecked_calls',
        'unchecked-send': 'unchecked_calls',
        'unused-return': 'unchecked_calls',
        'calls-loop': 'denial_of_service',
        'costly-loop': 'denial_of_service',
        'array-by-reference': 'denial_of_service'
    }
    
    # Severity mapping from Slither impact levels
    SEVERITY_MAPPING = {
        'High': 'Critical',
        'Medium': 'High', 
        'Low': 'Medium',
        'Informational': 'Low'
    }
    
    def __init__(self, 
                 slither_path: str = 'slither',
                 timeout: int = 60,
                 solc_version: Optional[str] = None):
        """
        Initialize the Slither analyzer.
        
        Args:
            slither_path: Path to Slither executable
            timeout: Maximum execution time in seconds
            solc_version: Specific Solidity compiler version to use
        """
        self.slither_path = slither_path
        self.timeout = timeout
        self.solc_version = solc_version
        self.is_available = self._check_availability()
        
    def _check_availability(self) -> bool:
        """
        Check if Slither is available and working.
        
        Returns:
            True if Slither is available, False otherwise
        """
        try:
            result = subprocess.run(
                [self.slither_path, '--version'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                logger.info(f"Slither is available: {result.stdout.strip()}")
                return True
            else:
                logger.warning(f"Slither check failed: {result.stderr}")
                return False
                
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError) as e:
            logger.warning(f"Slither not available: {e}")
            return False
    
    def analyze_contract(self, contract_code: str, filename: str = "contract.sol") -> SlitherResult:
        """
        Analyze a smart contract using Slither.
        
        Args:
            contract_code: Solidity source code
            filename: Name for the temporary contract file
            
        Returns:
            SlitherResult object with analysis results
        """
        if not self.is_available:
            return SlitherResult(
                success=False,
                execution_time=0.0,
                findings=[],
                error_message="Slither is not available",
                raw_output="",
                contract_path="",
                timestamp=datetime.now()
            )
        
        start_time = time.time()
        
        # Create temporary file for the contract
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sol', delete=False) as temp_file:
            temp_file.write(contract_code)
            temp_path = temp_file.name
        
        try:
            # Run Slither analysis
            result = self._run_slither(temp_path)
            execution_time = time.time() - start_time
            
            if result['success']:
                # Parse the JSON output
                findings = self._parse_slither_output(result['output'], contract_code)
                
                return SlitherResult(
                    success=True,
                    execution_time=execution_time,
                    findings=findings,
                    error_message=None,
                    raw_output=result['output'],
                    contract_path=temp_path,
                    timestamp=datetime.now()
                )
            else:
                return SlitherResult(
                    success=False,
                    execution_time=execution_time,
                    findings=[],
                    error_message=result['error'],
                    raw_output=result['output'],
                    contract_path=temp_path,
                    timestamp=datetime.now()
                )
                
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_path)
            except OSError:
                pass
    
    def _run_slither(self, contract_path: str) -> Dict[str, Any]:
        """
        Execute Slither on the contract file.
        
        Args:
            contract_path: Path to the contract file
            
        Returns:
            Dictionary with execution results
        """
        # Build Slither command
        cmd = [self.slither_path, contract_path, '--json', '-']
        
        # Add Solidity version if specified
        if self.solc_version:
            cmd.extend(['--solc', self.solc_version])
        
        # Add common flags for better analysis
        cmd.extend([
            '--disable-color',
            '--exclude-informational',
            '--exclude-low',
            '--exclude-optimization'
        ])
        
        try:
            logger.info(f"Running Slither: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=os.path.dirname(contract_path)
            )
            
            # Slither returns non-zero exit code when vulnerabilities are found
            # So we consider it successful if it's not a critical error
            if result.returncode in [0, 1, 2]:  # 0=no issues, 1=issues found, 2=warnings
                return {
                    'success': True,
                    'output': result.stdout,
                    'error': result.stderr
                }
            else:
                return {
                    'success': False,
                    'output': result.stdout,
                    'error': result.stderr
                }
                
        except subprocess.TimeoutExpired:
            logger.error(f"Slither analysis timed out after {self.timeout} seconds")
            return {
                'success': False,
                'output': "",
                'error': f"Analysis timed out after {self.timeout} seconds"
            }
        except Exception as e:
            logger.error(f"Error running Slither: {e}")
            return {
                'success': False,
                'output': "",
                'error': str(e)
            }
    
    def _parse_slither_output(self, json_output: str, contract_code: str) -> List[SlitherFinding]:
        """
        Parse Slither JSON output and convert to normalized findings.
        
        Args:
            json_output: Raw JSON output from Slither
            contract_code: Original contract source code
            
        Returns:
            List of SlitherFinding objects
        """
        findings = []
        
        if not json_output.strip():
            return findings
        
        try:
            # Parse JSON output
            data = json.loads(json_output)
            
            # Extract results from the JSON structure
            results = data.get('results', {})
            detectors = results.get('detectors', [])
            
            for detector in detectors:
                finding = self._convert_detector_to_finding(detector, contract_code)
                if finding:
                    findings.append(finding)
                    
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Slither JSON output: {e}")
            logger.debug(f"Raw output: {json_output}")
        except Exception as e:
            logger.error(f"Error processing Slither output: {e}")
        
        return findings
    
    def _convert_detector_to_finding(self, detector: Dict[str, Any], contract_code: str) -> Optional[SlitherFinding]:
        """
        Convert a Slither detector result to a SlitherFinding.
        
        Args:
            detector: Detector result from Slither JSON
            contract_code: Original contract source code
            
        Returns:
            SlitherFinding object or None if conversion fails
        """
        try:
            # Extract basic information
            check = detector.get('check', 'unknown')
            impact = detector.get('impact', 'Low')
            confidence = detector.get('confidence', 'Low')
            description = detector.get('description', 'No description available')
            
            # Map detector to vulnerability type
            vulnerability_type = self.DETECTOR_MAPPING.get(check, 'other')
            
            # Map severity
            severity = self.SEVERITY_MAPPING.get(impact, 'Low')
            
            # Extract location information
            elements = detector.get('elements', [])
            line_number = 1
            function_name = 'unknown'
            contract_name = 'unknown'
            code_snippet = ''
            
            if elements:
                first_element = elements[0]
                
                # Extract source mapping if available
                source_mapping = first_element.get('source_mapping', {})
                if source_mapping:
                    line_number = source_mapping.get('lines', [1])[0]
                    
                    # Extract code snippet
                    start = source_mapping.get('start', 0)
                    length = source_mapping.get('length', 0)
                    if start >= 0 and length > 0 and start + length <= len(contract_code):
                        code_snippet = contract_code[start:start + length]
                
                # Extract function and contract names
                if first_element.get('type') == 'function':
                    function_name = first_element.get('name', 'unknown')
                    contract_name = first_element.get('type_specific_fields', {}).get('parent', {}).get('name', 'unknown')
                elif first_element.get('type') == 'contract':
                    contract_name = first_element.get('name', 'unknown')
            
            # Generate recommendation based on vulnerability type
            recommendation = self._generate_recommendation(vulnerability_type, check)
            
            return SlitherFinding(
                vulnerability_type=vulnerability_type,
                severity=severity,
                confidence=confidence,
                description=description,
                line_number=line_number,
                function_name=function_name,
                contract_name=contract_name,
                code_snippet=code_snippet.strip(),
                recommendation=recommendation
            )
            
        except Exception as e:
            logger.error(f"Error converting detector to finding: {e}")
            return None
    
    def _generate_recommendation(self, vulnerability_type: str, detector_check: str) -> str:
        """
        Generate security recommendations based on vulnerability type.
        
        Args:
            vulnerability_type: Type of vulnerability
            detector_check: Specific Slither detector that found the issue
            
        Returns:
            Recommendation string
        """
        recommendations = {
            'reentrancy': 'Use the checks-effects-interactions pattern. Move external calls to the end of functions after state changes. Consider using reentrancy guards.',
            'access_control': 'Implement proper access control mechanisms. Use modifiers like onlyOwner and validate msg.sender. Avoid using tx.origin for authorization.',
            'arithmetic': 'Use SafeMath library for arithmetic operations (Solidity < 0.8.0) or upgrade to Solidity 0.8+ which has built-in overflow protection.',
            'bad_randomness': 'Do not use block.timestamp, block.number, or blockhash for randomness. Use a secure random number generator or oracle service.',
            'unchecked_calls': 'Always check the return value of external calls. Use require() statements or handle failures appropriately.',
            'denial_of_service': 'Avoid unbounded loops and expensive operations. Implement gas limits and consider using pull-over-push patterns.',
            'other': 'Review the code carefully and follow Solidity security best practices.'
        }
        
        return recommendations.get(vulnerability_type, recommendations['other'])
    
    def analyze_file(self, file_path: str) -> SlitherResult:
        """
        Analyze a contract file directly.
        
        Args:
            file_path: Path to the Solidity file
            
        Returns:
            SlitherResult object with analysis results
        """
        if not os.path.exists(file_path):
            return SlitherResult(
                success=False,
                execution_time=0.0,
                findings=[],
                error_message=f"File not found: {file_path}",
                raw_output="",
                contract_path=file_path,
                timestamp=datetime.now()
            )
        
        # Read the contract code
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                contract_code = f.read()
        except Exception as e:
            return SlitherResult(
                success=False,
                execution_time=0.0,
                findings=[],
                error_message=f"Error reading file: {e}",
                raw_output="",
                contract_path=file_path,
                timestamp=datetime.now()
            )
        
        # Use the existing analyze_contract method
        result = self.analyze_contract(contract_code, os.path.basename(file_path))
        result.contract_path = file_path
        
        return result
    
    def get_supported_detectors(self) -> List[str]:
        """
        Get list of supported Slither detectors.
        
        Returns:
            List of detector names
        """
        return list(self.DETECTOR_MAPPING.keys())
    
    def get_vulnerability_types(self) -> List[str]:
        """
        Get list of vulnerability types that can be detected.
        
        Returns:
            List of vulnerability type names
        """
        return list(set(self.DETECTOR_MAPPING.values()))
    
    def is_available(self) -> bool:
        """
        Check if Slither is available for analysis.
        
        Returns:
            True if available, False otherwise
        """
        return self.is_available
    
    def get_version(self) -> Optional[str]:
        """
        Get Slither version information.
        
        Returns:
            Version string or None if not available
        """
        if not self.is_available:
            return None
        
        try:
            result = subprocess.run(
                [self.slither_path, '--version'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                return None
                
        except Exception:
            return None


def main():
    """Example usage of SlitherAnalyzer."""
    # Sample vulnerable contract
    vulnerable_contract = """
    pragma solidity ^0.8.0;
    
    contract VulnerableContract {
        mapping(address => uint256) public balances;
        
        function withdraw(uint256 amount) public {
            require(balances[msg.sender] >= amount, "Insufficient balance");
            
            // Vulnerable to reentrancy - external call before state change
            (bool success, ) = msg.sender.call{value: amount}("");
            require(success, "Transfer failed");
            
            balances[msg.sender] -= amount;  // State change after external call
        }
        
        function deposit() public payable {
            balances[msg.sender] += msg.value;
        }
        
        // Vulnerable to timestamp manipulation
        function randomNumber() public view returns (uint256) {
            return uint256(keccak256(abi.encodePacked(block.timestamp))) % 100;
        }
        
        // Missing access control
        function emergencyWithdraw() public {
            payable(msg.sender).transfer(address(this).balance);
        }
    }
    """
    
    # Initialize Slither analyzer
    analyzer = SlitherAnalyzer()
    
    if not analyzer.is_available():
        print("Slither is not available. Please install it first:")
        print("pip install slither-analyzer")
        return
    
    print(f"Slither version: {analyzer.get_version()}")
    print(f"Supported detectors: {len(analyzer.get_supported_detectors())}")
    print(f"Vulnerability types: {analyzer.get_vulnerability_types()}")
    
    # Analyze the contract
    print("\nAnalyzing vulnerable contract...")
    result = analyzer.analyze_contract(vulnerable_contract)
    
    print(f"Analysis completed in {result.execution_time:.2f} seconds")
    print(f"Success: {result.success}")
    
    if result.success:
        print(f"Found {len(result.findings)} vulnerabilities:")
        
        for i, finding in enumerate(result.findings, 1):
            print(f"\n{i}. {finding.vulnerability_type.upper()}")
            print(f"   Severity: {finding.severity}")
            print(f"   Confidence: {finding.confidence}")
            print(f"   Line: {finding.line_number}")
            print(f"   Function: {finding.function_name}")
            print(f"   Description: {finding.description}")
            print(f"   Recommendation: {finding.recommendation}")
            
            if finding.code_snippet:
                print(f"   Code: {finding.code_snippet[:100]}...")
        
        # Show severity distribution
        severity_dist = result._get_severity_distribution()
        print(f"\nSeverity Distribution: {severity_dist}")
        
    else:
        print(f"Analysis failed: {result.error_message}")


if __name__ == "__main__":
    main()