"""
Mythril security analysis tool integration for smart contract security analysis.
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
import re

logger = logging.getLogger(__name__)


@dataclass
class MythrilFinding:
    """Represents a vulnerability finding from Mythril analysis."""
    vulnerability_type: str
    severity: str
    confidence: str
    description: str
    line_number: int
    function_name: str
    contract_name: str
    code_snippet: str
    recommendation: str
    swc_id: Optional[str] = None
    
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
            'recommendation': self.recommendation,
            'swc_id': self.swc_id
        }


@dataclass
class MythrilResult:
    """Represents the complete result of Mythril analysis."""
    success: bool
    execution_time: float
    findings: List[MythrilFinding]
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
        distribution = {'High': 0, 'Medium': 0, 'Low': 0}
        for finding in self.findings:
            severity = finding.severity
            if severity in distribution:
                distribution[severity] += 1
        return distribution


class MythrilAnalyzer:
    """
    Integrates with Mythril security analysis tool for smart contract analysis.
    
    Features:
    - Execute Mythril analysis on Solidity contracts
    - Parse JSON output and normalize results
    - Handle execution timeouts and errors
    - Map Mythril issue types to vulnerability categories
    - Provide detailed vulnerability findings with SWC mappings
    """
    
    # Mapping of Mythril issue types to vulnerability categories
    ISSUE_TYPE_MAPPING = {
        'Reentrancy': 'reentrancy',
        'State change after external call': 'reentrancy',
        'External call to user-supplied address': 'reentrancy',
        'Unprotected Ether Withdrawal': 'access_control',
        'Unprotected SELFDESTRUCT': 'access_control',
        'Authorization through tx.origin': 'access_control',
        'Missing or insufficient access controls': 'access_control',
        'Integer Overflow': 'arithmetic',
        'Integer Underflow': 'arithmetic',
        'Division by zero': 'arithmetic',
        'Weak Sources of Randomness from Chain Attributes': 'bad_randomness',
        'Dependence on predictable environment variable': 'bad_randomness',
        'Use of block timestamp': 'bad_randomness',
        'Unchecked return value from external call': 'unchecked_calls',
        'Exception disorder': 'unchecked_calls',
        'DoS with block gas limit': 'denial_of_service',
        'DoS with (unexpected) revert': 'denial_of_service',
        'Costly operations in a loop': 'denial_of_service'
    }
    
    # SWC ID to vulnerability type mapping
    SWC_MAPPING = {
        'SWC-107': 'reentrancy',
        'SWC-105': 'access_control',
        'SWC-106': 'access_control',
        'SWC-115': 'access_control',
        'SWC-101': 'arithmetic',
        'SWC-104': 'unchecked_calls',
        'SWC-120': 'bad_randomness',
        'SWC-116': 'bad_randomness',
        'SWC-128': 'denial_of_service'
    }
    
    # Severity mapping
    SEVERITY_MAPPING = {
        'High': 'Critical',
        'Medium': 'High',
        'Low': 'Medium'
    }
    
    def __init__(self, 
                 mythril_path: str = 'myth',
                 timeout: int = 120,
                 max_depth: int = 22,
                 execution_timeout: int = 86400):
        """
        Initialize the Mythril analyzer.
        
        Args:
            mythril_path: Path to Mythril executable
            timeout: Maximum execution time in seconds
            max_depth: Maximum symbolic execution depth
            execution_timeout: Maximum execution timeout for Mythril
        """
        self.mythril_path = mythril_path
        self.timeout = timeout
        self.max_depth = max_depth
        self.execution_timeout = execution_timeout
        self.is_available = self._check_availability()
        
    def _check_availability(self) -> bool:
        """
        Check if Mythril is available and working.
        
        Returns:
            True if Mythril is available, False otherwise
        """
        try:
            result = subprocess.run(
                [self.mythril_path, 'version'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                logger.info(f"Mythril is available: {result.stdout.strip()}")
                return True
            else:
                logger.warning(f"Mythril check failed: {result.stderr}")
                return False
                
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError) as e:
            logger.warning(f"Mythril not available: {e}")
            return False
    
    def analyze_contract(self, contract_code: str, filename: str = "contract.sol") -> MythrilResult:
        """
        Analyze a smart contract using Mythril.
        
        Args:
            contract_code: Solidity source code
            filename: Name for the temporary contract file
            
        Returns:
            MythrilResult object with analysis results
        """
        if not self.is_available:
            return MythrilResult(
                success=False,
                execution_time=0.0,
                findings=[],
                error_message="Mythril is not available",
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
            # Run Mythril analysis
            result = self._run_mythril(temp_path)
            execution_time = time.time() - start_time
            
            if result['success']:
                # Parse the JSON output
                findings = self._parse_mythril_output(result['output'], contract_code)
                
                return MythrilResult(
                    success=True,
                    execution_time=execution_time,
                    findings=findings,
                    error_message=None,
                    raw_output=result['output'],
                    contract_path=temp_path,
                    timestamp=datetime.now()
                )
            else:
                return MythrilResult(
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
    
    def _run_mythril(self, contract_path: str) -> Dict[str, Any]:
        """
        Execute Mythril on the contract file.
        
        Args:
            contract_path: Path to the contract file
            
        Returns:
            Dictionary with execution results
        """
        # Build Mythril command
        cmd = [
            self.mythril_path, 'analyze',
            contract_path,
            '--output', 'json',
            '--max-depth', str(self.max_depth),
            '--execution-timeout', str(self.execution_timeout)
        ]
        
        try:
            logger.info(f"Running Mythril: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=os.path.dirname(contract_path)
            )
            
            # Mythril returns different exit codes:
            # 0: No issues found
            # 1: Issues found
            # 2: Error occurred
            if result.returncode in [0, 1]:
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
            logger.error(f"Mythril analysis timed out after {self.timeout} seconds")
            return {
                'success': False,
                'output': "",
                'error': f"Analysis timed out after {self.timeout} seconds"
            }
        except Exception as e:
            logger.error(f"Error running Mythril: {e}")
            return {
                'success': False,
                'output': "",
                'error': str(e)
            }
    
    def _parse_mythril_output(self, json_output: str, contract_code: str) -> List[MythrilFinding]:
        """
        Parse Mythril JSON output and convert to normalized findings.
        
        Args:
            json_output: Raw JSON output from Mythril
            contract_code: Original contract source code
            
        Returns:
            List of MythrilFinding objects
        """
        findings = []
        
        if not json_output.strip():
            return findings
        
        try:
            # Parse JSON output
            data = json.loads(json_output)
            
            # Mythril output structure can vary, handle different formats
            issues = []
            if isinstance(data, dict):
                if 'issues' in data:
                    issues = data['issues']
                elif 'result' in data and isinstance(data['result'], dict):
                    issues = data['result'].get('issues', [])
            elif isinstance(data, list):
                issues = data
            
            for issue in issues:
                finding = self._convert_issue_to_finding(issue, contract_code)
                if finding:
                    findings.append(finding)
                    
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Mythril JSON output: {e}")
            logger.debug(f"Raw output: {json_output}")
            
            # Try to parse text output if JSON parsing fails
            findings = self._parse_text_output(json_output, contract_code)
            
        except Exception as e:
            logger.error(f"Error processing Mythril output: {e}")
        
        return findings
    
    def _convert_issue_to_finding(self, issue: Dict[str, Any], contract_code: str) -> Optional[MythrilFinding]:
        """
        Convert a Mythril issue to a MythrilFinding.
        
        Args:
            issue: Issue from Mythril JSON output
            contract_code: Original contract source code
            
        Returns:
            MythrilFinding object or None if conversion fails
        """
        try:
            # Extract basic information
            title = issue.get('title', 'Unknown Issue')
            description = issue.get('description', 'No description available')
            severity = issue.get('severity', 'Medium')
            swc_id = issue.get('swc-id', None)
            
            # Map issue type to vulnerability category
            vulnerability_type = self._map_issue_to_vulnerability_type(title, swc_id)
            
            # Map severity
            mapped_severity = self.SEVERITY_MAPPING.get(severity, severity)
            
            # Extract location information
            locations = issue.get('locations', [])
            line_number = 1
            function_name = 'unknown'
            contract_name = 'unknown'
            code_snippet = ''
            
            if locations:
                first_location = locations[0]
                source_map = first_location.get('sourceMap', '')
                
                # Parse source map to get line number
                if source_map:
                    line_number = self._parse_source_map_line(source_map, contract_code)
                
                # Extract code snippet from source map
                code_snippet = self._extract_code_snippet(source_map, contract_code)
            
            # Extract function and contract names from description or code
            function_name, contract_name = self._extract_names_from_description(description)
            
            # Generate recommendation
            recommendation = self._generate_recommendation(vulnerability_type, title)
            
            # Set confidence based on Mythril's analysis depth
            confidence = 'High'  # Mythril generally provides high-confidence results
            
            return MythrilFinding(
                vulnerability_type=vulnerability_type,
                severity=mapped_severity,
                confidence=confidence,
                description=description,
                line_number=line_number,
                function_name=function_name,
                contract_name=contract_name,
                code_snippet=code_snippet.strip(),
                recommendation=recommendation,
                swc_id=swc_id
            )
            
        except Exception as e:
            logger.error(f"Error converting issue to finding: {e}")
            return None
    
    def _parse_text_output(self, text_output: str, contract_code: str) -> List[MythrilFinding]:
        """
        Parse Mythril text output when JSON parsing fails.
        
        Args:
            text_output: Raw text output from Mythril
            contract_code: Original contract source code
            
        Returns:
            List of MythrilFinding objects
        """
        findings = []
        
        # Split output into sections
        sections = re.split(r'={3,}', text_output)
        
        for section in sections:
            if 'Title:' in section or 'Issue:' in section:
                finding = self._parse_text_section(section, contract_code)
                if finding:
                    findings.append(finding)
        
        return findings
    
    def _parse_text_section(self, section: str, contract_code: str) -> Optional[MythrilFinding]:
        """
        Parse a single text section from Mythril output.
        
        Args:
            section: Text section containing issue information
            contract_code: Original contract source code
            
        Returns:
            MythrilFinding object or None
        """
        try:
            lines = section.strip().split('\n')
            
            title = 'Unknown Issue'
            description = 'No description available'
            severity = 'Medium'
            swc_id = None
            line_number = 1
            
            for line in lines:
                line = line.strip()
                if line.startswith('Title:') or line.startswith('Issue:'):
                    title = line.split(':', 1)[1].strip()
                elif line.startswith('Description:'):
                    description = line.split(':', 1)[1].strip()
                elif line.startswith('Severity:'):
                    severity = line.split(':', 1)[1].strip()
                elif line.startswith('SWC ID:'):
                    swc_id = line.split(':', 1)[1].strip()
                elif 'line' in line.lower() and any(char.isdigit() for char in line):
                    # Try to extract line number
                    numbers = re.findall(r'\d+', line)
                    if numbers:
                        line_number = int(numbers[0])
            
            # Map to vulnerability type
            vulnerability_type = self._map_issue_to_vulnerability_type(title, swc_id)
            
            # Generate other fields
            recommendation = self._generate_recommendation(vulnerability_type, title)
            
            return MythrilFinding(
                vulnerability_type=vulnerability_type,
                severity=self.SEVERITY_MAPPING.get(severity, severity),
                confidence='High',
                description=description,
                line_number=line_number,
                function_name='unknown',
                contract_name='unknown',
                code_snippet='',
                recommendation=recommendation,
                swc_id=swc_id
            )
            
        except Exception as e:
            logger.error(f"Error parsing text section: {e}")
            return None
    
    def _map_issue_to_vulnerability_type(self, title: str, swc_id: Optional[str]) -> str:
        """
        Map Mythril issue to vulnerability type.
        
        Args:
            title: Issue title
            swc_id: SWC identifier
            
        Returns:
            Vulnerability type string
        """
        # First try SWC mapping
        if swc_id and swc_id in self.SWC_MAPPING:
            return self.SWC_MAPPING[swc_id]
        
        # Then try title mapping
        for pattern, vuln_type in self.ISSUE_TYPE_MAPPING.items():
            if pattern.lower() in title.lower():
                return vuln_type
        
        # Default to 'other' if no mapping found
        return 'other'
    
    def _parse_source_map_line(self, source_map: str, contract_code: str) -> int:
        """
        Parse source map to extract line number.
        
        Args:
            source_map: Source map string
            contract_code: Original contract code
            
        Returns:
            Line number
        """
        try:
            # Source map format: start:length:file_index
            parts = source_map.split(':')
            if len(parts) >= 2:
                start = int(parts[0])
                # Count newlines up to start position
                line_number = contract_code[:start].count('\n') + 1
                return line_number
        except (ValueError, IndexError):
            pass
        
        return 1
    
    def _extract_code_snippet(self, source_map: str, contract_code: str) -> str:
        """
        Extract code snippet from source map.
        
        Args:
            source_map: Source map string
            contract_code: Original contract code
            
        Returns:
            Code snippet string
        """
        try:
            parts = source_map.split(':')
            if len(parts) >= 2:
                start = int(parts[0])
                length = int(parts[1])
                
                if start >= 0 and length > 0 and start + length <= len(contract_code):
                    return contract_code[start:start + length]
        except (ValueError, IndexError):
            pass
        
        return ''
    
    def _extract_names_from_description(self, description: str) -> Tuple[str, str]:
        """
        Extract function and contract names from description.
        
        Args:
            description: Issue description
            
        Returns:
            Tuple of (function_name, contract_name)
        """
        function_name = 'unknown'
        contract_name = 'unknown'
        
        # Look for function names in description
        func_match = re.search(r'function\s+(\w+)', description, re.IGNORECASE)
        if func_match:
            function_name = func_match.group(1)
        
        # Look for contract names in description
        contract_match = re.search(r'contract\s+(\w+)', description, re.IGNORECASE)
        if contract_match:
            contract_name = contract_match.group(1)
        
        return function_name, contract_name
    
    def _generate_recommendation(self, vulnerability_type: str, issue_title: str) -> str:
        """
        Generate security recommendations based on vulnerability type.
        
        Args:
            vulnerability_type: Type of vulnerability
            issue_title: Specific issue title from Mythril
            
        Returns:
            Recommendation string
        """
        recommendations = {
            'reentrancy': 'Implement the checks-effects-interactions pattern. Use reentrancy guards or mutex locks. Move external calls to the end of functions.',
            'access_control': 'Implement proper access control mechanisms. Use modifiers for authorization. Validate msg.sender and avoid tx.origin.',
            'arithmetic': 'Use SafeMath library (Solidity < 0.8.0) or upgrade to Solidity 0.8+ with built-in overflow protection. Add bounds checking.',
            'bad_randomness': 'Use secure randomness sources. Avoid block.timestamp and block.number for random values. Consider using oracles or commit-reveal schemes.',
            'unchecked_calls': 'Always check return values of external calls. Use require() statements or handle failures appropriately.',
            'denial_of_service': 'Implement gas limits for loops. Use pull-over-push patterns. Avoid unbounded operations.',
            'other': 'Review the code carefully and follow Solidity security best practices. Consider getting a professional security audit.'
        }
        
        return recommendations.get(vulnerability_type, recommendations['other'])
    
    def analyze_file(self, file_path: str) -> MythrilResult:
        """
        Analyze a contract file directly.
        
        Args:
            file_path: Path to the Solidity file
            
        Returns:
            MythrilResult object with analysis results
        """
        if not os.path.exists(file_path):
            return MythrilResult(
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
            return MythrilResult(
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
    
    def get_supported_swc_ids(self) -> List[str]:
        """
        Get list of supported SWC IDs.
        
        Returns:
            List of SWC identifiers
        """
        return list(self.SWC_MAPPING.keys())
    
    def get_vulnerability_types(self) -> List[str]:
        """
        Get list of vulnerability types that can be detected.
        
        Returns:
            List of vulnerability type names
        """
        return list(set(self.SWC_MAPPING.values()))
    
    def is_available(self) -> bool:
        """
        Check if Mythril is available for analysis.
        
        Returns:
            True if available, False otherwise
        """
        return self.is_available
    
    def get_version(self) -> Optional[str]:
        """
        Get Mythril version information.
        
        Returns:
            Version string or None if not available
        """
        if not self.is_available:
            return None
        
        try:
            result = subprocess.run(
                [self.mythril_path, 'version'],
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
    """Example usage of MythrilAnalyzer."""
    # Sample vulnerable contract
    vulnerable_contract = """
    pragma solidity ^0.8.0;
    
    contract VulnerableContract {
        mapping(address => uint256) public balances;
        address public owner;
        
        constructor() {
            owner = msg.sender;
        }
        
        function withdraw(uint256 amount) public {
            require(balances[msg.sender] >= amount, "Insufficient balance");
            
            // Vulnerable to reentrancy
            (bool success, ) = msg.sender.call{value: amount}("");
            require(success, "Transfer failed");
            
            balances[msg.sender] -= amount;
        }
        
        function deposit() public payable {
            balances[msg.sender] += msg.value;
        }
        
        // Vulnerable to bad randomness
        function lottery() public view returns (bool) {
            return block.timestamp % 2 == 0;
        }
        
        // Missing access control
        function emergencyWithdraw() public {
            payable(msg.sender).transfer(address(this).balance);
        }
        
        // Potential integer overflow (if using older Solidity)
        function unsafeAdd(uint256 a, uint256 b) public pure returns (uint256) {
            return a + b;  // Could overflow in Solidity < 0.8.0
        }
    }
    """
    
    # Initialize Mythril analyzer
    analyzer = MythrilAnalyzer()
    
    if not analyzer.is_available():
        print("Mythril is not available. Please install it first:")
        print("pip install mythril")
        return
    
    print(f"Mythril version: {analyzer.get_version()}")
    print(f"Supported SWC IDs: {analyzer.get_supported_swc_ids()}")
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
            print(f"   SWC ID: {finding.swc_id}")
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