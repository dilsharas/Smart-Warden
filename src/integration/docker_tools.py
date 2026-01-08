"""
Docker-based external tool integration for Smart Warden.
Provides Slither and Mythril integration using Docker containers.
"""

import subprocess
import json
import tempfile
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

class DockerToolRunner:
    """Base class for Docker-based security tool integration."""
    
    def __init__(self):
        self.docker_available = self._check_docker()
    
    def _check_docker(self) -> bool:
        """Check if Docker is available."""
        try:
            result = subprocess.run(['docker', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                logger.info("Docker is available")
                return True
            else:
                logger.warning("Docker is not available")
                return False
        except (FileNotFoundError, subprocess.TimeoutExpired):
            logger.warning("Docker is not installed or not accessible")
            return False
    
    def _run_docker_command(self, image: str, command: List[str], 
                           contract_content: str, timeout: int = 60) -> Dict[str, Any]:
        """Run a Docker command with contract content."""
        if not self.docker_available:
            return {'success': False, 'error': 'Docker not available'}
        
        try:
            # Create temporary file for contract
            with tempfile.NamedTemporaryFile(mode='w', suffix='.sol', delete=False) as f:
                f.write(contract_content)
                temp_file = f.name
            
            # Prepare Docker command
            docker_cmd = [
                'docker', 'run', '--rm',
                '-v', f'{os.path.dirname(temp_file)}:/workspace',
                image
            ] + command + [f'/workspace/{os.path.basename(temp_file)}']
            
            # Run Docker command
            result = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            # Clean up temporary file
            os.unlink(temp_file)
            
            return {
                'success': result.returncode == 0,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'returncode': result.returncode
            }
            
        except subprocess.TimeoutExpired:
            logger.error(f"Docker command timed out after {timeout} seconds")
            return {'success': False, 'error': 'Timeout'}
        except Exception as e:
            logger.error(f"Docker command failed: {e}")
            return {'success': False, 'error': str(e)}

class DockerSlitherRunner(DockerToolRunner):
    """Docker-based Slither integration."""
    
    def __init__(self):
        super().__init__()
        self.image = "trailofbits/slither"
        self._pull_image()
    
    def _pull_image(self):
        """Pull Slither Docker image if not available."""
        if not self.docker_available:
            return
        
        try:
            logger.info("Checking Slither Docker image...")
            result = subprocess.run(
                ['docker', 'images', self.image, '--format', '{{.Repository}}'],
                capture_output=True, text=True, timeout=10
            )
            
            if self.image not in result.stdout:
                logger.info("Pulling Slither Docker image...")
                subprocess.run(['docker', 'pull', self.image], timeout=300)
                logger.info("Slither image pulled successfully")
            else:
                logger.info("Slither image already available")
                
        except Exception as e:
            logger.error(f"Failed to pull Slither image: {e}")
    
    def analyze_contract(self, contract_code: str) -> Dict[str, Any]:
        """Analyze contract using Slither."""
        if not self.docker_available:
            return {
                'tool': 'slither',
                'available': False,
                'error': 'Docker not available',
                'vulnerabilities': []
            }
        
        logger.info("Running Slither analysis via Docker...")
        
        # Run Slither with JSON output
        result = self._run_docker_command(
            self.image,
            ['--json', '-'],
            contract_code,
            timeout=120
        )
        
        if not result['success']:
            return {
                'tool': 'slither',
                'available': False,
                'error': result.get('error', 'Analysis failed'),
                'vulnerabilities': []
            }
        
        # Parse Slither JSON output
        vulnerabilities = self._parse_slither_output(result['stdout'])
        
        return {
            'tool': 'slither',
            'available': True,
            'vulnerabilities': vulnerabilities,
            'raw_output': result['stdout']
        }
    
    def _parse_slither_output(self, output: str) -> List[Dict[str, Any]]:
        """Parse Slither JSON output."""
        vulnerabilities = []
        
        try:
            if not output.strip():
                return vulnerabilities
            
            # Try to parse JSON output
            data = json.loads(output)
            
            if 'results' in data and 'detectors' in data['results']:
                for detector in data['results']['detectors']:
                    vuln = {
                        'type': detector.get('check', 'unknown'),
                        'severity': detector.get('impact', 'unknown').lower(),
                        'confidence': detector.get('confidence', 'unknown').lower(),
                        'description': detector.get('description', ''),
                        'line_number': self._extract_line_number(detector),
                        'recommendation': f"Review {detector.get('check', 'issue')} pattern"
                    }
                    vulnerabilities.append(vuln)
            
        except json.JSONDecodeError:
            # If JSON parsing fails, try to extract basic info
            logger.warning("Failed to parse Slither JSON output, using fallback")
            if 'INFO:Detectors:' in output:
                vulnerabilities.append({
                    'type': 'slither_detection',
                    'severity': 'medium',
                    'confidence': 'medium',
                    'description': 'Slither detected potential issues',
                    'line_number': 0,
                    'recommendation': 'Review Slither output for details'
                })
        
        return vulnerabilities
    
    def _extract_line_number(self, detector: Dict) -> int:
        """Extract line number from Slither detector result."""
        try:
            if 'elements' in detector and detector['elements']:
                element = detector['elements'][0]
                if 'source_mapping' in element:
                    return element['source_mapping'].get('lines', [0])[0]
        except (KeyError, IndexError):
            pass
        return 0

class DockerMythrilRunner(DockerToolRunner):
    """Docker-based Mythril integration."""
    
    def __init__(self):
        super().__init__()
        self.image = "mythril/myth"
        self._pull_image()
    
    def _pull_image(self):
        """Pull Mythril Docker image if not available."""
        if not self.docker_available:
            return
        
        try:
            logger.info("Checking Mythril Docker image...")
            result = subprocess.run(
                ['docker', 'images', self.image, '--format', '{{.Repository}}'],
                capture_output=True, text=True, timeout=10
            )
            
            if self.image not in result.stdout:
                logger.info("Pulling Mythril Docker image...")
                subprocess.run(['docker', 'pull', self.image], timeout=300)
                logger.info("Mythril image pulled successfully")
            else:
                logger.info("Mythril image already available")
                
        except Exception as e:
            logger.error(f"Failed to pull Mythril image: {e}")
    
    def analyze_contract(self, contract_code: str) -> Dict[str, Any]:
        """Analyze contract using Mythril."""
        if not self.docker_available:
            return {
                'tool': 'mythril',
                'available': False,
                'error': 'Docker not available',
                'vulnerabilities': []
            }
        
        logger.info("Running Mythril analysis via Docker...")
        
        # Run Mythril analysis
        result = self._run_docker_command(
            self.image,
            ['analyze', '--solv', '0.8.0'],
            contract_code,
            timeout=180
        )
        
        if not result['success']:
            return {
                'tool': 'mythril',
                'available': False,
                'error': result.get('error', 'Analysis failed'),
                'vulnerabilities': []
            }
        
        # Parse Mythril output
        vulnerabilities = self._parse_mythril_output(result['stdout'])
        
        return {
            'tool': 'mythril',
            'available': True,
            'vulnerabilities': vulnerabilities,
            'raw_output': result['stdout']
        }
    
    def _parse_mythril_output(self, output: str) -> List[Dict[str, Any]]:
        """Parse Mythril output."""
        vulnerabilities = []
        
        try:
            # Mythril output parsing (simplified)
            lines = output.split('\n')
            current_vuln = {}
            
            for line in lines:
                line = line.strip()
                
                if line.startswith('==== '):
                    # New vulnerability found
                    if current_vuln:
                        vulnerabilities.append(current_vuln)
                    
                    vuln_type = line.replace('====', '').strip()
                    current_vuln = {
                        'type': vuln_type.lower().replace(' ', '_'),
                        'severity': 'medium',
                        'confidence': 'medium',
                        'description': vuln_type,
                        'line_number': 0,
                        'recommendation': f'Review {vuln_type} vulnerability'
                    }
                
                elif line.startswith('SWC ID:'):
                    if current_vuln:
                        current_vuln['swc_id'] = line.replace('SWC ID:', '').strip()
                
                elif line.startswith('Severity:'):
                    if current_vuln:
                        severity = line.replace('Severity:', '').strip().lower()
                        current_vuln['severity'] = severity
            
            # Add last vulnerability
            if current_vuln:
                vulnerabilities.append(current_vuln)
        
        except Exception as e:
            logger.error(f"Error parsing Mythril output: {e}")
            # Fallback: if analysis ran but parsing failed
            if output and 'analysis' in output.lower():
                vulnerabilities.append({
                    'type': 'mythril_detection',
                    'severity': 'medium',
                    'confidence': 'medium',
                    'description': 'Mythril completed analysis',
                    'line_number': 0,
                    'recommendation': 'Review Mythril output for details'
                })
        
        return vulnerabilities

class FallbackToolRunner:
    """Fallback tool runner when Docker is not available."""
    
    def __init__(self):
        self.available = False
    
    def analyze_contract(self, contract_code: str, tool_name: str) -> Dict[str, Any]:
        """Provide fallback analysis when external tools are not available."""
        logger.info(f"Using fallback analysis for {tool_name}")
        
        # Simple pattern-based detection as fallback
        vulnerabilities = []
        
        # Check for common patterns
        if 'call{value:' in contract_code or '.call(' in contract_code:
            vulnerabilities.append({
                'type': 'external_call',
                'severity': 'medium',
                'confidence': 'low',
                'description': 'External call detected',
                'line_number': 0,
                'recommendation': 'Review external call for reentrancy risks'
            })
        
        if 'block.timestamp' in contract_code or 'now' in contract_code:
            vulnerabilities.append({
                'type': 'bad_randomness',
                'severity': 'medium',
                'confidence': 'medium',
                'description': 'Potential weak randomness source',
                'line_number': 0,
                'recommendation': 'Avoid using block.timestamp for randomness'
            })
        
        return {
            'tool': tool_name,
            'available': False,
            'fallback': True,
            'vulnerabilities': vulnerabilities,
            'note': 'Fallback pattern-based analysis used'
        }

# Global instances
docker_slither = DockerSlitherRunner()
docker_mythril = DockerMythrilRunner()
fallback_runner = FallbackToolRunner()

def analyze_with_slither(contract_code: str) -> Dict[str, Any]:
    """Analyze contract with Slither (Docker, native, or fallback)."""
    # Try Docker first
    if docker_slither.docker_available:
        try:
            result = docker_slither.analyze_contract(contract_code)
            if result.get('available'):
                return result
        except Exception as e:
            logger.warning(f"Docker Slither failed: {e}")
    
    # Fall back to native implementation
    try:
        from .native_tools import analyze_with_native_slither
        return analyze_with_native_slither(contract_code)
    except Exception as e:
        logger.warning(f"Native Slither failed: {e}")
        return fallback_runner.analyze_contract(contract_code, 'slither')

def analyze_with_mythril(contract_code: str) -> Dict[str, Any]:
    """Analyze contract with Mythril (Docker, native, or fallback)."""
    # Try Docker first
    if docker_mythril.docker_available:
        try:
            result = docker_mythril.analyze_contract(contract_code)
            if result.get('available'):
                return result
        except Exception as e:
            logger.warning(f"Docker Mythril failed: {e}")
    
    # Fall back to native implementation
    try:
        from .native_tools import analyze_with_native_mythril
        return analyze_with_native_mythril(contract_code)
    except Exception as e:
        logger.warning(f"Native Mythril failed: {e}")
        return fallback_runner.analyze_contract(contract_code, 'mythril')

def check_tools_availability() -> Dict[str, bool]:
    """Check availability of external tools."""
    return {
        'docker': docker_slither.docker_available,
        'slither': docker_slither.docker_available,
        'mythril': docker_mythril.docker_available
    }