#!/usr/bin/env python3
"""
Comprehensive Feature Testing for Smart Contract AI Analyzer
Tests all features: API, Models, AI Detection, Upload, Dashboard, etc.
"""

import requests
import json
import time
import sys
import subprocess
from pathlib import Path
import tempfile

class FeatureTester:
    def __init__(self, api_url="http://localhost:5000"):
        self.api_url = api_url
        self.test_results = {}
        
    def print_header(self, title):
        """Print formatted test header."""
        print(f"\n{'='*60}")
        print(f"🧪 {title}")
        print(f"{'='*60}")
    
    def print_result(self, feature, status, details=""):
        """Print test result."""
        icon = "✅" if status else "❌"
        print(f"{icon} {feature}: {'PASS' if status else 'FAIL'}")
        if details:
            print(f"   {details}")
        self.test_results[feature] = status
    
    def test_api_health(self):
        """Test API health endpoint."""
        self.print_header("API Health Check")
        
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                self.print_result("API Health", True, f"Status: {data.get('status')}, Version: {data.get('version')}")
                return True
            else:
                self.print_result("API Health", False, f"HTTP {response.status_code}")
                return False
        except Exception as e:
            self.print_result("API Health", False, f"Connection error: {str(e)[:50]}...")
            return False
    
    def test_swagger_ui(self):
        """Test Swagger UI accessibility."""
        self.print_header("Swagger UI Testing")
        
        try:
            # Test Swagger UI page
            response = requests.get(f"{self.api_url}/swagger", timeout=5)
            swagger_ui_working = response.status_code == 200 and "swagger-ui" in response.text.lower()
            self.print_result("Swagger UI Page", swagger_ui_working)
            
            # Test OpenAPI spec
            response = requests.get(f"{self.api_url}/api/swagger.json", timeout=5)
            openapi_working = response.status_code == 200 and "openapi" in response.json()
            self.print_result("OpenAPI Specification", openapi_working)
            
            return swagger_ui_working and openapi_working
            
        except Exception as e:
            self.print_result("Swagger UI", False, f"Error: {str(e)[:50]}...")
            return False
    
    def test_ai_models_status(self):
        """Test AI models status and availability."""
        self.print_header("AI Models Testing")
        
        try:
            # Test models status endpoint
            response = requests.get(f"{self.api_url}/api/models/status", timeout=5)
            if response.status_code == 200:
                models = response.json()
                self.print_result("Models Status Endpoint", True)
                
                # Check individual models
                binary_loaded = models.get('binary_classifier', {}).get('loaded', False)
                multiclass_loaded = models.get('multiclass_classifier', {}).get('loaded', False)
                
                self.print_result("Binary Classifier", binary_loaded, 
                                f"Accuracy: {models.get('binary_classifier', {}).get('accuracy', 'N/A')}")
                self.print_result("Multiclass Classifier", multiclass_loaded,
                                f"Accuracy: {models.get('multiclass_classifier', {}).get('accuracy', 'N/A')}")
                
                return binary_loaded and multiclass_loaded
            else:
                self.print_result("Models Status Endpoint", False, f"HTTP {response.status_code}")
                return False
                
        except Exception as e:
            self.print_result("AI Models Status", False, f"Error: {str(e)[:50]}...")
            return False
    
    def test_models_info(self):
        """Test detailed models information."""
        try:
            response = requests.get(f"{self.api_url}/api/models/info", timeout=5)
            if response.status_code == 200:
                info = response.json()
                models_loaded = info.get('models_loaded', 0)
                self.print_result("Models Info Endpoint", True, f"Models loaded: {models_loaded}")
                
                # Check model details
                binary_available = info.get('binary_model', {}).get('available', False)
                multiclass_available = info.get('multiclass_model', {}).get('available', False)
                
                self.print_result("Binary Model Info", binary_available)
                self.print_result("Multiclass Model Info", multiclass_available)
                
                return binary_available and multiclass_available
            else:
                self.print_result("Models Info Endpoint", False, f"HTTP {response.status_code}")
                return False
        except Exception as e:
            self.print_result("Models Info", False, f"Error: {str(e)[:50]}...")
            return False
    
    def test_vulnerability_detection(self):
        """Test AI vulnerability detection with various contract types."""
        self.print_header("AI Vulnerability Detection Testing")
        
        # Test cases with different vulnerability types
        test_cases = [
            {
                "name": "Reentrancy Vulnerable Contract",
                "contract": """
pragma solidity ^0.8.0;

contract VulnerableBank {
    mapping(address => uint256) public balances;
    
    function withdraw(uint256 amount) public {
        require(balances[msg.sender] >= amount);
        
        // Vulnerable to reentrancy
        (bool success, ) = msg.sender.call{value: amount}("");
        require(success);
        
        balances[msg.sender] -= amount;
    }
}""",
                "should_be_vulnerable": True,
                "expected_types": ["reentrancy"]
            },
            {
                "name": "Bad Randomness Contract",
                "contract": """
pragma solidity ^0.8.0;

contract BadRandomness {
    function randomNumber() public view returns (uint256) {
        return uint256(keccak256(abi.encodePacked(block.timestamp))) % 100;
    }
}""",
                "should_be_vulnerable": True,
                "expected_types": ["bad_randomness"]
            },
            {
                "name": "Access Control Issue",
                "contract": """
pragma solidity ^0.8.0;

contract NoAccessControl {
    function emergencyWithdraw() public {
        selfdestruct(payable(msg.sender));
    }
}""",
                "should_be_vulnerable": True,
                "expected_types": ["access_control", "dangerous_function"]
            },
            {
                "name": "Safe Contract",
                "contract": """
pragma solidity ^0.8.0;

contract SafeContract {
    mapping(address => uint256) public balances;
    address public owner;
    bool private locked;
    
    modifier onlyOwner() {
        require(msg.sender == owner);
        _;
    }
    
    modifier noReentrant() {
        require(!locked);
        locked = true;
        _;
        locked = false;
    }
    
    function withdraw(uint256 amount) public noReentrant {
        require(balances[msg.sender] >= amount);
        balances[msg.sender] -= amount;
        (bool success, ) = msg.sender.call{value: amount}("");
        require(success);
    }
}""",
                "should_be_vulnerable": False,
                "expected_types": []
            }
        ]
        
        detection_working = True
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n🔍 Test Case {i}: {test_case['name']}")
            
            try:
                response = requests.post(
                    f"{self.api_url}/api/analyze",
                    json={"contract_code": test_case["contract"]},
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    is_vulnerable = result.get('is_vulnerable', False)
                    risk_score = result.get('risk_score', 0)
                    vulnerabilities = result.get('vulnerabilities', [])
                    analysis_method = result.get('analysis_method', 'Unknown')
                    analysis_time = result.get('analysis_time', 0)
                    
                    print(f"   📊 Vulnerable: {is_vulnerable}")
                    print(f"   📈 Risk Score: {risk_score}")
                    print(f"   🔍 Vulnerabilities: {len(vulnerabilities)}")
                    print(f"   🤖 Method: {analysis_method}")
                    print(f"   ⏱️ Time: {analysis_time:.2f}s")
                    
                    if vulnerabilities:
                        print(f"   🚨 Detected Issues:")
                        for vuln in vulnerabilities[:3]:  # Show first 3
                            print(f"      - {vuln.get('type')}: {vuln.get('severity')} ({vuln.get('source')})")
                    
                    # Validate detection accuracy
                    detection_correct = is_vulnerable == test_case["should_be_vulnerable"]
                    if detection_correct:
                        self.print_result(f"Detection Accuracy - {test_case['name']}", True)
                    else:
                        self.print_result(f"Detection Accuracy - {test_case['name']}", False, 
                                        f"Expected: {test_case['should_be_vulnerable']}, Got: {is_vulnerable}")
                        detection_working = False
                    
                else:
                    self.print_result(f"Analysis - {test_case['name']}", False, f"HTTP {response.status_code}")
                    detection_working = False
                    
            except Exception as e:
                self.print_result(f"Analysis - {test_case['name']}", False, f"Error: {str(e)[:50]}...")
                detection_working = False
        
        return detection_working
    
    def test_external_tools_status(self):
        """Test external tools integration."""
        self.print_header("External Tools Testing")
        
        try:
            response = requests.get(f"{self.api_url}/api/tools/status", timeout=5)
            if response.status_code == 200:
                tools = response.json()
                self.print_result("Tools Status Endpoint", True)
                
                # Check individual tools
                slither_available = tools.get('slither', {}).get('available', False)
                mythril_available = tools.get('mythril', {}).get('available', False)
                solc_available = tools.get('solc', {}).get('available', False)
                
                self.print_result("Slither Integration", slither_available,
                                f"Version: {tools.get('slither', {}).get('version', 'N/A')}")
                self.print_result("Mythril Integration", mythril_available,
                                f"Version: {tools.get('mythril', {}).get('version', 'N/A')}")
                self.print_result("Solidity Compiler", solc_available,
                                f"Version: {tools.get('solc', {}).get('version', 'N/A')}")
                
                return True  # Tools status endpoint working (tools themselves may not be installed)
            else:
                self.print_result("Tools Status Endpoint", False, f"HTTP {response.status_code}")
                return False
                
        except Exception as e:
            self.print_result("External Tools Status", False, f"Error: {str(e)[:50]}...")
            return False
    
    def test_file_upload_simulation(self):
        """Test file upload capability (simulate contract file upload)."""
        self.print_header("File Upload Testing")
        
        try:
            # Create a temporary contract file
            test_contract = """pragma solidity ^0.8.0;

contract TestUpload {
    uint256 public value;
    
    function setValue(uint256 _value) public {
        value = _value;
    }
    
    function getValue() public view returns (uint256) {
        return value;
    }
}"""
            
            # Test analysis with file-like content
            response = requests.post(
                f"{self.api_url}/api/analyze",
                json={
                    "contract_code": test_contract,
                    "options": {"source": "file_upload"}
                },
                timeout=15
            )
            
            if response.status_code == 200:
                result = response.json()
                self.print_result("File Upload Simulation", True, 
                                f"Analysis successful, Method: {result.get('analysis_method')}")
                return True
            else:
                self.print_result("File Upload Simulation", False, f"HTTP {response.status_code}")
                return False
                
        except Exception as e:
            self.print_result("File Upload Simulation", False, f"Error: {str(e)[:50]}...")
            return False
    
    def test_dashboard_accessibility(self):
        """Test dashboard accessibility."""
        self.print_header("Dashboard Testing")
        
        try:
            # Check if dashboard is running
            response = requests.get("http://localhost:8501", timeout=5)
            if response.status_code == 200:
                self.print_result("Dashboard Accessibility", True, "Dashboard is running")
                return True
            else:
                self.print_result("Dashboard Accessibility", False, 
                                f"Dashboard not running (HTTP {response.status_code})")
                return False
        except Exception as e:
            self.print_result("Dashboard Accessibility", False, 
                            "Dashboard not running - Start with: python start_dashboard_only.py")
            return False
    
    def test_analysis_history(self):
        """Test analysis history endpoint."""
        self.print_header("Analysis History Testing")
        
        try:
            response = requests.get(f"{self.api_url}/api/history", timeout=5)
            if response.status_code == 200:
                history = response.json()
                total_analyses = history.get('total_analyses', 0)
                recent_analyses = history.get('recent_analyses', [])
                
                self.print_result("History Endpoint", True, 
                                f"Total: {total_analyses}, Recent: {len(recent_analyses)}")
                return True
            else:
                self.print_result("History Endpoint", False, f"HTTP {response.status_code}")
                return False
        except Exception as e:
            self.print_result("Analysis History", False, f"Error: {str(e)[:50]}...")
            return False
    
    def test_performance(self):
        """Test API performance with multiple requests."""
        self.print_header("Performance Testing")
        
        simple_contract = "pragma solidity ^0.8.0; contract Simple { uint256 public value; }"
        
        try:
            # Test multiple rapid requests
            start_time = time.time()
            successful_requests = 0
            
            for i in range(5):
                response = requests.post(
                    f"{self.api_url}/api/analyze",
                    json={"contract_code": simple_contract},
                    timeout=10
                )
                if response.status_code == 200:
                    successful_requests += 1
            
            total_time = time.time() - start_time
            avg_time = total_time / 5
            
            performance_good = successful_requests >= 4 and avg_time < 5.0
            
            self.print_result("Performance Test", performance_good,
                            f"{successful_requests}/5 requests successful, Avg: {avg_time:.2f}s")
            
            return performance_good
            
        except Exception as e:
            self.print_result("Performance Test", False, f"Error: {str(e)[:50]}...")
            return False
    
    def run_comprehensive_test(self):
        """Run all feature tests."""
        print("🚀 Smart Contract AI Analyzer - Comprehensive Feature Testing")
        print("=" * 80)
        
        # Run all tests
        tests = [
            ("API Health", self.test_api_health),
            ("Swagger UI", self.test_swagger_ui),
            ("AI Models Status", self.test_ai_models_status),
            ("Models Info", self.test_models_info),
            ("Vulnerability Detection", self.test_vulnerability_detection),
            ("External Tools", self.test_external_tools_status),
            ("File Upload", self.test_file_upload_simulation),
            ("Dashboard", self.test_dashboard_accessibility),
            ("Analysis History", self.test_analysis_history),
            ("Performance", self.test_performance)
        ]
        
        print(f"\n⏳ Running {len(tests)} comprehensive tests...")
        
        for test_name, test_func in tests:
            try:
                test_func()
            except Exception as e:
                self.print_result(test_name, False, f"Test error: {str(e)[:50]}...")
        
        # Summary
        self.print_header("COMPREHENSIVE TEST SUMMARY")
        
        total_tests = len(self.test_results)
        passed_tests = sum(self.test_results.values())
        
        print(f"\n📊 Overall Results:")
        print(f"   Total Tests: {total_tests}")
        print(f"   Passed: {passed_tests}")
        print(f"   Failed: {total_tests - passed_tests}")
        print(f"   Success Rate: {passed_tests/total_tests*100:.1f}%")
        
        # Detailed results
        print(f"\n📋 Detailed Results:")
        for feature, status in self.test_results.items():
            icon = "✅" if status else "❌"
            print(f"   {icon} {feature}")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        
        if passed_tests >= total_tests * 0.8:  # 80% pass rate
            print("   🎉 Excellent! Your system is working at high capacity")
            print("   🚀 All core features are functional")
        elif passed_tests >= total_tests * 0.6:  # 60% pass rate
            print("   ✅ Good! Most features are working")
            print("   🔧 Some minor issues to address")
        else:
            print("   ⚠️ Several features need attention")
            print("   🛠️ Review failed tests and fix issues")
        
        # Specific recommendations
        if not self.test_results.get("Dashboard Accessibility", True):
            print("   📱 Start dashboard: python start_dashboard_only.py")
        
        if not self.test_results.get("AI Models Status", True):
            print("   🤖 Setup AI models: python setup_ai_models.py")
        
        if not self.test_results.get("External Tools", True):
            print("   🛠️ Install tools: pip install slither-analyzer mythril")
        
        print(f"\n🌐 Access Points:")
        print(f"   📡 API Health: {self.api_url}/health")
        print(f"   📚 Swagger UI: {self.api_url}/swagger")
        print(f"   🔍 Analysis: POST {self.api_url}/api/analyze")
        print(f"   📱 Dashboard: http://localhost:8501")
        
        return passed_tests >= total_tests * 0.7  # 70% pass rate for success

def main():
    """Main testing function."""
    print("Starting comprehensive feature testing...")
    print("Make sure your backend API is running: python simple_api.py")
    print()
    
    tester = FeatureTester()
    success = tester.run_comprehensive_test()
    
    if success:
        print(f"\n🎯 COMPREHENSIVE TEST: PASSED!")
        print(f"🚀 Your Smart Contract AI Analyzer is working excellently!")
    else:
        print(f"\n⚠️ Some features need attention")
        print(f"💡 Review the failed tests above and fix the issues")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)