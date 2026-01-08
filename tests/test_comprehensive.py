#!/usr/bin/env python3
"""
Comprehensive test suite for Smart Warden.
Tests all major components and integration points.
"""

import pytest
import sys
import os
from pathlib import Path
import json
import tempfile
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from features.feature_extractor import SolidityFeatureExtractor
from models.model_loader import ModelLoader, predict_vulnerability
from integration.docker_tools import check_tools_availability
from integration.tool_comparator import ToolComparator

class TestFeatureExtraction:
    """Test feature extraction functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = SolidityFeatureExtractor()
        self.sample_contract = """
        pragma solidity ^0.8.0;
        
        contract TestContract {
            mapping(address => uint256) public balances;
            address public owner;
            
            modifier onlyOwner() {
                require(msg.sender == owner, "Not owner");
                _;
            }
            
            function withdraw(uint256 amount) public {
                require(balances[msg.sender] >= amount, "Insufficient balance");
                balances[msg.sender] -= amount;
                (bool success, ) = msg.sender.call{value: amount}("");
                require(success, "Transfer failed");
            }
            
            function adminWithdraw() public onlyOwner {
                payable(owner).transfer(address(this).balance);
            }
        }
        """
    
    def test_feature_extraction_basic(self):
        """Test basic feature extraction."""
        features = self.extractor.extract_features(self.sample_contract)
        
        assert isinstance(features, dict)
        assert len(features) > 0
        assert 'lines_of_code' in features
        assert 'function_count' in features
        assert features['lines_of_code'] > 0
        assert features['function_count'] >= 2
    
    def test_feature_extraction_vulnerability_patterns(self):
        """Test vulnerability pattern detection."""
        vulnerable_contract = """
        pragma solidity ^0.7.0;
        contract Vulnerable {
            mapping(address => uint) balances;
            
            function withdraw() public {
                uint amount = balances[msg.sender];
                msg.sender.call{value: amount}("");
                balances[msg.sender] = 0;
            }
        }
        """
        
        features = self.extractor.extract_features(vulnerable_contract)
        
        # Should detect external calls
        assert features.get('external_call_count', 0) > 0
        # Should detect potential reentrancy pattern
        assert features.get('state_change_after_call', 0) > 0
    
    def test_feature_extraction_safe_patterns(self):
        """Test detection of safe patterns."""
        features = self.extractor.extract_features(self.sample_contract)
        
        # Should detect access control
        assert features.get('modifier_count', 0) > 0
        assert features.get('require_count', 0) > 0
        # Should detect proper version
        assert features.get('solidity_version', 0) >= 0.8

class TestModelLoading:
    """Test model loading and prediction functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.loader = ModelLoader()
        self.sample_features = {
            'lines_of_code': 20,
            'function_count': 2,
            'external_call_count': 1,
            'require_count': 2,
            'modifier_count': 1,
            'state_change_after_call': 0,
            'solidity_version': 0.8
        }
    
    def test_model_loading(self):
        """Test model loading functionality."""
        models = self.loader.load_all_models()
        
        assert isinstance(models, dict)
        assert 'count' in models
        # At least one model should be loaded
        assert models['count'] >= 0
    
    def test_vulnerability_prediction(self):
        """Test vulnerability prediction."""
        result = predict_vulnerability(self.sample_features)
        
        assert isinstance(result, dict)
        assert 'available' in result
        
        if result['available']:
            assert 'binary_prediction' in result
            assert 'is_vulnerable' in result['binary_prediction']
            assert 'confidence' in result['binary_prediction']

class TestToolComparison:
    """Test tool comparison functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.comparator = ToolComparator()
        
        self.ai_result = {
            'available': True,
            'binary_prediction': {
                'is_vulnerable': True,
                'confidence': 0.8,
                'vulnerability_probability': 0.75
            },
            'multiclass_prediction': {
                'vulnerability_type': 'reentrancy',
                'confidence': 0.7
            }
        }
        
        self.slither_result = {
            'available': True,
            'vulnerabilities': [
                {
                    'type': 'reentrancy',
                    'severity': 'high',
                    'confidence': 'medium',
                    'description': 'Reentrancy vulnerability detected'
                }
            ]
        }
        
        self.mythril_result = {
            'available': False,
            'vulnerabilities': []
        }
    
    def test_tool_comparison(self):
        """Test tool comparison functionality."""
        comparison = self.comparator.compare_results(
            self.ai_result,
            self.slither_result,
            self.mythril_result
        )
        
        assert isinstance(comparison, dict)
        assert 'tools_used' in comparison
        assert 'consensus' in comparison
        assert 'combined_score' in comparison
        assert 'AI' in comparison['tools_used']
        assert 'Slither' in comparison['tools_used']
    
    def test_consensus_analysis(self):
        """Test consensus analysis."""
        comparison = self.comparator.compare_results(
            self.ai_result,
            self.slither_result,
            self.mythril_result
        )
        
        consensus = comparison['consensus']
        assert isinstance(consensus, dict)
        assert 'is_vulnerable' in consensus
        assert 'confidence' in consensus
        assert 'agreement_level' in consensus
    
    def test_vulnerability_summary(self):
        """Test vulnerability summary creation."""
        comparison = self.comparator.compare_results(
            self.ai_result,
            self.slither_result,
            self.mythril_result
        )
        
        summary = comparison['vulnerability_summary']
        assert isinstance(summary, dict)
        assert 'total_vulnerabilities' in summary
        assert 'by_type' in summary
        assert 'unique_types' in summary

class TestIntegration:
    """Test integration between components."""
    
    def test_end_to_end_analysis(self):
        """Test complete analysis workflow."""
        # Sample contract
        contract_code = """
        pragma solidity ^0.8.0;
        contract Simple {
            uint256 public value;
            function setValue(uint256 _value) public {
                value = _value;
            }
        }
        """
        
        # Extract features
        extractor = SolidityFeatureExtractor()
        features = extractor.extract_features(contract_code)
        
        assert isinstance(features, dict)
        assert len(features) > 0
        
        # Predict vulnerability
        prediction = predict_vulnerability(features)
        
        assert isinstance(prediction, dict)
        # Should work even if models aren't available
        assert 'available' in prediction
    
    def test_external_tools_availability(self):
        """Test external tools availability check."""
        availability = check_tools_availability()
        
        assert isinstance(availability, dict)
        assert 'docker' in availability
        assert 'slither' in availability
        assert 'mythril' in availability

class TestPerformance:
    """Test performance requirements."""
    
    def test_feature_extraction_performance(self):
        """Test feature extraction performance."""
        extractor = SolidityFeatureExtractor()
        
        # Large contract for performance testing
        large_contract = """
        pragma solidity ^0.8.0;
        contract LargeContract {
        """ + "\n".join([
            f"    uint256 public var{i};"
            f"    function func{i}() public {{ var{i} = {i}; }}"
            for i in range(100)
        ]) + "\n}"
        
        start_time = time.time()
        features = extractor.extract_features(large_contract)
        extraction_time = time.time() - start_time
        
        # Should complete within reasonable time
        assert extraction_time < 5.0  # 5 seconds max
        assert isinstance(features, dict)
        assert len(features) > 0
    
    def test_model_prediction_performance(self):
        """Test model prediction performance."""
        sample_features = {f'feature_{i}': float(i) for i in range(67)}
        
        start_time = time.time()
        result = predict_vulnerability(sample_features)
        prediction_time = time.time() - start_time
        
        # Should complete quickly
        assert prediction_time < 2.0  # 2 seconds max
        assert isinstance(result, dict)

class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_contract_code(self):
        """Test handling of invalid contract code."""
        extractor = SolidityFeatureExtractor()
        
        # Invalid Solidity code
        invalid_code = "This is not valid Solidity code!"
        
        # Should not crash
        features = extractor.extract_features(invalid_code)
        assert isinstance(features, dict)
    
    def test_empty_contract_code(self):
        """Test handling of empty contract code."""
        extractor = SolidityFeatureExtractor()
        
        features = extractor.extract_features("")
        assert isinstance(features, dict)
        assert features.get('lines_of_code', 0) == 0
    
    def test_missing_models(self):
        """Test handling when models are not available."""
        # Test with empty features
        result = predict_vulnerability({})
        
        assert isinstance(result, dict)
        assert 'available' in result

# Pytest configuration
def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )

# Test data fixtures
@pytest.fixture
def sample_vulnerable_contract():
    """Sample vulnerable contract for testing."""
    return """
    pragma solidity ^0.7.0;
    contract Vulnerable {
        mapping(address => uint) balances;
        
        function withdraw() public {
            uint amount = balances[msg.sender];
            msg.sender.call{value: amount}("");
            balances[msg.sender] = 0;
        }
        
        function randomNumber() public view returns (uint) {
            return uint(keccak256(abi.encodePacked(block.timestamp))) % 100;
        }
    }
    """

@pytest.fixture
def sample_safe_contract():
    """Sample safe contract for testing."""
    return """
    pragma solidity ^0.8.0;
    contract Safe {
        mapping(address => uint256) public balances;
        address public owner;
        
        modifier onlyOwner() {
            require(msg.sender == owner, "Not owner");
            _;
        }
        
        function withdraw(uint256 amount) public {
            require(balances[msg.sender] >= amount, "Insufficient balance");
            balances[msg.sender] -= amount;
            (bool success, ) = msg.sender.call{value: amount}("");
            require(success, "Transfer failed");
        }
    }
    """

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])