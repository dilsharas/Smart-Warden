"""
Unit tests for the SolidityFeatureExtractor class.
"""

import pytest
import pandas as pd
from features.feature_extractor import SolidityFeatureExtractor, FeatureConfig


class TestSolidityFeatureExtractor:
    """Test cases for SolidityFeatureExtractor."""
    
    def test_initialization(self):
        """Test feature extractor initialization."""
        extractor = SolidityFeatureExtractor()
        assert extractor.config is not None
        assert len(extractor.dangerous_functions) > 0
        assert len(extractor.access_modifiers) > 0
        assert len(extractor.vulnerability_keywords) > 0
    
    def test_initialization_with_config(self):
        """Test feature extractor initialization with custom config."""
        config = FeatureConfig(
            include_basic_metrics=True,
            include_function_analysis=False,
            include_dangerous_patterns=True
        )
        extractor = SolidityFeatureExtractor(config)
        assert extractor.config == config
    
    def test_extract_features_basic_contract(self, feature_extractor):
        """Test feature extraction from a basic contract."""
        contract_code = """
        pragma solidity ^0.8.0;
        
        contract SimpleContract {
            uint256 public value;
            
            function setValue(uint256 _value) public {
                value = _value;
            }
            
            function getValue() public view returns (uint256) {
                return value;
            }
        }
        """
        
        features = feature_extractor.extract_features(contract_code)
        
        # Check that features are extracted
        assert isinstance(features, dict)
        assert len(features) > 0
        
        # Check basic metrics
        assert 'lines_of_code' in features
        assert 'function_count' in features
        assert 'char_count' in features
        
        # Verify expected values
        assert features['function_count'] == 2
        assert features['public_function_count'] == 1
        assert features['view_function_count'] == 1
    
    def test_extract_features_vulnerable_contract(self, feature_extractor, vulnerability_test_cases):
        """Test feature extraction from vulnerable contracts."""
        for vuln_type, test_case in vulnerability_test_cases.items():
            features = feature_extractor.extract_features(test_case['code'])
            
            # Check that expected features are present
            for feature_name, expected_value in test_case['expected_features'].items():
                assert feature_name in features
                assert features[feature_name] == expected_value
    
    def test_basic_metrics_extraction(self, feature_extractor):
        """Test basic metrics extraction."""
        contract_code = """
        pragma solidity ^0.8.0;
        // This is a comment
        contract Test {
            uint256 value; // Another comment
            
            function test() public {
                value = 1;
            }
        }
        """
        
        features = feature_extractor.extract_features(contract_code)
        
        assert features['lines_of_code'] > 0
        assert features['char_count'] > 0
        assert features['comment_ratio'] > 0
        assert features['non_comment_lines'] > 0
    
    def test_function_analysis(self, feature_extractor):
        """Test function analysis features."""
        contract_code = """
        pragma solidity ^0.8.0;
        
        contract FunctionTest {
            function publicFunc() public {}
            function privateFunc() private {}
            function externalFunc() external {}
            function internalFunc() internal {}
            function payableFunc() public payable {}
            function viewFunc() public view returns (uint256) { return 1; }
            function pureFunc() public pure returns (uint256) { return 1; }
        }
        """
        
        features = feature_extractor.extract_features(contract_code)
        
        assert features['function_count'] == 7
        assert features['public_function_count'] >= 1
        assert features['private_function_count'] >= 1
        assert features['external_function_count'] >= 1
        assert features['payable_function_count'] >= 1
        assert features['view_function_count'] >= 1
        assert features['pure_function_count'] >= 1
    
    def test_dangerous_patterns_detection(self, feature_extractor):
        """Test dangerous pattern detection."""
        contract_code = """
        pragma solidity ^0.8.0;
        
        contract DangerousContract {
            function dangerous() public {
                selfdestruct(payable(msg.sender));
                address(0).delegatecall("");
                msg.sender.call("");
            }
        }
        """
        
        features = feature_extractor.extract_features(contract_code)
        
        assert features['has_selfdestruct'] == 1
        assert features['has_delegatecall'] == 1
        assert features['external_call_count'] >= 1
        assert features['dangerous_function_count'] >= 2
    
    def test_reentrancy_pattern_detection(self, feature_extractor):
        """Test reentrancy pattern detection."""
        contract_code = """
        pragma solidity ^0.8.0;
        
        contract ReentrancyTest {
            mapping(address => uint256) balances;
            
            function withdraw() public {
                uint256 amount = balances[msg.sender];
                msg.sender.call{value: amount}("");
                balances[msg.sender] = 0;
            }
        }
        """
        
        features = feature_extractor.extract_features(contract_code)
        
        assert features['external_call_count'] >= 1
        assert features['state_change_after_call'] >= 1
    
    def test_access_control_features(self, feature_extractor):
        """Test access control feature extraction."""
        contract_code = """
        pragma solidity ^0.8.0;
        
        contract AccessControlTest {
            address public owner;
            
            modifier onlyOwner() {
                require(msg.sender == owner);
                _;
            }
            
            function restricted() public onlyOwner {
                require(msg.sender != address(0));
                assert(owner != address(0));
            }
        }
        """
        
        features = feature_extractor.extract_features(contract_code)
        
        assert features['modifier_count'] >= 1
        assert features['require_count'] >= 1
        assert features['assert_count'] >= 1
        assert features['msg_sender_checks'] >= 1
    
    def test_arithmetic_features(self, feature_extractor):
        """Test arithmetic operation feature extraction."""
        contract_code = """
        pragma solidity ^0.8.0;
        
        contract ArithmeticTest {
            function calculate(uint256 a, uint256 b) public pure returns (uint256) {
                uint256 sum = a + b;
                uint256 product = a * b;
                uint256 quotient = a / b;
                sum += 10;
                return sum;
            }
        }
        """
        
        features = feature_extractor.extract_features(contract_code)
        
        assert features['addition_count'] >= 1
        assert features['multiplication_count'] >= 1
        assert features['division_count'] >= 1
        assert features['compound_assignment_count'] >= 1
    
    def test_randomness_features(self, feature_extractor):
        """Test randomness-related feature extraction."""
        contract_code = """
        pragma solidity ^0.8.0;
        
        contract RandomnessTest {
            function badRandom() public view returns (uint256) {
                return uint256(keccak256(abi.encodePacked(
                    block.timestamp,
                    block.number,
                    blockhash(block.number - 1)
                ))) % 100;
            }
        }
        """
        
        features = feature_extractor.extract_features(contract_code)
        
        assert features['uses_block_timestamp'] == 1
        assert features['uses_block_number'] == 1
        assert features['uses_blockhash'] == 1
        assert features['keccak256_count'] >= 1
    
    def test_solidity_version_extraction(self, feature_extractor):
        """Test Solidity version extraction."""
        contract_code_old = """
        pragma solidity ^0.7.6;
        contract OldContract {}
        """
        
        contract_code_new = """
        pragma solidity ^0.8.0;
        contract NewContract {}
        """
        
        features_old = feature_extractor.extract_features(contract_code_old)
        features_new = feature_extractor.extract_features(contract_code_new)
        
        assert features_old['solidity_version'] == 0.7
        assert features_old['is_safe_version'] == 0
        
        assert features_new['solidity_version'] == 0.8
        assert features_new['is_safe_version'] == 1
    
    def test_extract_batch(self, feature_extractor, sample_contracts_dataframe):
        """Test batch feature extraction."""
        features_df = feature_extractor.extract_batch(sample_contracts_dataframe)
        
        assert isinstance(features_df, pd.DataFrame)
        assert len(features_df) == len(sample_contracts_dataframe)
        assert 'filename' in features_df.columns
        assert 'vulnerability' in features_df.columns
        assert 'label' in features_df.columns
        
        # Check that numerical features are present
        numerical_features = [col for col in features_df.columns 
                            if col not in ['filename', 'vulnerability', 'label']]
        assert len(numerical_features) > 0
    
    def test_get_feature_names(self, feature_extractor):
        """Test getting feature names."""
        feature_names = feature_extractor.get_feature_names()
        
        assert isinstance(feature_names, list)
        assert len(feature_names) > 0
        assert 'lines_of_code' in feature_names
        assert 'function_count' in feature_names
        
        # Ensure no metadata columns are included
        assert 'filename' not in feature_names
        assert 'vulnerability' not in feature_names
        assert 'label' not in feature_names
    
    def test_validate_features(self, feature_extractor):
        """Test feature validation."""
        # Valid features
        valid_features = {
            'lines_of_code': 10,
            'function_count': 2,
            'char_count': 500,
            'public_function_ratio': 0.5
        }
        
        assert feature_extractor.validate_features(valid_features) == True
        
        # Invalid features - missing required
        invalid_features = {
            'function_count': 2,
            'char_count': 500
        }
        
        assert feature_extractor.validate_features(invalid_features) == False
        
        # Invalid features - bad ratio
        invalid_ratio_features = {
            'lines_of_code': 10,
            'function_count': 2,
            'char_count': 500,
            'public_function_ratio': 1.5  # Invalid ratio > 1
        }
        
        assert feature_extractor.validate_features(invalid_ratio_features) == False
    
    def test_empty_contract(self, feature_extractor):
        """Test feature extraction from empty contract."""
        empty_contract = ""
        
        features = feature_extractor.extract_features(empty_contract)
        
        # Should still return features, mostly zeros
        assert isinstance(features, dict)
        assert features['lines_of_code'] == 1  # Empty string has 1 line
        assert features['function_count'] == 0
        assert features['char_count'] == 0
    
    def test_malformed_contract(self, feature_extractor):
        """Test feature extraction from malformed contract."""
        malformed_contract = "this is not valid solidity code"
        
        # Should not raise exception, but extract what it can
        features = feature_extractor.extract_features(malformed_contract)
        
        assert isinstance(features, dict)
        assert features['lines_of_code'] == 1
        assert features['function_count'] == 0
        assert features['solidity_version'] == 0.0  # No pragma found
    
    def test_feature_config_filtering(self):
        """Test that feature config properly filters features."""
        config = FeatureConfig(
            include_basic_metrics=True,
            include_function_analysis=False,
            include_dangerous_patterns=False,
            include_control_flow=False,
            include_access_control=False,
            include_arithmetic=False,
            include_randomness=False
        )
        
        extractor = SolidityFeatureExtractor(config)
        
        contract_code = """
        pragma solidity ^0.8.0;
        contract Test {
            function test() public { selfdestruct(payable(msg.sender)); }
        }
        """
        
        features = extractor.extract_features(contract_code)
        
        # Should have basic metrics
        assert 'lines_of_code' in features
        assert 'char_count' in features
        
        # Should not have function analysis features
        assert 'function_count' not in features
        assert 'has_selfdestruct' not in features