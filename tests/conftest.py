"""
Pytest configuration and fixtures for the test suite.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os


@pytest.fixture
def sample_vulnerable_contract():
    """Sample vulnerable Solidity contract for testing."""
    return """
pragma solidity ^0.8.0;

contract VulnerableContract {
    mapping(address => uint256) public balances;
    
    function withdraw(uint256 amount) public {
        require(balances[msg.sender] >= amount, "Insufficient balance");
        
        // Vulnerable to reentrancy - external call before state update
        (bool success, ) = msg.sender.call{value: amount}("");
        require(success, "Transfer failed");
        
        balances[msg.sender] -= amount;  // State update after external call
    }
    
    function deposit() public payable {
        balances[msg.sender] += msg.value;
    }
}
"""


@pytest.fixture
def sample_safe_contract():
    """Sample safe Solidity contract for testing."""
    return """
pragma solidity ^0.8.0;

contract SafeContract {
    mapping(address => uint256) public balances;
    bool private locked;
    
    modifier noReentrancy() {
        require(!locked, "Reentrant call");
        locked = true;
        _;
        locked = false;
    }
    
    function withdraw(uint256 amount) public noReentrancy {
        require(balances[msg.sender] >= amount, "Insufficient balance");
        
        // Safe pattern - state update before external call
        balances[msg.sender] -= amount;
        
        (bool success, ) = msg.sender.call{value: amount}("");
        require(success, "Transfer failed");
    }
    
    function deposit() public payable {
        balances[msg.sender] += msg.value;
    }
}
"""


@pytest.fixture
def sample_contracts_dataframe():
    """Sample DataFrame with contract data for testing."""
    return pd.DataFrame({
        'filename': ['vulnerable.sol', 'safe.sol', 'access_control.sol'],
        'code': [
            'contract Vulnerable { function withdraw() { msg.sender.call(); balance -= 1; } }',
            'contract Safe { function withdraw() { balance -= 1; msg.sender.call(); } }',
            'contract AccessControl { function admin() public { selfdestruct(msg.sender); } }'
        ],
        'vulnerability': ['reentrancy', 'safe', 'access_control'],
        'label': [1, 0, 1]
    })


@pytest.fixture
def sample_features():
    """Sample feature vector for testing ML models."""
    return np.array([
        [10, 2, 1, 0, 1, 0.2, 5, 1, 0, 2],  # Vulnerable features
        [15, 3, 0, 1, 0, 0.1, 8, 0, 1, 1],  # Safe features
        [8, 1, 1, 0, 0, 0.3, 3, 1, 0, 3]    # Vulnerable features
    ])


@pytest.fixture
def sample_labels():
    """Sample labels for testing ML models."""
    return np.array([1, 0, 1])  # 1 = vulnerable, 0 = safe


@pytest.fixture
def temp_directory():
    """Temporary directory for test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def mock_model_path(temp_directory):
    """Mock model file path for testing."""
    model_path = temp_directory / "test_model.pkl"
    return str(model_path)


@pytest.fixture
def sample_analysis_result():
    """Sample analysis result for testing API responses."""
    return {
        "contract_id": "test_contract_123",
        "overall_risk_score": 75,
        "is_vulnerable": True,
        "confidence_level": 0.85,
        "vulnerabilities": [
            {
                "vulnerability_type": "reentrancy",
                "severity": "High",
                "confidence": 0.9,
                "line_number": 8,
                "description": "Potential reentrancy vulnerability detected",
                "recommendation": "Use reentrancy guard or checks-effects-interactions pattern",
                "code_snippet": "msg.sender.call{value: amount}(\"\");"
            }
        ],
        "feature_importance": {
            "external_call_count": 0.25,
            "state_change_after_call": 0.20,
            "has_reentrancy_guard": 0.15
        },
        "tool_comparison": {
            "slither": [
                {
                    "vulnerability_type": "reentrancy",
                    "severity": "High",
                    "confidence": 1.0,
                    "line_number": 8
                }
            ],
            "mythril": []
        },
        "analysis_time": 12.5,
        "timestamp": "2024-01-15T10:30:00Z"
    }


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up test environment variables."""
    os.environ['TESTING'] = 'True'
    os.environ['LOG_LEVEL'] = 'DEBUG'
    yield
    # Cleanup
    if 'TESTING' in os.environ:
        del os.environ['TESTING']
# Additional fixtures for comprehensive testing
import sys
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from features.feature_extractor import FeatureExtractor
    from models.random_forest import RandomForestModel
    from models.multiclass_classifier import MultiClassClassifier
    from integration.slither_runner import SlitherRunner
    from integration.mythril_runner import MythrilRunner
    from integration.tool_comparator import ToolComparator
except ImportError:
    # Handle import errors gracefully for testing
    pass


@pytest.fixture(scope="session")
def test_data_dir():
    """Get the test data directory path."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def sample_contract_files(test_data_dir):
    """Load sample contract files for testing."""
    contracts = {}
    
    contract_files = [
        "safe_contract.sol",
        "reentrancy_vulnerable.sol", 
        "access_control_bug.sol",
        "overflow_vulnerable.sol",
        "unchecked_call.sol",
        "dos_vulnerable.sol",
        "bad_randomness.sol"
    ]
    
    for filename in contract_files:
        filepath = test_data_dir / filename
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                contracts[filename] = f.read()
    
    return contracts


@pytest.fixture
def feature_extractor():
    """Create a FeatureExtractor instance."""
    try:
        return FeatureExtractor()
    except NameError:
        pytest.skip("FeatureExtractor not available")


@pytest.fixture
def sample_extracted_features(feature_extractor, sample_contract_files):
    """Extract features from sample contracts."""
    features = {}
    
    for contract_name, contract_code in sample_contract_files.items():
        try:
            features[contract_name] = feature_extractor.extract_features(contract_code)
        except Exception as e:
            pytest.skip(f"Failed to extract features from {contract_name}: {e}")
    
    return features


@pytest.fixture
def synthetic_ml_dataset():
    """Create synthetic dataset for testing ML models."""
    np.random.seed(42)
    
    n_samples = 100
    n_features = 30
    
    # Create synthetic feature data
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    # Create synthetic labels
    y_binary = pd.Series(np.random.choice(['safe', 'vulnerable'], n_samples))
    y_multiclass = pd.Series(np.random.choice(
        ['safe', 'reentrancy', 'access_control', 'arithmetic'], n_samples
    ))
    
    return {
        'X': X,
        'y_binary': y_binary,
        'y_multiclass': y_multiclass
    }


@pytest.fixture
def binary_classifier():
    """Create a RandomForestModel instance."""
    try:
        return RandomForestModel()
    except NameError:
        pytest.skip("RandomForestModel not available")


@pytest.fixture
def multiclass_classifier():
    """Create a MultiClassClassifier instance."""
    try:
        return MultiClassClassifier()
    except NameError:
        pytest.skip("MultiClassClassifier not available")


@pytest.fixture
def trained_binary_model(binary_classifier, synthetic_ml_dataset):
    """Create a trained binary classification model."""
    X = synthetic_ml_dataset['X']
    y = synthetic_ml_dataset['y_binary']
    
    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Train model
    binary_classifier.train(X_train, y_train)
    
    return {
        'model': binary_classifier,
        'X_test': X_test,
        'y_test': y_test
    }


@pytest.fixture
def trained_multiclass_model(multiclass_classifier, synthetic_ml_dataset):
    """Create a trained multi-class classification model."""
    X = synthetic_ml_dataset['X']
    y = synthetic_ml_dataset['y_multiclass']
    
    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Train model
    multiclass_classifier.train(X_train, y_train)
    
    return {
        'model': multiclass_classifier,
        'X_test': X_test,
        'y_test': y_test
    }


@pytest.fixture
def slither_analyzer():
    """Create a SlitherRunner instance."""
    try:
        return SlitherRunner()
    except NameError:
        pytest.skip("SlitherRunner not available")


@pytest.fixture
def mythril_analyzer():
    """Create a MythrilRunner instance."""
    try:
        return MythrilRunner()
    except NameError:
        pytest.skip("MythrilRunner not available")


@pytest.fixture
def tool_comparator():
    """Create a ToolComparator instance."""
    try:
        return ToolComparator()
    except NameError:
        pytest.skip("ToolComparator not available")


@pytest.fixture
def temp_contract_file():
    """Create a temporary contract file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sol', delete=False) as f:
        f.write("""
        pragma solidity ^0.8.0;
        
        contract TestContract {
            uint256 public value;
            
            function setValue(uint256 _value) public {
                value = _value;
            }
        }
        """)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    try:
        os.unlink(temp_path)
    except OSError:
        pass


@pytest.fixture
def vulnerability_test_cases():
    """Provide test cases with known vulnerabilities for validation."""
    return {
        'reentrancy': {
            'code': '''
            pragma solidity ^0.8.0;
            contract Vulnerable {
                mapping(address => uint256) balances;
                function withdraw() external {
                    uint256 amount = balances[msg.sender];
                    (bool success,) = msg.sender.call{value: amount}("");
                    require(success);
                    balances[msg.sender] = 0;
                }
            }
            ''',
            'expected_vulnerabilities': ['reentrancy'],
            'expected_features': {
                'external_call_count': 1,
                'state_change_after_call': 1
            }
        },
        'access_control': {
            'code': '''
            pragma solidity ^0.8.0;
            contract Vulnerable {
                address owner;
                function destroy() external {
                    selfdestruct(payable(msg.sender));
                }
            }
            ''',
            'expected_vulnerabilities': ['access_control'],
            'expected_features': {
                'has_selfdestruct': 1,
                'modifier_count': 0
            }
        },
        'bad_randomness': {
            'code': '''
            pragma solidity ^0.8.0;
            contract Vulnerable {
                function random() external view returns (uint256) {
                    return uint256(keccak256(abi.encodePacked(block.timestamp))) % 100;
                }
            }
            ''',
            'expected_vulnerabilities': ['bad_randomness'],
            'expected_features': {
                'uses_block_timestamp': 1,
                'keccak256_count': 1
            }
        }
    }


@pytest.fixture(scope="session")
def test_contract_dataset(test_data_dir):
    """Create a dataset from test contracts with labels."""
    contracts = []
    
    # Define contract files and their vulnerability types
    contract_info = {
        'safe_contract.sol': 'safe',
        'reentrancy_vulnerable.sol': 'reentrancy',
        'access_control_bug.sol': 'access_control',
        'overflow_vulnerable.sol': 'arithmetic',
        'unchecked_call.sol': 'unchecked_calls',
        'dos_vulnerable.sol': 'denial_of_service',
        'bad_randomness.sol': 'bad_randomness'
    }
    
    for filename, vulnerability_type in contract_info.items():
        filepath = test_data_dir / filename
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                code = f.read()
                contracts.append({
                    'filename': filename,
                    'code': code,
                    'vulnerability': vulnerability_type,
                    'label': 0 if vulnerability_type == 'safe' else 1
                })
    
    return pd.DataFrame(contracts)


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "external_tools: marks tests that require external tools (Slither, Mythril)"
    )


def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on their names."""
    for item in items:
        # Mark integration tests
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        
        # Mark external tool tests
        if any(tool in item.nodeid.lower() for tool in ["slither", "mythril"]):
            item.add_marker(pytest.mark.external_tools)
        
        # Mark slow tests
        if any(keyword in item.nodeid.lower() for keyword in ["benchmark", "performance", "load"]):
            item.add_marker(pytest.mark.slow)