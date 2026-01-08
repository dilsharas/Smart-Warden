#!/usr/bin/env python3
"""
Multi-class vulnerability classifier training for Smart Warden.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
from datetime import datetime
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

# Add src to path
sys.path.insert(0, 'src')

from features.feature_extractor import SolidityFeatureExtractor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_multiclass_dataset():
    """Create a dataset with specific vulnerability types."""
    logger.info("Creating multi-class vulnerability dataset...")
    
    # Vulnerability type samples
    vulnerability_samples = {
        'reentrancy': [
            """
            pragma solidity ^0.8.0;
            contract ReentrancyVuln {
                mapping(address => uint256) public balances;
                
                function withdraw(uint256 amount) public {
                    require(balances[msg.sender] >= amount);
                    (bool success, ) = msg.sender.call{value: amount}("");
                    require(success);
                    balances[msg.sender] -= amount; // State change after external call
                }
            }
            """,
            """
            pragma solidity ^0.7.0;
            contract ReentrancyExample {
                mapping(address => uint) balances;
                
                function withdraw() public {
                    uint amount = balances[msg.sender];
                    msg.sender.call{value: amount}(""); // Reentrancy vulnerability
                    balances[msg.sender] = 0;
                }
            }
            """
        ],
        'access_control': [
            """
            pragma solidity ^0.8.0;
            contract AccessControlVuln {
                address public owner;
                
                function withdraw() public {
                    // Missing access control - anyone can withdraw
                    payable(msg.sender).transfer(address(this).balance);
                }
                
                function changeOwner(address newOwner) public {
                    // Missing onlyOwner modifier
                    owner = newOwner;
                }
            }
            """,
            """
            pragma solidity ^0.8.0;
            contract MissingModifier {
                address owner;
                
                function sensitiveFunction() public {
                    // Should have onlyOwner modifier
                    selfdestruct(payable(msg.sender));
                }
            }
            """
        ],
        'arithmetic': [
            """
            pragma solidity ^0.7.0;
            contract ArithmeticVuln {
                uint256 public total;
                
                function add(uint256 amount) public {
                    total += amount; // No overflow protection in 0.7.0
                }
                
                function multiply(uint256 a, uint256 b) public pure returns (uint256) {
                    return a * b; // Potential overflow
                }
            }
            """,
            """
            pragma solidity ^0.6.0;
            contract OverflowExample {
                uint8 public count = 255;
                
                function increment() public {
                    count++; // Will overflow to 0
                }
            }
            """
        ],
        'unchecked_calls': [
            """
            pragma solidity ^0.8.0;
            contract UncheckedCallVuln {
                function sendEther(address to, uint256 amount) public {
                    to.call{value: amount}(""); // Unchecked external call
                }
                
                function delegateCall(address target, bytes memory data) public {
                    target.delegatecall(data); // Unchecked delegatecall
                }
            }
            """,
            """
            pragma solidity ^0.8.0;
            contract IgnoredReturn {
                function transfer(address to, uint amount) public {
                    to.call{value: amount}(""); // Return value ignored
                }
            }
            """
        ],
        'dos': [
            """
            pragma solidity ^0.8.0;
            contract DoSVuln {
                address[] public users;
                
                function addUser(address user) public {
                    users.push(user);
                }
                
                function payAllUsers() public {
                    for (uint i = 0; i < users.length; i++) {
                        // Unbounded loop - DoS vulnerability
                        payable(users[i]).transfer(1 ether);
                    }
                }
            }
            """,
            """
            pragma solidity ^0.8.0;
            contract GasLimit {
                mapping(address => uint) balances;
                address[] users;
                
                function distributeRewards() public {
                    for (uint i = 0; i < users.length; i++) {
                        // Can run out of gas with many users
                        balances[users[i]] += 100;
                    }
                }
            }
            """
        ],
        'bad_randomness': [
            """
            pragma solidity ^0.8.0;
            contract BadRandomnessVuln {
                function randomNumber() public view returns (uint256) {
                    return uint256(keccak256(abi.encodePacked(block.timestamp))) % 100;
                }
                
                function lottery() public view returns (address) {
                    uint random = uint(keccak256(abi.encodePacked(block.difficulty, now)));
                    return msg.sender; // Predictable randomness
                }
            }
            """,
            """
            pragma solidity ^0.8.0;
            contract WeakRandom {
                function getRandomNumber() public view returns (uint) {
                    return uint(keccak256(abi.encodePacked(block.number))) % 10;
                }
            }
            """
        ],
        'safe': [
            """
            pragma solidity ^0.8.0;
            contract SafeContract {
                mapping(address => uint256) public balances;
                address public owner;
                
                modifier onlyOwner() {
                    require(msg.sender == owner, "Not owner");
                    _;
                }
                
                function withdraw(uint256 amount) public {
                    require(balances[msg.sender] >= amount, "Insufficient balance");
                    balances[msg.sender] -= amount; // State change before external call
                    (bool success, ) = msg.sender.call{value: amount}("");
                    require(success, "Transfer failed");
                }
                
                function adminWithdraw() public onlyOwner {
                    payable(owner).transfer(address(this).balance);
                }
            }
            """,
            """
            pragma solidity ^0.8.0;
            contract SecureContract {
                uint256 public total;
                address public owner;
                
                modifier onlyOwner() {
                    require(msg.sender == owner);
                    _;
                }
                
                function add(uint256 amount) public {
                    total += amount; // Safe in 0.8.0+
                }
                
                function checkedCall(address to, uint256 amount) public onlyOwner {
                    (bool success, ) = to.call{value: amount}("");
                    require(success, "Transfer failed");
                }
            }
            """
        ]
    }
    
    # Create dataset
    contracts = []
    labels = []
    
    for vuln_type, samples in vulnerability_samples.items():
        for sample in samples:
            contracts.append(sample.strip())
            labels.append(vuln_type)
    
    # Create DataFrame
    df = pd.DataFrame({
        'contract_code': contracts,
        'vulnerability_type': labels
    })
    
    logger.info(f"Created multi-class dataset with {len(df)} contracts")
    logger.info(f"Vulnerability distribution: {df['vulnerability_type'].value_counts().to_dict()}")
    return df

def extract_features(df):
    """Extract features from contracts."""
    logger.info("Extracting features for multi-class training...")
    
    extractor = SolidityFeatureExtractor()
    features_list = []
    
    for idx, contract_code in enumerate(df['contract_code']):
        try:
            features = extractor.extract_features(contract_code)
            features_list.append(features)
            logger.info(f"Extracted {len(features)} features from contract {idx + 1}")
        except Exception as e:
            logger.error(f"Error extracting features from contract {idx + 1}: {e}")
            features_list.append({})
    
    # Convert to DataFrame
    features_df = pd.DataFrame(features_list)
    features_df = features_df.fillna(0)
    
    logger.info(f"Feature extraction complete. Shape: {features_df.shape}")
    return features_df

def train_multiclass_model(X, y):
    """Train multi-class classification model."""
    logger.info("Training multi-class vulnerability classifier...")
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Initialize model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        class_weight='balanced'
    )
    
    # Train model
    model.fit(X, y_encoded)
    
    # Get predictions for evaluation
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_encoded, predictions)
    logger.info(f"Training accuracy: {accuracy:.3f}")
    
    # Classification report
    class_names = label_encoder.classes_
    report = classification_report(y_encoded, predictions, target_names=class_names)
    logger.info(f"Classification Report:\n{report}")
    
    return model, label_encoder, accuracy

def save_multiclass_model(model, label_encoder, features_df, accuracy):
    """Save trained multi-class model and metadata."""
    logger.info("Saving multi-class model...")
    
    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Save model
    model_path = models_dir / "multiclass_classifier.joblib"
    joblib.dump({
        'model': model,
        'label_encoder': label_encoder,
        'feature_names': list(features_df.columns)
    }, model_path)
    logger.info(f"Multi-class model saved to {model_path}")
    
    # Update metadata
    metadata_path = models_dir / "metadata.json"
    
    # Load existing metadata or create new
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {}
    
    # Add multi-class model info
    metadata.update({
        "multiclass_model": {
            "model_type": "RandomForest",
            "model_version": "1.0.0",
            "training_date": datetime.now().isoformat(),
            "accuracy": float(accuracy),
            "n_features": len(features_df.columns),
            "feature_names": list(features_df.columns),
            "vulnerability_types": list(label_encoder.classes_),
            "model_parameters": {
                "n_estimators": model.n_estimators,
                "max_depth": model.max_depth,
                "random_state": model.random_state
            }
        }
    })
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Metadata updated at {metadata_path}")

def main():
    """Main training function."""
    logger.info("🚀 Starting Multi-Class Vulnerability Classifier Training")
    logger.info("=" * 60)
    
    try:
        # Create multi-class dataset
        df = create_multiclass_dataset()
        
        # Extract features
        features_df = extract_features(df)
        
        if features_df.empty:
            logger.error("No features extracted. Cannot train model.")
            return False
        
        # Prepare training data
        X = features_df
        y = df['vulnerability_type']
        
        logger.info(f"Training data shape: X={X.shape}, y={y.shape}")
        
        # Train multi-class model
        model, label_encoder, accuracy = train_multiclass_model(X, y)
        
        # Save model
        save_multiclass_model(model, label_encoder, features_df, accuracy)
        
        logger.info("=" * 60)
        logger.info("🎉 Multi-class model training completed successfully!")
        logger.info(f"📊 Final accuracy: {accuracy:.3f}")
        logger.info(f"📁 Model saved to: models/multiclass_classifier.joblib")
        logger.info(f"🏷️ Vulnerability types: {list(label_encoder.classes_)}")
        
        return True
        
    except Exception as e:
        logger.error(f"Multi-class training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)