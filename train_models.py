#!/usr/bin/env python3
"""
Simple model training script for Smart Warden AI models.
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

# Add src to path
sys.path.insert(0, 'src')

from features.feature_extractor import SolidityFeatureExtractor
from models.random_forest import RandomForestVulnerabilityDetector

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_dataset():
    """Create a sample dataset for training."""
    logger.info("Creating sample dataset...")
    
    # Sample vulnerable contracts
    vulnerable_contracts = [
        """
        pragma solidity ^0.8.0;
        contract Vulnerable {
            mapping(address => uint256) public balances;
            
            function withdraw(uint256 amount) public {
                require(balances[msg.sender] >= amount);
                (bool success, ) = msg.sender.call{value: amount}("");
                require(success);
                balances[msg.sender] -= amount; // State change after external call - reentrancy
            }
        }
        """,
        """
        pragma solidity ^0.7.0;
        contract BadMath {
            uint256 public total;
            
            function add(uint256 amount) public {
                total += amount; // No overflow protection in 0.7.0
            }
            
            function randomNumber() public view returns (uint256) {
                return uint256(keccak256(abi.encodePacked(block.timestamp))) % 100; // Bad randomness
            }
        }
        """,
        """
        pragma solidity ^0.8.0;
        contract AccessControl {
            address public owner;
            
            function withdraw() public {
                // Missing access control
                payable(msg.sender).transfer(address(this).balance);
            }
        }
        """,
        """
        pragma solidity ^0.8.0;
        contract UncheckedCall {
            function sendEther(address to, uint256 amount) public {
                to.call{value: amount}(""); // Unchecked external call
            }
        }
        """
    ]
    
    # Sample safe contracts
    safe_contracts = [
        """
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
        contract SafeMath {
            uint256 public total;
            
            function add(uint256 amount) public {
                total += amount; // Safe in 0.8.0+
            }
            
            function secureRandom() public view returns (uint256) {
                // This is still not truly random, but better than block.timestamp
                return uint256(keccak256(abi.encodePacked(
                    block.difficulty,
                    block.timestamp,
                    msg.sender
                ))) % 100;
            }
        }
        """,
        """
        pragma solidity ^0.8.0;
        contract CheckedCall {
            function sendEther(address to, uint256 amount) public {
                (bool success, ) = to.call{value: amount}("");
                require(success, "Transfer failed"); // Checked external call
            }
        }
        """
    ]
    
    # Create dataset
    contracts = []
    labels = []
    
    # Add vulnerable contracts
    for contract in vulnerable_contracts:
        contracts.append(contract.strip())
        labels.append(1)  # 1 = vulnerable
    
    # Add safe contracts  
    for contract in safe_contracts:
        contracts.append(contract.strip())
        labels.append(0)  # 0 = safe
    
    # Create DataFrame
    df = pd.DataFrame({
        'contract_code': contracts,
        'label': labels
    })
    
    logger.info(f"Created dataset with {len(df)} contracts ({sum(labels)} vulnerable, {len(labels) - sum(labels)} safe)")
    return df

def extract_features(df):
    """Extract features from contracts."""
    logger.info("Extracting features...")
    
    extractor = SolidityFeatureExtractor()
    features_list = []
    
    for idx, contract_code in enumerate(df['contract_code']):
        try:
            features = extractor.extract_features(contract_code)
            features_list.append(features)
            logger.info(f"Extracted {len(features)} features from contract {idx + 1}")
        except Exception as e:
            logger.error(f"Error extracting features from contract {idx + 1}: {e}")
            # Create dummy features
            features_list.append({})
    
    # Convert to DataFrame
    features_df = pd.DataFrame(features_list)
    
    # Fill NaN values with 0
    features_df = features_df.fillna(0)
    
    logger.info(f"Feature extraction complete. Shape: {features_df.shape}")
    return features_df

def train_binary_model(X, y):
    """Train binary classification model."""
    logger.info("Training binary classification model...")
    
    # Initialize model
    model = RandomForestVulnerabilityDetector(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    
    # Train model
    model.train(X, y)
    
    # Get predictions for evaluation
    predictions, probabilities = model.predict(X)
    
    # Calculate accuracy
    accuracy = np.mean(predictions == y)
    logger.info(f"Training accuracy: {accuracy:.3f}")
    
    return model

def save_model(model, features_df, accuracy):
    """Save trained model and metadata."""
    logger.info("Saving model...")
    
    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Save model
    model_path = models_dir / "binary_classifier.joblib"
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Save metadata
    metadata = {
        "model_type": "RandomForest",
        "model_version": "1.0.0",
        "training_date": datetime.now().isoformat(),
        "accuracy": float(accuracy),
        "n_features": len(features_df.columns),
        "feature_names": list(features_df.columns),
        "model_parameters": {
            "n_estimators": model.model.n_estimators,
            "max_depth": model.model.max_depth,
            "random_state": model.model.random_state
        }
    }
    
    metadata_path = models_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Metadata saved to {metadata_path}")

def main():
    """Main training function."""
    logger.info("🚀 Starting Smart Warden AI Model Training")
    logger.info("=" * 50)
    
    try:
        # Create sample dataset
        df = create_sample_dataset()
        
        # Extract features
        features_df = extract_features(df)
        
        if features_df.empty:
            logger.error("No features extracted. Cannot train model.")
            return False
        
        # Prepare training data
        X = features_df
        y = df['label']
        
        logger.info(f"Training data shape: X={X.shape}, y={y.shape}")
        
        # Train binary model
        model = train_binary_model(X, y)
        
        # Calculate final accuracy
        predictions, _ = model.predict(X)
        accuracy = np.mean(predictions == y)
        
        # Save model
        save_model(model, features_df, accuracy)
        
        logger.info("=" * 50)
        logger.info("🎉 Model training completed successfully!")
        logger.info(f"📊 Final accuracy: {accuracy:.3f}")
        logger.info(f"📁 Model saved to: models/binary_classifier.joblib")
        logger.info(f"📄 Metadata saved to: models/metadata.json")
        
        return True
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)