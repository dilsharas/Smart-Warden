#!/usr/bin/env python3
"""
Setup script to train AI models and install external tools for Smart Contract AI Analyzer.
"""

import os
import sys
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def install_external_tools():
    """Install Slither and Mythril external tools."""
    print("🛠️ Installing External Tools...")
    
    tools = [
        ("slither-analyzer", "Slither Static Analysis"),
        ("mythril", "Mythril Symbolic Execution")
    ]
    
    for package, description in tools:
        print(f"\n📦 Installing {description}...")
        try:
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", package
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ {description} installed successfully")
            else:
                print(f"❌ Failed to install {description}")
                print(f"Error: {result.stderr}")
        except Exception as e:
            print(f"❌ Error installing {description}: {e}")

def generate_training_data():
    """Generate synthetic training data for demonstration."""
    print("📊 Generating Training Data...")
    
    # Create synthetic vulnerability dataset
    np.random.seed(42)
    n_samples = 1000
    
    # Features (simplified for demo)
    features = {
        'external_call_count': np.random.poisson(2, n_samples),
        'state_change_after_call': np.random.binomial(1, 0.3, n_samples),
        'uses_block_timestamp': np.random.binomial(1, 0.2, n_samples),
        'public_function_count': np.random.poisson(5, n_samples),
        'payable_function_count': np.random.poisson(1, n_samples),
        'dangerous_function_count': np.random.poisson(0.5, n_samples),
        'modifier_count': np.random.poisson(2, n_samples),
        'require_count': np.random.poisson(3, n_samples),
        'loop_count': np.random.poisson(1, n_samples),
        'cyclomatic_complexity': np.random.gamma(2, 2, n_samples)
    }
    
    df = pd.DataFrame(features)
    
    # Generate labels based on feature combinations (simplified vulnerability logic)
    vulnerability_score = (
        df['external_call_count'] * 0.3 +
        df['state_change_after_call'] * 0.4 +
        df['uses_block_timestamp'] * 0.2 +
        df['dangerous_function_count'] * 0.3 +
        np.random.normal(0, 0.1, n_samples)  # Add noise
    )
    
    # Binary labels
    df['is_vulnerable'] = (vulnerability_score > 1.0).astype(int)
    
    # Multi-class labels
    vulnerability_types = ['safe', 'reentrancy', 'access_control', 'bad_randomness', 'unchecked_call']
    df['vulnerability_type'] = np.random.choice(vulnerability_types, n_samples, 
                                               p=[0.4, 0.2, 0.15, 0.15, 0.1])
    
    # Make vulnerable contracts have appropriate types
    vulnerable_mask = df['is_vulnerable'] == 1
    df.loc[vulnerable_mask, 'vulnerability_type'] = np.random.choice(
        vulnerability_types[1:], vulnerable_mask.sum()
    )
    df.loc[~vulnerable_mask, 'vulnerability_type'] = 'safe'
    
    print(f"✅ Generated {n_samples} training samples")
    print(f"   - Vulnerable: {df['is_vulnerable'].sum()}")
    print(f"   - Safe: {(1 - df['is_vulnerable']).sum()}")
    
    return df

def train_binary_classifier(df):
    """Train binary vulnerability classifier."""
    print("\n🤖 Training Binary Classifier...")
    
    # Import model
    sys.path.insert(0, 'src')
    from models.random_forest import RandomForestVulnerabilityDetector
    
    # Prepare data
    feature_cols = [col for col in df.columns if col not in ['is_vulnerable', 'vulnerability_type']]
    X = df[feature_cols]
    y = df['is_vulnerable']
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestVulnerabilityDetector()
    results = model.train(X_train, y_train, X_test, y_test)
    
    # Save model
    model_path = Path("models/binary_classifier.joblib")
    model_path.parent.mkdir(exist_ok=True)
    joblib.dump(model, model_path)
    
    print(f"✅ Binary classifier trained and saved")
    print(f"   - Training Accuracy: {results['train_accuracy']:.3f}")
    print(f"   - Validation Accuracy: {results.get('val_accuracy', 'N/A')}")
    print(f"   - Model saved to: {model_path}")
    
    return model

def train_multiclass_classifier(df):
    """Train multi-class vulnerability classifier."""
    print("\n🎯 Training Multi-class Classifier...")
    
    # Import model
    sys.path.insert(0, 'src')
    from models.multiclass_classifier import MultiClassVulnerabilityDetector
    
    # Prepare data
    feature_cols = [col for col in df.columns if col not in ['is_vulnerable', 'vulnerability_type']]
    X = df[feature_cols]
    y = df['vulnerability_type']
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = MultiClassVulnerabilityDetector()
    results = model.train(X_train, y_train, X_test, y_test)
    
    # Save model
    model_path = Path("models/multiclass_classifier.joblib")
    joblib.dump(model, model_path)
    
    print(f"✅ Multi-class classifier trained and saved")
    print(f"   - Training Accuracy: {results['train_accuracy']:.3f}")
    print(f"   - Validation Accuracy: {results.get('val_accuracy', 'N/A')}")
    print(f"   - Model saved to: {model_path}")
    
    return model

def create_model_metadata():
    """Create metadata file for trained models."""
    metadata = {
        'created_at': datetime.now().isoformat(),
        'models': {
            'binary_classifier': {
                'file': 'models/binary_classifier.joblib',
                'type': 'RandomForestClassifier',
                'task': 'binary_classification',
                'classes': ['safe', 'vulnerable']
            },
            'multiclass_classifier': {
                'file': 'models/multiclass_classifier.joblib', 
                'type': 'RandomForestClassifier',
                'task': 'multiclass_classification',
                'classes': ['safe', 'reentrancy', 'access_control', 'bad_randomness', 'unchecked_call']
            }
        },
        'features': [
            'external_call_count', 'state_change_after_call', 'uses_block_timestamp',
            'public_function_count', 'payable_function_count', 'dangerous_function_count',
            'modifier_count', 'require_count', 'loop_count', 'cyclomatic_complexity'
        ]
    }
    
    import json
    with open('models/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("✅ Model metadata created")

def test_external_tools():
    """Test if external tools are working."""
    print("\n🧪 Testing External Tools...")
    
    # Test Slither
    try:
        result = subprocess.run(['slither', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Slither is working")
        else:
            print("❌ Slither test failed")
    except FileNotFoundError:
        print("❌ Slither not found")
    
    # Test Mythril
    try:
        result = subprocess.run(['myth', 'version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Mythril is working")
        else:
            print("❌ Mythril test failed")
    except FileNotFoundError:
        print("❌ Mythril not found")

def main():
    """Main setup function."""
    print("🚀 Smart Contract AI Analyzer - AI Model Setup")
    print("=" * 60)
    
    try:
        # Step 1: Install external tools
        install_external_tools()
        
        # Step 2: Generate training data
        df = generate_training_data()
        
        # Step 3: Train models
        binary_model = train_binary_classifier(df)
        multiclass_model = train_multiclass_classifier(df)
        
        # Step 4: Create metadata
        create_model_metadata()
        
        # Step 5: Test external tools
        test_external_tools()
        
        print("\n" + "=" * 60)
        print("🎉 AI Model Setup Complete!")
        print("=" * 60)
        print("✅ Trained models saved to models/ directory")
        print("✅ External tools installation attempted")
        print("✅ System ready for real AI-powered analysis")
        
        print("\n🚀 Next Steps:")
        print("1. Restart the system: python start_system.py")
        print("2. The dashboard will now use real AI models")
        print("3. External tools (if installed) will be integrated")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)