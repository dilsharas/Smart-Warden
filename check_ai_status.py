#!/usr/bin/env python3
"""
Check AI integration status for Smart Contract AI Analyzer.
"""

import sys
import subprocess
from pathlib import Path
import json

def check_trained_models():
    """Check if AI models are trained and available."""
    print("🤖 Checking AI Models...")
    
    models_dir = Path("models")
    if not models_dir.exists():
        print("  ❌ Models directory not found")
        return False
    
    # Check for model files
    model_files = [
        "binary_classifier.joblib",
        "multiclass_classifier.joblib",
        "metadata.json"
    ]
    
    all_present = True
    for model_file in model_files:
        model_path = models_dir / model_file
        if model_path.exists():
            print(f"  ✅ {model_file}")
        else:
            print(f"  ❌ {model_file} - Missing")
            all_present = False
    
    if all_present:
        # Check metadata
        try:
            with open(models_dir / "metadata.json", 'r') as f:
                metadata = json.load(f)
            print(f"  ✅ Models created: {metadata.get('created_at', 'Unknown')}")
            print(f"  ✅ Features: {len(metadata.get('features', []))}")
        except Exception as e:
            print(f"  ⚠️ Metadata error: {e}")
    
    return all_present

def check_external_tools():
    """Check if external tools are available."""
    print("\n🛠️ Checking External Tools...")
    
    tools_available = {}
    
    # Check native tools first
    try:
        sys.path.insert(0, 'src')
        from integration.native_tools import check_native_tools_availability
        from integration.docker_tools import check_tools_availability
        
        native_status = check_native_tools_availability()
        docker_status = check_tools_availability()
        
        # Check Slither
        if docker_status.get('slither') and docker_status.get('docker'):
            print("  ✅ Slither Static Analysis - Docker available")
            tools_available['slither'] = True
        elif native_status.get('native_slither'):
            print("  ✅ Slither Static Analysis - Native implementation")
            tools_available['slither'] = True
        else:
            print("  ❌ Slither Static Analysis - Not available")
            tools_available['slither'] = False
        
        # Check Mythril
        if docker_status.get('mythril') and docker_status.get('docker'):
            print("  ✅ Mythril Symbolic Execution - Docker available")
            tools_available['mythril'] = True
        elif native_status.get('native_mythril'):
            print("  ✅ Mythril Symbolic Execution - Native implementation")
            tools_available['mythril'] = True
        else:
            print("  ❌ Mythril Symbolic Execution - Not available")
            tools_available['mythril'] = False
            
    except Exception as e:
        print(f"  ❌ Tool check failed: {e}")
        tools_available['slither'] = False
        tools_available['mythril'] = False
    
    return tools_available

def check_feature_extraction():
    """Check if feature extraction is working."""
    print("\n🔍 Checking Feature Extraction...")
    
    try:
        sys.path.insert(0, 'src')
        from features.feature_extractor import SolidityFeatureExtractor
        
        # Test with simple contract
        test_contract = "pragma solidity ^0.8.0; contract Test { function test() public {} }"
        
        extractor = SolidityFeatureExtractor()
        features = extractor.extract_features(test_contract)
        
        print(f"  ✅ Feature extraction working")
        print(f"  ✅ Extracted {len(features)} features")
        return True
        
    except Exception as e:
        print(f"  ❌ Feature extraction failed: {e}")
        return False

def check_model_loading():
    """Check if model loading is working."""
    print("\n📦 Checking Model Loading...")
    
    try:
        sys.path.insert(0, 'src')
        from models.model_loader import ModelLoader, predict_vulnerability
        
        loader = ModelLoader()
        models = loader.load_all_models()
        
        if models and models.get('count', 0) > 0:
            print("  ✅ Model loading working")
            print(f"  ✅ Loaded {models['count']} models")
            
            # Test prediction
            sample_features = {'lines_of_code': 10, 'function_count': 1}
            result = predict_vulnerability(sample_features)
            
            if result.get('available'):
                print("  ✅ Model prediction working")
                if result.get('binary_prediction'):
                    print("  ✅ Binary classification available")
                if result.get('multiclass_prediction'):
                    print("  ✅ Multi-class classification available")
            else:
                print("  ⚠️ Models loaded but prediction not available")
            
            return True
        else:
            print("  ❌ No models loaded")
            return False
            
    except Exception as e:
        print(f"  ❌ Model loading failed: {e}")
        return False

def check_api_integration():
    """Check if API can use AI models."""
    print("\n🔗 Checking API Integration...")
    
    try:
        # Test the analysis function
        sys.path.insert(0, '.')
        from simple_api import analyze_contract_with_ai
        
        test_contract = "pragma solidity ^0.8.0; contract Test { function test() public {} }"
        result = analyze_contract_with_ai(test_contract)
        
        if result.get('success'):
            method = result.get('analysis_method', 'Unknown')
            print(f"  ✅ API analysis working")
            print(f"  ✅ Analysis method: {method}")
            return True
        else:
            print(f"  ❌ API analysis failed")
            return False
            
    except Exception as e:
        print(f"  ❌ API integration failed: {e}")
        return False

def main():
    """Run complete AI status check."""
    print("🔍 Smart Contract AI Analyzer - AI Integration Status")
    print("=" * 60)
    
    checks = [
        ("Trained Models", check_trained_models),
        ("External Tools", check_external_tools),
        ("Feature Extraction", check_feature_extraction),
        ("Model Loading", check_model_loading),
        ("API Integration", check_api_integration)
    ]
    
    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"  ❌ {name} check failed: {e}")
            results[name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 AI INTEGRATION STATUS")
    print("=" * 60)
    
    ai_models_ready = results.get("Trained Models", False)
    external_tools_ready = any([
        results.get("External Tools", {}).get("slither", False),
        results.get("External Tools", {}).get("myth", False)
    ]) if isinstance(results.get("External Tools"), dict) else False
    
    print(f"🤖 AI Models:        {'✅ READY' if ai_models_ready else '❌ NOT READY'}")
    print(f"🛠️ External Tools:   {'✅ READY' if external_tools_ready else '❌ NOT READY'}")
    print(f"🔗 Integration:      {'✅ WORKING' if results.get('API Integration') else '❌ NOT WORKING'}")
    
    if ai_models_ready:
        print("\n🎉 AI models are trained and ready!")
        print("   The system will use real AI-powered analysis")
    else:
        print("\n⚠️ AI models not ready")
        print("   Run: python setup_ai_models.py")
        print("   The system will use pattern-based analysis")
    
    if not external_tools_ready:
        print("\n💡 To enable external tools:")
        print("   pip install slither-analyzer mythril")
    
    # Count successful components (handle mixed types)
    successful_components = 0
    for result in results.values():
        if isinstance(result, bool):
            successful_components += int(result)
        elif isinstance(result, dict):
            successful_components += int(any(result.values()))
        else:
            successful_components += int(bool(result))
    
    print(f"\n📈 Overall Status: {successful_components}/{len(results)} components ready")
    
    return ai_models_ready

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)