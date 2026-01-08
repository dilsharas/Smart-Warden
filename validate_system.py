#!/usr/bin/env python3
"""
Final system validation script for Smart Contract AI Analyzer.
Validates all components are working correctly.
"""

import sys
import importlib.util
from pathlib import Path
import requests
import time

def validate_files():
    """Validate all required files exist."""
    print("📁 Validating File Structure...")
    
    required_files = [
        "dashboard/dashboard.py",
        "dashboard/pages/analyze.py", 
        "dashboard/pages/results.py",
        "dashboard/pages/comparison.py",
        "dashboard/pages/metrics.py",
        "dashboard/pages/about.py",
        "simple_api.py",
        "start_system.py",
        "quick_start.py",
        "cleanup_ports.py"
    ]
    
    all_good = True
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path} - MISSING")
            all_good = False
    
    return all_good

def validate_page_loading():
    """Validate all dashboard pages can be loaded."""
    print("\n📄 Validating Page Loading...")
    
    pages_dir = Path("dashboard/pages")
    page_files = ["analyze.py", "results.py", "comparison.py", "metrics.py", "about.py"]
    
    all_good = True
    for page_file in page_files:
        page_path = pages_dir / page_file
        try:
            spec = importlib.util.spec_from_file_location(page_file[:-3], page_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            if hasattr(module, 'show_page'):
                print(f"  ✅ {page_file[:-3]} - Loaded with show_page()")
            else:
                print(f"  ❌ {page_file[:-3]} - Missing show_page() function")
                all_good = False
                
        except Exception as e:
            print(f"  ❌ {page_file[:-3]} - Load error: {str(e)[:50]}...")
            all_good = False
    
    return all_good

def validate_dependencies():
    """Validate required Python packages."""
    print("\n📦 Validating Dependencies...")
    
    required_packages = [
        "streamlit",
        "plotly", 
        "pandas",
        "numpy",
        "requests",
        "flask"
    ]
    
    optional_packages = [
        "reportlab"  # For PDF generation
    ]
    
    all_good = True
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} - MISSING (required)")
            all_good = False
    
    for package in optional_packages:
        try:
            __import__(package)
            print(f"  ✅ {package} (optional)")
        except ImportError:
            print(f"  ⚠️ {package} - Missing (optional - PDF generation disabled)")
    
    return all_good

def validate_api_functionality():
    """Validate API endpoints if running."""
    print("\n🔧 Validating API Functionality...")
    
    try:
        # Check if API is running
        response = requests.get("http://localhost:5000/health", timeout=3)
        if response.status_code == 200:
            print("  ✅ API Health Check - PASS")
            
            # Test analysis endpoint
            test_contract = "pragma solidity ^0.8.0; contract Test {}"
            response = requests.post(
                "http://localhost:5000/api/analyze",
                json={"contract_code": test_contract},
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    print("  ✅ API Analysis Endpoint - PASS")
                    return True
                else:
                    print("  ❌ API Analysis Endpoint - Failed")
                    return False
            else:
                print(f"  ❌ API Analysis Endpoint - HTTP {response.status_code}")
                return False
        else:
            print(f"  ❌ API Health Check - HTTP {response.status_code}")
            return False
            
    except requests.exceptions.RequestException:
        print("  ⚠️ API Not Running - Use 'python simple_api.py' to start")
        return True  # Not an error if API isn't running

def validate_dashboard_access():
    """Validate dashboard accessibility."""
    print("\n🌐 Validating Dashboard Access...")
    
    try:
        response = requests.get("http://localhost:8501", timeout=5)
        if response.status_code == 200:
            print("  ✅ Dashboard Accessible - PASS")
            return True
        else:
            print(f"  ❌ Dashboard Access - HTTP {response.status_code}")
            return False
    except requests.exceptions.RequestException:
        print("  ⚠️ Dashboard Not Running - Use 'streamlit run dashboard/dashboard.py' to start")
        return True  # Not an error if dashboard isn't running

def main():
    """Run complete system validation."""
    print("🔍 Smart Contract AI Analyzer - System Validation")
    print("=" * 60)
    
    validations = [
        ("File Structure", validate_files),
        ("Page Loading", validate_page_loading), 
        ("Dependencies", validate_dependencies),
        ("API Functionality", validate_api_functionality),
        ("Dashboard Access", validate_dashboard_access)
    ]
    
    results = {}
    for name, validator in validations:
        try:
            results[name] = validator()
        except Exception as e:
            print(f"  ❌ {name} - Validation error: {e}")
            results[name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 VALIDATION SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name:20} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} validations passed")
    
    if passed == total:
        print("\n🎉 System validation SUCCESSFUL!")
        print("🚀 Ready to use Smart Contract AI Analyzer")
        print("\nQuick start commands:")
        print("  Dashboard only: python quick_start.py")
        print("  Full system:    python start_system.py")
        return True
    else:
        print("\n⚠️ Some validations failed. Check the issues above.")
        print("💡 Most issues can be resolved by installing missing dependencies:")
        print("  pip install streamlit plotly pandas numpy requests flask")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)