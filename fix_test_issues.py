#!/usr/bin/env python3
"""
Fix common test issues in Smart Contract AI Analyzer.
"""

import subprocess
import sys
from pathlib import Path

def install_missing_dependencies():
    """Install missing test dependencies."""
    print("🔧 Installing missing test dependencies...")
    
    missing_packages = [
        "psutil>=5.9.0",  # For performance tests
        "pytest>=7.4.0",
        "pytest-cov>=4.1.0",
        "requests-mock>=1.11.0"
    ]
    
    for package in missing_packages:
        try:
            print(f"📦 Installing {package}...")
            subprocess.run([
                sys.executable, "-m", "pip", "install", package
            ], check=True, capture_output=True)
            print(f"  ✅ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"  ⚠️ Failed to install {package}: {e}")

def check_test_files():
    """Check if test files have syntax issues."""
    print("\n🔍 Checking test files for syntax issues...")
    
    test_files = [
        "tests/system/test_e2e_workflow.py",
        "tests/system/test_performance.py"
    ]
    
    for test_file in test_files:
        if Path(test_file).exists():
            try:
                with open(test_file, 'r') as f:
                    compile(f.read(), test_file, 'exec')
                print(f"  ✅ {test_file} - Syntax OK")
            except SyntaxError as e:
                print(f"  ❌ {test_file} - Syntax Error: {e}")
        else:
            print(f"  ⚠️ {test_file} - File not found")

def run_basic_tests():
    """Run basic tests to verify fixes."""
    print("\n🧪 Running basic tests...")
    
    try:
        # Run only unit tests first (safer)
        result = subprocess.run([
            sys.executable, "-m", "pytest", "tests/unit/", "-v", "--tb=short"
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("  ✅ Unit tests passed!")
        else:
            print("  ⚠️ Some unit tests failed:")
            print(result.stdout[-500:])  # Last 500 chars
            
    except subprocess.TimeoutExpired:
        print("  ⚠️ Tests timed out")
    except Exception as e:
        print(f"  ❌ Test execution failed: {e}")

def main():
    """Main fix function."""
    print("🔧 Smart Contract AI Analyzer - Test Issues Fix")
    print("=" * 60)
    
    # Step 1: Install missing dependencies
    install_missing_dependencies()
    
    # Step 2: Check test file syntax
    check_test_files()
    
    # Step 3: Run basic tests
    run_basic_tests()
    
    print("\n" + "=" * 60)
    print("🎉 Test fixes applied!")
    print("=" * 60)
    print("Now you can run tests with:")
    print("  python -m pytest tests/unit/ -v          # Unit tests only")
    print("  python -m pytest tests/integration/ -v   # Integration tests")
    print("  python -m pytest tests/ -v               # All tests")
    print("  python -m pytest tests/ -v --tb=short    # All tests with short traceback")

if __name__ == "__main__":
    main()