#!/usr/bin/env python3
"""
Pre-deployment checklist and setup script.
"""

import sys
import subprocess
import importlib.util
from pathlib import Path

def check_python_version():
    """Check Python version."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"  ✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
        return True
    else:
        print(f"  ❌ Python {version.major}.{version.minor}.{version.micro} - Need Python 3.8+")
        return False

def check_pip():
    """Check if pip is available."""
    print("📦 Checking pip...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "--version"], 
                      capture_output=True, check=True)
        print("  ✅ pip is available")
        return True
    except subprocess.CalledProcessError:
        print("  ❌ pip not available")
        return False

def install_requirements():
    """Install required packages."""
    print("📥 Installing requirements...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True)
        print("  ✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ❌ Failed to install requirements: {e}")
        return False

def check_key_packages():
    """Check if key packages are installed."""
    print("🔍 Checking key packages...")
    packages = ["streamlit", "flask", "pandas", "numpy", "plotly", "requests"]
    
    all_good = True
    for package in packages:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} - Missing")
            all_good = False
    
    return all_good

def create_directories():
    """Create necessary directories."""
    print("📁 Creating directories...")
    dirs = ["logs", "models", "data/processed", "results", "temp"]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"  ✅ {dir_path}")
    
    return True

def main():
    """Run pre-deployment checklist."""
    print("🔧 Smart Contract AI Analyzer - Pre-Deployment Checklist")
    print("=" * 60)
    
    checks = [
        ("Python Version", check_python_version),
        ("Pip Availability", check_pip),
        ("Install Requirements", install_requirements),
        ("Key Packages", check_key_packages),
        ("Create Directories", create_directories)
    ]
    
    all_passed = True
    for name, check_func in checks:
        try:
            result = check_func()
            if not result:
                all_passed = False
        except Exception as e:
            print(f"  ❌ {name} failed: {e}")
            all_passed = False
        print()
    
    if all_passed:
        print("🎉 Pre-deployment checklist PASSED!")
        print("🚀 Ready to deploy the system")
        return True
    else:
        print("❌ Some checks failed. Please fix the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)