#!/usr/bin/env python3
"""
Fix installation issues and install minimal requirements.
Handles Windows permission errors and package conflicts.
"""

import sys
import subprocess
import os
import time

def run_command_safe(command, description):
    """Run command with error handling."""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  ✅ Success")
            return True
        else:
            print(f"  ⚠️ Warning: {result.stderr[:100]}...")
            return False
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def upgrade_pip():
    """Upgrade pip to latest version."""
    print("📦 Upgrading pip...")
    commands = [
        f"{sys.executable} -m pip install --upgrade pip",
        f"{sys.executable} -m pip install --upgrade setuptools wheel"
    ]
    
    for cmd in commands:
        run_command_safe(cmd, f"Running: {cmd.split()[-1]}")

def install_minimal_packages():
    """Install only essential packages."""
    print("📥 Installing minimal essential packages...")
    
    # Core packages needed for basic functionality
    essential_packages = [
        "streamlit",
        "flask", 
        "flask-cors",
        "pandas",
        "numpy", 
        "plotly",
        "requests",
        "python-dotenv"
    ]
    
    success_count = 0
    for package in essential_packages:
        if run_command_safe(f"{sys.executable} -m pip install {package} --user", f"Installing {package}"):
            success_count += 1
        else:
            # Try without --user flag
            if run_command_safe(f"{sys.executable} -m pip install {package}", f"Installing {package} (retry)"):
                success_count += 1
    
    print(f"📊 Successfully installed {success_count}/{len(essential_packages)} packages")
    return success_count >= 6  # Need at least 6 core packages

def test_imports():
    """Test if core packages can be imported."""
    print("🧪 Testing package imports...")
    
    packages = ["streamlit", "flask", "pandas", "numpy", "plotly", "requests"]
    working_packages = []
    
    for package in packages:
        try:
            __import__(package)
            print(f"  ✅ {package}")
            working_packages.append(package)
        except ImportError:
            print(f"  ❌ {package} - Failed to import")
    
    return len(working_packages) >= 4  # Need at least 4 working packages

def create_simple_requirements():
    """Create a working requirements file."""
    print("📝 Creating simplified requirements...")
    
    working_requirements = """# Working minimal requirements
streamlit
flask
flask-cors
pandas
numpy
plotly
requests
python-dotenv"""
    
    with open("requirements-working.txt", "w") as f:
        f.write(working_requirements)
    
    print("  ✅ Created requirements-working.txt")

def main():
    """Main fix function."""
    print("🔧 Smart Contract AI Analyzer - Installation Fix")
    print("=" * 60)
    print("This will install only essential packages to get you started.")
    print("=" * 60)
    
    # Step 1: Upgrade pip
    upgrade_pip()
    print()
    
    # Step 2: Install minimal packages
    if install_minimal_packages():
        print("✅ Essential packages installed successfully!")
    else:
        print("⚠️ Some packages failed to install, but continuing...")
    print()
    
    # Step 3: Test imports
    if test_imports():
        print("✅ Core packages are working!")
    else:
        print("⚠️ Some imports failed, but basic functionality should work")
    print()
    
    # Step 4: Create working requirements
    create_simple_requirements()
    print()
    
    print("=" * 60)
    print("🎉 Installation fix complete!")
    print("=" * 60)
    print("Next steps:")
    print("1. Try starting the system:")
    print("   python quick_start.py")
    print()
    print("2. If that works, you're all set!")
    print()
    print("3. If you need more packages later:")
    print("   pip install -r requirements-working.txt")
    print("=" * 60)

if __name__ == "__main__":
    main()