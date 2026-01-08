#!/usr/bin/env python3
"""
Start dashboard only - bypasses API issues.
This is the most reliable way to start the system.
"""

import subprocess
import sys
import time
from pathlib import Path

def main():
    """Start dashboard only."""
    print("🚀 Smart Contract AI Analyzer - Dashboard Only")
    print("=" * 50)
    print("🌐 Starting Dashboard at: http://localhost:8501")
    print("⚠️ API Backend not started - using mock analysis")
    print("🛑 Press Ctrl+C to stop")
    print("=" * 50)
    
    # Check if dashboard exists
    if not Path("dashboard/dashboard.py").exists():
        print("❌ dashboard/dashboard.py not found!")
        print("💡 Make sure you're in the project root directory")
        return False
    
    try:
        # Start dashboard with streamlit
        result = subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "dashboard/dashboard.py",
            "--server.address", "localhost",
            "--server.port", "8501",
            "--browser.gatherUsageStats", "false",
            "--server.headless", "true"
        ])
        
        return result.returncode == 0
        
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
        return True
    except FileNotFoundError:
        print("❌ Streamlit not found!")
        print("💡 Install with: pip install streamlit")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Failed to start dashboard")
        print("💡 Try installing streamlit: pip install streamlit")
        sys.exit(1)
    else:
        print("\n✅ Dashboard session ended")