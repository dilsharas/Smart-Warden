#!/usr/bin/env python3
"""
Cleanup script to free ports used by the Smart Contract AI Analyzer.
"""

import subprocess
import sys
import time

def kill_process_on_port(port):
    """Kill process running on specified port (Windows)."""
    try:
        print(f"🔍 Checking port {port}...")
        
        # Find process using the port
        result = subprocess.run(
            f'netstat -ano | findstr :{port}',
            shell=True, capture_output=True, text=True
        )
        
        if result.stdout:
            lines = result.stdout.strip().split('\n')
            killed_any = False
            
            for line in lines:
                if f':{port}' in line and 'LISTENING' in line:
                    parts = line.split()
                    if len(parts) >= 5:
                        pid = parts[-1]
                        print(f"🔪 Killing process {pid} on port {port}")
                        kill_result = subprocess.run(
                            f'taskkill /F /PID {pid}', 
                            shell=True, capture_output=True
                        )
                        if kill_result.returncode == 0:
                            print(f"✅ Process {pid} killed successfully")
                            killed_any = True
                        else:
                            print(f"❌ Failed to kill process {pid}")
            
            if killed_any:
                time.sleep(2)
                return True
            else:
                print(f"ℹ️ No processes found on port {port}")
                return True
        else:
            print(f"✅ Port {port} is free")
            return True
            
    except Exception as e:
        print(f"❌ Error checking port {port}: {e}")
        return False

def main():
    """Clean up ports used by the system."""
    print("🧹 Smart Contract AI Analyzer - Port Cleanup")
    print("=" * 50)
    
    ports_to_clean = [5000, 8501]
    
    for port in ports_to_clean:
        kill_process_on_port(port)
    
    print("\n✅ Port cleanup complete!")
    print("🚀 You can now run: python start_system.py")

if __name__ == "__main__":
    main()