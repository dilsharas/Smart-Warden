#!/usr/bin/env python3
"""
Complete system startup script for Smart Contract AI Analyzer.
This script starts both the API backend and the dashboard frontend.
"""

import subprocess
import sys
import time
import os
import signal
import threading
from pathlib import Path
import requests

class SystemManager:
    def __init__(self):
        self.api_process = None
        self.dashboard_process = None
        self.running = True
    
    def check_port(self, port):
        """Check if a port is available."""
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) != 0
    
    def kill_process_on_port(self, port):
        """Kill process running on specified port (Windows)."""
        try:
            import subprocess
            # Find process using the port
            result = subprocess.run(
                f'netstat -ano | findstr :{port}',
                shell=True, capture_output=True, text=True
            )
            
            if result.stdout:
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if f':{port}' in line and 'LISTENING' in line:
                        parts = line.split()
                        if len(parts) >= 5:
                            pid = parts[-1]
                            print(f"🔪 Killing process {pid} on port {port}")
                            subprocess.run(f'taskkill /F /PID {pid}', shell=True)
                            time.sleep(2)
                            return True
            return False
        except Exception as e:
            print(f"⚠️ Could not kill process on port {port}: {e}")
            return False
    
    def wait_for_api(self, timeout=30):
        """Wait for API to be ready."""
        print("⏳ Waiting for API to start...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                response = requests.get("http://localhost:5000/health", timeout=2)
                if response.status_code == 200:
                    print("✅ API is ready!")
                    return True
            except:
                pass
            time.sleep(1)
        
        print("⚠️ API startup timeout, continuing anyway...")
        return False
    
    def start_api(self):
        """Start the API backend."""
        print("🚀 Starting API Backend...")
        
        if not self.check_port(5000):
            print("⚠️ Port 5000 is already in use")
            print("🔄 Attempting to free port 5000...")
            if self.kill_process_on_port(5000):
                print("✅ Port 5000 freed")
                time.sleep(2)
            else:
                print("❌ Could not free port 5000")
                return False
        
        try:
            self.api_process = subprocess.Popen([
                sys.executable, "simple_api.py"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Wait a moment for startup
            time.sleep(2)
            
            if self.api_process.poll() is None:
                print("✅ API Backend started successfully")
                return True
            else:
                print("❌ API Backend failed to start")
                return False
                
        except Exception as e:
            print(f"❌ Failed to start API: {e}")
            return False
    
    def start_dashboard(self):
        """Start the dashboard frontend."""
        print("🌐 Starting Dashboard...")
        
        if not self.check_port(8501):
            print("⚠️ Port 8501 is already in use")
            print("🔄 Attempting to free port 8501...")
            if self.kill_process_on_port(8501):
                print("✅ Port 8501 freed")
                time.sleep(2)
            else:
                print("❌ Could not free port 8501")
                return False
        
        try:
            self.dashboard_process = subprocess.Popen([
                sys.executable, "-m", "streamlit", "run", 
                "dashboard/dashboard.py",
                "--server.address", "localhost",
                "--server.port", "8501",
                "--browser.gatherUsageStats", "false"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Wait a moment for startup
            time.sleep(3)
            
            if self.dashboard_process.poll() is None:
                print("✅ Dashboard started successfully")
                return True
            else:
                print("❌ Dashboard failed to start")
                return False
                
        except Exception as e:
            print(f"❌ Failed to start dashboard: {e}")
            return False
    
    def stop_services(self):
        """Stop all services."""
        print("\n🛑 Stopping services...")
        
        if self.api_process:
            self.api_process.terminate()
            try:
                self.api_process.wait(timeout=5)
                print("✅ API Backend stopped")
            except subprocess.TimeoutExpired:
                self.api_process.kill()
                print("🔪 API Backend force killed")
        
        if self.dashboard_process:
            self.dashboard_process.terminate()
            try:
                self.dashboard_process.wait(timeout=5)
                print("✅ Dashboard stopped")
            except subprocess.TimeoutExpired:
                self.dashboard_process.kill()
                print("🔪 Dashboard force killed")
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        print(f"\n📡 Received signal {signum}")
        self.running = False
        self.stop_services()
        sys.exit(0)
    
    def monitor_processes(self):
        """Monitor running processes."""
        while self.running:
            time.sleep(5)
            
            # Check API process
            if self.api_process and self.api_process.poll() is not None:
                print("⚠️ API Backend process died")
                self.running = False
                break
            
            # Check Dashboard process
            if self.dashboard_process and self.dashboard_process.poll() is not None:
                print("⚠️ Dashboard process died")
                self.running = False
                break
    
    def run(self):
        """Run the complete system."""
        print("🚀 Smart Contract AI Analyzer - System Startup")
        print("=" * 60)
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        # Check prerequisites
        if not Path("simple_api.py").exists():
            print("❌ simple_api.py not found!")
            return False
        
        if not Path("dashboard/dashboard.py").exists():
            print("❌ dashboard/dashboard.py not found!")
            return False
        
        # Start API Backend
        if not self.start_api():
            print("❌ Failed to start API backend")
            return False
        
        # Wait for API to be ready
        self.wait_for_api()
        
        # Start Dashboard
        if not self.start_dashboard():
            print("❌ Failed to start dashboard")
            self.stop_services()
            return False
        
        print("\n" + "=" * 60)
        print("🎉 System Started Successfully!")
        print("📡 API Backend: http://localhost:5000")
        print("🌐 Dashboard: http://localhost:8501")
        print("🔗 Health Check: http://localhost:5000/health")
        print("🛑 Press Ctrl+C to stop all services")
        print("=" * 60)
        
        # Monitor processes
        try:
            monitor_thread = threading.Thread(target=self.monitor_processes)
            monitor_thread.daemon = True
            monitor_thread.start()
            
            # Keep main thread alive
            while self.running:
                time.sleep(1)
                
        except KeyboardInterrupt:
            pass
        finally:
            self.stop_services()
        
        return True

def main():
    """Main function."""
    manager = SystemManager()
    success = manager.run()
    
    if not success:
        print("\n❌ System startup failed")
        sys.exit(1)
    else:
        print("\n✅ System shutdown complete")

if __name__ == "__main__":
    main()