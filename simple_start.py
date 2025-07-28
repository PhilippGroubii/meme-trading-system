#!/usr/bin/env python3
"""
Simple script to manually start the API bridge
Just upload this file and run it manually
"""

import subprocess
import os
import time

def start_api_bridge():
    print("🚀 Starting API Bridge on port 8080...")
    
    # Change to the correct directory
    os.chdir('/opt/meme-trading')
    
    # Pull latest code
    print("📥 Pulling latest code...")
    subprocess.run(['git', 'pull', 'origin', 'main'], check=True)
    
    # Install dependencies
    print("📦 Installing dependencies...")
    subprocess.run(['venv/bin/pip', 'install', 'psutil'], check=False)
    
    # Stop any existing API bridge
    print("🛑 Stopping existing processes...")
    subprocess.run(['pkill', '-f', 'api_bridge.py'], check=False)
    
    # Start the API bridge
    print("▶️ Starting API bridge...")
    subprocess.Popen(['venv/bin/python', 'api_bridge.py'])
    
    time.sleep(3)
    
    # Test if it's working
    print("🧪 Testing API bridge...")
    result = subprocess.run(['curl', '-s', 'http://localhost:8080/health'], 
                          capture_output=True, text=True)
    
    if 'api_bridge_healthy' in result.stdout:
        print("✅ API Bridge started successfully!")
        print("🌐 Access at: http://54.162.71.225:8080/health")
    else:
        print("❌ API Bridge failed to start")
    
    return True

if __name__ == '__main__':
    start_api_bridge()
