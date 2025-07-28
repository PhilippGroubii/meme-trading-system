#!/usr/bin/env python3
"""
Auto-deployment script that can be triggered via HTTP
"""

from flask import Flask, jsonify
import subprocess
import os
import threading
import time

app = Flask(__name__)

@app.route('/deploy', methods=['POST', 'GET'])
def deploy():
    """Deploy latest code from GitHub"""
    try:
        # Run deployment in background
        def run_deployment():
            os.chdir('/opt/meme-trading')
            
            # Pull latest code
            subprocess.run(['git', 'pull', 'origin', 'main'], check=True)
            
            # Install dependencies
            subprocess.run(['venv/bin/pip', 'install', '-r', 'requirements.txt'], check=True)
            
            # Stop existing web_server
            subprocess.run(['pkill', '-f', 'web_server.py'], check=False)
            
            # Start new web_server
            subprocess.Popen(['venv/bin/python', 'web_server.py'])
            
        thread = threading.Thread(target=run_deployment)
        thread.start()
        
        return jsonify({"status": "deployment_started", "message": "Deploying latest code..."})
        
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/health')
def health():
    return jsonify({"status": "auto_deploy_healthy"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
