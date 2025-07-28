#!/usr/bin/env python3
"""
Simple API bridge for trading bot control
Runs on port 8080 to avoid conflicts
"""

from flask import Flask, jsonify
import subprocess
import os
import signal
import psutil

app = Flask(__name__)

@app.route('/health')
def health():
    return jsonify({"status": "api_bridge_healthy", "port": 8080})

@app.route('/api/status')
def trading_status():
    # Check if enhanced_simple_trader.py is running
    running = False
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if 'enhanced_simple_trader.py' in ' '.join(proc.info['cmdline']):
                running = True
                break
        except:
            pass
    
    return jsonify({
        "trading_active": running,
        "message": "Enhanced Simple Trader status",
        "timestamp": "2025-07-28T00:45:00"
    })

@app.route('/api/start_trading')
def start_trading():
    try:
        # Start the enhanced simple trader
        os.chdir('/opt/meme-trading')
        proc = subprocess.Popen([
            'venv/bin/python', 'enhanced_simple_trader.py'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        return jsonify({
            "status": "started",
            "message": "Enhanced Simple Trader started",
            "pid": proc.pid
        })
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/api/stop_trading')
def stop_trading():
    try:
        # Stop enhanced_simple_trader.py processes
        subprocess.run(['pkill', '-f', 'enhanced_simple_trader.py'], check=False)
        return jsonify({
            "status": "stopped",
            "message": "Enhanced Simple Trader stopped"
        })
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/api/run_optimized_session')
def run_optimized():
    try:
        # Run optimized trading session
        result = subprocess.run([
            'venv/bin/python', 'optimized_paper_trading.py'
        ], capture_output=True, text=True, timeout=300, cwd='/opt/meme-trading')
        
        return jsonify({
            "status": "completed",
            "message": "Optimized session completed",
            "output": result.stdout[-500:] if result.stdout else "No output"
        })
    except subprocess.TimeoutExpired:
        return jsonify({"status": "timeout", "message": "Session timed out"})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
