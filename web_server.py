#!/usr/bin/env python3
"""
Web Server for Meme Coin Trading System
"""

from flask import Flask, jsonify, render_template
import subprocess
import threading
import time
from datetime import datetime
import json
import os

app = Flask(__name__)

# Global variables to track trading status
trading_process = None
trading_status = {"running": False, "last_update": None}

@app.route('/health')
def health_check():
    return jsonify({
        "status": "healthy", 
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "trading_active": trading_status["running"]
    })

@app.route('/')
def dashboard():
    return jsonify({
        "message": "Meme Coin Trading System - Web Interface",
        "status": "running",
        "trading_bot_status": "active" if trading_status["running"] else "stopped",
        "last_update": trading_status["last_update"],
        "endpoints": {
            "health": "/health",
            "start_trading": "/api/start_trading",
            "stop_trading": "/api/stop_trading",
            "trading_status": "/api/status",
            "run_optimized_session": "/api/run_optimized_session"
        }
    })

@app.route('/api/status')
def get_status():
    return jsonify({
        "trading_active": trading_status["running"],
        "last_update": trading_status["last_update"],
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/start_trading')
def start_trading():
    global trading_process, trading_status
    
    if trading_status["running"]:
        return jsonify({"message": "Trading already running", "status": "running"})
    
    try:
        # Start the trading bot in background
        trading_process = subprocess.Popen(
            ["python", "enhanced_simple_trader.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        trading_status["running"] = True
        trading_status["last_update"] = datetime.now().isoformat()
        
        return jsonify({
            "message": "Trading bot started successfully",
            "status": "started",
            "pid": trading_process.pid
        })
    except Exception as e:
        return jsonify({"error": f"Failed to start trading: {str(e)}"})

@app.route('/api/stop_trading')
def stop_trading():
    global trading_process, trading_status
    
    if not trading_status["running"]:
        return jsonify({"message": "Trading not running", "status": "stopped"})
    
    try:
        if trading_process:
            trading_process.terminate()
            trading_process.wait(timeout=10)
        
        trading_status["running"] = False
        trading_status["last_update"] = datetime.now().isoformat()
        
        return jsonify({
            "message": "Trading bot stopped successfully",
            "status": "stopped"
        })
    except Exception as e:
        return jsonify({"error": f"Failed to stop trading: {str(e)}"})

@app.route('/api/run_optimized_session')
def run_optimized_session():
    try:
        # Run optimized trading session
        result = subprocess.run(
            ["python", "optimized_paper_trading.py"],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0:
            return jsonify({
                "status": "completed",
                "message": "Optimized trading session completed",
                "output": result.stdout[-500:]  # Last 500 characters
            })
        else:
            return jsonify({
                "status": "error",
                "message": "Optimized trading session failed",
                "error": result.stderr[-500:]
            })
    except subprocess.TimeoutExpired:
        return jsonify({
            "status": "timeout",
            "message": "Optimized trading session timed out"
        })
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    print("🌐 Starting Meme Coin Trading Web Interface")
    print("📊 Access dashboard at: http://localhost:5000")
    print("🔧 API endpoints available at /api/*")
    app.run(debug=False, host='0.0.0.0', port=5000)
