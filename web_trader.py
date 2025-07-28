#!/usr/bin/env python3
from flask import Flask, jsonify, render_template_string
import psycopg2
import os
from datetime import datetime
from dotenv import load_dotenv
import threading
import time

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Database connection
def get_db_connection():
    return psycopg2.connect(os.getenv('DATABASE_URL'))

# Simple HTML template
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Meme Coin Trading System</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
        .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #333; text-align: center; }
        .status { padding: 15px; margin: 20px 0; border-radius: 5px; }
        .success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .info { background: #d1ecf1; color: #0c5460; border: 1px solid #bee5eb; }
        .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }
        .stat-card { background: #f8f9fa; padding: 20px; border-radius: 8px; text-align: center; }
        .stat-number { font-size: 2em; font-weight: bold; color: #007bff; }
        button { background: #007bff; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; margin: 5px; }
        button:hover { background: #0056b3; }
        .footer { text-align: center; margin-top: 30px; color: #666; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 Meme Coin Trading System</h1>
        
        <div class="status success">
            <strong>✅ System Status: ONLINE</strong><br>
            Server: {{ server_info }}<br>
            Database: Connected<br>
            Environment: {{ environment }}
        </div>

        <div class="stats">
            <div class="stat-card">
                <div class="stat-number">{{ total_trades }}</div>
                <div>Total Trades</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">${{ total_pnl }}</div>
                <div>Total P&L</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{{ uptime }}</div>
                <div>Uptime</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{{ last_update }}</div>
                <div>Last Update</div>
            </div>
        </div>

        <div class="status info">
            <strong>📊 Trading Bot Status</strong><br>
            Ready for deployment. Your meme coin trading system is configured and running on AWS.<br>
            Domain: <strong>{{ domain }}</strong>
        </div>

        <div style="text-align: center; margin: 30px 0;">
            <button onclick="location.reload()">🔄 Refresh Status</button>
            <button onclick="window.open('/health', '_blank')">🏥 Health Check</button>
            <button onclick="window.open('/api/trades', '_blank')">📈 View Trades</button>
        </div>

        <div class="footer">
            <p>Meme Coin Trading System v1.0 | Deployed on AWS | {{ current_time }}</p>
        </div>
    </div>
</body>
</html>
'''

# Routes
@app.route('/')
def dashboard():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get trade statistics
        cursor.execute("SELECT COUNT(*) FROM trades")
        total_trades = cursor.fetchone()[0]
        
        cursor.execute("SELECT COALESCE(SUM(profit_loss), 0) FROM trades")
        total_pnl = cursor.fetchone()[0] or 0
        
        cursor.close()
        conn.close()
        
        return render_template_string(HTML_TEMPLATE,
            server_info=f"AWS EC2 ({os.uname().nodename})",
            environment=os.getenv('ENVIRONMENT', 'production'),
            domain=os.getenv('DOMAIN', 'groubiiconsulting.com'),
            total_trades=total_trades,
            total_pnl=f"{float(total_pnl):.2f}",
            uptime="Running",
            last_update=datetime.now().strftime('%H:%M:%S'),
            current_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
        )
    except Exception as e:
        return f"Error: {e}", 500

@app.route('/health')
def health_check():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.close()
        conn.close()
        
        return jsonify({
            "status": "healthy",
            "database": "connected",
            "timestamp": datetime.now().isoformat(),
            "environment": os.getenv('ENVIRONMENT'),
            "domain": os.getenv('DOMAIN')
        })
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 503

@app.route('/api/trades')
def get_trades():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM trades ORDER BY timestamp DESC LIMIT 10")
        trades = cursor.fetchall()
        cursor.close()
        conn.close()
        
        return jsonify({
            "trades": [
                {
                    "id": trade[0],
                    "symbol": trade[1],
                    "side": trade[2],
                    "quantity": float(trade[3]),
                    "price": float(trade[4]),
                    "timestamp": trade[5].isoformat() if trade[5] else None,
                    "profit_loss": float(trade[6]) if trade[6] else None
                }
                for trade in trades
            ]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)