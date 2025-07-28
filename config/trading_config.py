"""
Trading Configuration
All settings for your meme coin trading system
"""

import os
from datetime import datetime

class TradingConfig:
    def __init__(self):
        # === PORTFOLIO SETTINGS ===
        self.STARTING_CAPITAL = 10000  # Start with $10k for testing
        self.MAX_POSITION_SIZE = 0.20  # Max 20% per position  
        self.MAX_PORTFOLIO_RISK = 0.15  # Max 15% total drawdown
        self.MIN_CASH_RESERVE = 0.10   # Keep 10% cash
        
        # === TRADING SETTINGS ===
        self.PROFIT_LEVELS = {
            'level_1': {'target': 0.08, 'take': 0.25},   # Take 25% at +8%
            'level_2': {'target': 0.15, 'take': 0.25},   # Take 25% at +15%
            'level_3': {'target': 0.30, 'take': 0.25},   # Take 25% at +30%
            'moon_bag': {'target': float('inf'), 'take': 0.25}  # Let 25% ride
        }
        
        self.STOP_LOSS = 0.15  # 15% stop loss
        self.MIN_CONFIDENCE = 0.60  # Minimum ML confidence to trade
        self.MIN_SENTIMENT = 0.30   # Minimum sentiment score
        
        # === TARGET COINS ===
        self.WATCHLIST = [
            'DOGE', 'SHIB', 'PEPE', 'BONK', 'FLOKI', 
            'WIF', 'MEME', 'WOJAK', 'TURBO', 'LADYS'
        ]
        
        # === API SETTINGS ===
        self.API_KEYS = {
            # Get these from the respective platforms
            'REDDIT_CLIENT_ID': os.getenv('REDDIT_CLIENT_ID', ''),
            'REDDIT_CLIENT_SECRET': os.getenv('REDDIT_CLIENT_SECRET', ''),
            'COINGECKO_API_KEY': os.getenv('COINGECKO_API_KEY', ''),
            'TWITTER_BEARER_TOKEN': os.getenv('TWITTER_BEARER_TOKEN', ''),
        }
        
        # === TIMING SETTINGS ===
        self.SCAN_INTERVAL = 300  # Scan every 5 minutes
        self.ACTIVE_HOURS = (9, 23)  # Trade 9am-11pm EST
        self.PAPER_TRADING = True  # Start with paper trading
        
        # === RISK LIMITS ===
        self.MAX_DAILY_TRADES = 5
        self.MAX_DAILY_LOSS = 0.05  # Max 5% daily loss
        self.CORRELATION_LIMIT = 0.70  # Max correlation between positions
        
        # === LOGGING ===
        self.LOG_LEVEL = 'INFO'
        self.LOG_TRADES = True
        self.SEND_ALERTS = True
        
    def validate_config(self):
        """Validate configuration settings"""
        issues = []
        
        if self.STARTING_CAPITAL < 1000:
            issues.append("Starting capital should be at least $1000")
            
        if self.MAX_POSITION_SIZE > 0.30:
            issues.append("Max position size should not exceed 30%")
            
        if not self.WATCHLIST:
            issues.append("Watchlist cannot be empty")
            
        # Check for API keys in paper trading mode
        if not self.PAPER_TRADING:
            missing_keys = [k for k, v in self.API_KEYS.items() if not v]
            if missing_keys:
                issues.append(f"Missing API keys: {missing_keys}")
        
        return issues

# Global config instance
CONFIG = TradingConfig()

if __name__ == "__main__":
    # Test configuration
    issues = CONFIG.validate_config()
    
    if issues:
        print("❌ Configuration Issues:")
        for issue in issues:
            print(f"   - {issue}")
    else:
        print("✅ Configuration Valid!")
        
    print(f"\n📊 Trading Setup:")
    print(f"   Starting Capital: ${CONFIG.STARTING_CAPITAL:,}")
    print(f"   Max Position: {CONFIG.MAX_POSITION_SIZE:.0%}")
    print(f"   Watchlist: {len(CONFIG.WATCHLIST)} coins")
    print(f"   Paper Trading: {CONFIG.PAPER_TRADING}")