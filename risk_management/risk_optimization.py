import random
from datetime import datetime

class EnhancedTradingConfig:
    STARTING_CAPITAL = 10000
    MAX_POSITION_SIZE = 0.15  # 15% max per position
    MIN_SENTIMENT_THRESHOLD = 0.3
    VOLUME_THRESHOLD_MULTIPLIER = 1.5

class OptimizedRiskManager:
    def __init__(self):
        self.session_trades = []
        self.recent_losses = 0
    
    def should_skip_trade(self, portfolio_value):
        if self.recent_losses >= 3:
            return True, "Too many recent losses"
        return False, ""
    
    def adjust_position_size(self, base_size, recent_performance):
        if recent_performance == 'loss':
            return base_size * 0.5
        return base_size
    
    def calculate_dynamic_stop_loss(self, price, volatility):
        stop_percentage = max(0.05, volatility * 0.8)
        return price * (1 - stop_percentage)
    
    def record_trade_result(self, pnl, trade_type):
        self.session_trades.append({'pnl': pnl, 'type': trade_type})
        if pnl < 0:
            self.recent_losses += 1
        else:
            self.recent_losses = 0
    
    def get_risk_assessment(self):
        if self.recent_losses >= 2:
            return {'risk_level': 'high'}
        elif self.recent_losses == 1:
            return {'risk_level': 'medium'}
        return {'risk_level': 'low'}

class VolatilityCalculator:
    def calculate_recent_volatility(self, prices):
        if len(prices) < 2:
            return 0.1
        returns = [(prices[i]/prices[i-1] - 1) for i in range(1, len(prices))]
        return (sum(r**2 for r in returns) / len(returns)) ** 0.5
