"""
Simple Risk Manager
Basic risk management to prevent overtrading and excessive losses
"""

import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
import logging

load_dotenv()
logger = logging.getLogger(__name__)

class SimpleRiskManager:
    def __init__(self):
        """Initialize risk manager"""
        # Load settings from env
        self.max_positions = int(os.getenv('MAX_POSITIONS', 12))
        self.max_daily_trades = int(os.getenv('MAX_DAILY_TRADES', 20))
        self.max_loss_percent = float(os.getenv('MAX_LOSS_PERCENT', 0.10))
        self.risk_per_trade = float(os.getenv('RISK_PER_TRADE', 0.025))
        
        # Tracking
        self.positions = {}
        self.daily_trades = []
        self.total_pnl = 0
        self.blocked_coins = set()
        
        logger.info(f"Risk manager initialized: max {self.max_positions} positions")
    
    def can_trade(self, coin_symbol, check_all=True):
        """Check if we can trade this coin"""
        
        # Check if coin is blocked
        if coin_symbol in self.blocked_coins:
            return False, f"{coin_symbol} is temporarily blocked"
        
        # Check position count
        if len(self.positions) >= self.max_positions:
            return False, f"Max positions reached ({self.max_positions})"
        
        # Check if already have position
        if coin_symbol in self.positions:
            return False, f"Already have position in {coin_symbol}"
        
        # Check daily trade limit
        self._clean_old_trades()
        if len(self.daily_trades) >= self.max_daily_trades:
            return False, f"Daily trade limit reached ({self.max_daily_trades})"
        
        # Check total loss limit
        if self.total_pnl < -self.max_loss_percent:
            return False, f"Max loss reached ({self.total_pnl:.1%})"
        
        # Check timing (no trades between 11 PM and 1 AM)
        hour = datetime.now().hour
        if hour >= 23 or hour <= 1:
            return False, "No trading during late night hours"
        
        # All checks passed
        return True, "OK to trade"
    
    def add_position(self, coin_symbol, size, entry_price):
        """Add a new position"""
        self.positions[coin_symbol] = {
            'size': size,
            'entry_price': entry_price,
            'entry_time': datetime.now(),
            'pnl': 0
        }
        
        self.daily_trades.append({
            'coin': coin_symbol,
            'time': datetime.now()
        })
        
        logger.info(f"Added position: {coin_symbol} size={size:.4f} @ ${entry_price:.6f}")
    
    def update_position(self, coin_symbol, current_price):
        """Update position with current price"""
        if coin_symbol not in self.positions:
            return None
        
        position = self.positions[coin_symbol]
        entry_price = position['entry_price']
        
        # Calculate PnL
        pnl = (current_price - entry_price) / entry_price
        position['pnl'] = pnl
        position['current_price'] = current_price
        
        # Check stop loss
        if pnl < -0.05:  # 5% loss
            return 'STOP_LOSS'
        
        # Check take profit
        if pnl > 0.15:  # 15% profit
            return 'TAKE_PROFIT'
        
        # Check time-based exit
        hold_time = (datetime.now() - position['entry_time']).seconds / 3600
        if hold_time > 24:  # 24 hours
            return 'TIME_EXIT'
        
        return 'HOLD'
    
    def close_position(self, coin_symbol, exit_price):
        """Close a position"""
        if coin_symbol not in self.positions:
            return None
        
        position = self.positions[coin_symbol]
        pnl = (exit_price - position['entry_price']) / position['entry_price']
        
        # Update total PnL
        self.total_pnl += pnl * position['size']
        
        # Remove position
        del self.positions[coin_symbol]
        
        # Block coin temporarily if it was a loss
        if pnl < 0:
            self.blocked_coins.add(coin_symbol)
            # Unblock after 1 hour
            self._schedule_unblock(coin_symbol, 3600)
        
        logger.info(f"Closed position: {coin_symbol} PnL={pnl:.1%}")
        
        return pnl
    
    def _clean_old_trades(self):
        """Remove trades older than 24 hours"""
        cutoff = datetime.now() - timedelta(hours=24)
        self.daily_trades = [
            trade for trade in self.daily_trades 
            if trade['time'] > cutoff
        ]
    
    def _schedule_unblock(self, coin_symbol, delay_seconds):
        """Schedule coin to be unblocked (simplified version)"""
        # In production, use proper scheduling
        # For now, just track the time
        pass
    
    def get_risk_status(self):
        """Get current risk status"""
        return {
            'positions': len(self.positions),
            'max_positions': self.max_positions,
            'daily_trades': len(self.daily_trades),
            'max_daily_trades': self.max_daily_trades,
            'total_pnl': self.total_pnl,
            'blocked_coins': list(self.blocked_coins),
            'can_trade_more': len(self.positions) < self.max_positions
        }
    
    def calculate_position_size(self, balance, coin_price, confidence=1.0):
        """Calculate position size based on risk"""
        # Base position size
        position_value = balance * self.risk_per_trade
        
        # Adjust for confidence (0.5 to 1.5x)
        confidence_multiplier = 0.5 + (confidence * 0.5)
        position_value *= confidence_multiplier
        
        # Adjust for current portfolio heat
        portfolio_heat = len(self.positions) / self.max_positions
        if portfolio_heat > 0.7:
            position_value *= 0.7  # Reduce size when portfolio is hot
        
        # Cap at max position size
        max_position_value = balance * 0.05  # 5% max
        position_value = min(position_value, max_position_value)
        
        # Calculate coin amount
        position_size = position_value / coin_price
        
        return position_size, position_value