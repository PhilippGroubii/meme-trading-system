"""
Multi-Level Profit Taking System
Implements systematic profit taking at multiple levels
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import logging

class ProfitOptimizer:
    def __init__(self):
        """Initialize multi-level profit taking system"""
        self.profit_levels = {
            'level_1': {'target': 0.08, 'take': 0.25},   # Take 25% at +8%
            'level_2': {'target': 0.15, 'take': 0.25},   # Take 25% at +15%
            'level_3': {'target': 0.30, 'take': 0.25},   # Take 25% at +30%
            'moon_bag': {'target': float('inf'), 'take': 0.25}  # Let 25% ride
        }
        
        self.active_positions = {}  # symbol -> position info
        self.profit_history = {}    # symbol -> profit taking history
        
    def add_position(self, symbol: str, entry_price: float, quantity: float, 
                    entry_time: datetime = None) -> None:
        """
        Add new position to track
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price
            quantity: Initial quantity
            entry_time: Entry timestamp
        """
        self.active_positions[symbol] = {
            'entry_price': entry_price,
            'current_quantity': quantity,
            'original_quantity': quantity,
            'entry_time': entry_time or datetime.now(),
            'profit_taken': {},  # level -> quantity taken
            'remaining_levels': list(self.profit_levels.keys())
        }
        
        self.profit_history[symbol] = []
        
    def calculate_profit_percentage(self, symbol: str, current_price: float) -> float:
        """Calculate current profit percentage for position"""
        if symbol not in self.active_positions:
            return 0.0
            
        entry_price = self.active_positions[symbol]['entry_price']
        return (current_price - entry_price) / entry_price
    
    def check_profit_levels(self, symbol: str, current_price: float) -> List[Dict]:
        """
        Check which profit levels should be triggered
        
        Returns:
            List of actions to take
        """
        if symbol not in self.active_positions:
            return []
            
        position = self.active_positions[symbol]
        profit_pct = self.calculate_profit_percentage(symbol, current_price)
        actions = []
        
        for level in position['remaining_levels']:
            if level == 'moon_bag':
                continue  # Never sell moon bag automatically
                
            level_info = self.profit_levels[level]
            
            if profit_pct >= level_info['target']:
                # Calculate quantity to sell
                sell_quantity = position['original_quantity'] * level_info['take']
                
                # Ensure we don't sell more than we have
                sell_quantity = min(sell_quantity, position['current_quantity'])
                
                if sell_quantity > 0:
                    actions.append({
                        'action': 'sell',
                        'symbol': symbol,
                        'quantity': sell_quantity,
                        'level': level,
                        'target_profit': level_info['target'],
                        'current_profit': profit_pct,
                        'price': current_price
                    })
        
        return actions
    
    def execute_profit_taking(self, symbol: str, level: str, quantity: float, 
                            price: float, timestamp: datetime = None) -> Dict:
        """
        Execute profit taking and update position
        
        Returns:
            Execution details
        """
        if symbol not in self.active_positions:
            raise ValueError(f"No position found for {symbol}")
            
        position = self.active_positions[symbol]
        
        # Update position
        position['current_quantity'] -= quantity
        position['profit_taken'][level] = quantity
        
        # Remove level from remaining levels
        if level in position['remaining_levels']:
            position['remaining_levels'].remove(level)
            
        # Calculate profit
        entry_price = position['entry_price']
        profit_amount = quantity * (price - entry_price)
        profit_pct = (price - entry_price) / entry_price
        
        # Record in history
        execution_record = {
            'timestamp': timestamp or datetime.now(),
            'level': level,
            'quantity': quantity,
            'price': price,
            'profit_amount': profit_amount,
            'profit_percentage': profit_pct,
            'remaining_quantity': position['current_quantity']
        }
        
        self.profit_history[symbol].append(execution_record)
        
        return execution_record
    
    def get_trailing_stop_price(self, symbol: str, current_price: float, 
                               trailing_pct: float = 0.02) -> Optional[float]:
        """
        Calculate trailing stop price for remaining position
        
        Args:
            symbol: Trading symbol
            current_price: Current market price
            trailing_pct: Trailing stop percentage (default 2%)
            
        Returns:
            Stop price or None if no position
        """
        if symbol not in self.active_positions:
            return None
            
        position = self.active_positions[symbol]
        entry_price = position['entry_price']
        
        # Only use trailing stop if we're in profit
        if current_price <= entry_price:
            return entry_price * 0.95  # 5% stop loss
            
        # Trailing stop from current price
        return current_price * (1 - trailing_pct)
    
    def should_move_stop_to_breakeven(self, symbol: str, current_price: float, 
                                    breakeven_trigger: float = 0.05) -> bool:
        """
        Check if stop should be moved to break-even
        
        Args:
            breakeven_trigger: Profit percentage to trigger break-even stop
        """
        profit_pct = self.calculate_profit_percentage(symbol, current_price)
        return profit_pct >= breakeven_trigger
    
    def get_position_status(self, symbol: str, current_price: float) -> Dict:
        """Get comprehensive position status"""
        if symbol not in self.active_positions:
            return {}
            
        position = self.active_positions[symbol]
        profit_pct = self.calculate_profit_percentage(symbol, current_price)
        
        # Calculate total profit taken
        total_profit_taken = sum(
            record['profit_amount'] for record in self.profit_history[symbol]
        )
        
        # Calculate unrealized profit on remaining position
        remaining_value = position['current_quantity'] * current_price
        remaining_cost = position['current_quantity'] * position['entry_price']
        unrealized_profit = remaining_value - remaining_cost
        
        return {
            'symbol': symbol,
            'entry_price': position['entry_price'],
            'current_price': current_price,
            'current_profit_pct': profit_pct,
            'original_quantity': position['original_quantity'],
            'current_quantity': position['current_quantity'],
            'quantity_sold_pct': 1 - (position['current_quantity'] / position['original_quantity']),
            'total_profit_taken': total_profit_taken,
            'unrealized_profit': unrealized_profit,
            'total_profit': total_profit_taken + unrealized_profit,
            'remaining_levels': position['remaining_levels'],
            'profit_history': self.profit_history[symbol]
        }
    
    def optimize_exit_timing(self, symbol: str, current_price: float, 
                           volume_data: Dict = None, 
                           momentum_data: Dict = None) -> Dict:
        """
        Optimize exit timing based on multiple factors
        
        Args:
            volume_data: Dict with volume metrics
            momentum_data: Dict with momentum indicators
            
        Returns:
            Exit recommendation
        """
        if symbol not in self.active_positions:
            return {'action': 'hold', 'reason': 'no_position'}
            
        profit_pct = self.calculate_profit_percentage(symbol, current_price)
        
        # Base recommendation on profit levels
        actions = self.check_profit_levels(symbol, current_price)
        
        if actions:
            return {
                'action': 'partial_sell',
                'details': actions,
                'reason': 'profit_target_hit'
            }
        
        # Check for early exit signals
        early_exit_score = 0
        
        # Volume analysis
        if volume_data:
            if volume_data.get('volume_trend', 'neutral') == 'decreasing':
                early_exit_score += 1
            if volume_data.get('volume_spike', False):
                early_exit_score += 1
                
        # Momentum analysis  
        if momentum_data:
            if momentum_data.get('rsi', 50) > 70:  # Overbought
                early_exit_score += 1
            if momentum_data.get('macd_divergence', False):
                early_exit_score += 2
                
        # Strong early exit signal
        if early_exit_score >= 3 and profit_pct > 0.03:  # At least 3% profit
            return {
                'action': 'partial_sell',
                'reason': 'early_exit_signal',
                'exit_score': early_exit_score,
                'suggested_percentage': 0.3  # Sell 30% early
            }
            
        return {'action': 'hold', 'reason': 'no_exit_signal'}


# Example usage
if __name__ == "__main__":
    optimizer = ProfitOptimizer()
    
    # Add a position
    optimizer.add_position('MEME', entry_price=100.0, quantity=1000)
    
    # Simulate price movements and profit taking
    prices = [100, 105, 108, 112, 118, 125, 135]
    
    for price in prices:
        print(f"\nPrice: ${price}")
        
        # Check for profit taking opportunities
        actions = optimizer.check_profit_levels('MEME', price)
        
        for action in actions:
            print(f"Action: {action}")
            
            # Execute the profit taking
            optimizer.execute_profit_taking(
                action['symbol'], 
                action['level'], 
                action['quantity'], 
                action['price']
            )
            
        # Show position status
        status = optimizer.get_position_status('MEME', price)
        print(f"Remaining quantity: {status['current_quantity']}")
        print(f"Total profit taken: ${status['total_profit_taken']:.2f}")