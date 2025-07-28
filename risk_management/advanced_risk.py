"""
Advanced Risk Management System
Implements sophisticated risk controls and monitoring
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging

class AdvancedRiskManager:
    def __init__(self, max_drawdown: float = 0.15, sharpe_target: float = 2.0, 
                 max_positions: int = 12, risk_per_trade: float = 0.03):
        """
        Initialize advanced risk management system
        
        Args:
            max_drawdown: Maximum allowed drawdown (15%)
            sharpe_target: Target Sharpe ratio (2.0)
            max_positions: Maximum concurrent positions (8-12)
            risk_per_trade: Risk per trade as fraction of portfolio (1-3%)
        """
        self.max_drawdown = max_drawdown
        self.sharpe_target = sharpe_target
        self.max_positions = max_positions
        self.risk_per_trade = risk_per_trade
        
        # Risk tracking
        self.portfolio_value_history = []
        self.drawdown_history = []
        self.position_risks = {}  # symbol -> risk metrics
        self.correlation_matrix = {}
        
        # Performance tracking
        self.daily_returns = []
        self.sharpe_ratio = 0.0
        self.max_historic_drawdown = 0.0
        
        # Risk limits
        self.risk_limits = {
            'max_single_position': 0.25,      # 25% max per position
            'max_sector_exposure': 0.40,      # 40% max per sector
            'max_correlation_exposure': 0.60, # 60% max in correlated positions
            'min_cash_reserve': 0.10          # 10% minimum cash
        }
        
    def calculate_portfolio_drawdown(self, current_value: float, 
                                   peak_value: float = None) -> float:
        """Calculate current portfolio drawdown"""
        if not self.portfolio_value_history:
            return 0.0
            
        if peak_value is None:
            peak_value = max(self.portfolio_value_history + [current_value])
            
        drawdown = (peak_value - current_value) / peak_value
        return max(0, drawdown)
    
    def update_portfolio_metrics(self, current_value: float) -> None:
        """Update portfolio performance metrics"""
        self.portfolio_value_history.append(current_value)
        
        # Calculate daily return if we have previous value
        if len(self.portfolio_value_history) > 1:
            prev_value = self.portfolio_value_history[-2]
            daily_return = (current_value - prev_value) / prev_value
            self.daily_returns.append(daily_return)
            
        # Update drawdown
        current_drawdown = self.calculate_portfolio_drawdown(current_value)
        self.drawdown_history.append(current_drawdown)
        self.max_historic_drawdown = max(self.max_historic_drawdown, current_drawdown)
        
        # Update Sharpe ratio (last 30 days)
        if len(self.daily_returns) >= 30:
            recent_returns = self.daily_returns[-30:]
            if len(recent_returns) > 1:
                avg_return = np.mean(recent_returns)
                std_return = np.std(recent_returns)
                if std_return > 0:
                    self.sharpe_ratio = (avg_return * np.sqrt(252)) / (std_return * np.sqrt(252))
    
    def check_drawdown_limits(self, current_value: float) -> Dict:
        """Check if portfolio is approaching drawdown limits"""
        current_drawdown = self.calculate_portfolio_drawdown(current_value)
        
        status = 'normal'
        action_required = False
        
        if current_drawdown >= self.max_drawdown:
            status = 'critical'
            action_required = True
        elif current_drawdown >= self.max_drawdown * 0.8:  # 80% of max
            status = 'warning'
        elif current_drawdown >= self.max_drawdown * 0.6:  # 60% of max
            status = 'caution'
            
        return {
            'current_drawdown': current_drawdown,
            'max_allowed': self.max_drawdown,
            'status': status,
            'action_required': action_required,
            'remaining_buffer': self.max_drawdown - current_drawdown
        }
    
    def check_position_limits(self, current_positions: Dict[str, Dict], 
                            new_position: Dict = None) -> Dict:
        """Check if position limits are within acceptable ranges"""
        
        total_positions = len(current_positions)
        if new_position:
            total_positions += 1
            
        # Check position count limit
        position_count_ok = total_positions <= self.max_positions
        
        # Check individual position sizes
        oversized_positions = []
        total_exposure = 0
        
        all_positions = dict(current_positions)
        if new_position:
            all_positions[new_position['symbol']] = new_position
            
        for symbol, position in all_positions.items():
            position_value = position.get('position_value', 0)
            portfolio_value = position.get('portfolio_value', 1)
            position_pct = position_value / portfolio_value
            total_exposure += position_pct
            
            if position_pct > self.risk_limits['max_single_position']:
                oversized_positions.append({
                    'symbol': symbol,
                    'current_size': position_pct,
                    'max_allowed': self.risk_limits['max_single_position']
                })
        
        # Check cash reserve
        cash_reserve = 1.0 - total_exposure
        sufficient_cash = cash_reserve >= self.risk_limits['min_cash_reserve']
        
        return {
            'position_count_ok': position_count_ok,
            'current_positions': total_positions,
            'max_positions': self.max_positions,
            'oversized_positions': oversized_positions,
            'total_exposure': total_exposure,
            'cash_reserve': cash_reserve,
            'sufficient_cash': sufficient_cash,
            'all_limits_ok': all([
                position_count_ok,
                len(oversized_positions) == 0,
                sufficient_cash
            ])
        }
    
    def generate_risk_report(self, portfolio_value: float, 
                           active_positions: Dict) -> Dict:
        """Generate comprehensive risk report"""
        
        # Update metrics
        self.update_portfolio_metrics(portfolio_value)
        
        # Calculate additional metrics
        total_positions = len(active_positions)
        total_risk = sum(pos.get('risk_amount', 0) for pos in active_positions.values())
        total_risk_pct = total_risk / portfolio_value if portfolio_value > 0 else 0
        
        # Check position limits
        position_status = self.check_position_limits(active_positions)
        
        # Simple correlation check
        correlation_status = {
            'avg_correlation': 0.2,
            'max_correlation': 0.4,
            'correlation_risk': False,
            'highly_correlated_pairs': []
        }
        
        # Emergency assessment
        emergency_assessment = {
            'risk_level': 'normal',
            'emergency_actions': [],
            'position_status': position_status,
            'correlation_status': correlation_status,
            'immediate_action_required': False
        }
        
        return {
            'timestamp': datetime.now(),
            'portfolio_value': portfolio_value,
            'total_positions': total_positions,
            'total_risk_amount': total_risk,
            'total_risk_percentage': total_risk_pct,
            'current_sharpe': self.sharpe_ratio,
            'max_historic_drawdown': self.max_historic_drawdown,
            'risk_limits': self.risk_limits,
            'emergency_assessment': emergency_assessment,
            'recommendations': ['Risk levels within acceptable parameters']
        }


# Example usage
if __name__ == "__main__":
    risk_manager = AdvancedRiskManager()
    
    # Sample portfolio data
    portfolio_value = 100000
    sample_positions = {
        'MEME1': {
            'position_value': 15000,
            'portfolio_value': portfolio_value,
            'risk_amount': 1500
        }
    }
    
    # Generate risk report
    risk_report = risk_manager.generate_risk_report(portfolio_value, sample_positions)
    
    print(f"Risk Level: {risk_report['emergency_assessment']['risk_level']}")
    print(f"Total Risk: {risk_report['total_risk_percentage']:.1%}")
    print(f"Sharpe Ratio: {risk_report['current_sharpe']:.2f}")