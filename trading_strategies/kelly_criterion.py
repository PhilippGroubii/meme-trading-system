"""
Kelly Criterion for Dynamic Position Sizing
Implements optimal bet sizing for maximum long-term growth
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import logging

class KellyCriterion:
    def __init__(self, confidence_threshold: float = 0.6, max_position: float = 0.25):
        """
        Initialize Kelly Criterion calculator
        
        Args:
            confidence_threshold: Minimum confidence for position sizing
            max_position: Maximum position size as fraction of portfolio
        """
        self.confidence_threshold = confidence_threshold
        self.max_position = max_position
        self.win_rate_history = []
        self.avg_win_loss_ratio = 1.0
        
    def calculate_kelly_fraction(self, win_prob: float, win_loss_ratio: float) -> float:
        """
        Calculate Kelly fraction: f = (bp - q) / b
        
        Args:
            win_prob: Probability of winning (0-1)
            win_loss_ratio: Average win / Average loss ratio
            
        Returns:
            Kelly fraction (0-1)
        """
        if win_prob <= 0 or win_loss_ratio <= 0:
            return 0.0
            
        # Kelly formula: f = (bp - q) / b
        # where b = win_loss_ratio, p = win_prob, q = 1 - win_prob
        kelly_fraction = (win_loss_ratio * win_prob - (1 - win_prob)) / win_loss_ratio
        
        # Cap at maximum position size and ensure non-negative
        return max(0, min(kelly_fraction, self.max_position))
    
    def update_statistics(self, trades: List[Dict]) -> None:
        """Update win rate and win/loss ratio from recent trades"""
        if not trades:
            return
            
        wins = [t for t in trades if t['pnl'] > 0]
        losses = [t for t in trades if t['pnl'] < 0]
        
        if len(trades) > 0:
            self.win_rate = len(wins) / len(trades)
            
        if wins and losses:
            avg_win = np.mean([t['pnl'] for t in wins])
            avg_loss = abs(np.mean([t['pnl'] for t in losses]))
            self.avg_win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 1.0
    
    def get_position_size(self, signal_confidence: float, portfolio_value: float, 
                         recent_trades: List[Dict] = None) -> float:
        """
        Calculate optimal position size using Kelly Criterion
        
        Args:
            signal_confidence: Model confidence (0-1)
            portfolio_value: Current portfolio value
            recent_trades: Recent trade history for statistics
            
        Returns:
            Position size in dollars
        """
        if signal_confidence < self.confidence_threshold:
            return 0.0
            
        # Update statistics if trades provided
        if recent_trades:
            self.update_statistics(recent_trades)
            
        # Use signal confidence as win probability
        # Adjust by historical performance if available
        if hasattr(self, 'win_rate') and self.win_rate > 0:
            adjusted_win_prob = (signal_confidence + self.win_rate) / 2
        else:
            adjusted_win_prob = signal_confidence
            
        # Calculate Kelly fraction
        kelly_fraction = self.calculate_kelly_fraction(
            adjusted_win_prob, self.avg_win_loss_ratio
        )
        
        # Apply confidence-based scaling
        confidence_scaling = min(signal_confidence / 0.8, 1.0)
        scaled_fraction = kelly_fraction * confidence_scaling
        
        return portfolio_value * scaled_fraction
    
    def get_portfolio_heat(self, active_positions: Dict[str, float], 
                          portfolio_value: float) -> float:
        """
        Calculate current portfolio heat (total risk exposure)
        
        Args:
            active_positions: Dict of symbol -> position_value
            portfolio_value: Total portfolio value
            
        Returns:
            Portfolio heat as fraction (0-1)
        """
        total_exposure = sum(abs(pos) for pos in active_positions.values())
        return total_exposure / portfolio_value if portfolio_value > 0 else 0.0
    
    def adjust_for_portfolio_heat(self, proposed_size: float, current_heat: float) -> float:
        """
        Adjust position size based on current portfolio heat
        
        Args:
            proposed_size: Proposed position size
            current_heat: Current portfolio heat (0-1)
            
        Returns:
            Adjusted position size
        """
        max_total_heat = 0.8  # Maximum 80% portfolio exposure
        
        if current_heat >= max_total_heat:
            return 0.0  # No new positions if at max heat
            
        # Scale down if adding this position would exceed max heat
        available_heat = max_total_heat - current_heat
        heat_scaling = min(available_heat / 0.2, 1.0)  # Gradual scaling
        
        return proposed_size * heat_scaling


class ConfidenceBasedSizing:
    """Confidence-based position scaling system"""
    
    def __init__(self):
        self.confidence_levels = {
            'very_high': (0.8, 1.0, 1.0),    # confidence range, size multiplier
            'high': (0.65, 0.8, 0.75),
            'medium': (0.5, 0.65, 0.5),
            'low': (0.35, 0.5, 0.25),
            'very_low': (0.0, 0.35, 0.0)
        }
    
    def get_confidence_level(self, confidence: float) -> str:
        """Determine confidence level from score"""
        for level, (min_conf, max_conf, _) in self.confidence_levels.items():
            if min_conf <= confidence < max_conf:
                return level
        return 'very_low'
    
    def get_size_multiplier(self, confidence: float) -> float:
        """Get position size multiplier based on confidence"""
        level = self.get_confidence_level(confidence)
        return self.confidence_levels[level][2]


class PortfolioHeatManager:
    """Manages overall portfolio risk exposure"""
    
    def __init__(self, max_heat: float = 0.8, target_heat: float = 0.6):
        self.max_heat = max_heat
        self.target_heat = target_heat
        self.position_heat = {}  # symbol -> heat contribution
        
    def update_position_heat(self, symbol: str, position_value: float, 
                           portfolio_value: float) -> None:
        """Update heat contribution for a position"""
        self.position_heat[symbol] = abs(position_value) / portfolio_value
        
    def get_total_heat(self) -> float:
        """Get total portfolio heat"""
        return sum(self.position_heat.values())
        
    def can_add_position(self, proposed_heat: float) -> bool:
        """Check if new position can be added without exceeding max heat"""
        return (self.get_total_heat() + proposed_heat) <= self.max_heat
        
    def get_heat_scaling_factor(self) -> float:
        """Get scaling factor based on current heat level"""
        current_heat = self.get_total_heat()
        
        if current_heat <= self.target_heat:
            return 1.0
        elif current_heat <= self.max_heat:
            # Linear scaling between target and max
            return 1.0 - (current_heat - self.target_heat) / (self.max_heat - self.target_heat)
        else:
            return 0.0  # Over max heat, no new positions


# Example usage and testing
if __name__ == "__main__":
    # Test Kelly Criterion
    kelly = KellyCriterion()
    
    # Sample recent trades
    sample_trades = [
        {'pnl': 100}, {'pnl': -50}, {'pnl': 200}, {'pnl': -75}, {'pnl': 150}
    ]
    
    portfolio_value = 10000
    signal_confidence = 0.75
    
    position_size = kelly.get_position_size(
        signal_confidence, portfolio_value, sample_trades
    )
    
    print(f"Recommended position size: ${position_size:.2f}")
    print(f"As percentage of portfolio: {position_size/portfolio_value:.1%}")