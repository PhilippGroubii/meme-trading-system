"""
Smart Entry Optimization System
Implements VWAP entries, momentum confirmation, and divergence detection
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging

class VWAPEntry:
    """Volume Weighted Average Price entry system"""
    
    def __init__(self, lookback_periods: int = 20):
        self.lookback_periods = lookback_periods
        self.vwap_cache = {}
        
    def calculate_vwap(self, prices: List[float], volumes: List[float], 
                      high_prices: List[float] = None, 
                      low_prices: List[float] = None) -> float:
        """
        Calculate VWAP (Volume Weighted Average Price)
        
        Args:
            prices: List of closing prices
            volumes: List of volumes
            high_prices: List of high prices (optional)
            low_prices: List of low prices (optional)
            
        Returns:
            VWAP value
        """
        if not prices or not volumes or len(prices) != len(volumes):
            return 0.0
            
        # Use typical price if high/low available, otherwise use close
        if high_prices and low_prices:
            typical_prices = [(h + l + c) / 3 for h, l, c in zip(high_prices, low_prices, prices)]
        else:
            typical_prices = prices
            
        # Calculate VWAP
        total_pv = sum(tp * v for tp, v in zip(typical_prices, volumes))
        total_volume = sum(volumes)
        
        return total_pv / total_volume if total_volume > 0 else prices[-1] if prices else 0.0
        
    def get_vwap_signal(self, current_price: float, vwap_price: float, 
                       tolerance: float = 0.005) -> Dict:
        """
        Get VWAP-based entry signal
        
        Args:
            current_price: Current market price
            vwap_price: Current VWAP value
            tolerance: Price tolerance around VWAP (0.5% default)
            
        Returns:
            Signal dictionary
        """
        # Handle edge cases
        if vwap_price <= 0 or current_price <= 0:
            return {
                'signal': 'no_signal',
                'reason': 'invalid_prices',
                'vwap_price': vwap_price,
                'current_price': current_price,
                'deviation_pct': 0,
                'entry_quality': 'poor'
            }
            
        price_diff_pct = (current_price - vwap_price) / vwap_price
        
        # Check if price is near VWAP
        if abs(price_diff_pct) <= tolerance:
            return {
                'signal': 'entry_ready',
                'reason': 'price_at_vwap',
                'vwap_price': vwap_price,
                'current_price': current_price,
                'deviation_pct': price_diff_pct,
                'entry_quality': 'high'
            }
        elif price_diff_pct < -tolerance:
            return {
                'signal': 'below_vwap', 
                'reason': 'price_below_vwap',
                'vwap_price': vwap_price,
                'deviation_pct': price_diff_pct,
                'entry_quality': 'excellent'  # Better entry below VWAP
            }
        else:
            return {
                'signal': 'above_vwap',
                'reason': 'price_above_vwap', 
                'vwap_price': vwap_price,
                'deviation_pct': price_diff_pct,
                'entry_quality': 'poor'
            }


class MomentumConfirmation:
    """Momentum confirmation system to avoid false breakouts"""
    
    def __init__(self):
        self.rsi_period = 14
        self.volume_sma_period = 20
        
    def calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """Calculate RSI indicator"""
        if len(prices) < period + 1:
            return 50.0
            
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]
        
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        
        if avg_loss == 0:
            return 100.0
            
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def check_volume_confirmation(self, current_volume: float, 
                                avg_volume: float, 
                                min_multiplier: float = 1.5) -> bool:
        """Check if volume confirms the move"""
        return current_volume >= avg_volume * min_multiplier
    
    def detect_breakout_momentum(self, prices: List[float], volumes: List[float],
                               resistance_level: float = None) -> Dict:
        """
        Detect if current price action shows strong breakout momentum
        
        Returns:
            Momentum analysis
        """
        if len(prices) < 5 or len(volumes) < 5:
            return {'momentum': 'insufficient_data'}
            
        current_price = prices[-1]
        prev_prices = prices[-5:-1]
        current_volume = volumes[-1]
        avg_volume = sum(volumes[-20:]) / min(len(volumes), 20)
        
        # Check price momentum
        price_change = (current_price - prices[-2]) / prices[-2]
        avg_change = sum((prices[i] - prices[i-1])/prices[i-1] for i in range(-4, 0)) / 4
        
        momentum_strength = price_change / avg_change if avg_change != 0 else 1
        
        # Volume confirmation
        volume_confirmed = self.check_volume_confirmation(current_volume, avg_volume)
        
        # RSI check (avoid overbought)
        rsi = self.calculate_rsi(prices)
        rsi_favorable = 30 < rsi < 70  # Not oversold or overbought
        
        # Breakout detection
        breakout_detected = False
        if resistance_level:
            breakout_detected = current_price > resistance_level
            
        return {
            'momentum': 'strong' if momentum_strength > 1.5 and volume_confirmed else 'weak',
            'momentum_strength': momentum_strength,
            'volume_confirmed': volume_confirmed,
            'volume_ratio': current_volume / avg_volume,
            'rsi': rsi,
            'rsi_favorable': rsi_favorable,
            'breakout_detected': breakout_detected,
            'entry_recommendation': all([
                momentum_strength > 1.2,
                volume_confirmed,
                rsi_favorable
            ])
        }


class DivergenceDetector:
    """Detect price/indicator divergences to spot reversals"""
    
    def __init__(self):
        self.lookback_periods = 20
        
    def find_peaks_valleys(self, data: List[float], min_distance: int = 3) -> Dict:
        """Find peaks and valleys in price data"""
        if len(data) < min_distance * 2 + 1:
            return {'peaks': [], 'valleys': []}
            
        peaks = []
        valleys = []
        
        for i in range(min_distance, len(data) - min_distance):
            # Check for peak
            is_peak = all(data[i] >= data[j] for j in range(i-min_distance, i+min_distance+1))
            if is_peak and data[i] > data[i-1] and data[i] > data[i+1]:
                peaks.append((i, data[i]))
                
            # Check for valley  
            is_valley = all(data[i] <= data[j] for j in range(i-min_distance, i+min_distance+1))
            if is_valley and data[i] < data[i-1] and data[i] < data[i+1]:
                valleys.append((i, data[i]))
                
        return {'peaks': peaks, 'valleys': valleys}
    
    def detect_bullish_divergence(self, prices: List[float], 
                                indicator: List[float]) -> Dict:
        """
        Detect bullish divergence (lower lows in price, higher lows in indicator)
        
        Args:
            prices: Price data
            indicator: Indicator data (RSI, MACD, etc.)
            
        Returns:
            Divergence analysis
        """
        if len(prices) != len(indicator) or len(prices) < 10:
            return {'divergence': False, 'strength': 0}
            
        # Find recent valleys
        price_extremes = self.find_peaks_valleys(prices)
        indicator_extremes = self.find_peaks_valleys(indicator)
        
        price_valleys = price_extremes['valleys']
        indicator_valleys = indicator_extremes['valleys']
        
        if len(price_valleys) < 2 or len(indicator_valleys) < 2:
            return {'divergence': False, 'strength': 0}
            
        # Get two most recent valleys
        recent_price_valleys = sorted(price_valleys, key=lambda x: x[0])[-2:]
        recent_indicator_valleys = sorted(indicator_valleys, key=lambda x: x[0])[-2:]
        
        # Check for bullish divergence
        price_lower_low = recent_price_valleys[1][1] < recent_price_valleys[0][1]
        indicator_higher_low = recent_indicator_valleys[1][1] > recent_indicator_valleys[0][1]
        
        divergence_detected = price_lower_low and indicator_higher_low
        
        # Calculate divergence strength
        if divergence_detected:
            price_decline = (recent_price_valleys[0][1] - recent_price_valleys[1][1]) / recent_price_valleys[0][1]
            indicator_improvement = (recent_indicator_valleys[1][1] - recent_indicator_valleys[0][1]) / recent_indicator_valleys[0][1]
            strength = min(price_decline + indicator_improvement, 1.0)
        else:
            strength = 0
            
        return {
            'divergence': divergence_detected,
            'type': 'bullish',
            'strength': strength,
            'price_valleys': recent_price_valleys,
            'indicator_valleys': recent_indicator_valleys
        }
    
    def detect_bearish_divergence(self, prices: List[float], 
                                indicator: List[float]) -> Dict:
        """Detect bearish divergence (higher highs in price, lower highs in indicator)"""
        if len(prices) != len(indicator) or len(prices) < 10:
            return {'divergence': False, 'strength': 0}
            
        # Find recent peaks
        price_extremes = self.find_peaks_valleys(prices)
        indicator_extremes = self.find_peaks_valleys(indicator)
        
        price_peaks = price_extremes['peaks']
        indicator_peaks = indicator_extremes['peaks']
        
        if len(price_peaks) < 2 or len(indicator_peaks) < 2:
            return {'divergence': False, 'strength': 0}
            
        # Get two most recent peaks
        recent_price_peaks = sorted(price_peaks, key=lambda x: x[0])[-2:]
        recent_indicator_peaks = sorted(indicator_peaks, key=lambda x: x[0])[-2:]
        
        # Check for bearish divergence
        price_higher_high = recent_price_peaks[1][1] > recent_price_peaks[0][1]
        indicator_lower_high = recent_indicator_peaks[1][1] < recent_indicator_peaks[0][1]
        
        divergence_detected = price_higher_high and indicator_lower_high
        
        # Calculate divergence strength
        if divergence_detected:
            price_gain = (recent_price_peaks[1][1] - recent_price_peaks[0][1]) / recent_price_peaks[0][1]
            indicator_decline = (recent_indicator_peaks[0][1] - recent_indicator_peaks[1][1]) / recent_indicator_peaks[0][1]
            strength = min(price_gain + indicator_decline, 1.0)
        else:
            strength = 0
            
        return {
            'divergence': divergence_detected,
            'type': 'bearish', 
            'strength': strength,
            'price_peaks': recent_price_peaks,
            'indicator_peaks': recent_indicator_peaks
        }


class SmartEntryOptimizer:
    """Main class combining all entry optimization techniques"""
    
    def __init__(self):
        self.vwap_entry = VWAPEntry()
        self.momentum_confirmation = MomentumConfirmation()
        self.divergence_detector = DivergenceDetector()
        
    def analyze_entry_opportunity(self, market_data: Dict) -> Dict:
        """
        Comprehensive entry analysis
        
        Args:
            market_data: Dict containing prices, volumes, indicators
            
        Returns:
            Complete entry analysis
        """
        required_fields = ['prices', 'volumes', 'high_prices', 'low_prices']
        if not all(field in market_data for field in required_fields):
            return {'error': 'insufficient_market_data'}
            
        prices = market_data['prices']
        volumes = market_data['volumes']
        current_price = prices[-1]
        
        # VWAP Analysis
        vwap_price = self.vwap_entry.calculate_vwap(
            prices, volumes, 
            market_data.get('high_prices'), 
            market_data.get('low_prices')
        )
        vwap_signal = self.vwap_entry.get_vwap_signal(current_price, vwap_price)
        
        # Momentum Analysis
        momentum_analysis = self.momentum_confirmation.detect_breakout_momentum(
            prices, volumes, market_data.get('resistance_level')
        )
        
        # Divergence Analysis
        rsi_data = [self.momentum_confirmation.calculate_rsi(prices[:i+1]) 
                   for i in range(len(prices))]
        
        bullish_div = self.divergence_detector.detect_bullish_divergence(prices, rsi_data)
        bearish_div = self.divergence_detector.detect_bearish_divergence(prices, rsi_data)
        
        # Combine all signals for final recommendation
        entry_score = 0
        entry_factors = []
        
        # VWAP scoring
        if vwap_signal['entry_quality'] == 'excellent':
            entry_score += 3
            entry_factors.append('excellent_vwap_entry')
        elif vwap_signal['entry_quality'] == 'high':
            entry_score += 2
            entry_factors.append('good_vwap_entry')
            
        # Momentum scoring
        if momentum_analysis.get('entry_recommendation', False):
            entry_score += 3
            entry_factors.append('strong_momentum')
        elif momentum_analysis.get('volume_confirmed', False):
            entry_score += 1
            entry_factors.append('volume_confirmed')
            
        # Divergence scoring
        if bullish_div['divergence'] and bullish_div['strength'] > 0.3:
            entry_score += 2
            entry_factors.append('bullish_divergence')
        if bearish_div['divergence'] and bearish_div['strength'] > 0.3:
            entry_score -= 2
            entry_factors.append('bearish_divergence_warning')
            
        # Final recommendation
        if entry_score >= 5:
            recommendation = 'strong_buy'
        elif entry_score >= 3:
            recommendation = 'buy'
        elif entry_score >= 1:
            recommendation = 'weak_buy'
        elif entry_score <= -2:
            recommendation = 'avoid'
        else:
            recommendation = 'wait'
            
        return {
            'recommendation': recommendation,
            'entry_score': entry_score,
            'entry_factors': entry_factors,
            'vwap_analysis': vwap_signal,
            'momentum_analysis': momentum_analysis,
            'bullish_divergence': bullish_div,
            'bearish_divergence': bearish_div,
            'current_price': current_price,
            'vwap_price': vwap_price
        }


# Example usage
if __name__ == "__main__":
    optimizer = SmartEntryOptimizer()
    
    # Sample market data
    sample_data = {
        'prices': [100, 101, 99, 102, 104, 103, 105, 107, 106, 108],
        'volumes': [1000, 1200, 800, 1500, 1800, 1000, 2000, 2200, 1500, 2500],
        'high_prices': [101, 102, 100, 103, 105, 104, 106, 108, 107, 109],
        'low_prices': [99, 100, 98, 101, 103, 102, 104, 106, 105, 107],
        'resistance_level': 105
    }
    
    analysis = optimizer.analyze_entry_opportunity(sample_data)
    print(f"Entry Recommendation: {analysis['recommendation']}")
    print(f"Entry Score: {analysis['entry_score']}")
    print(f"Key Factors: {analysis['entry_factors']}")