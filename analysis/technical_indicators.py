"""
Technical analysis indicators for better entry/exit timing
"""

import pandas as pd
import numpy as np
from ta import add_all_ta_features
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.volume import OnBalanceVolumeIndicator, VolumeWeightedAveragePrice
import logging

logger = logging.getLogger(__name__)

class TechnicalAnalyzer:
    def __init__(self):
        self.min_periods = 50  # Minimum periods needed for analysis
        
    def analyze(self, price_data):
        """Perform complete technical analysis"""
        if len(price_data) < self.min_periods:
            logger.warning(f"Insufficient data for TA: {len(price_data)} periods")
            return self._empty_analysis()
        
        try:
            # Ensure we have required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in required_cols:
                if col not in price_data.columns:
                    logger.error(f"Missing required column: {col}")
                    return self._empty_analysis()
            
            # Calculate all indicators
            analysis = {
                'momentum': self._analyze_momentum(price_data),
                'trend': self._analyze_trend(price_data),
                'volatility': self._analyze_volatility(price_data),
                'volume': self._analyze_volume(price_data),
                'support_resistance': self._find_support_resistance(price_data),
                'patterns': self._detect_patterns(price_data),
                'entry_signals': self._generate_entry_signals(price_data),
                'exit_signals': self._generate_exit_signals(price_data)
            }
            
            # Overall score
            analysis['overall_score'] = self._calculate_overall_score(analysis)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Technical analysis error: {e}")
            return self._empty_analysis()
    
    def _analyze_momentum(self, df):
        """Analyze momentum indicators"""
        close = df['close']
        high = df['high']
        low = df['low']
        
        # RSI
        rsi = RSIIndicator(close=close, window=14)
        current_rsi = rsi.rsi().iloc[-1]
        
        # Stochastic
        stoch = StochasticOscillator(high=high, low=low, close=close)
        current_stoch = stoch.stoch().iloc[-1]
        
        # MACD
        macd = MACD(close=close)
        macd_line = macd.macd().iloc[-1]
        signal_line = macd.macd_signal().iloc[-1]
        macd_diff = macd.macd_diff().iloc[-1]
        
        # Momentum score
        momentum_score = 0
        
        # RSI signals
        if current_rsi < 30:
            momentum_score += 20  # Oversold
        elif current_rsi > 70:
            momentum_score -= 20  # Overbought
        else:
            momentum_score += 10  # Neutral zone
        
        # MACD signals
        if macd_diff > 0 and macd_line > signal_line:
            momentum_score += 15  # Bullish crossover
        elif macd_diff < 0 and macd_line < signal_line:
            momentum_score -= 15  # Bearish crossover
        
        return {
            'rsi': current_rsi,
            'rsi_signal': 'oversold' if current_rsi < 30 else ('overbought' if current_rsi > 70 else 'neutral'),
            'stochastic': current_stoch,
            'macd': macd_line,
            'macd_signal': signal_line,
            'macd_diff': macd_diff,
            'momentum_score': momentum_score
        }
    
    def _analyze_trend(self, df):
        """Analyze trend indicators"""
        close = df['close']
        
        # Moving averages
        sma_20 = SMAIndicator(close=close, window=20).sma_indicator().iloc[-1]
        sma_50 = SMAIndicator(close=close, window=50).sma_indicator().iloc[-1]
        ema_12 = EMAIndicator(close=close, window=12).ema_indicator().iloc[-1]
        ema_26 = EMAIndicator(close=close, window=26).ema_indicator().iloc[-1]
        
        current_price = close.iloc[-1]
        
        # Trend strength
        trend_score = 0
        
        # Price vs MAs
        if current_price > sma_20 > sma_50:
            trend_score += 20  # Strong uptrend
        elif current_price > sma_20:
            trend_score += 10  # Uptrend
        elif current_price < sma_20 < sma_50:
            trend_score -= 20  # Strong downtrend
        elif current_price < sma_20:
            trend_score -= 10  # Downtrend
        
        # EMA crossover
        if ema_12 > ema_26:
            trend_score += 10  # Bullish
        else:
            trend_score -= 10  # Bearish
        
        # Trend direction over last 5 periods
        recent_trend = (close.iloc[-1] - close.iloc[-5]) / close.iloc[-5]
        
        return {
            'sma_20': sma_20,
            'sma_50': sma_50,
            'ema_12': ema_12,
            'ema_26': ema_26,
            'price_vs_sma20': (current_price - sma_20) / sma_20,
            'price_vs_sma50': (current_price - sma_50) / sma_50,
            'trend_direction': 'bullish' if trend_score > 0 else 'bearish',
            'trend_strength': abs(trend_score),
            'recent_trend': recent_trend,
            'trend_score': trend_score
        }
    
    def _analyze_volatility(self, df):
        """Analyze volatility indicators"""
        close = df['close']
        high = df['high']
        low = df['low']
        
        # Bollinger Bands
        bb = BollingerBands(close=close)
        bb_upper = bb.bollinger_hband().iloc[-1]
        bb_lower = bb.bollinger_lband().iloc[-1]
        bb_middle = bb.bollinger_mavg().iloc[-1]
        
        current_price = close.iloc[-1]
        
        # ATR
        atr = AverageTrueRange(high=high, low=low, close=close).average_true_range().iloc[-1]
        
        # Volatility metrics
        bb_width = (bb_upper - bb_lower) / bb_middle
        bb_position = (current_price - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5
        
        # Historical volatility
        returns = close.pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)  # Annualized
        
        return {
            'bb_upper': bb_upper,
            'bb_middle': bb_middle,
            'bb_lower': bb_lower,
            'bb_width': bb_width,
            'bb_position': bb_position,
            'atr': atr,
            'volatility': volatility,
            'volatility_signal': 'high' if bb_width > 0.1 else ('low' if bb_width < 0.05 else 'normal')
        }
    
    def _analyze_volume(self, df):
        """Analyze volume indicators"""
        close = df['close']
        volume = df['volume']
        high = df['high']
        low = df['low']
        
        # OBV
        obv = OnBalanceVolumeIndicator(close=close, volume=volume).on_balance_volume()
        obv_slope = (obv.iloc[-1] - obv.iloc[-10]) / 10  # 10-period slope
        
        # VWAP
        vwap = VolumeWeightedAveragePrice(
            high=high, low=low, close=close, volume=volume
        ).volume_weighted_average_price().iloc[-1]
        
        # Volume analysis
        avg_volume = volume.rolling(20).mean().iloc[-1]
        current_volume = volume.iloc[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        
        # Volume trend
        volume_trend = 'increasing' if obv_slope > 0 else 'decreasing'
        
        # Price vs VWAP
        current_price = close.iloc[-1]
        price_vs_vwap = (current_price - vwap) / vwap if vwap > 0 else 0
        
        return {
            'current_volume': current_volume,
            'avg_volume': avg_volume,
            'volume_ratio': volume_ratio,
            'obv_slope': obv_slope,
            'volume_trend': volume_trend,
            'vwap': vwap,
            'price_vs_vwap': price_vs_vwap,
            'volume_signal': 'high' if volume_ratio > 2 else ('low' if volume_ratio < 0.5 else 'normal')
        }
    
    def _find_support_resistance(self, df):
        """Find support and resistance levels"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Simple pivot points
        pivot = (high.iloc[-1] + low.iloc[-1] + close.iloc[-1]) / 3
        
        # Support levels
        support1 = 2 * pivot - high.iloc[-1]
        support2 = pivot - (high.iloc[-1] - low.iloc[-1])
        
        # Resistance levels
        resistance1 = 2 * pivot - low.iloc[-1]
        resistance2 = pivot + (high.iloc[-1] - low.iloc[-1])
        
        # Find recent swing highs/lows
        recent_high = high.tail(20).max()
        recent_low = low.tail(20).min()
        
        current_price = close.iloc[-1]
        
        return {
            'pivot': pivot,
            'support1': support1,
            'support2': support2,
            'resistance1': resistance1,
            'resistance2': resistance2,
            'recent_high': recent_high,
            'recent_low': recent_low,
            'nearest_support': max([s for s in [support1, support2, recent_low] if s < current_price], default=0),
            'nearest_resistance': min([r for r in [resistance1, resistance2, recent_high] if r > current_price], default=float('inf'))
        }
    
    def _detect_patterns(self, df):
        """Detect chart patterns"""
        close = df['close']
        patterns = []
        
        # Breakout detection
        recent_high = close.tail(20).max()
        if close.iloc[-1] > recent_high * 0.98:
            patterns.append('potential_breakout')
        
        # Double bottom
        recent_prices = close.tail(50)
        lows = recent_prices[recent_prices == recent_prices.rolling(5).min()]
        if len(lows) >= 2:
            if abs(lows.iloc[-1] - lows.iloc[-2]) / lows.iloc[-2] < 0.03:  # Within 3%
                patterns.append('double_bottom')
        
        # Ascending triangle (simplified)
        if len(close) > 20:
            highs = close.tail(20).rolling(5).max()
            lows = close.tail(20).rolling(5).min()
            
            if highs.iloc[-1] > highs.iloc[-10] * 0.98 and lows.iloc[-1] > lows.iloc[-10]:
                patterns.append('ascending_triangle')
        
        return patterns
    
    def _generate_entry_signals(self, df):
        """Generate entry signals based on technical indicators"""
        signals = []
        strength = 0
        
        # Get current analysis
        momentum = self._analyze_momentum(df)
        trend = self._analyze_trend(df)
        volatility = self._analyze_volatility(df)
        volume = self._analyze_volume(df)
        
        # RSI oversold
        if momentum['rsi'] < 30:
            signals.append('rsi_oversold')
            strength += 20
        
        # Bollinger Band squeeze
        if volatility['bb_position'] < 0.2:
            signals.append('bb_squeeze_low')
            strength += 15
        
        # Volume spike
        if volume['volume_ratio'] > 2:
            signals.append('volume_spike')
            strength += 15
        
        # Uptrend confirmation
        if trend['trend_direction'] == 'bullish':
            signals.append('uptrend')
            strength += 20
        
        # MACD bullish
        if momentum['macd_diff'] > 0:
            signals.append('macd_bullish')
            strength += 10
        
        # Price above VWAP
        if volume['price_vs_vwap'] > 0:
            signals.append('above_vwap')
            strength += 10
        
        return {
            'signals': signals,
            'strength': strength,
            'action': 'strong_buy' if strength > 60 else ('buy' if strength > 40 else 'wait')
        }
    
    def _generate_exit_signals(self, df):
        """Generate exit signals based on technical indicators"""
        signals = []
        strength = 0
        
        # Get current analysis
        momentum = self._analyze_momentum(df)
        trend = self._analyze_trend(df)
        volatility = self._analyze_volatility(df)
        
        # RSI overbought
        if momentum['rsi'] > 70:
            signals.append('rsi_overbought')
            strength += 20
        
        # At resistance
        if volatility['bb_position'] > 0.8:
            signals.append('at_bb_upper')
            strength += 15
        
        # Trend reversal
        if trend['trend_direction'] == 'bearish' and trend['recent_trend'] < -0.05:
            signals.append('trend_reversal')
            strength += 25
        
        # MACD bearish
        if momentum['macd_diff'] < 0:
            signals.append('macd_bearish')
            strength += 15
        
        return {
            'signals': signals,
            'strength': strength,
            'action': 'strong_sell' if strength > 50 else ('sell' if strength > 30 else 'hold')
        }
    
    def _calculate_overall_score(self, analysis):
        """Calculate overall technical score"""
        score = 50  # Start neutral
        
        # Momentum
        score += analysis['momentum']['momentum_score'] * 0.3
        
        # Trend
        score += analysis['trend']['trend_score'] * 0.3
        
        # Entry signals
        score += analysis['entry_signals']['strength'] * 0.2
        
        # Exit signals (negative)
        score -= analysis['exit_signals']['strength'] * 0.2
        
        return max(0, min(100, score))
    
    def _empty_analysis(self):
        """Return empty analysis structure"""
        return {
            'momentum': {'rsi': 50, 'momentum_score': 0},
            'trend': {'trend_direction': 'neutral', 'trend_score': 0},
            'volatility': {'bb_position': 0.5, 'volatility': 0},
            'volume': {'volume_ratio': 1, 'volume_signal': 'normal'},
            'support_resistance': {},
            'patterns': [],
            'entry_signals': {'signals': [], 'strength': 0, 'action': 'wait'},
            'exit_signals': {'signals': [], 'strength': 0, 'action': 'hold'},
            'overall_score': 50
        }