# analysis/technical.py - Improved version with better error handling
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import math
import warnings

# Suppress pandas warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)

# Import TA libraries (alternatives to TA-Lib)
try:
    import ta
    TA_AVAILABLE = True
    print("✓ Using 'ta' library for technical analysis")
except ImportError:
    TA_AVAILABLE = False
    print("⚠ 'ta' library not available, using custom implementations")

try:
    import pandas_ta as pta
    PANDAS_TA_AVAILABLE = True
    print("✓ Using 'pandas_ta' library for additional indicators")
except ImportError:
    PANDAS_TA_AVAILABLE = False
    print("⚠ 'pandas_ta' library not available")

@dataclass
class TechnicalIndicators:
    # Basic indicators
    rsi: float = 50.0
    rsi_signal: str = "NEUTRAL"
    
    # MACD
    macd: float = 0.0
    macd_signal: float = 0.0
    macd_histogram: float = 0.0
    macd_signal_line: str = "NEUTRAL"
    
    # Bollinger Bands
    bb_upper: float = 0.0
    bb_lower: float = 0.0
    bb_middle: float = 0.0
    bb_position: float = 0.5  # 0-1 scale, where price sits in bands
    bb_squeeze: bool = False
    
    # Moving Averages
    ema_12: float = 0.0
    ema_26: float = 0.0
    sma_20: float = 0.0
    sma_50: float = 0.0
    sma_200: float = 0.0
    
    # Volume indicators
    volume_sma: float = 0.0
    volume_ratio: float = 1.0  # Current volume vs average
    
    # Price action
    price_change_24h: float = 0.0
    price_change_7d: float = 0.0
    volatility: float = 0.0
    
    # Support/Resistance
    support_level: float = 0.0
    resistance_level: float = 0.0
    
    # Advanced indicators
    stoch_k: float = 50.0
    stoch_d: float = 50.0
    williams_r: float = -50.0
    atr: float = 0.0  # Average True Range
    
    # Momentum
    momentum: float = 0.0
    rate_of_change: float = 0.0

class TechnicalAnalysis:
    def __init__(self):
        self.indicators = {}
        self.min_periods = {
            'rsi': 14,
            'macd': 26,
            'bollinger': 20,
            'stochastic': 14,
            'williams_r': 14,
            'atr': 14
        }
    
    def _prepare_dataframe(self, price_data: List[Dict]) -> pd.DataFrame:
        """Convert price data to pandas DataFrame for TA libraries"""
        try:
            if not price_data:
                return pd.DataFrame()
            
            # Extract data based on available fields
            df_data = []
            for item in price_data:
                if isinstance(item, dict):
                    # Try different field names
                    price = float(item.get('price', item.get('close', item.get('c', 0))))
                    volume = float(item.get('volume', item.get('vol', item.get('v', 0))))
                    high = float(item.get('high', item.get('h', price)))
                    low = float(item.get('low', item.get('l', price)))
                    open_price = float(item.get('open', item.get('o', price)))
                    
                    # Use timestamp if available, otherwise create index
                    timestamp = item.get('timestamp', item.get('time', item.get('date')))
                    
                    df_data.append({
                        'open': open_price,
                        'high': high,
                        'low': low,
                        'close': price,
                        'volume': volume,
                        'timestamp': timestamp
                    })
                else:
                    # Handle simple price list
                    price = float(item)
                    df_data.append({
                        'open': price,
                        'high': price,
                        'low': price,
                        'close': price,
                        'volume': 0,
                        'timestamp': None
                    })
            
            df = pd.DataFrame(df_data)
            
            # Create index
            if 'timestamp' in df.columns and df['timestamp'].notna().any():
                try:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df.set_index('timestamp', inplace=True)
                except:
                    df.index = range(len(df))
            else:
                df.index = range(len(df))
            
            return df
            
        except Exception as e:
            print(f"Error preparing DataFrame: {e}")
            return pd.DataFrame()
    
    def calculate_rsi_with_ta(self, df: pd.DataFrame, period: int = 14) -> Tuple[float, str]:
        """Calculate RSI using TA library with better error handling"""
        try:
            if len(df) < period:
                return 50.0, "NEUTRAL"
            
            if TA_AVAILABLE:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    rsi_series = ta.momentum.RSIIndicator(df['close'], window=period).rsi()
                    rsi = rsi_series.iloc[-1] if not rsi_series.empty and not pd.isna(rsi_series.iloc[-1]) else 50.0
            else:
                # Fallback to custom implementation
                rsi = self.calculate_rsi_custom(df['close'].tolist(), period)[0]
            
            # Generate signal
            if rsi > 70:
                signal = "SELL"
            elif rsi < 30:
                signal = "BUY"
            else:
                signal = "NEUTRAL"
            
            return float(rsi), signal
            
        except Exception as e:
            print(f"Error calculating RSI: {e}")
            return 50.0, "NEUTRAL"
    
    def calculate_macd_with_ta(self, df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[float, float, float, str]:
        """Calculate MACD using TA library with better error handling"""
        try:
            if len(df) < slow:
                return 0.0, 0.0, 0.0, "NEUTRAL"
            
            if TA_AVAILABLE:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    macd_indicator = ta.trend.MACD(df['close'], window_fast=fast, window_slow=slow, window_sign=signal)
                    
                    macd_line = 0.0
                    signal_line = 0.0
                    histogram = 0.0
                    
                    if not macd_indicator.macd().empty:
                        macd_val = macd_indicator.macd().iloc[-1]
                        if not pd.isna(macd_val):
                            macd_line = float(macd_val)
                    
                    if not macd_indicator.macd_signal().empty:
                        signal_val = macd_indicator.macd_signal().iloc[-1]
                        if not pd.isna(signal_val):
                            signal_line = float(signal_val)
                    
                    if not macd_indicator.macd_diff().empty:
                        hist_val = macd_indicator.macd_diff().iloc[-1]
                        if not pd.isna(hist_val):
                            histogram = float(hist_val)
            else:
                # Fallback to custom implementation
                macd_line, signal_line, histogram, _ = self.calculate_macd_custom(df['close'].tolist(), fast, slow, signal)
            
            # Generate signal
            if macd_line > signal_line and histogram > 0:
                macd_signal = "BUY"
            elif macd_line < signal_line and histogram < 0:
                macd_signal = "SELL"
            else:
                macd_signal = "NEUTRAL"
            
            return macd_line, signal_line, histogram, macd_signal
            
        except Exception as e:
            print(f"Error calculating MACD: {e}")
            return 0.0, 0.0, 0.0, "NEUTRAL"
    
    def calculate_atr_with_ta(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range using TA library with better error handling"""
        try:
            if len(df) < max(2, period):
                return 0.0
            
            if TA_AVAILABLE:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # Ensure we have enough data
                    if len(df) >= period:
                        atr_series = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=period).average_true_range()
                        if not atr_series.empty:
                            atr_val = atr_series.iloc[-1]
                            if not pd.isna(atr_val):
                                return float(atr_val)
            
            # Fallback to custom implementation
            return self.calculate_atr_custom(df['high'].tolist(), df['low'].tolist(), df['close'].tolist(), period)
            
        except Exception as e:
            # Silently handle ATR errors for minimal data
            return 0.0
    
    def calculate_williams_r_with_ta(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Williams %R using TA library with better error handling"""
        try:
            if len(df) < period:
                return -50.0
            
            if TA_AVAILABLE:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    williams_series = ta.momentum.WilliamsRIndicator(df['high'], df['low'], df['close'], lbp=period).williams_r()
                    if not williams_series.empty:
                        williams_val = williams_series.iloc[-1]
                        if not pd.isna(williams_val):
                            return float(williams_val) * 100  # Convert to percentage
            
            # Fallback to custom implementation
            return self.calculate_williams_r_custom(df['high'].tolist(), df['low'].tolist(), df['close'].tolist(), period)
            
        except Exception as e:
            print(f"Error calculating Williams %R: {e}")
            return -50.0
    
    def calculate_bollinger_bands_with_ta(self, df: pd.DataFrame, period: int = 20, std_dev: float = 2) -> Tuple[float, float, float, float, bool]:
        """Calculate Bollinger Bands using TA library with better error handling"""
        try:
            if len(df) < period:
                current_price = df['close'].iloc[-1] if not df.empty else 0
                return current_price * 1.02, current_price, current_price * 0.98, 0.5, False
            
            if TA_AVAILABLE:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    bb_indicator = ta.volatility.BollingerBands(df['close'], window=period, window_dev=std_dev)
                    
                    upper = df['close'].iloc[-1] * 1.02  # Default values
                    middle = df['close'].iloc[-1]
                    lower = df['close'].iloc[-1] * 0.98
                    
                    if not bb_indicator.bollinger_hband().empty:
                        upper_val = bb_indicator.bollinger_hband().iloc[-1]
                        if not pd.isna(upper_val):
                            upper = float(upper_val)
                    
                    if not bb_indicator.bollinger_mavg().empty:
                        middle_val = bb_indicator.bollinger_mavg().iloc[-1]
                        if not pd.isna(middle_val):
                            middle = float(middle_val)
                    
                    if not bb_indicator.bollinger_lband().empty:
                        lower_val = bb_indicator.bollinger_lband().iloc[-1]
                        if not pd.isna(lower_val):
                            lower = float(lower_val)
            else:
                # Fallback to custom implementation
                upper, middle, lower, _, _ = self.calculate_bollinger_bands_custom(df['close'].tolist(), period, std_dev)
            
            # Calculate current position within bands
            current_price = df['close'].iloc[-1]
            if upper != lower and upper > lower:
                position = (current_price - lower) / (upper - lower)
                position = max(0.0, min(1.0, position))  # Clamp between 0 and 1
            else:
                position = 0.5
            
            # Detect squeeze (bands are narrow)
            band_width = (upper - lower) / middle if middle != 0 else 0
            squeeze = band_width < 0.1  # 10% width indicates squeeze
            
            return upper, middle, lower, position, squeeze
            
        except Exception as e:
            print(f"Error calculating Bollinger Bands: {e}")
            current_price = df['close'].iloc[-1] if not df.empty else 0
            return current_price * 1.02, current_price, current_price * 0.98, 0.5, False
    
    def calculate_moving_averages_with_ta(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate various moving averages using TA library with better error handling"""
        try:
            if df.empty:
                return {'ema_12': 0, 'ema_26': 0, 'sma_20': 0, 'sma_50': 0, 'sma_200': 0}
            
            mas = {}
            
            if TA_AVAILABLE:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    
                    # EMA 12
                    if len(df) >= 12:
                        ema12_series = ta.trend.EMAIndicator(df['close'], window=12).ema_indicator()
                        if not ema12_series.empty:
                            ema12_val = ema12_series.iloc[-1]
                            mas['ema_12'] = float(ema12_val) if not pd.isna(ema12_val) else 0
                        else:
                            mas['ema_12'] = 0
                    else:
                        mas['ema_12'] = 0
                    
                    # EMA 26
                    if len(df) >= 26:
                        ema26_series = ta.trend.EMAIndicator(df['close'], window=26).ema_indicator()
                        if not ema26_series.empty:
                            ema26_val = ema26_series.iloc[-1]
                            mas['ema_26'] = float(ema26_val) if not pd.isna(ema26_val) else 0
                        else:
                            mas['ema_26'] = 0
                    else:
                        mas['ema_26'] = 0
                    
                    # SMA 20
                    if len(df) >= 20:
                        sma20_series = ta.trend.SMAIndicator(df['close'], window=20).sma_indicator()
                        if not sma20_series.empty:
                            sma20_val = sma20_series.iloc[-1]
                            mas['sma_20'] = float(sma20_val) if not pd.isna(sma20_val) else 0
                        else:
                            mas['sma_20'] = 0
                    else:
                        mas['sma_20'] = 0
                    
                    # SMA 50
                    if len(df) >= 50:
                        sma50_series = ta.trend.SMAIndicator(df['close'], window=50).sma_indicator()
                        if not sma50_series.empty:
                            sma50_val = sma50_series.iloc[-1]
                            mas['sma_50'] = float(sma50_val) if not pd.isna(sma50_val) else 0
                        else:
                            mas['sma_50'] = 0
                    else:
                        mas['sma_50'] = 0
                    
                    # SMA 200
                    if len(df) >= 200:
                        sma200_series = ta.trend.SMAIndicator(df['close'], window=200).sma_indicator()
                        if not sma200_series.empty:
                            sma200_val = sma200_series.iloc[-1]
                            mas['sma_200'] = float(sma200_val) if not pd.isna(sma200_val) else 0
                        else:
                            mas['sma_200'] = 0
                    else:
                        mas['sma_200'] = 0
            else:
                # Fallback to custom implementation
                prices = df['close'].tolist()
                mas = {
                    'ema_12': self._calculate_ema(np.array(prices), 12) if len(prices) >= 12 else 0,
                    'ema_26': self._calculate_ema(np.array(prices), 26) if len(prices) >= 26 else 0,
                    'sma_20': self._calculate_sma(prices, 20) if len(prices) >= 20 else 0,
                    'sma_50': self._calculate_sma(prices, 50) if len(prices) >= 50 else 0,
                    'sma_200': self._calculate_sma(prices, 200) if len(prices) >= 200 else 0,
                }
            
            return mas
            
        except Exception as e:
            print(f"Error calculating moving averages: {e}")
            return {'ema_12': 0, 'ema_26': 0, 'sma_20': 0, 'sma_50': 0, 'sma_200': 0}
    
    def calculate_stochastic_with_ta(self, df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Tuple[float, float]:
        """Calculate Stochastic Oscillator using TA library with better error handling"""
        try:
            if len(df) < k_period:
                return 50.0, 50.0
            
            if TA_AVAILABLE:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    stoch_indicator = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'], window=k_period, smooth_window=d_period)
                    
                    stoch_k = 50.0
                    stoch_d = 50.0
                    
                    if not stoch_indicator.stoch().empty:
                        k_val = stoch_indicator.stoch().iloc[-1]
                        if not pd.isna(k_val):
                            stoch_k = float(k_val)
                    
                    if not stoch_indicator.stoch_signal().empty:
                        d_val = stoch_indicator.stoch_signal().iloc[-1]
                        if not pd.isna(d_val):
                            stoch_d = float(d_val)
            else:
                # Fallback to custom implementation
                stoch_k, stoch_d = self.calculate_stochastic_custom(df['high'].tolist(), df['low'].tolist(), df['close'].tolist(), k_period, d_period)
            
            return stoch_k, stoch_d
            
        except Exception as e:
            print(f"Error calculating Stochastic: {e}")
            return 50.0, 50.0
    
    def analyze(self, price_data: List[Dict], symbol: str = "UNKNOWN") -> TechnicalIndicators:
        """Perform complete technical analysis using TA libraries with better error handling"""
        if not price_data:
            return self._default_indicators()
        
        try:
            # Prepare DataFrame
            df = self._prepare_dataframe(price_data)
            if df.empty:
                return self._default_indicators()
            
            # Calculate indicators using TA library when available
            rsi, rsi_signal = self.calculate_rsi_with_ta(df)
            macd, macd_signal, macd_histogram, macd_signal_line = self.calculate_macd_with_ta(df)
            bb_upper, bb_middle, bb_lower, bb_position, bb_squeeze = self.calculate_bollinger_bands_with_ta(df)
            
            # Moving averages
            mas = self.calculate_moving_averages_with_ta(df)
            
            # Volume analysis
            volume_sma = 0
            current_volume = 0
            volume_ratio = 1.0
            
            if 'volume' in df.columns and len(df) >= 20:
                try:
                    volume_sma = df['volume'].rolling(window=20).mean().iloc[-1]
                    current_volume = df['volume'].iloc[-1]
                    if volume_sma > 0:
                        volume_ratio = current_volume / volume_sma
                except:
                    pass
            
            # Price changes
            prices = df['close'].tolist()
            price_change_24h = 0
            price_change_7d = 0
            
            try:
                if len(prices) >= 24 and prices[-24] != 0:
                    price_change_24h = ((prices[-1] - prices[-24]) / prices[-24] * 100)
                if len(prices) >= 168 and prices[-168] != 0:
                    price_change_7d = ((prices[-1] - prices[-168]) / prices[-168] * 100)
            except:
                pass
            
            # Volatility (standard deviation of recent returns)
            volatility = 0
            try:
                if len(prices) >= 20:
                    returns = df['close'].pct_change().dropna()
                    if len(returns) >= 10:
                        volatility = returns.tail(20).std() * 100
            except:
                pass
            
            # Support and resistance
            support_level, resistance_level = self.calculate_support_resistance(prices)
            
            # Advanced indicators
            stoch_k, stoch_d = self.calculate_stochastic_with_ta(df)
            williams_r = self.calculate_williams_r_with_ta(df)
            atr = self.calculate_atr_with_ta(df)
            
            # Momentum indicators
            momentum = 0
            rate_of_change = 0
            
            try:
                if len(prices) >= 10 and prices[-10] != 0:
                    momentum = ((prices[-1] - prices[-10]) / prices[-10] * 100)
                if len(prices) >= 12 and prices[-12] != 0:
                    rate_of_change = ((prices[-1] - prices[-12]) / prices[-12] * 100)
            except:
                pass
            
            return TechnicalIndicators(
                rsi=rsi,
                rsi_signal=rsi_signal,
                macd=macd,
                macd_signal=macd_signal,
                macd_histogram=macd_histogram,
                macd_signal_line=macd_signal_line,
                bb_upper=bb_upper,
                bb_lower=bb_lower,
                bb_middle=bb_middle,
                bb_position=bb_position,
                bb_squeeze=bb_squeeze,
                ema_12=mas['ema_12'],
                ema_26=mas['ema_26'],
                sma_20=mas['sma_20'],
                sma_50=mas['sma_50'],
                sma_200=mas['sma_200'],
                volume_sma=volume_sma,
                volume_ratio=volume_ratio,
                price_change_24h=price_change_24h,
                price_change_7d=price_change_7d,
                volatility=volatility,
                support_level=support_level,
                resistance_level=resistance_level,
                stoch_k=stoch_k,
                stoch_d=stoch_d,
                williams_r=williams_r,
                atr=atr,
                momentum=momentum,
                rate_of_change=rate_of_change
            )
        
        except Exception as e:
            print(f"Error in technical analysis for {symbol}: {e}")
            return self._default_indicators()
    
    # Keep all the existing fallback custom implementations and other methods...
    # (The rest of the methods remain the same as in the previous version)
    
    def calculate_rsi_custom(self, prices: List[float], period: int = 14) -> Tuple[float, str]:
        """Custom RSI calculation fallback"""
        if len(prices) < period + 1:
            return 50.0, "NEUTRAL"
        
        try:
            prices_array = np.array(prices)
            deltas = np.diff(prices_array)
            
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            avg_gain = np.mean(gains[-period:])
            avg_loss = np.mean(losses[-period:])
            
            if avg_loss == 0:
                rsi = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            
            signal = "SELL" if rsi > 70 else "BUY" if rsi < 30 else "NEUTRAL"
            return rsi, signal
        except Exception:
            return 50.0, "NEUTRAL"
    
    def calculate_macd_custom(self, prices: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[float, float, float, str]:
        """Custom MACD calculation fallback"""
        if len(prices) < slow:
            return 0.0, 0.0, 0.0, "NEUTRAL"
        
        try:
            prices_array = np.array(prices)
            ema_fast = self._calculate_ema(prices_array, fast)
            ema_slow = self._calculate_ema(prices_array, slow)
            
            macd_line = ema_fast - ema_slow
            signal_line = macd_line  # Simplified
            histogram = macd_line - signal_line
            
            macd_signal = "BUY" if macd_line > signal_line else "SELL" if macd_line < signal_line else "NEUTRAL"
            return macd_line, signal_line, histogram, macd_signal
        except Exception:
            return 0.0, 0.0, 0.0, "NEUTRAL"
    
    def calculate_bollinger_bands_custom(self, prices: List[float], period: int = 20, std_dev: float = 2) -> Tuple[float, float, float, float, bool]:
        """Custom Bollinger Bands calculation fallback"""
        if len(prices) < period:
            current_price = prices[-1] if prices else 0
            return current_price * 1.02, current_price, current_price * 0.98, 0.5, False
        
        try:
            recent_prices = np.array(prices[-period:])
            middle = np.mean(recent_prices)
            std = np.std(recent_prices)
            
            upper = middle + (std_dev * std)
            lower = middle - (std_dev * std)
            
            current_price = prices[-1]
            position = (current_price - lower) / (upper - lower) if upper != lower else 0.5
            squeeze = (upper - lower) / middle < 0.1 if middle != 0 else False
            
            return upper, middle, lower, position, squeeze
        except Exception:
            current_price = prices[-1] if prices else 0
            return current_price * 1.02, current_price, current_price * 0.98, 0.5, False
    
    def calculate_stochastic_custom(self, highs: List[float], lows: List[float], closes: List[float], k_period: int = 14, d_period: int = 3) -> Tuple[float, float]:
        """Custom Stochastic calculation fallback"""
        if len(closes) < k_period:
            return 50.0, 50.0
        
        try:
            if not highs:
                highs = closes
            if not lows:
                lows = closes
            
            period_high = max(highs[-k_period:])
            period_low = min(lows[-k_period:])
            current_close = closes[-1]
            
            if period_high != period_low:
                stoch_k = ((current_close - period_low) / (period_high - period_low)) * 100
            else:
                stoch_k = 50.0
            
            stoch_d = stoch_k  # Simplified
            return stoch_k, stoch_d
        except Exception:
            return 50.0, 50.0
    
    def calculate_williams_r_custom(self, highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> float:
        """Custom Williams %R calculation fallback"""
        if len(closes) < period:
            return -50.0
        
        try:
            if not highs:
                highs = closes
            if not lows:
                lows = closes
            
            highest_high = max(highs[-period:])
            lowest_low = min(lows[-period:])
            current_close = closes[-1]
            
            if highest_high != lowest_low:
                williams_r = ((highest_high - current_close) / (highest_high - lowest_low)) * -100
            else:
                williams_r = -50.0
            
            return williams_r
        except Exception:
            return -50.0
    
    def calculate_atr_custom(self, highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> float:
        """Custom ATR calculation fallback"""
        if len(closes) < 2:
            return 0.0
        
        try:
            if not highs:
                highs = closes
            if not lows:
                lows = closes
            
            true_ranges = []
            for i in range(1, min(len(closes), len(highs), len(lows))):
                high_low = highs[i] - lows[i]
                high_close_prev = abs(highs[i] - closes[i - 1])
                low_close_prev = abs(lows[i] - closes[i - 1])
                
                true_range = max(high_low, high_close_prev, low_close_prev)
                true_ranges.append(true_range)
            
            if len(true_ranges) >= period:
                atr = np.mean(true_ranges[-period:])
            elif true_ranges:
                atr = np.mean(true_ranges)
            else:
                atr = 0.0
            
            return atr
        except Exception:
            return 0.0
    
    def _calculate_ema(self, prices: np.ndarray, period: int) -> float:
        """Calculate Exponential Moving Average"""
        if len(prices) == 0:
            return 0.0
        
        alpha = 2 / (period + 1)
        ema = prices[0]
        
        for price in prices[1:]:
            ema = alpha * price + (1 - alpha) * ema
        
        return ema
    
    def _calculate_sma(self, prices: List[float], period: int) -> float:
        """Calculate Simple Moving Average"""
        if len(prices) < period:
            return np.mean(prices) if prices else 0.0
        
        return np.mean(prices[-period:])
    
    def calculate_support_resistance(self, prices: List[float], volume: List[float] = None) -> Tuple[float, float]:
        """Calculate support and resistance levels"""
        if len(prices) < 10:
            current_price = prices[-1] if prices else 0
            return current_price * 0.95, current_price * 1.05
        
        try:
            prices_array = np.array(prices)
            recent_prices = prices_array[-50:]  # Last 50 data points
            
            support = np.percentile(recent_prices, 20)  # 20th percentile as support
            resistance = np.percentile(recent_prices, 80)  # 80th percentile as resistance
            
            return support, resistance
        except Exception:
            current_price = prices[-1] if prices else 0
            return current_price * 0.95, current_price * 1.05
    
    def _default_indicators(self) -> TechnicalIndicators:
        """Return default indicators when calculation fails"""
        return TechnicalIndicators()
    
    def get_trading_signals(self, indicators: TechnicalIndicators) -> Dict[str, str]:
        """Generate trading signals based on indicators"""
        signals = {}
        
        # RSI signals
        signals['rsi'] = indicators.rsi_signal
        
        # MACD signals
        signals['macd'] = indicators.macd_signal_line
        
        # Bollinger Bands signals
        if indicators.bb_position > 0.8:
            signals['bollinger'] = 'SELL'
        elif indicators.bb_position < 0.2:
            signals['bollinger'] = 'BUY'
        else:
            signals['bollinger'] = 'HOLD'
        
        # EMA crossover signals
        if indicators.ema_12 > indicators.ema_26:
            signals['ema_crossover'] = 'BUY'
        elif indicators.ema_12 < indicators.ema_26:
            signals['ema_crossover'] = 'SELL'
        else:
            signals['ema_crossover'] = 'HOLD'
        
        # Moving average trend signals
        current_price = indicators.bb_middle  # Use middle BB as proxy for current price
        if current_price > indicators.sma_20 > indicators.sma_50:
            signals['trend'] = 'BULLISH'
        elif current_price < indicators.sma_20 < indicators.sma_50:
            signals['trend'] = 'BEARISH'
        else:
            signals['trend'] = 'SIDEWAYS'
        
        # Stochastic signals
        if indicators.stoch_k > 80 and indicators.stoch_d > 80:
            signals['stochastic'] = 'SELL'
        elif indicators.stoch_k < 20 and indicators.stoch_d < 20:
            signals['stochastic'] = 'BUY'
        else:
            signals['stochastic'] = 'HOLD'
        
        # Volume confirmation
        if indicators.volume_ratio > 1.5:
            signals['volume'] = 'HIGH'
        elif indicators.volume_ratio < 0.5:
            signals['volume'] = 'LOW'
        else:
            signals['volume'] = 'NORMAL'
        
        # Momentum signals
        if indicators.momentum > 5:
            signals['momentum'] = 'STRONG_BUY'
        elif indicators.momentum > 2:
            signals['momentum'] = 'BUY'
        elif indicators.momentum < -5:
            signals['momentum'] = 'STRONG_SELL'
        elif indicators.momentum < -2:
            signals['momentum'] = 'SELL'
        else:
            signals['momentum'] = 'NEUTRAL'
        
        return signals
    
    def get_overall_signal(self, indicators: TechnicalIndicators) -> Dict[str, Any]:
        """Get overall trading signal based on all indicators"""
        signals = self.get_trading_signals(indicators)
        
        # Count bullish and bearish signals
        buy_signals = 0
        sell_signals = 0
        neutral_signals = 0
        
        signal_weights = {
            'rsi': 1.5,
            'macd': 2.0,
            'bollinger': 1.0,
            'ema_crossover': 1.5,
            'stochastic': 1.0,
            'momentum': 1.2
        }
        
        total_weight = 0
        weighted_score = 0
        
        for signal_name, signal_value in signals.items():
            if signal_name in signal_weights:
                weight = signal_weights[signal_name]
                total_weight += weight
                
                if signal_value in ['BUY', 'STRONG_BUY', 'BULLISH']:
                    weighted_score += weight
                    buy_signals += 1
                elif signal_value in ['SELL', 'STRONG_SELL', 'BEARISH']:
                    weighted_score -= weight
                    sell_signals += 1
                else:
                    neutral_signals += 1
        
        # Calculate overall score (-1 to 1)
        overall_score = weighted_score / total_weight if total_weight > 0 else 0
        
        # Determine overall signal
        if overall_score > 0.3:
            overall_signal = 'STRONG_BUY'
        elif overall_score > 0.1:
            overall_signal = 'BUY'
        elif overall_score < -0.3:
            overall_signal = 'STRONG_SELL'
        elif overall_score < -0.1:
            overall_signal = 'SELL'
        else:
            overall_signal = 'HOLD'
        
        # Calculate confidence based on signal agreement
        total_signals = buy_signals + sell_signals + neutral_signals
        max_agreement = max(buy_signals, sell_signals, neutral_signals)
        confidence = max_agreement / total_signals if total_signals > 0 else 0
        
        return {
            'overall_signal': overall_signal,
            'score': overall_score,
            'confidence': confidence,
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'neutral_signals': neutral_signals,
            'individual_signals': signals,
            'strength': self._get_signal_strength(abs(overall_score))
        }
    
    def _get_signal_strength(self, score: float) -> str:
        """Convert signal score to strength description"""
        if score >= 0.7:
            return "VERY_STRONG"
        elif score >= 0.5:
            return "STRONG"
        elif score >= 0.3:
            return "MODERATE"
        elif score >= 0.1:
            return "WEAK"
        else:
            return "VERY_WEAK"
    
    def get_risk_metrics(self, indicators: TechnicalIndicators) -> Dict[str, float]:
        """Calculate risk metrics based on technical indicators"""
        try:
            # Volatility risk
            volatility_risk = min(indicators.volatility / 10.0, 1.0)  # Normalize to 0-1
            
            # Support/resistance risk
            current_price = indicators.bb_middle
            if current_price > 0 and indicators.resistance_level > indicators.support_level:
                price_position = (current_price - indicators.support_level) / (indicators.resistance_level - indicators.support_level)
                sr_risk = abs(price_position - 0.5) * 2  # Risk increases at extremes
            else:
                sr_risk = 0.5
            
            # RSI risk (extreme values indicate higher risk)
            rsi_risk = 0
            if indicators.rsi > 70 or indicators.rsi < 30:
                rsi_risk = abs(indicators.rsi - 50) / 50.0
            
            # Bollinger Band risk
            bb_risk = abs(indicators.bb_position - 0.5) * 2
            
            # ATR risk (normalized)
            atr_risk = min(indicators.atr / current_price, 0.1) * 10 if current_price > 0 else 0
            
            # Overall risk score
            risk_components = [volatility_risk, sr_risk, rsi_risk, bb_risk, atr_risk]
            overall_risk = sum(risk_components) / len(risk_components)
            
            return {
                'overall_risk': overall_risk,
                'volatility_risk': volatility_risk,
                'support_resistance_risk': sr_risk,
                'rsi_risk': rsi_risk,
                'bollinger_risk': bb_risk,
                'atr_risk': atr_risk,
                'risk_level': self._get_risk_level(overall_risk)
            }
        
        except Exception as e:
            print(f"Error calculating risk metrics: {e}")
            return {
                'overall_risk': 0.5,
                'volatility_risk': 0.5,
                'support_resistance_risk': 0.5,
                'rsi_risk': 0.5,
                'bollinger_risk': 0.5,
                'atr_risk': 0.5,
                'risk_level': 'MEDIUM'
            }
    
    def _get_risk_level(self, risk_score: float) -> str:
        """Convert risk score to risk level"""
        if risk_score >= 0.8:
            return "VERY_HIGH"
        elif risk_score >= 0.6:
            return "HIGH"
        elif risk_score >= 0.4:
            return "MEDIUM"
        elif risk_score >= 0.2:
            return "LOW"
        else:
            return "VERY_LOW"
    
    def generate_analysis_report(self, indicators: TechnicalIndicators, symbol: str) -> Dict[str, Any]:
        """Generate comprehensive technical analysis report"""
        signals = self.get_overall_signal(indicators)
        risk_metrics = self.get_risk_metrics(indicators)
        
        # Key observations
        observations = []
        
        if indicators.rsi > 70:
            observations.append("RSI indicates overbought conditions")
        elif indicators.rsi < 30:
            observations.append("RSI indicates oversold conditions")
        
        if indicators.bb_squeeze:
            observations.append("Bollinger Bands show squeeze - potential breakout")
        
        if indicators.volume_ratio > 2:
            observations.append("Unusually high volume detected")
        elif indicators.volume_ratio < 0.5:
            observations.append("Below average volume")
        
        if abs(indicators.momentum) > 10:
            direction = "upward" if indicators.momentum > 0 else "downward"
            observations.append(f"Strong {direction} momentum detected")
        
        # Price levels
        current_price = indicators.bb_middle
        support_distance = ((current_price - indicators.support_level) / current_price * 100) if current_price > 0 else 0
        resistance_distance = ((indicators.resistance_level - current_price) / current_price * 100) if current_price > 0 else 0
        
        return {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'ta_library_used': 'ta' if TA_AVAILABLE else 'custom',
            'overall_signal': signals['overall_signal'],
            'confidence': signals['confidence'],
            'risk_level': risk_metrics['risk_level'],
            'current_price': current_price,
            'support_level': indicators.support_level,
            'resistance_level': indicators.resistance_level,
            'support_distance_pct': support_distance,
            'resistance_distance_pct': resistance_distance,
            'key_indicators': {
                'rsi': indicators.rsi,
                'macd': indicators.macd,
                'bollinger_position': indicators.bb_position,
                'volume_ratio': indicators.volume_ratio,
                'volatility': indicators.volatility,
                'momentum': indicators.momentum
            },
            'signals': signals['individual_signals'],
            'risk_metrics': risk_metrics,
            'observations': observations,
            'trend_analysis': {
                'short_term': 'BULLISH' if indicators.ema_12 > indicators.ema_26 else 'BEARISH',
                'medium_term': 'BULLISH' if current_price > indicators.sma_20 else 'BEARISH',
                'long_term': 'BULLISH' if current_price > indicators.sma_50 else 'BEARISH'
            }
        }