"""
Feature engineering for meme coin ML models
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    def __init__(self):
        """Initialize feature engineering pipeline"""
        self.scalers = {}
        self.feature_selectors = {}
        self.feature_importance = {}
        self.feature_columns = []
        self.generated_features = []
        self.is_fitted = False
        
    def create_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create price-based features"""
        features_df = df.copy()
        
        if 'close' not in df.columns:
            return features_df
        
        # Moving averages
        for window in [5, 10, 20, 50]:
            features_df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
            features_df[f'ema_{window}'] = df['close'].ewm(span=window).mean()
        
        # Price ratios to moving averages
        for window in [5, 10, 20]:
            sma_col = f'sma_{window}'
            if sma_col in features_df.columns:
                features_df[f'price_to_sma_{window}'] = df['close'] / features_df[sma_col]
                features_df[f'sma_{window}_slope'] = features_df[sma_col].diff()
        
        # Price changes (returns)
        for period in [1, 4, 24, 168]:  # 1h, 4h, 1d, 1w
            features_df[f'return_{period}h'] = df['close'].pct_change(period)
            features_df[f'log_return_{period}h'] = np.log(df['close'] / df['close'].shift(period))
        
        # Volatility
        for window in [5, 10, 20]:
            features_df[f'volatility_{window}'] = df['close'].rolling(window=window).std()
            features_df[f'volatility_{window}_norm'] = (features_df[f'volatility_{window}'] / 
                                                       features_df[f'sma_{window}'])
        
        # Price momentum
        for period in [5, 10, 20]:
            features_df[f'momentum_{period}'] = df['close'] / df['close'].shift(period) - 1
            features_df[f'roc_{period}'] = (df['close'] - df['close'].shift(period)) / df['close'].shift(period) * 100
        
        # Support and resistance levels
        for window in [10, 20, 50]:
            features_df[f'resistance_{window}'] = df['close'].rolling(window=window, center=True).max()
            features_df[f'support_{window}'] = df['close'].rolling(window=window, center=True).min()
            
            # Distance to support/resistance
            features_df[f'dist_to_resistance_{window}'] = (features_df[f'resistance_{window}'] - df['close']) / df['close']
            features_df[f'dist_to_support_{window}'] = (df['close'] - features_df[f'support_{window}']) / df['close']
        
        # Price channel position
        features_df['price_channel_position'] = ((df['close'] - features_df['support_20']) / 
                                               (features_df['resistance_20'] - features_df['support_20']))
        
        return features_df
    
    def create_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create volume-based features"""
        features_df = df.copy()
        
        if 'volume' not in df.columns:
            return features_df
        
        # Volume moving averages
        for window in [5, 10, 20, 50]:
            features_df[f'volume_sma_{window}'] = df['volume'].rolling(window=window).mean()
        
        # Volume ratios
        for window in [5, 10, 20]:
            vol_sma_col = f'volume_sma_{window}'
            if vol_sma_col in features_df.columns:
                features_df[f'volume_ratio_{window}'] = df['volume'] / features_df[vol_sma_col]
        
        # Volume momentum
        for period in [1, 4, 24]:
            features_df[f'volume_change_{period}h'] = df['volume'].pct_change(period)
            features_df[f'volume_momentum_{period}'] = df['volume'] / df['volume'].shift(period) - 1
        
        # Volume volatility
        for window in [10, 20]:
            features_df[f'volume_volatility_{window}'] = df['volume'].rolling(window=window).std()
            features_df[f'volume_cv_{window}'] = (features_df[f'volume_volatility_{window}'] / 
                                                features_df[f'volume_sma_{window}'])
        
        # On-Balance Volume (OBV)
        if 'close' in df.columns:
            features_df['obv'] = self._calculate_obv(df['close'], df['volume'])
            features_df['obv_sma_10'] = features_df['obv'].rolling(window=10).mean()
            features_df['obv_momentum'] = features_df['obv'] / features_df['obv'].shift(10) - 1
        
        # Volume-Price Trend (VPT)
        if 'close' in df.columns:
            features_df['vpt'] = self._calculate_vpt(df['close'], df['volume'])
            features_df['vpt_sma_10'] = features_df['vpt'].rolling(window=10).mean()
        
        # Accumulation/Distribution Line
        if all(col in df.columns for col in ['high', 'low', 'close']):
            features_df['ad_line'] = self._calculate_ad_line(df['high'], df['low'], df['close'], df['volume'])
        
        return features_df
    
    def create_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create technical indicator features"""
        features_df = df.copy()
        
        if 'close' not in df.columns:
            return features_df
        
        # RSI
        for period in [7, 14, 21]:
            features_df[f'rsi_{period}'] = self._calculate_rsi(df['close'], period)
        
        # MACD
        macd_result = self._calculate_macd(df['close'])
        features_df.update(macd_result)
        
        # Bollinger Bands
        for window in [10, 20]:
            bb_result = self._calculate_bollinger_bands(df['close'], window)
            for key, value in bb_result.items():
                features_df[f'{key}_{window}'] = value
        
        # Stochastic Oscillator
        if all(col in df.columns for col in ['high', 'low', 'close']):
            stoch_result = self._calculate_stochastic(df['high'], df['low'], df['close'])
            features_df.update(stoch_result)
        
        # Williams %R
        if all(col in df.columns for col in ['high', 'low', 'close']):
            for period in [14, 21]:
                features_df[f'williams_r_{period}'] = self._calculate_williams_r(
                    df['high'], df['low'], df['close'], period)
        
        # Commodity Channel Index (CCI)
        if all(col in df.columns for col in ['high', 'low', 'close']):
            features_df['cci'] = self._calculate_cci(df['high'], df['low'], df['close'])
        
        # Average True Range (ATR)
        if all(col in df.columns for col in ['high', 'low', 'close']):
            features_df['atr'] = self._calculate_atr(df['high'], df['low'], df['close'])
            features_df['atr_ratio'] = features_df['atr'] / df['close']
        
        return features_df
    
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features"""
        features_df = df.copy()
        
        if not isinstance(df.index, pd.DatetimeIndex):
            return features_df
        
        # Basic time features
        features_df['hour'] = df.index.hour
        features_df['day_of_week'] = df.index.dayofweek
        features_df['day_of_month'] = df.index.day
        features_df['month'] = df.index.month
        features_df['quarter'] = df.index.quarter
        
        # Cyclical encoding
        features_df['hour_sin'] = np.sin(2 * np.pi * features_df['hour'] / 24)
        features_df['hour_cos'] = np.cos(2 * np.pi * features_df['hour'] / 24)
        features_df['day_sin'] = np.sin(2 * np.pi * features_df['day_of_week'] / 7)
        features_df['day_cos'] = np.cos(2 * np.pi * features_df['day_of_week'] / 7)
        features_df['month_sin'] = np.sin(2 * np.pi * features_df['month'] / 12)
        features_df['month_cos'] = np.cos(2 * np.pi * features_df['month'] / 12)
        
        # Trading session indicators
        features_df['is_weekend'] = (features_df['day_of_week'] >= 5).astype(int)
        features_df['is_trading_hours'] = ((features_df['hour'] >= 9) & 
                                         (features_df['hour'] <= 16)).astype(int)
        features_df['is_asian_hours'] = ((features_df['hour'] >= 1) & 
                                       (features_df['hour'] <= 8)).astype(int)
        features_df['is_european_hours'] = ((features_df['hour'] >= 8) & 
                                          (features_df['hour'] <= 16)).astype(int)
        features_df['is_american_hours'] = ((features_df['hour'] >= 14) & 
                                          (features_df['hour'] <= 22)).astype(int)
        
        return features_df
    
    def create_lag_features(self, df: pd.DataFrame, target_column: str = 'close', 
                          lags: List[int] = [1, 2, 3, 6, 12, 24]) -> pd.DataFrame:
        """Create lag features"""
        features_df = df.copy()
        
        if target_column not in df.columns:
            return features_df
        
        # Basic lag features
        for lag in lags:
            features_df[f'{target_column}_lag_{lag}'] = df[target_column].shift(lag)
        
        # Rolling statistics of lags
        for window in [3, 6, 12]:
            if window <= max(lags):
                lag_cols = [f'{target_column}_lag_{lag}' for lag in lags if lag <= window]
                if lag_cols:
                    lag_data = features_df[lag_cols]
                    features_df[f'{target_column}_lag_mean_{window}'] = lag_data.mean(axis=1)
                    features_df[f'{target_column}_lag_std_{window}'] = lag_data.std(axis=1)
                    features_df[f'{target_column}_lag_min_{window}'] = lag_data.min(axis=1)
                    features_df[f'{target_column}_lag_max_{window}'] = lag_data.max(axis=1)
        
        # Lag differences
        for lag in lags[:4]:  # Only use first few lags for differences
            features_df[f'{target_column}_diff_lag_{lag}'] = (df[target_column] - 
                                                            features_df[f'{target_column}_lag_{lag}'])
        
        return features_df
    
    def create_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create sentiment-based features"""
        features_df = df.copy()
        
        # If sentiment columns exist
        sentiment_columns = [col for col in df.columns if 'sentiment' in col.lower()]
        
        for col in sentiment_columns:
            # Moving averages
            for window in [5, 10, 20]:
                features_df[f'{col}_sma_{window}'] = df[col].rolling(window=window).mean()
            
            # Sentiment momentum
            for period in [1, 4, 24]:
                features_df[f'{col}_change_{period}'] = df[col].diff(period)
                features_df[f'{col}_momentum_{period}'] = df[col] / df[col].shift(period) - 1
            
            # Sentiment volatility
            for window in [10, 20]:
                features_df[f'{col}_volatility_{window}'] = df[col].rolling(window=window).std()
            
            # Sentiment extremes
            features_df[f'{col}_extreme_positive'] = (df[col] > 0.7).astype(int)
            features_df[f'{col}_extreme_negative'] = (df[col] < -0.7).astype(int)
        
        return features_df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between different data types"""
        features_df = df.copy()
        
        # Price-Volume interactions
        if all(col in df.columns for col in ['close', 'volume']):
            features_df['price_volume_product'] = df['close'] * df['volume']
            features_df['price_volume_ratio'] = df['close'] / (df['volume'] + 1e-8)
            
            # Volume-weighted price
            features_df['vwap_5'] = (df['close'] * df['volume']).rolling(5).sum() / df['volume'].rolling(5).sum()
            features_df['vwap_20'] = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
            
            # Price position relative to VWAP
            features_df['price_vs_vwap_5'] = df['close'] / features_df['vwap_5']
            features_df['price_vs_vwap_20'] = df['close'] / features_df['vwap_20']
        
        # Volatility-Volume interactions
        if 'volatility_20' in features_df.columns and 'volume_ratio_20' in features_df.columns:
            features_df['volatility_volume_interaction'] = (features_df['volatility_20'] * 
                                                          features_df['volume_ratio_20'])
        
        # RSI-Volume interactions
        if 'rsi_14' in features_df.columns and 'volume_ratio_20' in features_df.columns:
            features_df['rsi_volume_divergence'] = features_df['rsi_14'] / features_df['volume_ratio_20']
        
        # Sentiment-Price interactions
        sentiment_cols = [col for col in features_df.columns if 'sentiment' in col.lower() 
                         and 'sma' not in col and 'change' not in col and 'momentum' not in col]
        
        if sentiment_cols and 'return_24h' in features_df.columns:
            for sent_col in sentiment_cols:
                features_df[f'{sent_col}_price_alignment'] = (features_df[sent_col] * 
                                                            features_df['return_24h'])
        
        return features_df
    
    def create_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create market regime features"""
        features_df = df.copy()
        
        if 'close' not in df.columns:
            return features_df
        
        # Volatility regime
        if 'volatility_20' in features_df.columns:
            vol_threshold_high = features_df['volatility_20'].quantile(0.75)
            vol_threshold_low = features_df['volatility_20'].quantile(0.25)
            
            features_df['high_volatility_regime'] = (features_df['volatility_20'] > vol_threshold_high).astype(int)
            features_df['low_volatility_regime'] = (features_df['volatility_20'] < vol_threshold_low).astype(int)
        
        # Trend regime
        if 'sma_20' in features_df.columns and 'sma_50' in features_df.columns:
            features_df['bullish_trend'] = (features_df['sma_20'] > features_df['sma_50']).astype(int)
            features_df['bearish_trend'] = (features_df['sma_20'] < features_df['sma_50']).astype(int)
        
        # Volume regime
        if 'volume_ratio_20' in features_df.columns:
            vol_ratio_high = features_df['volume_ratio_20'].quantile(0.8)
            features_df['high_volume_regime'] = (features_df['volume_ratio_20'] > vol_ratio_high).astype(int)
        
        # Price momentum regime
        if 'momentum_20' in features_df.columns:
            momentum_threshold = 0.05  # 5% threshold
            features_df['strong_momentum_up'] = (features_df['momentum_20'] > momentum_threshold).astype(int)
            features_df['strong_momentum_down'] = (features_df['momentum_20'] < -momentum_threshold).astype(int)
        
        return features_df
    
    
    def _clean_features(self, df):
        """Clean features by handling NaN and infinite values"""
        import numpy as np
        import pandas as pd
        
        print(f"Cleaning features - initial NaN count: {df.isnull().sum().sum()}")
        
        # Replace infinite values with NaN first
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Get numeric columns only
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # For each numeric column, fill NaN values
        for col in numeric_cols:
            if df[col].isnull().any():
                if col.endswith('_ratio') or col.endswith('_change') or col.endswith('_return'):
                    # For ratios and changes, use 0
                    df[col].fillna(0, inplace=True)
                elif 'volume' in col.lower():
                    # For volume features, use median
                    df[col].fillna(df[col].median(), inplace=True)
                elif any(indicator in col.lower() for indicator in ['rsi', 'macd', 'bb_', 'williams']):
                    # For technical indicators, use forward fill then backward fill
                    df[col].fillna(method='ffill', inplace=True)
                    df[col].fillna(method='bfill', inplace=True)
                    # If still NaN, use reasonable defaults
                    if 'rsi' in col.lower():
                        df[col].fillna(50, inplace=True)  # Neutral RSI
                    else:
                        df[col].fillna(0, inplace=True)
                else:
                    # For other features, use forward fill then median
                    df[col].fillna(method='ffill', inplace=True)
                    df[col].fillna(df[col].median(), inplace=True)
        
        # Final check - any remaining NaN values get filled with 0
        df.fillna(0, inplace=True)
        
        print(f"Cleaning complete - final NaN count: {df.isnull().sum().sum()}")
        
        return df

    def engineer_features(self, df: pd.DataFrame, target_column: str = 'close') -> pd.DataFrame:
        """Complete feature engineering pipeline"""
        features_df = df.copy()
        
        print("Engineering features...")
        
        # Create all feature types
        features_df = self.create_price_features(features_df)
        features_df = self.create_volume_features(features_df)
        features_df = self.create_technical_features(features_df)
        features_df = self.create_time_features(features_df)
        features_df = self.create_lag_features(features_df, target_column)
        features_df = self.create_sentiment_features(features_df)
        features_df = self.create_interaction_features(features_df)
        features_df = self.create_regime_features(features_df)
        
        # Store generated features (excluding original columns)
        original_columns = set(df.columns)
        self.generated_features = [col for col in features_df.columns if col not in original_columns]
        
        print(f"Generated {len(self.generated_features)} new features")
        
        return features_df
    
    def select_features(self, df: pd.DataFrame, target_column: str, 
                       method: str = 'importance', n_features: int = 50) -> pd.DataFrame:
        """Feature selection"""
        # Get feature columns (exclude target and metadata)
        exclude_columns = [target_column, 'timestamp'] + [col for col in df.columns 
                          if col.startswith('Unnamed')]
        feature_columns = [col for col in df.columns if col not in exclude_columns]
        
        X = df[feature_columns]
        y = df[target_column]
        
        # Remove rows with NaN values
        mask = ~(X.isna().any(axis=1) | y.isna())
        X_clean = X[mask]
        y_clean = y[mask]
        
        if len(X_clean) == 0:
            print("No clean data available for feature selection")
            return df[feature_columns[:n_features]]
        
        if method == 'importance':
            # Use Random Forest feature importance
            rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
            rf.fit(X_clean.fillna(X_clean.median()), y_clean)
            
            importance_scores = dict(zip(feature_columns, rf.feature_importances_))
            selected_features = sorted(importance_scores.keys(), 
                                     key=importance_scores.get, reverse=True)[:n_features]
            
            self.feature_importance = sorted(importance_scores.items(), 
                                           key=lambda x: x[1], reverse=True)
            
        elif method == 'univariate':
            # Use univariate statistical tests
            selector = SelectKBest(score_func=f_regression, k=n_features)
            selector.fit(X_clean.fillna(X_clean.median()), y_clean)
            
            selected_features = [feature_columns[i] for i in selector.get_support(indices=True)]
            
        elif method == 'mutual_info':
            # Use mutual information
            selector = SelectKBest(score_func=mutual_info_regression, k=n_features)
            selector.fit(X_clean.fillna(X_clean.median()), y_clean)
            
            selected_features = [feature_columns[i] for i in selector.get_support(indices=True)]
        
        else:
            # Default: use all features up to n_features
            selected_features = feature_columns[:n_features]
        
        self.feature_columns = selected_features
        print(f"Selected {len(selected_features)} features using {method} method")
        
        return df[selected_features + [target_column]]
    
    def scale_features(self, df: pd.DataFrame, method: str = 'standard', 
                      feature_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Scale features"""
        if feature_columns is None:
            feature_columns = self.feature_columns
        
        if not feature_columns:
            feature_columns = [col for col in df.columns 
                             if col not in ['timestamp'] and not col.startswith('Unnamed')]
        
        scaled_df = df.copy()
        
        # Choose scaler
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            print(f"Unknown scaling method: {method}")
            return df
        
        # Fit and transform
        feature_data = df[feature_columns]
        
        # Handle NaN values
        feature_data_clean = feature_data.fillna(feature_data.median())
        
        scaled_features = scaler.fit_transform(feature_data_clean)
        scaled_df[feature_columns] = scaled_features
        
        self.scalers[method] = scaler
        self.is_fitted = True
        
        print(f"Scaled features using {method} scaler")
        
        return scaled_df
    
    def get_feature_importance(self, top_n: int = 20) -> List[Tuple[str, float]]:
        """Get feature importance rankings"""
        if not self.feature_importance:
            return []
        
        return self.feature_importance[:top_n]
    
    # Technical indicator calculation methods
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict:
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        macd_histogram = macd - macd_signal
        
        return {
            'macd': macd,
            'macd_signal': macd_signal,
            'macd_histogram': macd_histogram
        }
    
    def _calculate_bollinger_bands(self, prices: pd.Series, window: int = 20, std_dev: int = 2) -> Dict:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        
        return {
            'bb_upper': sma + (std * std_dev),
            'bb_middle': sma,
            'bb_lower': sma - (std * std_dev),
            'bb_width': (std * std_dev * 2) / sma,
            'bb_position': (prices - (sma - std * std_dev)) / (std * std_dev * 2)
        }
    
    def _calculate_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                            k_period: int = 14, d_period: int = 3) -> Dict:
        """Calculate Stochastic Oscillator"""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        
        return {
            'stoch_k': k_percent,
            'stoch_d': d_percent
        }
    
    def _calculate_williams_r(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                            period: int = 14) -> pd.Series:
        """Calculate Williams %R"""
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        return -100 * (highest_high - close) / (highest_high - lowest_low)
    
    def _calculate_cci(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                      period: int = 20) -> pd.Series:
        """Calculate Commodity Channel Index"""
        typical_price = (high + low + close) / 3
        sma_tp = typical_price.rolling(window=period).mean()
        mad = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
        return (typical_price - sma_tp) / (0.015 * mad)
    
    def _calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                      period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        tr1 = high - low
        tr2 = np.abs(high - close.shift(1))
        tr3 = np.abs(low - close.shift(1))
        true_range = np.maximum(tr1, np.maximum(tr2, tr3))
        return true_range.rolling(window=period).mean()
    
    def _calculate_obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate On-Balance Volume"""
        obv = pd.Series(index=close.index, dtype=float)
        obv.iloc[0] = volume.iloc[0]
        
        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
            elif close.iloc[i] < close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv
    
    def _calculate_vpt(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate Volume-Price Trend"""
        return (volume * close.pct_change()).cumsum()
    
    def _calculate_ad_line(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                          volume: pd.Series) -> pd.Series:
        """Calculate Accumulation/Distribution Line"""
        clv = ((close - low) - (high - close)) / (high - low)
        clv = clv.fillna(0)  # Handle division by zero
        ad_volume = clv * volume
        return ad_volume.cumsum()


def main():
    """Test the feature engineer"""
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=200, freq='H')
    
    # Generate realistic meme coin data
    returns = np.random.normal(0.001, 0.03, 200)
    prices = 100 * np.exp(np.cumsum(returns))
    
    # Add some volatility clustering
    volatility = np.random.gamma(1, 0.02, 200)
    prices += np.random.normal(0, volatility * prices)
    
    # Ensure positive prices
    prices = np.maximum(prices, 0.01)
    
    volume = np.random.lognormal(10, 1, 200)
    high = prices * (1 + np.abs(np.random.normal(0, 0.01, 200)))
    low = prices * (1 - np.abs(np.random.normal(0, 0.01, 200)))
    sentiment = np.random.normal(0, 0.3, 200)
    
    df = pd.DataFrame({
        'close': prices,
        'high': high,
        'low': low,
        'volume': volume,
        'sentiment_score': sentiment
    }, index=dates)
    
    # Initialize feature engineer
    fe = FeatureEngineer()
    
    print("Testing Feature Engineer...")
    print("="*50)
    
    # Engineer features
    features_df = fe.engineer_features(df, target_column='close')
    print(f"Original features: {len(df.columns)}")
    print(f"Total features after engineering: {len(features_df.columns)}")
    print(f"New features generated: {len(fe.generated_features)}")
    
    # Feature selection
    selected_df = fe.select_features(features_df, target_column='close', 
                                   method='importance', n_features=30)
    print(f"Selected features: {len(fe.feature_columns)}")
    
    # Feature importance
    print("\nTop 10 Important Features:")
    importance = fe.get_feature_importance(10)
    for feature, score in importance:
        print(f"{feature}: {score:.4f}")
    
    # Scale features
    scaled_df = fe.scale_features(selected_df, method='standard')
    print(f"\nFeatures scaled successfully")
    
    # Show some statistics
    print(f"\nFeature Statistics:")
    print(f"Features with NaN: {scaled_df.isna().any().sum()}")
    print(f"Feature correlation with target (mean): {scaled_df.corr()['close'].abs().mean():.3f}")


if __name__ == "__main__":
    main()