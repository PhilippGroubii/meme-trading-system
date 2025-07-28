"""
Price prediction model for meme coins using multiple ML approaches
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')


class PricePredictor:
    def __init__(self, model_type: str = 'ensemble'):
        """Initialize price prediction model"""
        self.model_type = model_type
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.is_trained = False
        
        # Initialize models
        self.models = {
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boost': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                random_state=42
            ),
            'linear': Ridge(alpha=1.0),
            'ensemble': None  # Will be created during training
        }
        
        # Initialize scalers
        self.scalers = {
            'features': StandardScaler(),
            'target': MinMaxScaler()
        }
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features for price prediction"""
        features_df = df.copy()
        
        # Basic price features
        features_df['price_ma_5'] = df['close'].rolling(window=5).mean()
        features_df['price_ma_10'] = df['close'].rolling(window=10).mean()
        features_df['price_ma_20'] = df['close'].rolling(window=20).mean()
        
        # Price ratios
        features_df['price_to_ma5'] = df['close'] / features_df['price_ma_5']
        features_df['price_to_ma10'] = df['close'] / features_df['price_ma_10']
        features_df['price_to_ma20'] = df['close'] / features_df['price_ma_20']
        
        # Volatility features
        features_df['volatility_5'] = df['close'].rolling(window=5).std()
        features_df['volatility_10'] = df['close'].rolling(window=10).std()
        features_df['volatility_20'] = df['close'].rolling(window=20).std()
        
        # Returns
        features_df['return_1h'] = df['close'].pct_change(1)
        features_df['return_4h'] = df['close'].pct_change(4)
        features_df['return_24h'] = df['close'].pct_change(24)
        
        # Volume features
        if 'volume' in df.columns:
            features_df['volume_ma_5'] = df['volume'].rolling(window=5).mean()
            features_df['volume_ma_10'] = df['volume'].rolling(window=10).mean()
            features_df['volume_ratio'] = df['volume'] / features_df['volume_ma_10']
            features_df['volume_price_trend'] = (df['volume'] * df['close']).rolling(window=5).mean()
        
        # Technical indicators
        features_df = self._add_technical_features(features_df)
        
        # Sentiment features (if available)
        if 'sentiment_score' in df.columns:
            features_df['sentiment_ma_5'] = df['sentiment_score'].rolling(window=5).mean()
            features_df['sentiment_change'] = df['sentiment_score'].diff()
        
        # Time-based features
        if isinstance(df.index, pd.DatetimeIndex):
            features_df['hour'] = df.index.hour
            features_df['day_of_week'] = df.index.dayofweek
            features_df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
        
        # Lag features
        for lag in [1, 2, 3, 6, 12, 24]:
            features_df[f'price_lag_{lag}'] = df['close'].shift(lag)
            features_df[f'return_lag_{lag}'] = features_df['return_1h'].shift(lag)
        
        return features_df
    
    def _add_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicator features"""
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        sma_20 = df['close'].rolling(window=20).mean()
        std_20 = df['close'].rolling(window=20).std()
        df['bb_upper'] = sma_20 + (2 * std_20)
        df['bb_lower'] = sma_20 - (2 * std_20)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Support and resistance levels
        df['resistance_level'] = df['close'].rolling(window=20, center=True).max()
        df['support_level'] = df['close'].rolling(window=20, center=True).min()
        df['price_to_resistance'] = df['close'] / df['resistance_level']
        df['price_to_support'] = df['close'] / df['support_level']
        
        return df
    
    def prepare_data(self, df: pd.DataFrame, target_column: str = 'close', 
                    prediction_horizon: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for training"""
        # Create features
        features_df = self.create_features(df)
        
        # Create target (future price)
        target = df[target_column].shift(-prediction_horizon)
        
        # Select feature columns (exclude target and non-feature columns)
        exclude_columns = ['close', 'open', 'high', 'low', 'volume', 'timestamp']
        feature_columns = [col for col in features_df.columns 
                          if col not in exclude_columns and not col.startswith('Unnamed')]
        
        # Get features and target
        X = features_df[feature_columns].values
        y = target.values
        
        # Remove rows with NaN values
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X = X[mask]
        y = y[mask]
        
        self.feature_columns = feature_columns
        
        return X, y
    
    def train(self, df: pd.DataFrame, target_column: str = 'close', 
              prediction_horizon: int = 1, test_size: float = 0.2) -> Dict:
        """Train the price prediction model"""
        # Prepare data
        X, y = self.prepare_data(df, target_column, prediction_horizon)
        
        if len(X) < 50:
            raise ValueError("Not enough data points for training (minimum 50 required)")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, shuffle=False
        )
        
        # Scale features
        X_train_scaled = self.scalers['features'].fit_transform(X_train)
        X_test_scaled = self.scalers['features'].transform(X_test)
        
        # Scale target
        y_train_scaled = self.scalers['target'].fit_transform(y_train.reshape(-1, 1)).flatten()
        y_test_scaled = self.scalers['target'].transform(y_test.reshape(-1, 1)).flatten()
        
        # Train individual models
        model_scores = {}
        predictions = {}
        
        for name, model in self.models.items():
            if name == 'ensemble':
                continue
                
            try:
                # Train model
                model.fit(X_train_scaled, y_train_scaled)
                
                # Predict
                y_pred_scaled = model.predict(X_test_scaled)
                y_pred = self.scalers['target'].inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
                
                # Calculate metrics
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                model_scores[name] = {
                    'mae': mae,
                    'mse': mse,
                    'rmse': np.sqrt(mse),
                    'r2': r2,
                    'mape': np.mean(np.abs((y_test - y_pred) / y_test)) * 100
                }
                
                predictions[name] = y_pred
                
                # Feature importance
                if hasattr(model, 'feature_importances_'):
                    importance = dict(zip(self.feature_columns, model.feature_importances_))
                    self.feature_importance[name] = sorted(importance.items(), 
                                                         key=lambda x: x[1], reverse=True)
                
            except Exception as e:
                print(f"Error training {name}: {e}")
                model_scores[name] = {'error': str(e)}
        
        # Create ensemble model
        ensemble_pred = np.zeros_like(y_test)
        weights = {'random_forest': 0.4, 'gradient_boost': 0.4, 'linear': 0.2}
        
        for name, weight in weights.items():
            if name in predictions:
                ensemble_pred += weight * predictions[name]
        
        # Ensemble metrics
        ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
        ensemble_mse = mean_squared_error(y_test, ensemble_pred)
        ensemble_r2 = r2_score(y_test, ensemble_pred)
        
        model_scores['ensemble'] = {
            'mae': ensemble_mae,
            'mse': ensemble_mse,
            'rmse': np.sqrt(ensemble_mse),
            'r2': ensemble_r2,
            'mape': np.mean(np.abs((y_test - ensemble_pred) / y_test)) * 100
        }
        
        self.is_trained = True
        
        return {
            'model_scores': model_scores,
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'feature_count': len(self.feature_columns),
            'prediction_horizon': prediction_horizon
        }
    
    
    def _handle_nan_values(self, X):
        """Handle NaN values in features before model training/prediction"""
        from sklearn.impute import SimpleImputer
        import numpy as np
        
        # Check if we have any NaN values
        if np.isnan(X).any():
            print(f"Warning: Found NaN values in features. Applying imputation...")
            
            # Use median imputation for numerical features
            if not hasattr(self, 'feature_imputer'):
                self.feature_imputer = SimpleImputer(strategy='median')
                X_imputed = self.feature_imputer.fit_transform(X)
            else:
                X_imputed = self.feature_imputer.transform(X)
            
            return X_imputed
        
        return X

    def predict(self, df: pd.DataFrame, model_name: str = 'ensemble') -> Dict:
        """Make price predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Prepare features
        features_df = self.create_features(df)
        X = features_df[self.feature_columns].values
        
        # Handle NaN values
        if np.isnan(X).any():
            # Use last valid values
            X = pd.DataFrame(X).fillna(method='ffill').fillna(method='bfill').values
        
        # Scale features
        X_scaled = self.scalers['features'].transform(X)
        
        if model_name == 'ensemble':
            # Ensemble prediction
            predictions_scaled = np.zeros(len(X))
            weights = {'random_forest': 0.4, 'gradient_boost': 0.4, 'linear': 0.2}
            
            for name, weight in weights.items():
                if name in self.models and self.models[name] is not None:
                    pred_scaled = self.models[name].predict(X_scaled)
                    predictions_scaled += weight * pred_scaled
        else:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not found")
            
            predictions_scaled = self.models[model_name].predict(X_scaled)
        
        # Inverse transform predictions
        predictions = self.scalers['target'].inverse_transform(
            predictions_scaled.reshape(-1, 1)
        ).flatten()
        
        # Calculate prediction confidence
        confidence = self._calculate_prediction_confidence(X_scaled, predictions_scaled)
        
        return {
            'predictions': predictions,
            'confidence': confidence,
            'model_used': model_name,
            'last_prediction': predictions[-1],
            'prediction_change': (predictions[-1] - df['close'].iloc[-1]) / df['close'].iloc[-1] * 100
        }
    
    def _calculate_prediction_confidence(self, X_scaled: np.ndarray, 
                                       predictions_scaled: np.ndarray) -> List[float]:
        """Calculate prediction confidence based on model agreement"""
        if len(X_scaled) == 0:
            return []
        
        # Get predictions from all models
        all_predictions = []
        for name, model in self.models.items():
            if name != 'ensemble' and model is not None:
                try:
                    pred = model.predict(X_scaled)
                    all_predictions.append(pred)
                except:
                    continue
        
        if len(all_predictions) < 2:
            return [0.5] * len(predictions_scaled)
        
        # Calculate confidence based on model agreement
        all_predictions = np.array(all_predictions)
        confidence = []
        
        for i in range(len(predictions_scaled)):
            # Calculate standard deviation of predictions
            std = np.std(all_predictions[:, i])
            # Convert to confidence (lower std = higher confidence)
            conf = max(0, min(1, 1 - (std / np.mean(np.abs(all_predictions[:, i])))))
            confidence.append(conf)
        
        return confidence
    
    def get_feature_importance(self, model_name: str = 'random_forest', top_n: int = 10) -> List[Tuple]:
        """Get feature importance for interpretability"""
        if model_name not in self.feature_importance:
            return []
        
        return self.feature_importance[model_name][:top_n]
    
    def save_model(self, filepath: str):
        """Save trained model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'models': self.models,
            'scalers': self.scalers,
            'feature_columns': self.feature_columns,
            'feature_importance': self.feature_importance,
            'is_trained': self.is_trained,
            'model_type': self.model_type
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model"""
        model_data = joblib.load(filepath)
        
        self.models = model_data['models']
        self.scalers = model_data['scalers']
        self.feature_columns = model_data['feature_columns']
        self.feature_importance = model_data.get('feature_importance', {})
        self.is_trained = model_data['is_trained']
        self.model_type = model_data['model_type']
        
        print(f"Model loaded from {filepath}")
    
    def backtest(self, df: pd.DataFrame, prediction_horizon: int = 1, 
                window_size: int = 100) -> Dict:
        """Backtest the model on historical data"""
        if len(df) < window_size + prediction_horizon + 50:
            raise ValueError("Not enough data for backtesting")
        
        predictions = []
        actual_prices = []
        timestamps = []
        
        for i in range(window_size, len(df) - prediction_horizon):
            # Use sliding window for training
            train_data = df.iloc[i-window_size:i]
            
            # Create temporary model
            temp_model = PricePredictor(self.model_type)
            
            try:
                # Train on window
                temp_model.train(train_data, prediction_horizon=prediction_horizon, test_size=0.1)
                
                # Predict next value
                pred_data = df.iloc[i:i+1]
                prediction = temp_model.predict(pred_data)
                
                predictions.append(prediction['last_prediction'])
                actual_prices.append(df['close'].iloc[i + prediction_horizon])
                timestamps.append(df.index[i + prediction_horizon])
                
            except Exception as e:
                print(f"Error in backtest at position {i}: {e}")
                continue
        
        if len(predictions) == 0:
            return {'error': 'No successful predictions in backtest'}
        
        # Calculate backtest metrics
        predictions = np.array(predictions)
        actual_prices = np.array(actual_prices)
        
        mae = mean_absolute_error(actual_prices, predictions)
        mse = mean_squared_error(actual_prices, predictions)
        r2 = r2_score(actual_prices, predictions)
        mape = np.mean(np.abs((actual_prices - predictions) / actual_prices)) * 100
        
        # Directional accuracy
        actual_direction = np.sign(np.diff(actual_prices))
        pred_direction = np.sign(predictions[1:] - actual_prices[:-1])
        directional_accuracy = np.mean(actual_direction == pred_direction) * 100
        
        return {
            'mae': mae,
            'mse': mse,
            'rmse': np.sqrt(mse),
            'r2': r2,
            'mape': mape,
            'directional_accuracy': directional_accuracy,
            'total_predictions': len(predictions),
            'predictions': predictions.tolist(),
            'actual_prices': actual_prices.tolist(),
            'timestamps': [str(ts) for ts in timestamps]
        }


def main():
    """Test the price predictor"""
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=500, freq='H')
    
    # Generate realistic meme coin price data
    base_price = 100
    trend = np.linspace(0, 0.5, 500)  # Slight upward trend
    noise = np.random.normal(0, 0.02, 500)  # 2% noise
    jumps = np.random.choice([0, 0.1, -0.1], 500, p=[0.98, 0.01, 0.01])  # Occasional jumps
    
    returns = trend + noise + jumps
    prices = base_price * np.exp(np.cumsum(returns))
    
    volume = np.random.lognormal(10, 1, 500)
    sentiment = np.random.normal(0, 0.3, 500)
    
    df = pd.DataFrame({
        'close': prices,
        'volume': volume,
        'sentiment_score': sentiment
    }, index=dates)
    
    # Initialize and train model
    predictor = PricePredictor()
    
    print("Training price prediction model...")
    results = predictor.train(df, prediction_horizon=1)
    
    print("Training Results:")
    for model, scores in results['model_scores'].items():
        if 'error' not in scores:
            print(f"{model}: R² = {scores['r2']:.3f}, MAPE = {scores['mape']:.2f}%")
    
    # Make predictions
    print("\nMaking predictions...")
    predictions = predictor.predict(df.tail(10))
    print(f"Last price: ${df['close'].iloc[-1]:.4f}")
    print(f"Predicted next price: ${predictions['last_prediction']:.4f}")
    print(f"Predicted change: {predictions['prediction_change']:.2f}%")
    
    # Feature importance
    print("\nTop features:")
    importance = predictor.get_feature_importance()
    for feature, score in importance[:5]:
        print(f"{feature}: {score:.3f}")


if __name__ == "__main__":
    main()