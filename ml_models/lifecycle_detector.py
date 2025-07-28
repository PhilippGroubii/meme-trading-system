"""
Meme coin lifecycle detection: Accumulation → Pump → Distribution → Dump
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from datetime import datetime, timedelta


class LifecycleDetector:
    def __init__(self):
        """Initialize lifecycle detector"""
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.is_trained = False
        
        # Lifecycle stages
        self.stages = {
            0: 'accumulation',
            1: 'pump',
            2: 'distribution', 
            3: 'dump'
        }
        
        # Stage characteristics for labeling
        self.stage_rules = {
            'accumulation': {
                'volume_increase': (1.0, 2.0),  # 1-2x normal volume
                'price_change': (-0.05, 0.1),   # -5% to +10% price change
                'social_activity': (0.2, 0.6),  # Low to medium social activity
                'holder_concentration': (0.6, 0.9)  # High concentration
            },
            'pump': {
                'volume_increase': (3.0, 10.0),  # 3-10x normal volume
                'price_change': (0.2, 2.0),      # +20% to +200% price change
                'social_activity': (0.7, 1.0),   # High social activity
                'holder_concentration': (0.4, 0.8)  # Medium concentration
            },
            'distribution': {
                'volume_increase': (2.0, 8.0),   # 2-8x normal volume
                'price_change': (-0.1, 0.3),     # -10% to +30% price change
                'social_activity': (0.6, 0.9),   # Medium to high social activity
                'holder_concentration': (0.2, 0.6)  # Decreasing concentration
            },
            'dump': {
                'volume_increase': (1.5, 5.0),   # 1.5-5x normal volume
                'price_change': (-0.8, -0.1),    # -80% to -10% price change
                'social_activity': (0.1, 0.5),   # Low social activity
                'holder_concentration': (0.1, 0.4)  # Low concentration
            }
        }
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features for lifecycle detection"""
        features_df = df.copy()
        
        # Volume patterns
        features_df['volume_ma_5'] = df['volume'].rolling(window=5).mean()
        features_df['volume_ma_20'] = df['volume'].rolling(window=20).mean()
        features_df['volume_ratio'] = df['volume'] / features_df['volume_ma_20']
        features_df['volume_trend'] = features_df['volume_ma_5'] / features_df['volume_ma_20']
        features_df['volume_volatility'] = df['volume'].rolling(window=10).std()
        
        # Price patterns
        features_df['price_ma_5'] = df['close'].rolling(window=5).mean()
        features_df['price_ma_10'] = df['close'].rolling(window=10).mean()
        features_df['price_ma_20'] = df['close'].rolling(window=20).mean()
        
        # Price changes
        features_df['price_change_1h'] = df['close'].pct_change(1)
        features_df['price_change_4h'] = df['close'].pct_change(4)
        features_df['price_change_24h'] = df['close'].pct_change(24)
        
        # Volatility
        features_df['volatility_5'] = df['close'].rolling(window=5).std()
        features_df['volatility_20'] = df['close'].rolling(window=20).std()
        features_df['volatility_ratio'] = features_df['volatility_5'] / features_df['volatility_20']
        
        # Price position relative to moving averages
        features_df['price_vs_ma5'] = df['close'] / features_df['price_ma_5']
        features_df['price_vs_ma10'] = df['close'] / features_df['price_ma_10']
        features_df['price_vs_ma20'] = df['close'] / features_df['price_ma_20']
        
        # Technical indicators
        features_df = self._add_technical_indicators(features_df)
        
        # Social metrics (if available)
        if 'social_activity' in df.columns:
            features_df['social_ma_5'] = df['social_activity'].rolling(window=5).mean()
            features_df['social_change'] = df['social_activity'].pct_change(1)
            features_df['social_momentum'] = features_df['social_ma_5'] / df['social_activity'].rolling(window=20).mean()
        else:
            # Create proxy social metrics from volume and price
            features_df['social_proxy'] = (features_df['volume_ratio'] * 
                                         np.abs(features_df['price_change_24h'])).rolling(window=5).mean()
        
        # Holder distribution (if available)
        if 'holder_distribution' in df.columns:
            features_df['holder_concentration'] = df['holder_distribution']
            features_df['holder_change'] = df['holder_distribution'].diff()
        else:
            # Create proxy from volume patterns
            features_df['holder_concentration_proxy'] = (1 / (1 + features_df['volume_ratio'])).rolling(window=10).mean()
        
        # Market structure
        features_df['support_level'] = df['close'].rolling(window=20, center=True).min()
        features_df['resistance_level'] = df['close'].rolling(window=20, center=True).max()
        features_df['price_position'] = ((df['close'] - features_df['support_level']) / 
                                       (features_df['resistance_level'] - features_df['support_level']))
        
        # Momentum indicators
        features_df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
        features_df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
        features_df['momentum_24'] = df['close'] / df['close'].shift(24) - 1
        
        # Volume-price relationship
        features_df['vp_ratio'] = (features_df['volume_ratio'] * 
                                 (1 + np.abs(features_df['price_change_24h'])))
        features_df['accumulation_signal'] = (features_df['volume_ratio'] * 
                                            (1 - np.abs(features_df['price_change_24h'])))
        
        return features_df
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators"""
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
        
        # Bollinger Bands position
        sma_20 = df['close'].rolling(window=20).mean()
        std_20 = df['close'].rolling(window=20).std()
        df['bb_upper'] = sma_20 + (2 * std_20)
        df['bb_lower'] = sma_20 - (2 * std_20)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Williams %R
        high_14 = df['close'].rolling(window=14).max()
        low_14 = df['close'].rolling(window=14).min()
        df['williams_r'] = -100 * (high_14 - df['close']) / (high_14 - low_14)
        
        return df
    
    def label_lifecycle_stages(self, df: pd.DataFrame) -> np.ndarray:
        """Automatically label lifecycle stages based on patterns"""
        features_df = self.create_features(df)
        labels = np.zeros(len(df))
        
        # Calculate rolling metrics for stage detection
        volume_increase = features_df['volume_ratio'].fillna(1)
        price_change = features_df['price_change_24h'].fillna(0)
        
        # Social activity proxy
        if 'social_activity' in features_df.columns:
            social_activity = features_df['social_activity'].fillna(0.5)
        else:
            social_activity = features_df['social_proxy'].fillna(0.5)
        
        # Holder concentration proxy
        if 'holder_concentration' in features_df.columns:
            holder_concentration = features_df['holder_concentration'].fillna(0.5)
        else:
            holder_concentration = features_df['holder_concentration_proxy'].fillna(0.5)
        
        for i in range(len(df)):
            vol_inc = volume_increase.iloc[i]
            price_chg = price_change.iloc[i]
            social_act = social_activity.iloc[i]
            holder_conc = holder_concentration.iloc[i]
            
            # Score each stage
            stage_scores = {}
            
            for stage_name, rules in self.stage_rules.items():
                score = 0
                
                # Volume score
                if rules['volume_increase'][0] <= vol_inc <= rules['volume_increase'][1]:
                    score += 1
                
                # Price change score
                if rules['price_change'][0] <= price_chg <= rules['price_change'][1]:
                    score += 1
                
                # Social activity score
                if rules['social_activity'][0] <= social_act <= rules['social_activity'][1]:
                    score += 1
                
                # Holder concentration score
                if rules['holder_concentration'][0] <= holder_conc <= rules['holder_concentration'][1]:
                    score += 1
                
                stage_scores[stage_name] = score
            
            # Assign stage with highest score
            best_stage = max(stage_scores, key=stage_scores.get)
            stage_num = list(self.stages.values()).index(best_stage)
            labels[i] = stage_num
        
        # Smooth labels to avoid rapid transitions
        labels = self._smooth_labels(labels)
        
        return labels
    
    def _smooth_labels(self, labels: np.ndarray, window: int = 5) -> np.ndarray:
        """Smooth label transitions"""
        smoothed = labels.copy()
        
        for i in range(window, len(labels) - window):
            # Use mode in window
            window_labels = labels[i-window:i+window+1]
            mode_label = max(set(window_labels), key=list(window_labels).count)
            smoothed[i] = mode_label
        
        return smoothed
    
    def train(self, df: pd.DataFrame, labels: Optional[np.ndarray] = None) -> Dict:
        """Train lifecycle detection model"""
        # Create features
        features_df = self.create_features(df)
        
        # Auto-label if no labels provided
        if labels is None:
            labels = self.label_lifecycle_stages(df)
        
        # Select feature columns
        exclude_columns = ['close', 'open', 'high', 'low', 'volume', 'timestamp', 
                          'social_activity', 'holder_distribution']
        feature_columns = [col for col in features_df.columns 
                          if col not in exclude_columns and not col.startswith('Unnamed')]
        
        self.feature_columns = feature_columns
        
        # Prepare data
        X = features_df[feature_columns].values
        y = labels
        
        # Remove NaN values
        mask = ~np.isnan(X).any(axis=1)
        X = X[mask]
        y = y[mask]
        
        if len(X) < 50:
            raise ValueError("Not enough data points for training")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        # Predictions for detailed metrics
        y_pred = self.model.predict(X_test_scaled)
        
        # Feature importance
        importance = dict(zip(feature_columns, self.model.feature_importances_))
        self.feature_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        
        self.is_trained = True
        
        return {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'classification_report': classification_report(y_test, y_pred, 
                                                         target_names=list(self.stages.values()), 
                                                         output_dict=True),
            'feature_importance': self.feature_importance[:10],
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'stage_distribution': {stage: int(np.sum(y == i)) for i, stage in self.stages.items()}
        }
    
    def predict(self, df: pd.DataFrame) -> Dict:
        """Predict lifecycle stage"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Create features
        features_df = self.create_features(df)
        X = features_df[self.feature_columns].values
        
        # Handle NaN values
        X = pd.DataFrame(X).fillna(method='ffill').fillna(method='bfill').values
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Predict
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        
        # Get stage names
        stage_names = [self.stages[pred] for pred in predictions]
        
        # Current stage (last prediction)
        current_stage = stage_names[-1]
        current_probs = probabilities[-1]
        
        # Stage confidence
        stage_confidence = {self.stages[i]: prob for i, prob in enumerate(current_probs)}
        
        # Trend analysis
        if len(predictions) > 5:
            recent_stages = predictions[-5:]
            stage_trend = self._analyze_stage_trend(recent_stages)
        else:
            stage_trend = 'insufficient_data'
        
        return {
            'current_stage': current_stage,
            'stage_confidence': stage_confidence,
            'stage_trend': stage_trend,
            'all_predictions': stage_names,
            'probabilities': probabilities.tolist(),
            'transition_risk': self._calculate_transition_risk(current_probs),
            'recommendations': self._get_stage_recommendations(current_stage, stage_confidence)
        }
    
    def _analyze_stage_trend(self, recent_stages: np.ndarray) -> str:
        """Analyze trend in recent stages"""
        if len(set(recent_stages)) == 1:
            return 'stable'
        
        # Check for progression
        stage_sequence = ['accumulation', 'pump', 'distribution', 'dump']
        recent_stage_names = [self.stages[stage] for stage in recent_stages]
        
        # Find direction
        first_stage_idx = stage_sequence.index(recent_stage_names[0]) if recent_stage_names[0] in stage_sequence else 0
        last_stage_idx = stage_sequence.index(recent_stage_names[-1]) if recent_stage_names[-1] in stage_sequence else 0
        
        if last_stage_idx > first_stage_idx:
            return 'progressing'
        elif last_stage_idx < first_stage_idx:
            return 'reversing'
        else:
            return 'oscillating'
    
    def _calculate_transition_risk(self, probabilities: np.ndarray) -> float:
        """Calculate risk of stage transition"""
        # Higher risk when probabilities are more evenly distributed
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
        max_entropy = np.log(len(probabilities))
        
        return entropy / max_entropy
    
    def _get_stage_recommendations(self, current_stage: str, stage_confidence: Dict) -> Dict:
        """Get trading recommendations based on stage"""
        recommendations = {
            'accumulation': {
                'action': 'BUY',
                'urgency': 'low',
                'rationale': 'Accumulation phase - good entry point before pump',
                'risk_level': 'medium',
                'position_size': 'small_to_medium'
            },
            'pump': {
                'action': 'HOLD_OR_PARTIAL_SELL',
                'urgency': 'medium',
                'rationale': 'Pump phase - ride the momentum but prepare for distribution',
                'risk_level': 'high',
                'position_size': 'reduce_gradually'
            },
            'distribution': {
                'action': 'SELL',
                'urgency': 'high',
                'rationale': 'Distribution phase - smart money is exiting',
                'risk_level': 'very_high',
                'position_size': 'exit_majority'
            },
            'dump': {
                'action': 'AVOID_OR_SHORT',
                'urgency': 'critical',
                'rationale': 'Dump phase - avoid or consider shorting',
                'risk_level': 'extreme',
                'position_size': 'no_position'
            }
        }
        
        base_rec = recommendations.get(current_stage, recommendations['accumulation'])
        
        # Adjust based on confidence
        max_confidence = max(stage_confidence.values())
        if max_confidence < 0.6:
            base_rec['confidence_warning'] = 'Low confidence in stage detection - use caution'
        
        return base_rec
    
    def save_model(self, filepath: str):
        """Save trained model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'feature_importance': self.feature_importance,
            'stages': self.stages,
            'stage_rules': self.stage_rules,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
        print(f"Lifecycle detector saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model"""
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        self.feature_importance = model_data.get('feature_importance', [])
        self.stages = model_data['stages']
        self.stage_rules = model_data['stage_rules']
        self.is_trained = model_data['is_trained']
        
        print(f"Lifecycle detector loaded from {filepath}")


def main():
    """Test the lifecycle detector"""
    # Create sample data representing different lifecycle stages
    np.random.seed(42)
    
    # Simulate meme coin lifecycle
    dates = pd.date_range('2024-01-01', periods=200, freq='H')
    
    # Create realistic lifecycle pattern
    accumulation_period = 50
    pump_period = 30
    distribution_period = 40
    dump_period = 80
    
    # Generate price data
    base_price = 100
    prices = []
    volumes = []
    
    # Accumulation phase
    for i in range(accumulation_period):
        price_change = np.random.normal(0.001, 0.02)  # Slow accumulation
        volume = np.random.lognormal(9, 0.5)  # Normal volume
        prices.append(price_change)
        volumes.append(volume)
    
    # Pump phase
    for i in range(pump_period):
        price_change = np.random.normal(0.05, 0.03)  # Strong upward movement
        volume = np.random.lognormal(11, 0.8)  # High volume
        prices.append(price_change)
        volumes.append(volume)
    
    # Distribution phase
    for i in range(distribution_period):
        price_change = np.random.normal(0.01, 0.04)  # Volatile sideways
        volume = np.random.lognormal(10.5, 0.7)  # High volume
        prices.append(price_change)
        volumes.append(volume)
    
    # Dump phase
    for i in range(dump_period):
        price_change = np.random.normal(-0.03, 0.05)  # Decline
        volume = np.random.lognormal(9.5, 0.6)  # Decreasing volume
        prices.append(price_change)
        volumes.append(volume)
    
    # Convert to actual prices
    price_series = base_price * np.exp(np.cumsum(prices))
    
    df = pd.DataFrame({
        'close': price_series,
        'volume': volumes
    }, index=dates)
    
    # Initialize and train detector
    detector = LifecycleDetector()
    
    print("Training lifecycle detector...")
    results = detector.train(df)
    
    print(f"Training Results:")
    print(f"Train Accuracy: {results['train_accuracy']:.3f}")
    print(f"Test Accuracy: {results['test_accuracy']:.3f}")
    print(f"Stage Distribution: {results['stage_distribution']}")
    
    # Make predictions
    print("\nMaking predictions...")
    predictions = detector.predict(df)
    
    print(f"Current Stage: {predictions['current_stage']}")
    print(f"Stage Confidence: {predictions['stage_confidence']}")
    print(f"Stage Trend: {predictions['stage_trend']}")
    print(f"Recommendations: {predictions['recommendations']}")
    
    # Feature importance
    print("\nTop Features:")
    for feature, importance in results['feature_importance'][:5]:
        print(f"{feature}: {importance:.3f}")


if __name__ == "__main__":
    main()