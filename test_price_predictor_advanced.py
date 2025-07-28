# File: test_price_predictor_advanced.py
from ml_models.price_predictor import PricePredictor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def test_advanced_scenarios():
    print("🚀 ADVANCED PRICE PREDICTOR TESTING")
    print("="*50)
    
    # Test different market scenarios
    scenarios = {
        'bull_market': generate_bull_market_data(),
        'bear_market': generate_bear_market_data(),
        'sideways_market': generate_sideways_market_data(),
        'volatile_meme': generate_volatile_meme_data()
    }
    
    for scenario_name, data in scenarios.items():
        print(f"\n📊 Testing {scenario_name.upper()} scenario...")
        
        predictor = PricePredictor('ensemble')
        
        try:
            # Train on scenario data
            results = predictor.train(data, prediction_horizon=4)  # 4-hour prediction
            
            # Evaluate performance
            best_model = max(results['model_scores'].items(), 
                           key=lambda x: x[1].get('r2', -1) if 'error' not in x[1] else -1)
            
            print(f"   Best Model: {best_model[0]} (R² = {best_model[1]['r2']:.3f})")
            
            # Test predictions
            predictions = predictor.predict(data.tail(10))
            print(f"   Prediction Confidence: {np.mean(predictions['confidence']):.3f}")
            print(f"   Predicted 4h Change: {predictions['prediction_change']:.2f}%")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")

def generate_bull_market_data():
    """Generate bull market scenario"""
    dates = pd.date_range('2024-01-01', periods=200, freq='H')
    trend = np.linspace(0, 1.0, 200)  # Strong uptrend
    noise = np.random.normal(0, 0.01, 200)
    prices = 0.05 * np.exp(trend + noise)
    volume = np.random.lognormal(14, 0.8, 200)
    sentiment = np.random.normal(0.5, 0.2, 200)  # Bullish sentiment
    
    return pd.DataFrame({
        'close': prices,
        'volume': volume,
        'sentiment_score': sentiment
    }, index=dates)

def generate_bear_market_data():
    """Generate bear market scenario"""
    dates = pd.date_range('2024-01-01', periods=200, freq='H')
    trend = np.linspace(0, -0.8, 200)  # Strong downtrend
    noise = np.random.normal(0, 0.015, 200)
    prices = 0.15 * np.exp(trend + noise)
    volume = np.random.lognormal(13, 0.6, 200)  # Lower volume
    sentiment = np.random.normal(-0.3, 0.2, 200)  # Bearish sentiment
    
    return pd.DataFrame({
        'close': prices,
        'volume': volume,
        'sentiment_score': sentiment
    }, index=dates)

def generate_sideways_market_data():
    """Generate sideways market scenario"""
    dates = pd.date_range('2024-01-01', periods=200, freq='H')
    trend = np.sin(np.arange(200) * 0.1) * 0.1  # Oscillating
    noise = np.random.normal(0, 0.008, 200)
    prices = 0.08 * (1 + trend + noise)
    volume = np.random.lognormal(13.5, 0.5, 200)
    sentiment = np.random.normal(0, 0.15, 200)  # Neutral sentiment
    
    return pd.DataFrame({
        'close': prices,
        'volume': volume,
        'sentiment_score': sentiment
    }, index=dates)

def generate_volatile_meme_data():
    """Generate volatile meme coin scenario"""
    dates = pd.date_range('2024-01-01', periods=200, freq='H')
    
    # Random large moves (pumps and dumps)
    jumps = np.random.choice([0, 0.3, -0.3, 0.5, -0.4], 200, 
                           p=[0.85, 0.05, 0.05, 0.025, 0.025])
    trend = np.cumsum(jumps) * 0.1
    noise = np.random.normal(0, 0.03, 200)
    
    prices = 0.001 * np.exp(trend + noise)  # Very low price like many meme coins
    volume = np.random.lognormal(16, 1.2, 200)  # Very high volume
    
    # Sentiment follows price movements with lag
    sentiment = np.roll(jumps, 2) * 2 + np.random.normal(0, 0.2, 200)
    sentiment = np.clip(sentiment, -1, 1)
    
    return pd.DataFrame({
        'close': prices,
        'volume': volume,
        'sentiment_score': sentiment
    }, index=dates)

if __name__ == "__main__":
    test_advanced_scenarios()