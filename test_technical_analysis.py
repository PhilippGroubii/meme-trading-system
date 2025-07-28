# test_technical_analysis.py
import random
from datetime import datetime, timedelta
from analysis.technical import TechnicalAnalysis

def generate_sample_price_data(days=30, base_price=100):
    """Generate sample price data for testing"""
    data = []
    current_price = base_price
    current_time = datetime.now() - timedelta(days=days)
    
    for i in range(days * 24):  # Hourly data
        # Simulate price movement
        change = random.uniform(-0.05, 0.05)  # ±5% change
        current_price *= (1 + change)
        
        # Simulate volume
        volume = random.uniform(10000, 100000)
        
        # Create OHLC data
        high = current_price * random.uniform(1.0, 1.02)
        low = current_price * random.uniform(0.98, 1.0)
        open_price = current_price * random.uniform(0.99, 1.01)
        
        data.append({
            'timestamp': current_time.isoformat(),
            'open': open_price,
            'high': high,
            'low': low,
            'close': current_price,
            'price': current_price,  # Alternative field name
            'volume': volume
        })
        
        current_time += timedelta(hours=1)
    
    return data

def test_technical_analysis():
    print("=== Testing Technical Analysis ===")
    
    # Initialize technical analysis
    ta = TechnicalAnalysis()
    
    # Generate sample data
    print("\n=== Generating Sample Data ===")
    sample_data = generate_sample_price_data(30, 0.08)  # 30 days, starting at $0.08
    print(f"Generated {len(sample_data)} data points")
    print(f"Price range: ${min(d['close'] for d in sample_data):.6f} - ${max(d['close'] for d in sample_data):.6f}")
    
    # Test full analysis
    print("\n=== Testing Full Technical Analysis ===")
    try:
        indicators = ta.analyze(sample_data, "DOGEUSDT")
        
        print("Technical Indicators:")
        print(f"  RSI: {indicators.rsi:.2f} ({indicators.rsi_signal})")
        print(f"  MACD: {indicators.macd:.6f}")
        print(f"  MACD Signal: {indicators.macd_signal:.6f}")
        print(f"  MACD Histogram: {indicators.macd_histogram:.6f}")
        print(f"  BB Upper: ${indicators.bb_upper:.6f}")
        print(f"  BB Middle: ${indicators.bb_middle:.6f}")
        print(f"  BB Lower: ${indicators.bb_lower:.6f}")
        print(f"  BB Position: {indicators.bb_position:.3f}")
        print(f"  BB Squeeze: {indicators.bb_squeeze}")
        print(f"  EMA 12: ${indicators.ema_12:.6f}")
        print(f"  EMA 26: ${indicators.ema_26:.6f}")
        print(f"  SMA 20: ${indicators.sma_20:.6f}")
        print(f"  Volume Ratio: {indicators.volume_ratio:.2f}")
        print(f"  Volatility: {indicators.volatility:.2f}%")
        print(f"  Stochastic K: {indicators.stoch_k:.2f}")
        print(f"  Stochastic D: {indicators.stoch_d:.2f}")
        print(f"  Williams %R: {indicators.williams_r:.2f}")
        print(f"  ATR: {indicators.atr:.6f}")
        print(f"  Momentum: {indicators.momentum:.2f}%")
        
    except Exception as e:
        print(f"Error in full analysis: {e}")
        import traceback
        traceback.print_exc()
    
    # Test trading signals
    print("\n=== Testing Trading Signals ===")
    try:
        signals = ta.get_trading_signals(indicators)
        print("Individual Signals:")
        for signal_name, signal_value in signals.items():
            print(f"  {signal_name}: {signal_value}")
    except Exception as e:
        print(f"Error getting trading signals: {e}")
    
    # Test overall signal
    print("\n=== Testing Overall Signal ===")
    try:
        overall = ta.get_overall_signal(indicators)
        print("Overall Analysis:")
        print(f"  Signal: {overall['overall_signal']}")
        print(f"  Score: {overall['score']:.3f}")
        print(f"  Confidence: {overall['confidence']:.3f}")
        print(f"  Strength: {overall['strength']}")
        print(f"  Buy signals: {overall['buy_signals']}")
        print(f"  Sell signals: {overall['sell_signals']}")
        print(f"  Neutral signals: {overall['neutral_signals']}")
    except Exception as e:
        print(f"Error getting overall signal: {e}")
    
    # Test risk metrics
    print("\n=== Testing Risk Metrics ===")
    try:
        risk = ta.get_risk_metrics(indicators)
        print("Risk Assessment:")
        print(f"  Overall risk: {risk['overall_risk']:.3f}")
        print(f"  Risk level: {risk['risk_level']}")
        print(f"  Volatility risk: {risk['volatility_risk']:.3f}")
        print(f"  RSI risk: {risk['rsi_risk']:.3f}")
        print(f"  Bollinger risk: {risk['bollinger_risk']:.3f}")
        print(f"  ATR risk: {risk['atr_risk']:.3f}")
    except Exception as e:
        print(f"Error calculating risk metrics: {e}")
    
    # Test comprehensive report
    print("\n=== Testing Analysis Report ===")
    try:
        report = ta.generate_analysis_report(indicators, "DOGEUSDT")
        print("Analysis Report Summary:")
        print(f"  Symbol: {report['symbol']}")
        print(f"  TA Library: {report.get('ta_library_used', 'unknown')}")
        print(f"  Overall Signal: {report['overall_signal']}")
        print(f"  Confidence: {report['confidence']:.3f}")
        print(f"  Risk Level: {report['risk_level']}")
        print(f"  Current Price: ${report['current_price']:.6f}")
        print(f"  Support: ${report['support_level']:.6f} ({report['support_distance_pct']:.1f}% away)")
        print(f"  Resistance: ${report['resistance_level']:.6f} ({report['resistance_distance_pct']:.1f}% away)")
        
        print("  Trend Analysis:")
        for timeframe, trend in report['trend_analysis'].items():
            print(f"    {timeframe}: {trend}")
        
        if report['observations']:
            print("  Key Observations:")
            for obs in report['observations']:
                print(f"    - {obs}")
    except Exception as e:
        print(f"Error generating analysis report: {e}")
    
    # Test individual indicators
    print("\n=== Testing Individual Indicators ===")
    
    # Test with simple price data
    simple_prices = [d['close'] for d in sample_data]
    
    # Test RSI
    try:
        if hasattr(ta, 'calculate_rsi_with_ta'):
            df = ta._prepare_dataframe(sample_data)
            rsi, rsi_signal = ta.calculate_rsi_with_ta(df)
            print(f"RSI (TA lib): {rsi:.2f} ({rsi_signal})")
        
        rsi_custom, signal_custom = ta.calculate_rsi_custom(simple_prices)
        print(f"RSI (custom): {rsi_custom:.2f} ({signal_custom})")
    except Exception as e:
        print(f"Error testing RSI: {e}")
    
    # Test MACD
    try:
        if hasattr(ta, 'calculate_macd_with_ta'):
            df = ta._prepare_dataframe(sample_data)
            macd, signal, hist, macd_sig = ta.calculate_macd_with_ta(df)
            print(f"MACD (TA lib): {macd:.6f}, Signal: {signal:.6f} ({macd_sig})")
        
        macd_c, signal_c, hist_c, sig_c = ta.calculate_macd_custom(simple_prices)
        print(f"MACD (custom): {macd_c:.6f}, Signal: {signal_c:.6f} ({sig_c})")
    except Exception as e:
        print(f"Error testing MACD: {e}")
    
    # Test with minimal data
    print("\n=== Testing with Minimal Data ===")
    try:
        minimal_data = sample_data[:5]  # Only 5 data points
        minimal_indicators = ta.analyze(minimal_data, "TEST")
        print(f"Minimal data analysis - RSI: {minimal_indicators.rsi:.2f}")
        print("✓ Minimal data handled gracefully")
    except Exception as e:
        print(f"Error with minimal data: {e}")
    
    # Test with empty data
    print("\n=== Testing with Empty Data ===")
    try:
        empty_indicators = ta.analyze([], "TEST")
        print(f"Empty data analysis - RSI: {empty_indicators.rsi:.2f}")
        print("✓ Empty data handled gracefully")
    except Exception as e:
        print(f"Error with empty data: {e}")
    
    # Test different data formats
    print("\n=== Testing Different Data Formats ===")
    
    # Test simple price list
    try:
        price_list = [0.08, 0.081, 0.079, 0.082, 0.084, 0.083, 0.085, 0.087, 0.086, 0.088]
        simple_indicators = ta.analyze([{'price': p} for p in price_list], "SIMPLE")
        print(f"Simple price list - RSI: {simple_indicators.rsi:.2f}")
        print("✓ Simple price format handled")
    except Exception as e:
        print(f"Error with simple format: {e}")
    
    print("\n✓ Technical analysis testing completed!")
    print(f"✓ Using {'ta library' if hasattr(ta, 'calculate_rsi_with_ta') else 'custom implementations'}")
    print("\n=== Technical Analysis Test Complete ===")

if __name__ == "__main__":
    test_technical_analysis()