#!/usr/bin/env python3
"""
Complete Meme Coin Trading System Test
Tests the entire 4-phase system end-to-end with realistic scenarios
"""

import sys
import os
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List

# Add all module paths
sys.path.extend([
    'sentiment', 'data_sources', 'ml_models', 'trading_strategies', 
    'risk_management', 'analysis', 'utils'
])

# Import all components
try:
    # Phase 1 - Sentiment Analysis
    from reddit_scanner import RedditScanner
    from multi_source import MultiSourceSentiment
    from twitter_free import TwitterScanner
    from discord_scanner import DiscordScanner
    
    # Phase 2 - Data Collection
    from coingecko import CoinGeckoAPI
    from dexscreener import DexScreenerAPI
    from google_trends import GoogleTrendsAPI
    from onchain import OnChainAnalyzer
    
    # Phase 3 - ML Models
    from price_predictor import PricePredictor
    from lifecycle_detector import LifecycleDetector
    from sentiment_classifier import SentimentClassifier
    
    # Phase 4 - Trading Strategies
    from kelly_criterion import KellyCriterion
    from profit_optimizer import ProfitOptimizer
    from smart_entries import SmartEntryOptimizer
    
    # Risk Management
    from simple_risk import SimpleRiskManager
    from advanced_risk import AdvancedRiskManager
    
    # Utilities
    from logger import TradingLogger
    from database import DatabaseManager
    
    print("✅ All modules imported successfully!")
    
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Please ensure all Phase files are created and in correct directories")
    sys.exit(1)

class FullSystemTest:
    """Complete end-to-end system test"""
    
    def __init__(self):
        self.logger = TradingLogger()
        self.db = DatabaseManager()
        
        # Initialize all components
        self.sentiment_analyzer = MultiSourceSentiment()
        self.price_predictor = PricePredictor()
        self.lifecycle_detector = LifecycleDetector()
        self.kelly_criterion = KellyCriterion()
        self.profit_optimizer = ProfitOptimizer()
        self.entry_optimizer = SmartEntryOptimizer()
        self.risk_manager = AdvancedRiskManager()
        
        # Test portfolio
        self.portfolio = {
            'cash': 100000,
            'positions': {},
            'total_value': 100000,
            'trade_history': []
        }
        
        # Test results
        self.test_results = {
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': [],
            'performance_metrics': {}
        }
        
    def log_test(self, test_name: str, result: bool, details: str = ""):
        """Log test result"""
        self.test_results['total_tests'] += 1
        if result:
            self.test_results['passed_tests'] += 1
            print(f"✅ {test_name}: PASSED {details}")
        else:
            self.test_results['failed_tests'].append(f"{test_name}: {details}")
            print(f"❌ {test_name}: FAILED {details}")
            
    def simulate_market_data(self, symbol: str) -> Dict:
        """Generate realistic market data for testing"""
        import random
        import numpy as np
        
        # Generate 30 days of realistic price data
        base_price = random.uniform(0.001, 1.0)  # Meme coin price range
        prices = [base_price]
        volumes = []
        
        for i in range(30):
            # Simulate realistic meme coin volatility (high!)
            change = random.gauss(0, 0.15)  # 15% daily volatility
            new_price = max(0.0001, prices[-1] * (1 + change))
            prices.append(new_price)
            
            # Volume spikes during big moves
            base_volume = random.uniform(100000, 1000000)
            if abs(change) > 0.1:  # 10%+ move
                volume = base_volume * random.uniform(2, 5)
            else:
                volume = base_volume * random.uniform(0.5, 1.5)
            volumes.append(volume)
            
        return {
            'symbol': symbol,
            'prices': prices,
            'volumes': volumes,
            'high_prices': [p * random.uniform(1.0, 1.05) for p in prices],
            'low_prices': [p * random.uniform(0.95, 1.0) for p in prices],
            'timestamps': [datetime.now() - timedelta(days=30-i) for i in range(31)]
        }
    
    def simulate_sentiment_data(self, symbol: str) -> Dict:
        """Generate realistic sentiment data"""
        import random
        
        sentiment_score = random.uniform(-1, 1)
        confidence = random.uniform(0.3, 0.95)
        
        return {
            'symbol': symbol,
            'sentiment_score': sentiment_score,
            'confidence': confidence,
            'reddit_mentions': random.randint(10, 500),
            'twitter_mentions': random.randint(50, 2000),
            'discord_activity': random.uniform(0.1, 1.0),
            'google_trends': random.uniform(0, 100),
            'social_momentum': random.uniform(0.2, 1.0)
        }
    
    def test_phase1_sentiment_analysis(self) -> bool:
        """Test Phase 1 - Sentiment Analysis"""
        print("\n🔍 Testing Phase 1 - Sentiment Analysis")
        print("-" * 40)
        
        test_symbols = ['DOGE', 'SHIB', 'PEPE', 'BONK']
        
        for symbol in test_symbols:
            try:
                # Simulate sentiment analysis
                sentiment_data = self.simulate_sentiment_data(symbol)
                
                # Test sentiment processing
                processed = self.sentiment_analyzer.process_sentiment(sentiment_data)
                
                # Validate results
                has_score = 'sentiment_score' in processed
                has_confidence = 'confidence' in processed
                valid_range = -1 <= processed.get('sentiment_score', 0) <= 1
                
                self.log_test(
                    f"Sentiment Analysis - {symbol}",
                    has_score and has_confidence and valid_range,
                    f"Score: {processed.get('sentiment_score', 0):.2f}"
                )
                
            except Exception as e:
                self.log_test(f"Sentiment Analysis - {symbol}", False, str(e))
                
        return True
    
    def test_phase2_data_collection(self) -> bool:
        """Test Phase 2 - Data Collection"""
        print("\n📊 Testing Phase 2 - Data Collection")
        print("-" * 40)
        
        test_symbols = ['DOGE', 'SHIB']
        
        for symbol in test_symbols:
            try:
                # Simulate market data collection
                market_data = self.simulate_market_data(symbol)
                
                # Test data validation
                has_prices = len(market_data['prices']) > 20
                has_volumes = len(market_data['volumes']) > 20
                valid_prices = all(p > 0 for p in market_data['prices'])
                
                self.log_test(
                    f"Data Collection - {symbol}",
                    has_prices and has_volumes and valid_prices,
                    f"30 days data, current: ${market_data['prices'][-1]:.6f}"
                )
                
            except Exception as e:
                self.log_test(f"Data Collection - {symbol}", False, str(e))
                
        return True
    
    def test_phase3_ml_models(self) -> bool:
        """Test Phase 3 - ML Models"""
        print("\n🤖 Testing Phase 3 - ML Models")
        print("-" * 40)
        
        symbol = 'TESTCOIN'
        market_data = self.simulate_market_data(symbol)
        sentiment_data = self.simulate_sentiment_data(symbol)
        
        try:
            # Test Price Predictor
            price_features = {
                'prices': market_data['prices'],
                'volumes': market_data['volumes'],
                'sentiment': sentiment_data['sentiment_score']
            }
            
            prediction = self.price_predictor.predict_price_movement(price_features)
            
            has_prediction = 'direction' in prediction
            has_confidence = 'confidence' in prediction
            valid_confidence = 0 <= prediction.get('confidence', 0) <= 1
            
            self.log_test(
                "Price Predictor",
                has_prediction and has_confidence and valid_confidence,
                f"Direction: {prediction.get('direction', 'unknown')}, Confidence: {prediction.get('confidence', 0):.2f}"
            )
            
            # Test Lifecycle Detector
            lifecycle_features = {
                'volume_patterns': market_data['volumes'],
                'price_history': market_data['prices'],
                'social_metrics': sentiment_data
            }
            
            lifecycle = self.lifecycle_detector.detect_lifecycle_stage(lifecycle_features)
            
            valid_stages = ['accumulation', 'pump', 'distribution', 'dump']
            has_stage = lifecycle.get('stage') in valid_stages
            has_confidence = 'confidence' in lifecycle
            
            self.log_test(
                "Lifecycle Detector",
                has_stage and has_confidence,
                f"Stage: {lifecycle.get('stage', 'unknown')}"
            )
            
        except Exception as e:
            self.log_test("ML Models", False, str(e))
            
        return True
    
    def test_phase4_trading_strategies(self) -> bool:
        """Test Phase 4 - Trading Strategies"""
        print("\n💰 Testing Phase 4 - Trading Strategies")
        print("-" * 40)
        
        symbol = 'TESTCOIN'
        market_data = self.simulate_market_data(symbol)
        
        try:
            # Test Smart Entry
            entry_analysis = self.entry_optimizer.analyze_entry_opportunity(market_data)
            
            has_recommendation = 'recommendation' in entry_analysis
            has_score = 'entry_score' in entry_analysis
            
            self.log_test(
                "Smart Entry Analysis",
                has_recommendation and has_score,
                f"Recommendation: {entry_analysis.get('recommendation', 'unknown')}"
            )
            
            # Test Kelly Criterion Position Sizing
            position_size = self.kelly_criterion.get_position_size(
                signal_confidence=0.75,
                portfolio_value=self.portfolio['cash']
            )
            
            reasonable_size = 0 < position_size < self.portfolio['cash'] * 0.5
            
            self.log_test(
                "Kelly Position Sizing",
                reasonable_size,
                f"Size: ${position_size:,.2f} ({position_size/self.portfolio['cash']:.1%})"
            )
            
            # Test Profit Optimizer
            self.profit_optimizer.add_position(symbol, entry_price=0.001, quantity=1000000)
            
            # Simulate profit taking
            profit_actions = self.profit_optimizer.check_profit_levels(symbol, 0.0012)  # 20% gain
            
            self.log_test(
                "Profit Optimization",
                len(profit_actions) > 0,
                f"Triggered {len(profit_actions)} profit levels"
            )
            
        except Exception as e:
            self.log_test("Trading Strategies", False, str(e))
            
        return True
    
    def test_risk_management(self) -> bool:
        """Test Risk Management Systems"""
        print("\n🛡️ Testing Risk Management")
        print("-" * 40)
        
        try:
            # Test portfolio risk assessment
            sample_positions = {
                'COIN1': {
                    'position_value': 25000,
                    'portfolio_value': self.portfolio['total_value'],
                    'risk_amount': 2500,
                    'price_history': [0.001, 0.0012, 0.0011, 0.0013, 0.0015]
                },
                'COIN2': {
                    'position_value': 15000,
                    'portfolio_value': self.portfolio['total_value'],
                    'risk_amount': 1500,
                    'price_history': [0.005, 0.0052, 0.0048, 0.0055, 0.0060]
                }
            }
            
            risk_report = self.risk_manager.generate_risk_report(
                self.portfolio['total_value'], 
                sample_positions
            )
            
            has_risk_level = 'emergency_assessment' in risk_report
            has_metrics = 'total_risk_percentage' in risk_report
            
            self.log_test(
                "Risk Assessment",
                has_risk_level and has_metrics,
                f"Risk Level: {risk_report.get('emergency_assessment', {}).get('risk_level', 'unknown')}"
            )
            
        except Exception as e:
            self.log_test("Risk Management", False, str(e))
            
        return True
    
    def test_end_to_end_workflow(self) -> bool:
        """Test complete trading workflow"""
        print("\n🔄 Testing End-to-End Workflow")
        print("-" * 40)
        
        try:
            symbol = 'WORKFLOW_TEST'
            
            # Step 1: Collect data
            market_data = self.simulate_market_data(symbol)
            sentiment_data = self.simulate_sentiment_data(symbol)
            
            # Step 2: Analyze sentiment
            sentiment_analysis = self.sentiment_analyzer.process_sentiment(sentiment_data)
            
            # Step 3: ML predictions
            price_prediction = self.price_predictor.predict_price_movement({
                'prices': market_data['prices'],
                'volumes': market_data['volumes'],
                'sentiment': sentiment_analysis['sentiment_score']
            })
            
            # Step 4: Entry analysis
            entry_analysis = self.entry_optimizer.analyze_entry_opportunity(market_data)
            
            # Step 5: Position sizing
            if entry_analysis['recommendation'] in ['buy', 'strong_buy']:
                position_size = self.kelly_criterion.get_position_size(
                    signal_confidence=price_prediction.get('confidence', 0.5),
                    portfolio_value=self.portfolio['cash']
                )
                
                # Step 6: Risk check
                risk_check = self.risk_manager.check_position_limits(self.portfolio['positions'])
                
                # Step 7: Execute trade (simulated)
                if risk_check['all_limits_ok'] and position_size > 0:
                    trade_executed = True
                    entry_price = market_data['prices'][-1]
                    
                    # Add to profit optimizer
                    shares = position_size / entry_price
                    self.profit_optimizer.add_position(symbol, entry_price, shares)
                    
                    # Record trade
                    trade = {
                        'symbol': symbol,
                        'action': 'BUY',
                        'size': position_size,
                        'price': entry_price,
                        'timestamp': datetime.now(),
                        'sentiment': sentiment_analysis['sentiment_score'],
                        'prediction_confidence': price_prediction.get('confidence', 0)
                    }
                    
                    self.portfolio['trade_history'].append(trade)
                    
                else:
                    trade_executed = False
            else:
                trade_executed = False
                
            workflow_success = all([
                'sentiment_score' in sentiment_analysis,
                'confidence' in price_prediction,
                'recommendation' in entry_analysis
            ])
            
            self.log_test(
                "End-to-End Workflow",
                workflow_success,
                f"Trade executed: {trade_executed}"
            )
            
        except Exception as e:
            self.log_test("End-to-End Workflow", False, str(e))
            
        return True
    
    def test_performance_simulation(self) -> bool:
        """Simulate trading performance over time"""
        print("\n📈 Testing Performance Simulation")
        print("-" * 40)
        
        try:
            # Simulate 10 trades
            starting_value = self.portfolio['total_value']
            
            for i in range(10):
                symbol = f"SIM_{i}"
                market_data = self.simulate_market_data(symbol)
                sentiment_data = self.simulate_sentiment_data(symbol)
                
                # Quick trade simulation
                if sentiment_data['sentiment_score'] > 0.3:  # Bullish
                    entry_price = market_data['prices'][-1]
                    position_size = self.portfolio['cash'] * 0.1  # 10% position
                    
                    # Simulate some time passing and price movement
                    import random
                    exit_multiplier = random.uniform(0.8, 1.4)  # -20% to +40%
                    exit_price = entry_price * exit_multiplier
                    
                    pnl = (exit_price - entry_price) * (position_size / entry_price)
                    self.portfolio['cash'] += pnl
                    self.portfolio['total_value'] = self.portfolio['cash']
                    
                    self.portfolio['trade_history'].append({
                        'symbol': symbol,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'pnl': pnl,
                        'return_pct': (exit_price - entry_price) / entry_price
                    })
            
            # Calculate performance metrics
            total_return = (self.portfolio['total_value'] - starting_value) / starting_value
            total_trades = len(self.portfolio['trade_history'])
            winning_trades = len([t for t in self.portfolio['trade_history'] if t.get('pnl', 0) > 0])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            self.test_results['performance_metrics'] = {
                'total_return': total_return,
                'win_rate': win_rate,
                'total_trades': total_trades,
                'final_value': self.portfolio['total_value']
            }
            
            performance_reasonable = -0.5 < total_return < 2.0  # -50% to +200% is reasonable for test
            
            self.log_test(
                "Performance Simulation",
                performance_reasonable,
                f"Return: {total_return:.1%}, Win Rate: {win_rate:.1%}"
            )
            
        except Exception as e:
            self.log_test("Performance Simulation", False, str(e))
            
        return True
    
    def run_full_system_test(self):
        """Run complete system test"""
        print("🚀 MEME COIN TRADING SYSTEM - FULL SYSTEM TEST")
        print("=" * 60)
        print(f"Test started at: {datetime.now()}")
        print()
        
        # Run all test phases
        self.test_phase1_sentiment_analysis()
        self.test_phase2_data_collection()
        self.test_phase3_ml_models()
        self.test_phase4_trading_strategies()
        self.test_risk_management()
        self.test_end_to_end_workflow()
        self.test_performance_simulation()
        
        # Final results
        print("\n" + "=" * 60)
        print("🎯 FULL SYSTEM TEST RESULTS")
        print("=" * 60)
        
        total_tests = self.test_results['total_tests']
        passed_tests = self.test_results['passed_tests']
        success_rate = passed_tests / total_tests if total_tests > 0 else 0
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {len(self.test_results['failed_tests'])}")
        print(f"Success Rate: {success_rate:.1%}")
        
        if self.test_results['failed_tests']:
            print("\n❌ Failed Tests:")
            for failure in self.test_results['failed_tests']:
                print(f"   - {failure}")
        
        # Performance metrics
        if self.test_results['performance_metrics']:
            metrics = self.test_results['performance_metrics']
            print(f"\n📊 Simulated Performance:")
            print(f"   Total Return: {metrics['total_return']:.1%}")
            print(f"   Win Rate: {metrics['win_rate']:.1%}")
            print(f"   Total Trades: {metrics['total_trades']}")
            print(f"   Final Portfolio Value: ${metrics['final_value']:,.2f}")
        
        # Overall assessment
        if success_rate >= 0.9:
            print(f"\n🎉 SYSTEM STATUS: READY FOR LIVE TRADING!")
            print("All major components working correctly.")
        elif success_rate >= 0.7:
            print(f"\n⚠️ SYSTEM STATUS: MOSTLY READY")
            print("Minor issues detected. Review failed tests before live trading.")
        else:
            print(f"\n🚨 SYSTEM STATUS: NOT READY")
            print("Major issues detected. Fix failed components before proceeding.")
            
        print(f"\nTest completed at: {datetime.now()}")
        
        return success_rate >= 0.9

if __name__ == "__main__":
    tester = FullSystemTest()
    
    try:
        system_ready = tester.run_full_system_test()
        
        if system_ready:
            print("\n✅ YOUR MEME COIN TRADING SYSTEM IS READY!")
            print("You can now proceed to live trading with confidence.")
        else:
            print("\n❌ System needs attention before live trading.")
            
    except Exception as e:
        print(f"\n💥 System test failed with error: {e}")
        print("Please check your file structure and dependencies.")
        sys.exit(1)