#!/usr/bin/env python3
"""
Paper Trading Engine
Start trading without real money to validate your system
"""

import sys
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List

# Add paths
sys.path.extend([
    'sentiment', 'data_sources', 'ml_models', 'trading_strategies', 
    'risk_management', 'config'
])

# Import your components
from trading_config import CONFIG
from kelly_criterion import KellyCriterion
from profit_optimizer import ProfitOptimizer
from smart_entries import SmartEntryOptimizer
from advanced_risk import AdvancedRiskManager

class PaperTradingEngine:
    def __init__(self):
        self.portfolio = {
            'cash': CONFIG.STARTING_CAPITAL,
            'positions': {},
            'total_value': CONFIG.STARTING_CAPITAL,
            'trades': [],
            'daily_pnl': 0,
            'total_pnl': 0
        }
        
        # Initialize trading components
        self.kelly = KellyCriterion()
        self.profit_optimizer = ProfitOptimizer()
        self.entry_optimizer = SmartEntryOptimizer()
        self.risk_manager = AdvancedRiskManager()
        
        # Paper trading state
        self.trading_active = True
        self.last_scan = datetime.now()
        
        print("🎯 PAPER TRADING ENGINE INITIALIZED")
        print(f"   Starting Capital: ${CONFIG.STARTING_CAPITAL:,}")
        print(f"   Watchlist: {CONFIG.WATCHLIST}")
        print(f"   Max Position Size: {CONFIG.MAX_POSITION_SIZE:.0%}")
    
    def simulate_market_data(self, symbol: str) -> Dict:
        """Simulate real-time market data for testing"""
        import random
        
        # Simulate realistic meme coin data
        base_price = random.uniform(0.00001, 0.1)  # Wide range for meme coins
        
        # Generate last 20 price points
        prices = [base_price]
        volumes = []
        
        for i in range(19):
            # Meme coins are volatile!
            change = random.gauss(0, 0.08)  # 8% volatility
            new_price = max(0.000001, prices[-1] * (1 + change))
            prices.append(new_price)
            
            # Higher volume on big moves
            base_volume = random.uniform(100000, 2000000)
            if abs(change) > 0.05:  # 5%+ move
                volume = base_volume * random.uniform(2, 8)
            else:
                volume = base_volume
            volumes.append(volume)
        
        return {
            'symbol': symbol,
            'current_price': prices[-1],
            'prices': prices,
            'volumes': volumes,
            'high_prices': [p * random.uniform(1.0, 1.03) for p in prices],
            'low_prices': [p * random.uniform(0.97, 1.0) for p in prices],
            'market_cap': prices[-1] * random.uniform(10000000, 100000000),
            'volume_24h': sum(volumes[-24:]) if len(volumes) >= 24 else sum(volumes)
        }
    
    def simulate_sentiment(self, symbol: str) -> Dict:
        """Simulate sentiment data"""
        import random
        
        # Meme coins have wild sentiment swings
        sentiment_score = random.uniform(-0.8, 0.8)
        confidence = random.uniform(0.4, 0.95)
        
        return {
            'symbol': symbol,
            'sentiment_score': sentiment_score,
            'confidence': confidence,
            'reddit_mentions': random.randint(10, 500),
            'social_volume': random.uniform(0.2, 1.0)
        }
    
    def analyze_opportunity(self, symbol: str) -> Dict:
        """Analyze trading opportunity for a symbol"""
        
        # Get market data
        market_data = self.simulate_market_data(symbol)
        sentiment_data = self.simulate_sentiment(symbol)
        
        print(f"\n🔍 Analyzing {symbol}")
        print(f"   Price: ${market_data['current_price']:.6f}")
        print(f"   Volume: {market_data['volume_24h']:,.0f}")
        print(f"   Sentiment: {sentiment_data['sentiment_score']:.2f}")
        
        # Entry analysis
        try:
            entry_analysis = self.entry_optimizer.analyze_entry_opportunity(market_data)
        except Exception as e:
            print(f"   ⚠️ Entry analysis error: {e}")
            entry_analysis = {'recommendation': 'wait', 'entry_score': 0}
        
        # Combine signals
        ml_confidence = 0.75  # Mock ML prediction
        
        overall_score = 0
        factors = []
        
        # Sentiment check
        if sentiment_data['sentiment_score'] > CONFIG.MIN_SENTIMENT:
            overall_score += 2
            factors.append("✅ Bullish sentiment")
        else:
            factors.append("❌ Weak sentiment")
        
        # Entry timing
        if entry_analysis.get('recommendation') in ['buy', 'strong_buy']:
            overall_score += 3
            factors.append("✅ Good entry timing")
        else:
            factors.append("❌ Poor entry timing")
        
        # ML confidence  
        if ml_confidence > CONFIG.MIN_CONFIDENCE:
            overall_score += 2
            factors.append("✅ High ML confidence")
        else:
            factors.append("❌ Low ML confidence")
        
        # Volume
        if market_data['volume_24h'] > 1000000:
            overall_score += 1
            factors.append("✅ Good volume")
        else:
            factors.append("❌ Low volume")
        
        recommendation = 'BUY' if overall_score >= 5 else 'WAIT'
        
        return {
            'symbol': symbol,
            'recommendation': recommendation,
            'score': overall_score,
            'factors': factors,
            'market_data': market_data,
            'sentiment_data': sentiment_data,
            'ml_confidence': ml_confidence,
            'entry_analysis': entry_analysis
        }
    
    def execute_paper_trade(self, analysis: Dict) -> bool:
        """Execute a paper trade based on analysis"""
        
        if analysis['recommendation'] != 'BUY':
            return False
        
        symbol = analysis['symbol']
        current_price = analysis['market_data']['current_price']
        
        # Calculate position size using Kelly Criterion
        position_size = self.kelly.get_position_size(
            signal_confidence=analysis['ml_confidence'],
            portfolio_value=self.portfolio['total_value']
        )
        
        # Risk check
        if position_size > self.portfolio['cash']:
            print(f"   ❌ Insufficient cash: need ${position_size:,.0f}, have ${self.portfolio['cash']:,.0f}")
            return False
        
        if position_size / self.portfolio['total_value'] > CONFIG.MAX_POSITION_SIZE:
            position_size = self.portfolio['total_value'] * CONFIG.MAX_POSITION_SIZE
            print(f"   ⚠️ Position size capped at {CONFIG.MAX_POSITION_SIZE:.0%}")
        
        # Execute trade
        shares = position_size / current_price
        entry_time = datetime.now()
        
        # Add to portfolio
        self.portfolio['positions'][symbol] = {
            'shares': shares,
            'entry_price': current_price,
            'entry_time': entry_time,
            'current_value': position_size,
            'stop_loss': current_price * (1 - CONFIG.STOP_LOSS)
        }
        
        # Update cash
        self.portfolio['cash'] -= position_size
        
        # Add to profit optimizer
        self.profit_optimizer.add_position(symbol, current_price, shares)
        
        # Record trade
        trade_record = {
            'symbol': symbol,
            'action': 'BUY',
            'shares': shares,
            'price': current_price,
            'value': position_size,
            'time': entry_time,
            'analysis_score': analysis['score']
        }
        
        self.portfolio['trades'].append(trade_record)
        
        print(f"   🎯 PAPER TRADE EXECUTED:")
        print(f"      Symbol: {symbol}")
        print(f"      Shares: {shares:,.0f}")
        print(f"      Price: ${current_price:.6f}")
        print(f"      Value: ${position_size:,.2f}")
        print(f"      Stop Loss: ${trade_record['stop_loss']:.6f}" if 'stop_loss' in locals() else "")
        
        return True
    
    def update_positions(self):
        """Update all position values and check for profit taking"""
        
        for symbol in list(self.portfolio['positions'].keys()):
            position = self.portfolio['positions'][symbol]
            
            # Simulate price movement
            market_data = self.simulate_market_data(symbol)
            current_price = market_data['current_price']
            
            # Update position value
            current_value = position['shares'] * current_price
            position['current_value'] = current_value
            
            # Calculate P&L
            cost_basis = position['shares'] * position['entry_price']
            pnl = current_value - cost_basis
            pnl_pct = (current_price - position['entry_price']) / position['entry_price']
            
            print(f"\n📊 {symbol} Update:")
            print(f"   Current Price: ${current_price:.6f}")
            print(f"   P&L: ${pnl:,.2f} ({pnl_pct:.1%})")
            
            # Check profit levels
            profit_actions = self.profit_optimizer.check_profit_levels(symbol, current_price)
            
            for action in profit_actions:
                print(f"   🎯 Taking profit: {action['level']} - ${action['quantity'] * current_price:,.2f}")
                
                # Execute profit taking
                self.profit_optimizer.execute_profit_taking(
                    action['symbol'], action['level'], 
                    action['quantity'], action['price']
                )
                
                # Update position
                position['shares'] -= action['quantity']
                profit_value = action['quantity'] * current_price
                self.portfolio['cash'] += profit_value
                
                # Record trade
                self.portfolio['trades'].append({
                    'symbol': symbol,
                    'action': f'SELL_{action["level"]}',
                    'shares': action['quantity'],
                    'price': current_price,
                    'value': profit_value,
                    'time': datetime.now()
                })
            
            # Check stop loss
            if current_price <= position['stop_loss']:
                print(f"   🛑 STOP LOSS triggered at ${current_price:.6f}")
                
                # Close position
                sell_value = position['shares'] * current_price
                self.portfolio['cash'] += sell_value
                
                self.portfolio['trades'].append({
                    'symbol': symbol,
                    'action': 'STOP_LOSS',
                    'shares': position['shares'],
                    'price': current_price,
                    'value': sell_value,
                    'time': datetime.now()
                })
                
                del self.portfolio['positions'][symbol]
    
    def print_portfolio_status(self):
        """Print current portfolio status"""
        
        # Calculate total value
        position_value = sum(pos['current_value'] for pos in self.portfolio['positions'].values())
        total_value = self.portfolio['cash'] + position_value
        total_pnl = total_value - CONFIG.STARTING_CAPITAL
        total_return = total_pnl / CONFIG.STARTING_CAPITAL
        
        print(f"\n" + "="*50)
        print(f"💰 PORTFOLIO STATUS")
        print(f"="*50)
        print(f"Cash: ${self.portfolio['cash']:,.2f}")
        print(f"Positions: ${position_value:,.2f}")
        print(f"Total Value: ${total_value:,.2f}")
        print(f"Total P&L: ${total_pnl:,.2f} ({total_return:.1%})")
        print(f"Active Positions: {len(self.portfolio['positions'])}")
        print(f"Total Trades: {len(self.portfolio['trades'])}")
        
        if self.portfolio['positions']:
            print(f"\n📊 Current Positions:")
            for symbol, pos in self.portfolio['positions'].items():
                pnl = pos['current_value'] - (pos['shares'] * pos['entry_price'])
                pnl_pct = pnl / (pos['shares'] * pos['entry_price'])
                print(f"   {symbol}: ${pos['current_value']:,.0f} ({pnl_pct:.1%})")
    
    def run_trading_session(self, duration_minutes: int = 60):
        """Run a paper trading session"""
        
        print(f"\n🚀 STARTING PAPER TRADING SESSION")
        print(f"Duration: {duration_minutes} minutes")
        print(f"Scan Interval: {CONFIG.SCAN_INTERVAL} seconds")
        
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=duration_minutes)
        scan_count = 0
        
        while datetime.now() < end_time and self.trading_active:
            scan_count += 1
            print(f"\n{'='*20} SCAN #{scan_count} {'='*20}")
            
            # Update existing positions
            if self.portfolio['positions']:
                self.update_positions()
            
            # Look for new opportunities
            if len(self.portfolio['positions']) < 5:  # Max 5 positions
                for symbol in CONFIG.WATCHLIST[:3]:  # Check first 3 coins
                    if symbol not in self.portfolio['positions']:
                        analysis = self.analyze_opportunity(symbol)
                        
                        if analysis['recommendation'] == 'BUY':
                            self.execute_paper_trade(analysis)
                            break  # Only one trade per scan
            
            # Print status
            self.print_portfolio_status()
            
            # Wait for next scan
            print(f"\n⏳ Waiting {CONFIG.SCAN_INTERVAL} seconds for next scan...")
            time.sleep(CONFIG.SCAN_INTERVAL)
        
        print(f"\n🏁 TRADING SESSION COMPLETE")
        self.print_portfolio_status()
        
        # Performance summary
        final_value = self.portfolio['cash'] + sum(pos['current_value'] for pos in self.portfolio['positions'].values())
        total_return = (final_value - CONFIG.STARTING_CAPITAL) / CONFIG.STARTING_CAPITAL
        
        print(f"\n📈 SESSION PERFORMANCE:")
        print(f"   Duration: {duration_minutes} minutes")
        print(f"   Scans: {scan_count}")
        print(f"   Trades: {len(self.portfolio['trades'])}")
        print(f"   Total Return: {total_return:.1%}")
        
        return total_return

if __name__ == "__main__":
    # Start paper trading
    engine = PaperTradingEngine()
    
    print("\n🎯 Choose your paper trading session:")
    print("1. Quick test (10 minutes)")
    print("2. Short session (30 minutes)")  
    print("3. Full session (60 minutes)")
    print("4. Custom duration")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == "1":
        duration = 10
    elif choice == "2":
        duration = 30
    elif choice == "3":
        duration = 60
    elif choice == "4":
        duration = int(input("Enter duration in minutes: "))
    else:
        duration = 10  # Default
    
    try:
        return_pct = engine.run_trading_session(duration)
        
        if return_pct > 0:
            print(f"\n🎉 Profitable session! +{return_pct:.1%}")
        else:
            print(f"\n📉 Learning session: {return_pct:.1%}")
            
        print(f"\n✅ Your system is working! Ready for live trading when you are.")
        
    except KeyboardInterrupt:
        print(f"\n⏹️ Trading session stopped by user")
        engine.print_portfolio_status()