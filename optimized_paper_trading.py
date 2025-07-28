#!/usr/bin/env python3
"""
Optimized Paper Trading Engine
Incorporates lessons from Test 2 (-25.4% session)
"""

import sys
sys.path.extend(['sentiment', 'data_sources', 'ml_models', 'trading_strategies', 'risk_management', 'config'])

from risk_optimization import OptimizedRiskManager, VolatilityCalculator, EnhancedTradingConfig
from kelly_criterion import KellyCriterion
from profit_optimizer import ProfitOptimizer
from smart_entries import SmartEntryOptimizer

class OptimizedPaperTrading:
    def __init__(self):
        self.config = EnhancedTradingConfig()
        self.risk_manager = OptimizedRiskManager()
        self.volatility_calc = VolatilityCalculator()
        
        # Trading components
        self.kelly = KellyCriterion()
        self.profit_optimizer = ProfitOptimizer()
        self.entry_optimizer = SmartEntryOptimizer()
        
        # Portfolio
        self.portfolio = {
            'cash': self.config.STARTING_CAPITAL,
            'positions': {},
            'total_value': self.config.STARTING_CAPITAL,
            'trades': []
        }
        
        print("🎯 OPTIMIZED PAPER TRADING ENGINE INITIALIZED")
        print(f"   Starting Capital: ${self.config.STARTING_CAPITAL:,}")
        print(f"   Max Position Size: {self.config.MAX_POSITION_SIZE:.0%}")
        print(f"   Enhanced Risk Management: ACTIVE")
        print(f"   Dynamic Stop Losses: ACTIVE")
    
    def analyze_opportunity_v2(self, symbol: str) -> dict:
        """Enhanced opportunity analysis with volatility consideration"""
        import random
        
        # Simulate market data
        base_price = random.uniform(0.001, 0.1)
        prices = [base_price * (1 + random.gauss(0, 0.15)) for _ in range(20)]
        volumes = [random.uniform(10000000, 100000000) for _ in range(20)]
        
        current_price = prices[-1]
        sentiment_score = random.uniform(-0.8, 0.8)
        
        # Calculate volatility
        recent_volatility = self.volatility_calc.calculate_recent_volatility(prices)
        
        print(f"\n🔍 Analyzing {symbol}")
        print(f"   Price: ${current_price:.6f}")
        print(f"   Volume: {volumes[-1]:,.0f}")
        print(f"   Sentiment: {sentiment_score:.2f}")
        print(f"   Recent Volatility: {recent_volatility:.1%}")
        
        # Enhanced scoring system
        score = 0
        factors = []
        
        # Sentiment (higher threshold)
        if sentiment_score > self.config.MIN_SENTIMENT_THRESHOLD:
            score += 3
            factors.append("✅ Strong sentiment")
        else:
            factors.append("❌ Weak sentiment")
        
        # Volatility consideration
        if recent_volatility < 0.2:  # Less than 20% volatility
            score += 2
            factors.append("✅ Reasonable volatility")
        else:
            score -= 1
            factors.append("⚠️ High volatility detected")
        
        # Volume check
        avg_volume = sum(volumes[-5:]) / 5
        if volumes[-1] > avg_volume * self.config.VOLUME_THRESHOLD_MULTIPLIER:
            score += 1
            factors.append("✅ Volume spike")
        
        # Risk assessment
        risk_assessment = self.risk_manager.get_risk_assessment()
        if risk_assessment['risk_level'] == 'high':
            score -= 3
            factors.append("🛑 High risk period")
        elif risk_assessment['risk_level'] == 'medium':
            score -= 1
            factors.append("⚠️ Medium risk period")
        
        recommendation = 'BUY' if score >= 4 else 'WAIT'
        
        return {
            'symbol': symbol,
            'recommendation': recommendation,
            'score': score,
            'factors': factors,
            'current_price': current_price,
            'sentiment_score': sentiment_score,
            'recent_volatility': recent_volatility,
            'risk_assessment': risk_assessment
        }
    
    def execute_optimized_trade(self, analysis: dict) -> bool:
        """Execute trade with enhanced risk management"""
        
        if analysis['recommendation'] != 'BUY':
            return False
        
        symbol = analysis['symbol']
        current_price = analysis['current_price']
        recent_volatility = analysis['recent_volatility']
        
        # Check risk limits
        should_skip, reason = self.risk_manager.should_skip_trade(self.portfolio['total_value'])
        if should_skip:
            print(f"   🛑 Trade skipped: {reason}")
            return False
        
        # Calculate position size
        base_size = self.kelly.get_position_size(0.7, self.portfolio['total_value'])
        
        # Adjust for recent performance
        recent_performance = 'normal'
        if len(self.portfolio['trades']) > 0:
            last_trade = self.portfolio['trades'][-1]
            recent_performance = 'profit' if last_trade.get('pnl', 0) > 0 else 'loss'
        
        adjusted_size = self.risk_manager.adjust_position_size(base_size, recent_performance)
        
        # Apply config limits
        max_allowed = self.portfolio['total_value'] * self.config.MAX_POSITION_SIZE
        position_size = min(adjusted_size, max_allowed, self.portfolio['cash'])
        
        if position_size < 100:  # Minimum trade size
            print("   ❌ Position size too small")
            return False
        
        # Calculate dynamic stop loss
        stop_loss = self.risk_manager.calculate_dynamic_stop_loss(current_price, recent_volatility)
        
        # Execute trade
        shares = position_size / current_price
        
        self.portfolio['positions'][symbol] = {
            'shares': shares,
            'entry_price': current_price,
            'current_value': position_size,
            'stop_loss': stop_loss,
            'entry_volatility': recent_volatility
        }
        
        self.portfolio['cash'] -= position_size
        
        # Add to profit optimizer
        self.profit_optimizer.add_position(symbol, current_price, shares)
        
        print(f"   🎯 OPTIMIZED TRADE EXECUTED:")
        print(f"      Symbol: {symbol}")
        print(f"      Shares: {shares:,.0f}")
        print(f"      Price: ${current_price:.6f}")
        print(f"      Value: ${position_size:,.2f}")
        print(f"      Stop Loss: ${stop_loss:.6f} ({((current_price-stop_loss)/current_price):.1%})")
        
        return True
    
    def run_optimized_session(self, duration_minutes: int = 30):
        """Run optimized trading session"""
        from datetime import datetime, timedelta
        import time
        
        print(f"\n🚀 STARTING OPTIMIZED TRADING SESSION")
        print(f"Duration: {duration_minutes} minutes")
        print(f"Enhanced Risk Management: ACTIVE")
        
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=duration_minutes)
        scan_count = 0
        
        while datetime.now() < end_time:
            scan_count += 1
            print(f"\n{'='*20} OPTIMIZED SCAN #{scan_count} {'='*20}")
            
            # Risk assessment
            risk_assessment = self.risk_manager.get_risk_assessment()
            print(f"📊 Risk Level: {risk_assessment['risk_level']}")
            
            # Update existing positions (simplified for demo)
            positions_to_close = []
            
            for symbol in list(self.portfolio['positions'].keys()):
                position = self.portfolio['positions'][symbol]
                
                # Simulate price movement
                import random
                volatility = position['entry_volatility']
                price_change = random.gauss(0, volatility)
                current_price = position['entry_price'] * (1 + price_change)
                
                current_value = position['shares'] * current_price
                pnl = current_value - (position['shares'] * position['entry_price'])
                pnl_pct = pnl / (position['shares'] * position['entry_price'])
                
                print(f"\n📊 {symbol} Update:")
                print(f"   Current Price: ${current_price:.6f}")
                print(f"   P&L: ${pnl:,.2f} ({pnl_pct:.1%})")
                
                # Check stop loss
                if current_price <= position['stop_loss']:
                    print(f"   🛑 STOP LOSS triggered")
                    
                    # Close position
                    self.portfolio['cash'] += current_value
                    self.risk_manager.record_trade_result(pnl, 'stop_loss')
                    positions_to_close.append(symbol)
                    
                    self.portfolio['trades'].append({
                        'symbol': symbol,
                        'action': 'STOP_LOSS',
                        'pnl': pnl,
                        'pnl_pct': pnl_pct
                    })
                
                # Check profit levels
                else:
                    profit_actions = self.profit_optimizer.check_profit_levels(symbol, current_price)
                    for action in profit_actions:
                        print(f"   🎯 Taking profit: {action['level']}")
                        
                        # Record partial profit
                        partial_pnl = action['quantity'] * (current_price - position['entry_price'])
                        self.risk_manager.record_trade_result(partial_pnl, 'profit_taking')
                        
                        # Update position
                        position['shares'] -= action['quantity']
                        self.portfolio['cash'] += action['quantity'] * current_price
            
            # Remove closed positions
            for symbol in positions_to_close:
                del self.portfolio['positions'][symbol]
            
            # Look for new opportunities (limit to avoid overtrading)
            if len(self.portfolio['positions']) < 2:  # Max 2 positions
                watchlist = ['DOGE', 'SHIB', 'PEPE'][:1]  # Only check 1 coin per scan
                
                for symbol in watchlist:
                    if symbol not in self.portfolio['positions']:
                        analysis = self.analyze_opportunity_v2(symbol)
                        
                        if analysis['recommendation'] == 'BUY':
                            self.execute_optimized_trade(analysis)
                            break  # Only one trade per scan
            
            # Portfolio status
            position_value = sum(pos['current_value'] for pos in self.portfolio['positions'].values())
            total_value = self.portfolio['cash'] + position_value
            total_pnl = total_value - self.config.STARTING_CAPITAL
            
            print(f"\n{'='*50}")
            print(f"💰 OPTIMIZED PORTFOLIO STATUS")
            print(f"{'='*50}")
            print(f"Cash: ${self.portfolio['cash']:,.2f}")
            print(f"Positions: ${position_value:,.2f}")
            print(f"Total Value: ${total_value:,.2f}")
            print(f"Total P&L: ${total_pnl:,.2f} ({total_pnl/self.config.STARTING_CAPITAL:.1%})")
            print(f"Risk Level: {risk_assessment['risk_level']}")
            
            time.sleep(300)  # 5 minute intervals
        
        # Final results
        final_value = self.portfolio['cash'] + sum(pos['current_value'] for pos in self.portfolio['positions'].values())
        final_return = (final_value - self.config.STARTING_CAPITAL) / self.config.STARTING_CAPITAL
        
        print(f"\n🏁 OPTIMIZED SESSION COMPLETE")
        print(f"Final Return: {final_return:.1%}")
        print(f"Risk Management: {len(self.risk_manager.session_trades)} decisions made")
        
        return final_return

if __name__ == "__main__":
    trader = OptimizedPaperTrading()
    trader.run_optimized_session(30)
