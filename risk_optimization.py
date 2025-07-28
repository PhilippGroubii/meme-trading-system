#!/usr/bin/env python3
"""
Risk Optimization After Test 2 Analysis
Fixes the issues identified in the -25.4% trading session
"""

import sys
sys.path.extend(['config', 'trading_strategies', 'risk_management'])

from trading_config import CONFIG

class OptimizedRiskManager:
    """Enhanced risk management based on test results"""
    
    def __init__(self):
        # Updated risk parameters based on meme coin volatility
        self.VOLATILITY_ADJUSTED_STOPS = {
            'low_vol': 0.15,    # 15% for stable periods
            'med_vol': 0.25,    # 25% for moderate volatility
            'high_vol': 0.35    # 35% for extreme volatility
        }
        
        self.DAILY_LOSS_LIMITS = {
            'max_daily_loss': 0.10,     # Max 10% daily loss
            'position_cooldown': 3600,   # 1 hour between failed trades
            'max_consecutive_losses': 3  # Stop after 3 losses in a row
        }
        
        self.POSITION_SIZING_ADJUSTMENTS = {
            'after_loss': 0.5,     # Reduce size by 50% after loss
            'after_profit': 1.2,   # Increase size by 20% after profit
            'max_heat': 0.3        # Max 30% portfolio in positions
        }
        
        # Session state tracking
        self.consecutive_losses = 0
        self.daily_pnl = 0
        self.last_trade_time = None
        self.session_trades = []
    
    def calculate_dynamic_stop_loss(self, entry_price: float, 
                                  recent_volatility: float) -> float:
        """Calculate stop loss based on current volatility"""
        
        # Determine volatility level
        if recent_volatility < 0.05:  # < 5% recent moves
            vol_level = 'low_vol'
        elif recent_volatility < 0.15:  # < 15% recent moves
            vol_level = 'med_vol'
        else:  # > 15% recent moves
            vol_level = 'high_vol'
        
        stop_percentage = self.VOLATILITY_ADJUSTED_STOPS[vol_level]
        stop_price = entry_price * (1 - stop_percentage)
        
        print(f"   🛡️ Dynamic stop loss: {stop_percentage:.0%} = ${stop_price:.6f}")
        return stop_price
    
    def should_skip_trade(self, current_portfolio_value: float) -> tuple:
        """Check if we should skip this trade due to risk limits"""
        
        # Check daily loss limit
        daily_loss_pct = self.daily_pnl / current_portfolio_value
        if daily_loss_pct <= -self.DAILY_LOSS_LIMITS['max_daily_loss']:
            return True, "Daily loss limit reached"
        
        # Check consecutive losses
        if self.consecutive_losses >= self.DAILY_LOSS_LIMITS['max_consecutive_losses']:
            return True, f"Too many consecutive losses ({self.consecutive_losses})"
        
        # Check cooldown period
        if self.last_trade_time and hasattr(self, '_trade_was_loss'):
            import time
            time_since_last = time.time() - self.last_trade_time
            if (self._trade_was_loss and 
                time_since_last < self.DAILY_LOSS_LIMITS['position_cooldown']):
                return True, f"Cooldown period ({time_since_last:.0f}s remaining)"
        
        return False, "Trade allowed"
    
    def adjust_position_size(self, base_size: float, 
                           recent_performance: str) -> float:
        """Adjust position size based on recent performance"""
        
        if recent_performance == 'loss':
            adjusted_size = base_size * self.POSITION_SIZING_ADJUSTMENTS['after_loss']
            print(f"   📉 Size reduced after loss: ${adjusted_size:,.0f}")
        elif recent_performance == 'profit':
            adjusted_size = base_size * self.POSITION_SIZING_ADJUSTMENTS['after_profit']
            print(f"   📈 Size increased after profit: ${adjusted_size:,.0f}")
        else:
            adjusted_size = base_size
        
        return adjusted_size
    
    def record_trade_result(self, pnl: float, trade_type: str = None):
        """Record trade result and update risk state"""
        import time
        
        self.daily_pnl += pnl
        self.last_trade_time = time.time()
        
        if pnl < 0:
            self.consecutive_losses += 1
            self._trade_was_loss = True
            print(f"   📉 Loss recorded. Consecutive losses: {self.consecutive_losses}")
        else:
            self.consecutive_losses = 0  # Reset on profit
            self._trade_was_loss = False
            print(f"   📈 Profit recorded. Consecutive losses reset.")
        
        self.session_trades.append({
            'pnl': pnl,
            'type': trade_type,
            'time': time.time(),
            'consecutive_losses': self.consecutive_losses
        })
    
    def get_risk_assessment(self) -> dict:
        """Get current risk assessment"""
        
        if not self.session_trades:
            return {'risk_level': 'normal', 'recommendation': 'continue'}
        
        recent_trades = self.session_trades[-5:]  # Last 5 trades
        win_rate = len([t for t in recent_trades if t['pnl'] > 0]) / len(recent_trades)
        
        if self.consecutive_losses >= 2:
            risk_level = 'high'
            recommendation = 'reduce_size'
        elif win_rate < 0.3:  # Less than 30% win rate
            risk_level = 'medium'
            recommendation = 'pause_new_positions'
        else:
            risk_level = 'normal'
            recommendation = 'continue'
        
        return {
            'risk_level': risk_level,
            'recommendation': recommendation,
            'consecutive_losses': self.consecutive_losses,
            'win_rate': win_rate,
            'daily_pnl': self.daily_pnl
        }


class VolatilityCalculator:
    """Calculate recent volatility for dynamic stop losses"""
    
    @staticmethod
    def calculate_recent_volatility(prices: list, periods: int = 5) -> float:
        """Calculate volatility over recent periods"""
        if len(prices) < periods + 1:
            return 0.1  # Default moderate volatility
        
        recent_prices = prices[-periods-1:]
        returns = []
        
        for i in range(1, len(recent_prices)):
            return_pct = (recent_prices[i] - recent_prices[i-1]) / recent_prices[i-1]
            returns.append(abs(return_pct))
        
        return sum(returns) / len(returns) if returns else 0.1


class EnhancedTradingConfig:
    """Enhanced configuration based on test learnings"""
    
    def __init__(self):
        # More conservative settings
        self.STARTING_CAPITAL = 10000
        self.MAX_POSITION_SIZE = 0.15  # Reduced from 20% to 15%
        self.MAX_DAILY_POSITIONS = 3   # Max 3 new positions per day
        self.MIN_SENTIMENT_THRESHOLD = 0.4  # Higher sentiment requirement
        
        # Improved profit taking
        self.PROFIT_LEVELS = {
            'level_1': {'target': 0.12, 'take': 0.30},   # 30% at +12%
            'level_2': {'target': 0.25, 'take': 0.30},   # 30% at +25%
            'level_3': {'target': 0.50, 'take': 0.25},   # 25% at +50%
            'moon_bag': {'target': float('inf'), 'take': 0.15}  # Keep 15%
        }
        
        # Trading windows (avoid volatile periods)
        self.SAFE_TRADING_HOURS = (10, 22)  # 10am-10pm EST
        self.AVOID_WEEKEND_VOLATILITY = True
        
        # Enhanced sentiment requirements
        self.SENTIMENT_CONFIDENCE_MIN = 0.7
        self.VOLUME_THRESHOLD_MULTIPLIER = 1.5  # Must be 50% above average


def create_optimized_paper_trading():
    """Create optimized paper trading with lessons learned"""
    
    optimized_script = '''#!/usr/bin/env python3
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
        
        print(f"\\n🔍 Analyzing {symbol}")
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
        
        print(f"\\n🚀 STARTING OPTIMIZED TRADING SESSION")
        print(f"Duration: {duration_minutes} minutes")
        print(f"Enhanced Risk Management: ACTIVE")
        
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=duration_minutes)
        scan_count = 0
        
        while datetime.now() < end_time:
            scan_count += 1
            print(f"\\n{'='*20} OPTIMIZED SCAN #{scan_count} {'='*20}")
            
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
                
                print(f"\\n📊 {symbol} Update:")
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
            
            print(f"\\n{'='*50}")
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
        
        print(f"\\n🏁 OPTIMIZED SESSION COMPLETE")
        print(f"Final Return: {final_return:.1%}")
        print(f"Risk Management: {len(self.risk_manager.session_trades)} decisions made")
        
        return final_return

if __name__ == "__main__":
    trader = OptimizedPaperTrading()
    trader.run_optimized_session(30)
'''
    
    with open('optimized_paper_trading.py', 'w') as f:
        f.write(optimized_script)
    
    print("✅ Optimized paper trading created")

if __name__ == "__main__":
    print("🔧 CREATING RISK OPTIMIZATIONS BASED ON TEST 2")
    print("=" * 50)
    
    # Create the enhanced components
    risk_manager = OptimizedRiskManager()
    config = EnhancedTradingConfig()
    create_optimized_paper_trading()
    
    print("\n✅ OPTIMIZATIONS COMPLETE!")
    print("\n🎯 Key Improvements:")
    print("   • Dynamic stop losses (15-35% based on volatility)")
    print("   • Position size reduction after losses")
    print("   • Daily loss limits (max 10%)")
    print("   • Consecutive loss protection")
    print("   • Higher sentiment thresholds")
    print("   • Volatility-aware trading")
    
    print("\n🚀 Run optimized version:")
    print("   python optimized_paper_trading.py")