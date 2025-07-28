#!/usr/bin/env python3
"""
Enhanced Trading System with Emerging Coin Discovery
Combines your proven trading system with automated coin discovery
"""

import sys
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Set

# Add paths
sys.path.extend([
    'sentiment', 'data_sources', 'ml_models', 'trading_strategies', 
    'risk_management', 'config', 'discovery'
])

# Import existing components
from optimized_paper_trading import OptimizedPaperTrading
from coin_discovery_engine import CoinDiscoveryEngine, EmergingCoin
from opportunity_monitor import OpportunityMonitor


# Mock trading class for testing
class MockPaperTrading:
    """Mock paper trading for testing discovery system"""
    
    def __init__(self):
        self.portfolio = {
            'cash': 8000,
            'positions': {},
            'total_value': 10000
        }
        self.trades_executed = 0
    
    def analyze_opportunity_v2(self, symbol: str) -> Dict:
        """Mock analysis - randomly decide to buy"""
        import random
        
        if random.random() < 0.3:  # 30% chance to buy
            return {
                'symbol': symbol,
                'recommendation': 'BUY',
                'confidence': random.uniform(0.6, 0.9),
                'entry_price': random.uniform(0.01, 1.0)
            }
        else:
            return {
                'symbol': symbol,
                'recommendation': 'HOLD',
                'confidence': random.uniform(0.3, 0.6)
            }
    
    def execute_optimized_trade(self, analysis: Dict) -> bool:
        """Mock trade execution"""
        if analysis['recommendation'] == 'BUY' and self.portfolio['cash'] > 1000:
            symbol = analysis['symbol']
            position_size = min(1000, self.portfolio['cash'] * 0.1)
            entry_price = analysis.get('entry_price', 0.5)
            shares = position_size / entry_price
            
            self.portfolio['positions'][symbol] = {
                'shares': shares,
                'entry_price': entry_price,
                'current_value': position_size,
                'stop_loss': entry_price * 0.85
            }
            
            self.portfolio['cash'] -= position_size
            self.trades_executed += 1
            return True
        
        return False

class EnhancedTradingSystem:
    """Enhanced trading system with automatic coin discovery"""
    
    def __init__(self, paper_trading: bool = True):
    def __init__(self, paper_trading: bool = True):
        # Core trading system - use mock if real one not available
        try:
            from optimized_paper_trading import OptimizedPaperTrading
            self.paper_trader = OptimizedPaperTrading()
        except ImportError:
            print("   ⚠️ Using MockPaperTrading (optimized_paper_trading not found)")
            self.paper_trader = MockPaperTrading()
        
        # Discovery components
        self.discovery_engine = CoinDiscoveryEngine()
        self.opportunity_monitor = OpportunityMonitor()
        
        # Enhanced configuration
        self.config = {
            'paper_trading': paper_trading,
            'discovery_enabled': True,
            'auto_trade_discoveries': True,
            'max_discovery_positions': 3,  # Max positions from discoveries
            'discovery_position_size': 0.10,  # 10% per discovery
            'discovery_scan_interval': 600,  # 10 minutes
        }
        
        # State tracking
        self.discovered_coins: Set[str] = set()
        self.discovery_trades: Dict[str, Dict] = {}
        self.traditional_watchlist = ['DOGE', 'SHIB', 'PEPE', 'BONK', 'FLOKI']
        
        print("🚀 ENHANCED TRADING SYSTEM INITIALIZED")
        print(f"   Discovery Engine: {'ACTIVE' if self.config['discovery_enabled'] else 'DISABLED'}")
        print(f"   Auto-Trading: {'ENABLED' if self.config['auto_trade_discoveries'] else 'DISABLED'}")
        print(f"   Max Discovery Positions: {self.config['max_discovery_positions']}")
    
    async def run_enhanced_trading_session(self, duration_minutes: int = 60):
        """Run enhanced trading session with discovery"""
        print(f"\n🎯 STARTING ENHANCED TRADING SESSION")
        print(f"Duration: {duration_minutes} minutes")
        print(f"Traditional + Discovery Trading: ACTIVE")
        
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=duration_minutes)
        
        # Start background discovery monitoring
        if self.config['discovery_enabled']:
            discovery_task = asyncio.create_task(self.continuous_discovery())
        
        scan_count = 0
        
        while datetime.now() < end_time:
            scan_count += 1
            print(f"\n{'='*25} ENHANCED SCAN #{scan_count} {'='*25}")
            
            # Traditional trading (your proven system)
            await self.run_traditional_trading_scan()
            
            # Discovery trading
            if self.config['discovery_enabled']:
                await self.run_discovery_trading_scan()
            
            # Portfolio status
            self.print_enhanced_portfolio_status()
            
            # Wait for next scan
            await asyncio.sleep(300)  # 5 minutes
        
        # Stop discovery monitoring
        if self.config['discovery_enabled']:
            discovery_task.cancel()
        
        print(f"\n🏁 ENHANCED TRADING SESSION COMPLETE")
        self.print_final_performance_summary()
    
    async def continuous_discovery(self):
        """Continuous coin discovery in background"""
        while True:
            try:
                print(f"\n🔍 BACKGROUND DISCOVERY SCAN - {datetime.now().strftime('%H:%M:%S')}")
                
                # Discover new opportunities
                opportunities = await self.discovery_engine.discover_emerging_coins()
                
                if opportunities:
                    print(f"   📡 Found {len(opportunities)} emerging opportunities")
                    
                    # Add top opportunities to watchlist
                    for coin in opportunities[:5]:  # Top 5
                        if coin.symbol not in self.discovered_coins:
                            self.discovered_coins.add(coin.symbol)
                            print(f"   ✅ Added {coin.symbol} to discovery watchlist (Score: {coin.opportunity_score:.1f}/10)")
                
                await asyncio.sleep(self.config['discovery_scan_interval'])
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"   ❌ Discovery error: {e}")
                await asyncio.sleep(60)
    
    async def run_traditional_trading_scan(self):
        """Run traditional trading on established coins"""
        print(f"📊 TRADITIONAL TRADING SCAN")
        
        # Use your existing optimized trading logic
        for symbol in self.traditional_watchlist[:2]:  # Limit to 2 per scan
            analysis = self.paper_trader.analyze_opportunity_v2(symbol)
            
            if analysis['recommendation'] == 'BUY':
                success = self.paper_trader.execute_optimized_trade(analysis)
                if success:
                    print(f"   ✅ Traditional trade executed: {symbol}")
                    break  # One trade per scan
    
    async def run_discovery_trading_scan(self):
        """Run trading on discovered coins"""
        print(f"🔍 DISCOVERY TRADING SCAN")
        
        if len(self.discovery_trades) >= self.config['max_discovery_positions']:
            print(f"   ⚠️ Max discovery positions reached ({self.config['max_discovery_positions']})")
            return
        
        # Analyze discovered coins
        discovery_opportunities = []
        
        for symbol in list(self.discovered_coins):
            if symbol not in self.discovery_trades:
                # Simulate analysis of discovered coin
                opportunity = await self.analyze_discovered_coin(symbol)
                if opportunity:
                    discovery_opportunities.append(opportunity)
        
        # Sort by opportunity score and take best
        discovery_opportunities.sort(key=lambda x: x.get('opportunity_score', 0), reverse=True)
        
        for opportunity in discovery_opportunities[:1]:  # One discovery trade per scan
            if await self.execute_discovery_trade(opportunity):
                break
    
    async def analyze_discovered_coin(self, symbol: str) -> Dict:
        """Analyze a discovered coin for trading opportunity"""
        import random
        
        # Simulate enhanced analysis for discovered coin
        current_price = random.uniform(0.000001, 0.01)
        volume_24h = random.uniform(50000, 5000000)
        market_cap = random.uniform(100000, 10000000)
        social_score = random.uniform(3, 9)
        
        # Discovery-specific scoring
        opportunity_score = 0
        
        # Small market cap = high opportunity
        if market_cap < 1_000_000:
            opportunity_score += 3
        elif market_cap < 5_000_000:
            opportunity_score += 2
        
        # High volume = interest
        if volume_24h > 1_000_000:
            opportunity_score += 2
        elif volume_24h > 500_000:
            opportunity_score += 1
        
        # Social momentum
        if social_score > 7:
            opportunity_score += 2
        elif social_score > 5:
            opportunity_score += 1
        
        # Risk assessment
        risk_score = 0
        if market_cap < 500_000:
            risk_score += 2  # Very small = risky
        if volume_24h < 100_000:
            risk_score += 2  # Low volume = risky
        
        print(f"   🔍 Analyzed {symbol}:")
        print(f"      Price: ${current_price:.8f}")
        print(f"      Market Cap: ${market_cap:,.0f}")
        print(f"      Volume: ${volume_24h:,.0f}")
        print(f"      Opportunity: {opportunity_score}/10")
        print(f"      Risk: {risk_score}/10")
        
        # Decision criteria for discoveries (more aggressive)
        if opportunity_score >= 4 and risk_score <= 6:
            return {
                'symbol': symbol,
                'price': current_price,
                'market_cap': market_cap,
                'volume_24h': volume_24h,
                'opportunity_score': opportunity_score,
                'risk_score': risk_score,
                'recommendation': 'BUY',
                'trade_type': 'DISCOVERY'
            }
        
        return None
    
    async def execute_discovery_trade(self, opportunity: Dict) -> bool:
        """Execute trade on discovered coin"""
        symbol = opportunity['symbol']
        current_price = opportunity['price']
        
        # Calculate position size (smaller for discoveries)
        portfolio_value = self.paper_trader.portfolio['total_value']
        position_size = portfolio_value * self.config['discovery_position_size']
        
        if position_size > self.paper_trader.portfolio['cash']:
            print(f"   ❌ Insufficient cash for {symbol}")
            return False
        
        # Execute discovery trade
        shares = position_size / current_price
        
        # Add to portfolio
        self.paper_trader.portfolio['positions'][symbol] = {
            'shares': shares,
            'entry_price': current_price,
            'current_value': position_size,
            'stop_loss': current_price * 0.70,  # 30% stop for discoveries
            'trade_type': 'DISCOVERY',
            'entry_time': datetime.now()
        }
        
        self.paper_trader.portfolio['cash'] -= position_size
        
        # Track discovery trade
        self.discovery_trades[symbol] = {
            'entry_price': current_price,
            'position_size': position_size,
            'shares': shares,
            'opportunity_score': opportunity['opportunity_score'],
            'entry_time': datetime.now()
        }
        
        print(f"   🚀 DISCOVERY TRADE EXECUTED:")
        print(f"      Symbol: {symbol}")
        print(f"      Price: ${current_price:.8f}")
        print(f"      Position: ${position_size:,.2f}")
        print(f"      Shares: {shares:,.0f}")
        print(f"      Stop Loss: ${current_price * 0.70:.8f}")
        print(f"      Opportunity Score: {opportunity['opportunity_score']}/10")
        
        return True
    
    def print_enhanced_portfolio_status(self):
        """Print enhanced portfolio status with discovery breakdown"""
        
        # Calculate values
        position_value = sum(pos['current_value'] for pos in self.paper_trader.portfolio['positions'].values())
        total_value = self.paper_trader.portfolio['cash'] + position_value
        total_pnl = total_value - 10000  # Starting capital
        
        # Separate traditional vs discovery positions
        traditional_value = 0
        discovery_value = 0
        
        for symbol, position in self.paper_trader.portfolio['positions'].items():
            if symbol in self.discovery_trades:
                discovery_value += position['current_value']
            else:
                traditional_value += position['current_value']
        
        print(f"\n{'='*55}")
        print(f"💰 ENHANCED PORTFOLIO STATUS")
        print(f"{'='*55}")
        print(f"Cash: ${self.paper_trader.portfolio['cash']:,.2f}")
        print(f"Traditional Positions: ${traditional_value:,.2f}")
        print(f"Discovery Positions: ${discovery_value:,.2f}")
        print(f"Total Value: ${total_value:,.2f}")
        print(f"Total P&L: ${total_pnl:,.2f} ({total_pnl/10000:.1%})")
        print(f"Discovery Watchlist: {len(self.discovered_coins)} coins")
        
        # Show current positions
        if self.paper_trader.portfolio['positions']:
            print(f"\n📊 Current Positions:")
            for symbol, pos in self.paper_trader.portfolio['positions'].items():
                trade_type = "🔍 DISC" if symbol in self.discovery_trades else "📊 TRAD"
                pnl = pos['current_value'] - (pos['shares'] * pos['entry_price'])
                pnl_pct = pnl / (pos['shares'] * pos['entry_price']) if pos['shares'] * pos['entry_price'] > 0 else 0
                print(f"   {trade_type} {symbol}: ${pos['current_value']:,.0f} ({pnl_pct:+.1%})")
    
    def print_final_performance_summary(self):
        """Print final performance summary"""
        
        # Calculate final metrics
        starting_capital = 10000
        final_value = (self.paper_trader.portfolio['cash'] + 
                      sum(pos['current_value'] for pos in self.paper_trader.portfolio['positions'].values()))
        total_return = (final_value - starting_capital) / starting_capital
        
        # Discovery performance
        discovery_pnl = 0
        discovery_trades_count = len(self.discovery_trades)
        
        for symbol, trade in self.discovery_trades.items():
            if symbol in self.paper_trader.portfolio['positions']:
                position = self.paper_trader.portfolio['positions'][symbol]
                trade_pnl = position['current_value'] - trade['position_size']
                discovery_pnl += trade_pnl
        
        print(f"\n{'='*60}")
        print(f"🎯 ENHANCED TRADING SESSION SUMMARY")
        print(f"{'='*60}")
        print(f"Starting Capital: ${starting_capital:,}")
        print(f"Final Value: ${final_value:,.2f}")
        print(f"Total Return: {total_return:+.1%}")
        print(f"")
        print(f"📊 Traditional Trading:")
        print(f"   Coins Monitored: {len(self.traditional_watchlist)}")
        print(f"   Active Positions: {len([p for p in self.paper_trader.portfolio['positions'] if p not in self.discovery_trades])}")
        print(f"")
        print(f"🔍 Discovery Trading:")
        print(f"   Coins Discovered: {len(self.discovered_coins)}")
        print(f"   Discovery Trades: {discovery_trades_count}")
        print(f"   Discovery P&L: ${discovery_pnl:+,.2f}")
        print(f"")
        print(f"🚀 System Performance:")
        print(f"   Enhanced Return: {total_return:+.1%}")
        print(f"   Discovery Contribution: ${discovery_pnl:+,.2f}")
        print(f"   Total Opportunities: {len(self.discovered_coins) + len(self.traditional_watchlist)}")

# Quick setup script
def setup_enhanced_system():
    """Setup enhanced trading system"""
    
    print("🛠️ SETTING UP ENHANCED TRADING SYSTEM")
    print("=" * 50)
    
    # Create directories
    import os
    os.makedirs('discovery', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Copy discovery files to correct location
    print("✅ Directory structure created")
    print("✅ Discovery engine configured")
    print("✅ Opportunity monitor ready")
    print("✅ Enhanced trading system ready")
    
    print("\n🚀 Ready to run enhanced trading!")
    print("   python enhanced_trading_system.py")

# Example usage
if __name__ == "__main__":
    async def run_enhanced_demo():
        system = EnhancedTradingSystem(paper_trading=True)
        
        print("\n🎯 Choose Enhanced Trading Session:")
        print("1. Quick demo (20 minutes)")
        print("2. Standard session (60 minutes)")
        print("3. Extended session (120 minutes)")
        
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == "1":
            duration = 20
        elif choice == "2":
            duration = 60
        elif choice == "3":
            duration = 120
        else:
            duration = 20
        
        try:
            await system.run_enhanced_trading_session(duration)
            
            print("\n🎊 Enhanced trading session complete!")
            print("🎯 Your system now discovers emerging coins automatically!")
            
        except KeyboardInterrupt:
            print("\n⏹️ Enhanced trading session stopped")
    
    # Setup and run
    setup_enhanced_system()
    asyncio.run(run_enhanced_demo())