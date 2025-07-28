#!/usr/bin/env python3
"""
Enhanced Trading System with Discovery
Simple working version
"""

import sys
import asyncio
import time
from datetime import datetime, timedelta

# Mock trading class
class MockPaperTrading:
    def __init__(self):
        self.portfolio = {
            'cash': 8000,
            'positions': {},
            'total_value': 10000
        }
        self.trades_executed = 0
    
    def analyze_opportunity_v2(self, symbol: str):
        import random
        return {
            'symbol': symbol,
            'recommendation': 'BUY' if random.random() < 0.3 else 'HOLD',
            'confidence': random.uniform(0.6, 0.9),
            'entry_price': random.uniform(0.01, 1.0)
        }
    
    def execute_optimized_trade(self, analysis):
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
    def __init__(self, paper_trading=True):
        self.paper_trader = MockPaperTrading()
        self.config = {
            'paper_trading': paper_trading,
            'discovery_enabled': True,
            'auto_trade_discoveries': True,
            'max_discovery_positions': 3,
            'discovery_position_size': 0.10,
            'discovery_scan_interval': 60,
        }
        self.discovered_coins = set()
        self.discovery_trades = {}
        self.traditional_watchlist = ['DOGE', 'SHIB', 'PEPE', 'BONK']
        
        print("🚀 ENHANCED TRADING SYSTEM READY")
        print(f"   Paper Trading: {paper_trading}")
        print(f"   Discovery: {'ENABLED' if self.config['discovery_enabled'] else 'DISABLED'}")
    
    async def run_enhanced_trading_session(self, duration_minutes=10):
        print(f"\n🎯 ENHANCED TRADING SESSION START")
        print(f"Duration: {duration_minutes} minutes")
        
        # Import discovery engine
        try:
            sys.path.append('discovery')
            from coin_discovery_engine import CoinDiscoveryEngine
            discovery_engine = CoinDiscoveryEngine()
            print("✅ Discovery engine loaded")
        except Exception as e:
            print(f"⚠️ Discovery engine failed: {e}")
            discovery_engine = None
        
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=duration_minutes)
        scan_count = 0
        
        while datetime.now() < end_time:
            scan_count += 1
            print(f"\n--- SCAN #{scan_count} ---")
            
            # Discovery scan
            if discovery_engine:
                try:
                    opportunities = await discovery_engine.discover_emerging_coins()
                    if opportunities:
                        print(f"🔍 Found {len(opportunities)} opportunities")
                        for coin in opportunities[:2]:
                            print(f"   📊 {coin.symbol} - Score: {coin.opportunity_score:.1f}/10")
                            self.discovered_coins.add(coin.symbol)
                    else:
                        print("🔍 No new opportunities found")
                except Exception as e:
                    print(f"🔍 Discovery scan failed: {e}")
            
            # Traditional trading
            for symbol in self.traditional_watchlist[:1]:
                analysis = self.paper_trader.analyze_opportunity_v2(symbol)
                if analysis['recommendation'] == 'BUY':
                    success = self.paper_trader.execute_optimized_trade(analysis)
                    if success:
                        print(f"✅ Trade executed: {symbol}")
            
            # Portfolio status
            total_value = (self.paper_trader.portfolio['cash'] + 
                          sum(pos['current_value'] for pos in self.paper_trader.portfolio['positions'].values()))
            pnl = total_value - 10000
            print(f"💰 Portfolio: ${total_value:,.0f} (P&L: ${pnl:+,.0f})")
            print(f"📊 Positions: {len(self.paper_trader.portfolio['positions'])}")
            print(f"🔍 Discovered: {len(self.discovered_coins)} coins")
            
            # Wait for next scan
            await asyncio.sleep(30)
        
        print(f"\n🏁 SESSION COMPLETE")
        final_value = (self.paper_trader.portfolio['cash'] + 
                      sum(pos['current_value'] for pos in self.paper_trader.portfolio['positions'].values()))
        final_pnl = final_value - 10000
        print(f"💰 Final Portfolio: ${final_value:,.0f}")
        print(f"📈 Total P&L: ${final_pnl:+,.0f} ({final_pnl/10000:+.1%})")
        print(f"🔍 Total Discovered: {len(self.discovered_coins)} coins")

if __name__ == "__main__":
    async def main():
        system = EnhancedTradingSystem()
        
        print("\n🎯 Choose session duration:")
        print("1. Quick demo (2 minutes)")
        print("2. Standard session (10 minutes)") 
        print("3. Extended session (30 minutes)")
        
        choice = input("\nChoice (1-3): ").strip()
        
        duration = {'1': 2, '2': 10, '3': 30}.get(choice, 2)
        
        await system.run_enhanced_trading_session(duration)
        
        print("\n🎉 Enhanced trading complete!")
    
    asyncio.run(main())
