#!/usr/bin/env python3
"""
Phase 2 Integrated Trader - All advanced data sources combined
"""

import asyncio
import os
from datetime import datetime
from dotenv import load_dotenv
import logging
import pandas as pd

# Coinbase
from coinbase.rest import RESTClient

# Import all our components
from sentiment.multi_source_aggregator import MultiSourceAggregator
from risk_management.simple_risk import SimpleRiskManager

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class Phase2IntegratedTrader:
    def __init__(self, paper_trading=True):
        # Initialize components
        self.aggregator = MultiSourceAggregator()
        self.risk_mgr = SimpleRiskManager()
        
        self.paper_trading = paper_trading
        
        # Coinbase setup
        self.api_key = os.getenv('COINBASE_API_KEY')
        self.api_secret = os.getenv('COINBASE_API_SECRET')
        
        if self.api_key and self.api_secret:
            self.client = RESTClient(api_key=self.api_key, api_secret=self.api_secret)
            logger.info("✅ Connected to Coinbase")
        else:
            self.client = None
            logger.warning("⚠️ No Coinbase - simulation mode")
        
        # Settings
        self.min_score = 65
        self.balance = 10000.0
        
        # Tracking
        self.active_positions = {}
        
        mode = "PAPER" if paper_trading else "LIVE"
        logger.info(f"🚀 Phase 2 Integrated Trader - {mode} MODE")
        logger.info("📊 Data Sources: Reddit + CoinGecko + DEXScreener + Technical Analysis")
    
    async def scan_and_trade(self):
        """Main scanning loop with all data sources"""
        logger.info(f"\n{'='*70}")
        logger.info(f"🔍 Multi-Source Scan at {datetime.now().strftime('%H:%M:%S')}")
        logger.info(f"Balance: ${self.balance:.2f} | Positions: {len(self.active_positions)}")
        logger.info(f"{'='*70}")
        
        # Get opportunities from all sources
        opportunities = await self.aggregator.scan_all_opportunities()
        
        if opportunities:
            logger.info(f"\n🎯 Found {len(opportunities)} opportunities:")
            
            for i, opp in enumerate(opportunities[:10], 1):
                logger.info(
                    f"{i:2d}. {opp['symbol']:6s} | Score: {opp['score']:5.1f} | "
                    f"Action: {opp['action']:12s} | Signals: {opp['signals']} | "
                    f"Sources: {', '.join(opp['sources'])}"
                )
            
            # Execute best trade if score is high enough
            for opp in opportunities:
                if opp['score'] >= self.min_score and opp['action'] in ['BUY', 'STRONG_BUY']:
                    # Check if we can trade
                    can_trade, reason = self.risk_mgr.can_trade(opp['symbol'])
                    
                    if can_trade and opp['symbol'] not in self.active_positions:
                        # Get detailed analysis
                        analysis = await self.aggregator.analyze_coin_comprehensive(opp['symbol'])
                        
                        # Execute trade
                        await self.execute_trade(opp['symbol'], 'BUY', opp['score'], analysis)
                        break  # One trade at a time
        else:
            logger.info("No opportunities found meeting criteria")
        
        # Check exits
        await self.check_exits()
    
    async def execute_trade(self, coin, action, score, analysis):
        """Execute trade with full analysis"""
        logger.info(f"\n💰 EXECUTING TRADE: {action} {coin}")
        logger.info(f"Score: {score:.1f}")
        
        # Show signals
        for signal in analysis['signals']:
            logger.info(f"  📍 {signal['message']}")
        
        # Paper trade for now
        if action == 'BUY':
            position_size = 1000  # $1000 position
            self.active_positions[coin] = {
                'entry_time': datetime.now(),
                'entry_score': score,
                'size': position_size
            }
            self.balance -= position_size
            logger.info(f"✅ Bought {coin} - Position size: ${position_size}")
    
    async def check_exits(self):
        """Check exit conditions for positions"""
        for coin in list(self.active_positions.keys()):
            # Re-analyze
            analysis = await self.aggregator.analyze_coin_comprehensive(coin)
            
            if analysis['action'] in ['SELL', 'AVOID'] or analysis['combined_score'] < 40:
                logger.info(f"\n📉 EXIT SIGNAL for {coin}")
                logger.info(f"  Score dropped to: {analysis['combined_score']:.1f}")
                logger.info(f"  Action: {analysis['action']}")
                
                # Exit position
                self.balance += self.active_positions[coin]['size'] * 1.1  # Assume 10% profit
                del self.active_positions[coin]
                logger.info(f"✅ Sold {coin}")
    
    async def run(self):
        """Main loop"""
        logger.info("\n🚀 Starting Phase 2 Integrated Trader")
        logger.info("Features: Multi-source analysis with technical indicators")
        
        while True:
            try:
                await self.scan_and_trade()
                
                logger.info("\n⏰ Next scan in 2 minutes...")
                await asyncio.sleep(120)
                
            except KeyboardInterrupt:
                logger.info("\n👋 Shutting down...")
                break
            except Exception as e:
                logger.error(f"Error: {e}")
                await asyncio.sleep(120)

async def main():
    trader = Phase2IntegratedTrader(paper_trading=True)
    await trader.run()

if __name__ == "__main__":
    asyncio.run(main())