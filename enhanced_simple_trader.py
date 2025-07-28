#!/usr/bin/env python3
"""
Enhanced simple trader with real Coinbase integration
This version can actually execute trades
"""

import os
os.environ['PRAW_ASYNC_WARNING'] = 'False'
import asyncio
import os
import json
from datetime import datetime
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template
import ccxt
from pycoingecko import CoinGeckoAPI
from pytrends.request import TrendReq

app = Flask(__name__)

@app.route('/health')
def health_check():
    """Health check endpoint for load balancer"""
    try:
        # Check database connection
        conn = get_db_connection()
        if conn:
            conn.close()
            db_status = "healthy"
        else:
            db_status = "unhealthy"
            
        # Check Redis connection (if using)
        redis_status = "healthy"  # Add actual Redis check
        
        status = {
            "status": "healthy" if db_status == "healthy" else "unhealthy",
            "database": db_status,
            "redis": redis_status,
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(status), 200 if status["status"] == "healthy" else 503
        
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 503

# ADD environment configuration:
import os

# Database configuration from environment
DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///trading.db')
ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')

# Coinbase Advanced Trade
from coinbase.rest import RESTClient

# Import our components
from sentiment.reddit_scanner import RedditScanner
from risk_management.simple_risk import SimpleRiskManager

load_dotenv()

class EnhancedSimpleTrader:
    """Simple meme trader that can execute real trades"""
    
    def __init__(self, paper_trading=True):
        self.reddit = RedditScanner()
        self.risk_mgr = SimpleRiskManager()
        self.paper_trading = paper_trading
        
        # Initialize Coinbase client
        self.api_key = os.getenv('COINBASE_API_KEY')
        self.api_secret = os.getenv('COINBASE_API_SECRET')
        
        if self.api_key and self.api_secret:
            self.client = RESTClient(api_key=self.api_key, api_secret=self.api_secret)
            print("✅ Connected to Coinbase")
        else:
            print("⚠️ No Coinbase credentials - running in simulation mode")
            self.client = None
        
        # Trading settings
        self.min_score = 0.65  # Minimum score to trade
        self.balance = 10000.0  # Starting balance for paper trading
        
        # Known meme coins on Coinbase
        self.meme_coins = ['DOGE', 'SHIB', 'PEPE', 'FLOKI', 'BONK', 'WIF']
        
        # Track our trades
        self.active_positions = {}
        self.trade_history = []
        
        mode = "PAPER TRADING" if paper_trading else "LIVE TRADING"
        print(f"🚀 Enhanced Simple Trader Ready! Mode: {mode}")
    
    async def get_coinbase_prices(self):
        """Get real prices from Coinbase"""
        prices = {}
        
        if self.client:
            try:
                for coin in self.meme_coins:
                    product_id = f"{coin}-USD"
                    ticker = self.client.get_product(product_id)
                    if ticker and 'price' in ticker:
                        prices[coin] = float(ticker['price'])
                        print(f"  {coin}: ${prices[coin]:.8f}")
            except Exception as e:
                print(f"❌ Error getting prices: {e}")
                # Fallback to dummy prices
                return self.get_dummy_prices()
        else:
            return self.get_dummy_prices()
        
        return prices
    
    def get_dummy_prices(self):
        """Dummy prices for testing"""
        return {
            'DOGE': 0.08234,
            'SHIB': 0.00001234,
            'PEPE': 0.00000123,
            'FLOKI': 0.00003456,
            'BONK': 0.00001567,
            'WIF': 0.00008901
        }
    
    async def execute_trade(self, coin, action, score):
        """Execute a trade (real or paper)"""
        
        # Get current price
        prices = await self.get_coinbase_prices()
        price = prices.get(coin, 0)
        
        if price == 0:
            print(f"❌ No price for {coin}")
            return
        
        # Calculate position size
        position_size, position_value = self.risk_mgr.calculate_position_size(
            self.balance, price, confidence=score
        )
        
        print(f"\n💰 TRADE SIGNAL: {action} {coin}")
        print(f"  Price: ${price:.8f}")
        print(f"  Score: {score:.2f}")
        print(f"  Position: {position_size:.2f} {coin} (${position_value:.2f})")
        
        if self.paper_trading:
            # Paper trade
            if action == 'BUY':
                self.active_positions[coin] = {
                    'size': position_size,
                    'entry_price': price,
                    'entry_time': datetime.now(),
                    'value': position_value
                }
                self.balance -= position_value
                print(f"  ✅ PAPER BUY executed")
            else:  # SELL
                if coin in self.active_positions:
                    position = self.active_positions[coin]
                    exit_value = position['size'] * price
                    pnl = exit_value - position['value']
                    pnl_percent = (pnl / position['value']) * 100
                    
                    self.balance += exit_value
                    del self.active_positions[coin]
                    
                    print(f"  ✅ PAPER SELL executed")
                    print(f"  PnL: ${pnl:.2f} ({pnl_percent:.1f}%)")
        else:
            # Real trade (be careful!)
            if self.client:
                try:
                    if action == 'BUY':
                        order = self.client.create_order(
                            product_id=f"{coin}-USD",
                            side="buy",
                            order_type="market",
                            funds=str(position_value)  # Buy with USD amount
                        )
                        print(f"  ✅ REAL BUY executed: {order}")
                    else:  # SELL
                        if coin in self.active_positions:
                            order = self.client.create_order(
                                product_id=f"{coin}-USD",
                                side="sell",
                                order_type="market",
                                size=str(position_size)  # Sell coin amount
                            )
                            print(f"  ✅ REAL SELL executed: {order}")
                except Exception as e:
                    print(f"  ❌ Trade failed: {e}")
        
        # Record trade
        self.trade_history.append({
            'time': datetime.now(),
            'coin': coin,
            'action': action,
            'price': price,
            'size': position_size,
            'value': position_value,
            'score': score
        })
    
    async def check_exits(self):
        """Check if any positions should be exited"""
        for coin, position in list(self.active_positions.items()):
            # Update position with current price
            current_price = (await self.get_coinbase_prices()).get(coin, position['entry_price'])
            
            # Check exit conditions
            exit_signal = self.risk_mgr.update_position(coin, current_price)
            
            if exit_signal != 'HOLD':
                print(f"\n🔔 EXIT SIGNAL for {coin}: {exit_signal}")
                await self.execute_trade(coin, 'SELL', 0.5)  # Exit with neutral score
    
    async def scan_and_trade(self):
        """Main scanning and trading logic"""
        print(f"\n{'='*60}")
        print(f"🔍 Scanning at {datetime.now().strftime('%H:%M:%S')}")
        print(f"Balance: ${self.balance:.2f} | Positions: {len(self.active_positions)}")
        print(f"{'='*60}")
        
        # Get Reddit hot coins
        reddit_hot = await self.reddit.get_hot_coins(min_mentions=2)
        
        # Get current prices
        print("\n📊 Current Prices:")
        prices = await self.get_coinbase_prices()
        
        # Find opportunities
        opportunities = []
        
        for coin in self.meme_coins:
            # Skip if we already have a position
            if coin in self.active_positions:
                continue
            
            # Check if we can trade
            can_trade, reason = self.risk_mgr.can_trade(coin)
            if not can_trade:
                continue
            
            # Calculate score
            base_score = 0.5
            
            # Add Reddit boost
            if coin in reddit_hot:
                reddit_data = reddit_hot[coin]
                boost = min(0.3, reddit_data['sentiment'] * 0.3)
                score = min(1.0, base_score + boost)
                
                opportunities.append({
                    'coin': coin,
                    'score': score,
                    'reddit_mentions': reddit_data['mentions'],
                    'reddit_sentiment': reddit_data['sentiment']
                })
        
        # Sort by score
        opportunities.sort(key=lambda x: x['score'], reverse=True)
        
        # Display opportunities
        if opportunities:
            print("\n🎯 Opportunities:")
            for opp in opportunities[:5]:
                reddit_info = f"Reddit: {opp['reddit_mentions']} mentions, " \
                             f"sentiment {opp['reddit_sentiment']:.2f}"
                print(f"  {opp['coin']}: Score {opp['score']:.2f} ({reddit_info})")
        
        # Execute best trade if score is high enough
        if opportunities and opportunities[0]['score'] >= self.min_score:
            best = opportunities[0]
            await self.execute_trade(best['coin'], 'BUY', best['score'])
        
        # Check exits for existing positions
        await self.check_exits()
        
        # Show current positions
        if self.active_positions:
            print("\n📈 Active Positions:")
            for coin, pos in self.active_positions.items():
                current_price = prices.get(coin, pos['entry_price'])
                pnl = ((current_price - pos['entry_price']) / pos['entry_price']) * 100
                print(f"  {coin}: {pos['size']:.2f} @ ${pos['entry_price']:.8f} "
                      f"(PnL: {pnl:+.1f}%)")
    
    async def run(self):
        """Main loop"""
        print("\n🎯 Starting Enhanced Simple Trader")
        print(f"Min score to trade: {self.min_score}")
        print(f"Risk per trade: {self.risk_mgr.risk_per_trade*100:.1f}%")
        
        while True:
            try:
                await self.scan_and_trade()
                
                # Show performance
                if self.trade_history:
                    wins = sum(1 for t in self.trade_history if t['action'] == 'SELL')
                    total_trades = len([t for t in self.trade_history if t['action'] == 'BUY'])
                    if total_trades > 0:
                        print(f"\n📊 Performance: {wins}/{total_trades} trades completed")
                
                print("\n⏰ Next scan in 60 seconds...")
                await asyncio.sleep(60)
                
            except KeyboardInterrupt:
                print("\n👋 Shutting down...")
                
                # Close all positions
                if self.active_positions:
                    print("\n📉 Closing all positions...")
                    for coin in list(self.active_positions.keys()):
                        await self.execute_trade(coin, 'SELL', 0.5)
                
                # Show final stats
                print(f"\n📊 Final Balance: ${self.balance:.2f}")
                print(f"Total Trades: {len(self.trade_history)}")
                
                break
                
            except Exception as e:
                print(f"❌ Error: {e}")
                await asyncio.sleep(60)

async def main():
    """Main entry point"""
    import sys
    
    # Check command line arguments
    paper_trading = True
    if len(sys.argv) > 1:
        if sys.argv[1] == 'live':
            response = input("⚠️  LIVE TRADING MODE - Are you sure? (yes/no): ")
            if response.lower() == 'yes':
                paper_trading = False
            else:
                print("Staying in paper trading mode")
    
    trader = EnhancedSimpleTrader(paper_trading=paper_trading)
    await trader.run()

if __name__ == "__main__":
    asyncio.run(main())
