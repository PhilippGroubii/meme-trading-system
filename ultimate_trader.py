#!/usr/bin/env python3
"""
Ultimate Meme Trader - All profitable features combined
Includes: Reddit, CoinGecko, Google Trends, Smart Profit Taking, Optimal Timing
"""

import asyncio
import os
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv
import logging

# Coinbase
from coinbase.rest import RESTClient

# Data sources
from pycoingecko import CoinGeckoAPI
from pytrends.request import TrendReq
import pandas as pd

# Our components
from sentiment.reddit_scanner import RedditScanner
from risk_management.simple_risk import SimpleRiskManager

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class UltimateMemeTrader:
    """Ultimate meme trader with all profitability features"""
    
    def __init__(self, paper_trading=True):
        # Initialize components
        self.reddit = RedditScanner()
        self.risk_mgr = SimpleRiskManager()
        self.coingecko = CoinGeckoAPI()
        self.pytrends = TrendReq(hl='en-US', tz=360)
        
        self.paper_trading = paper_trading
        
        # Coinbase setup
        self.api_key = os.getenv('COINBASE_API_KEY')
        self.api_secret = os.getenv('COINBASE_API_SECRET')
        
        if self.api_key and self.api_secret:
            self.client = RESTClient(api_key=self.api_key, api_secret=self.api_secret)
            logger.info("✅ Connected to Coinbase")
        else:
            logger.warning("⚠️ No Coinbase credentials - simulation mode")
            self.client = None
        
        # Trading settings
        self.min_score = 0.65
        self.balance = 10000.0
        
        # Profit taking levels
        self.profit_levels = [
            {'threshold': 0.08, 'sell_percent': 0.25},  # Sell 25% at +8%
            {'threshold': 0.15, 'sell_percent': 0.25},  # Sell 25% at +15%
            {'threshold': 0.30, 'sell_percent': 0.25},  # Sell 25% at +30%
            # Keep 25% as moon bag
        ]
        
        # Trading hours (UTC)
        self.trading_hours = {
            'start': 13,  # 9 AM EST
            'end': 3      # 11 PM EST (next day)
        }
        
        # Expanded meme coins list
        self.meme_coins = ['DOGE', 'SHIB', 'PEPE', 'FLOKI', 'BONK', 'WIF', 'TURBO', 'LADYS']
        
        # Tracking
        self.active_positions = {}
        self.profit_taken = {}  # Track which profit levels we've taken
        self.trending_coins = {}
        self.google_trends_cache = {}
        
        mode = "PAPER TRADING" if paper_trading else "LIVE TRADING"
        logger.info(f"🚀 Ultimate Meme Trader Ready! Mode: {mode}")
    
    def is_trading_hours(self):
        """Check if current time is within trading hours"""
        current_hour = datetime.utcnow().hour
        
        if self.trading_hours['end'] < self.trading_hours['start']:
            # Handles case where end time is next day
            return current_hour >= self.trading_hours['start'] or current_hour <= self.trading_hours['end']
        else:
            return self.trading_hours['start'] <= current_hour <= self.trading_hours['end']
    
    async def get_coingecko_trending(self):
        """Get trending coins from CoinGecko"""
        try:
            logger.info("🦎 Checking CoinGecko trending...")
            trending = self.coingecko.get_search_trending()
            
            trending_memes = {}
            for coin in trending['coins']:
                item = coin['item']
                symbol = item['symbol'].upper()
                
                # Check if it's a meme coin
                if self.is_likely_meme(symbol, item['name']):
                    trending_memes[symbol] = {
                        'name': item['name'],
                        'market_cap_rank': item.get('market_cap_rank', 999),
                        'score': item.get('score', 0),
                        'price_btc': item.get('price_btc', 0)
                    }
                    logger.info(f"  🔥 {symbol} trending on CoinGecko!")
            
            self.trending_coins = trending_memes
            return trending_memes
            
        except Exception as e:
            logger.error(f"CoinGecko error: {e}")
            return {}
    
    def is_likely_meme(self, symbol, name):
        """Check if coin is likely a meme"""
        meme_keywords = ['doge', 'shib', 'inu', 'pepe', 'moon', 'safe', 'baby', 'floki', 'elon', 'punk']
        
        symbol_lower = symbol.lower()
        name_lower = name.lower()
        
        return any(keyword in symbol_lower or keyword in name_lower for keyword in meme_keywords)
    
    async def get_google_trends(self, coin_symbol):
        """Check Google Trends for FOMO detection"""
        cache_key = f"{coin_symbol}_{datetime.now().date()}"
        
        # Check cache
        if cache_key in self.google_trends_cache:
            return self.google_trends_cache[cache_key]
        
        try:
            logger.info(f"📈 Checking Google Trends for {coin_symbol}...")
            
            # Build search query
            kw_list = [f"{coin_symbol} crypto", f"buy {coin_symbol}"]
            
            # Get trend data for last 7 days
            self.pytrends.build_payload(kw_list, timeframe='now 7-d')
            interest = self.pytrends.interest_over_time()
            
            if not interest.empty:
                # Calculate trend momentum
                recent_avg = interest.tail(24).mean().mean()  # Last 24 hours
                older_avg = interest.head(-24).mean().mean()  # Before that
                
                if older_avg > 0:
                    trend_momentum = recent_avg / older_avg
                else:
                    trend_momentum = 1.0
                
                result = {
                    'momentum': trend_momentum,
                    'current_interest': recent_avg,
                    'is_spiking': trend_momentum > 1.5
                }
                
                # Cache result
                self.google_trends_cache[cache_key] = result
                
                if result['is_spiking']:
                    logger.info(f"  🚀 {coin_symbol} Google searches spiking! ({trend_momentum:.1f}x)")
                
                return result
            
        except Exception as e:
            logger.error(f"Google Trends error: {e}")
        
        return {'momentum': 1.0, 'current_interest': 0, 'is_spiking': False}
    
    async def get_coinbase_prices(self):
        """Get real prices from Coinbase"""
        prices = {}
        
        if self.client:
            try:
                for coin in self.meme_coins:
                    try:
                        product_id = f"{coin}-USD"
                        ticker = self.client.get_product(product_id)
                        if ticker and 'price' in ticker:
                            prices[coin] = float(ticker['price'])
                    except:
                        continue
            except Exception as e:
                logger.error(f"Coinbase price error: {e}")
        
        # Fallback prices for testing
        if not prices:
            prices = {
                'DOGE': 0.08234,
                'SHIB': 0.00001234,
                'PEPE': 0.00000123,
                'FLOKI': 0.00003456,
                'BONK': 0.00001567,
                'WIF': 0.00008901
            }
        
        return prices
    
    async def calculate_coin_score(self, coin):
        """Calculate comprehensive score for a coin"""
        base_score = 0.4
        
        # Reddit boost (up to +0.3)
        reddit_hot = await self.reddit.get_hot_coins(min_mentions=2)
        if coin in reddit_hot:
            reddit_data = reddit_hot[coin]
            reddit_boost = min(0.3, reddit_data['sentiment'] * 0.3)
            base_score += reddit_boost
            logger.info(f"  📱 Reddit boost for {coin}: +{reddit_boost:.2f}")
        
        # CoinGecko boost (up to +0.2)
        if coin in self.trending_coins:
            coingecko_boost = 0.2
            base_score += coingecko_boost
            logger.info(f"  🦎 CoinGecko trending boost for {coin}: +{coingecko_boost:.2f}")
        
        # Google Trends boost (up to +0.2)
        trends = await self.get_google_trends(coin)
        if trends['is_spiking']:
            google_boost = 0.2
            base_score += google_boost
            logger.info(f"  📈 Google Trends spike boost for {coin}: +{google_boost:.2f}")
        
        return min(1.0, base_score)
    
    async def execute_trade(self, coin, action, score, size_override=None):
        """Execute a trade with smart features"""
        
        # Check trading hours
        if not self.is_trading_hours():
            logger.warning(f"❌ Outside trading hours - skipping {action} {coin}")
            return
        
        prices = await self.get_coinbase_prices()
        price = prices.get(coin, 0)
        
        if price == 0:
            return
        
        # Calculate position size
        if size_override:
            position_size = size_override
            position_value = position_size * price
        else:
            position_size, position_value = self.risk_mgr.calculate_position_size(
                self.balance, price, confidence=score
            )
        
        logger.info(f"\n💰 TRADE SIGNAL: {action} {coin}")
        logger.info(f"  Price: ${price:.8f}")
        logger.info(f"  Score: {score:.2f}")
        logger.info(f"  Size: {position_size:.2f} {coin} (${position_value:.2f})")
        
        if self.paper_trading:
            if action == 'BUY':
                self.active_positions[coin] = {
                    'size': position_size,
                    'entry_price': price,
                    'entry_time': datetime.now(),
                    'value': position_value,
                    'highest_price': price
                }
                self.profit_taken[coin] = []
                self.balance -= position_value
                logger.info(f"  ✅ PAPER BUY executed")
                
            else:  # SELL
                if coin in self.active_positions:
                    position = self.active_positions[coin]
                    exit_value = position_size * price
                    
                    # Calculate PnL
                    if size_override:  # Partial sell
                        pnl = (price - position['entry_price']) / position['entry_price']
                        profit = exit_value - (position_size * position['entry_price'])
                    else:  # Full sell
                        pnl = (exit_value - position['value']) / position['value']
                        profit = exit_value - position['value']
                    
                    self.balance += exit_value
                    
                    # Update or remove position
                    if size_override and size_override < position['size']:
                        position['size'] -= size_override
                        position['value'] = position['size'] * position['entry_price']
                    else:
                        del self.active_positions[coin]
                    
                    logger.info(f"  ✅ PAPER SELL executed")
                    logger.info(f"  PnL: ${profit:.2f} ({pnl*100:.1f}%)")
        
        # Record trade
        self.risk_mgr.add_position(coin, position_size, price)
    
    async def check_profit_taking(self):
        """Check if any positions should take profits"""
        for coin, position in list(self.active_positions.items()):
            prices = await self.get_coinbase_prices()
            current_price = prices.get(coin, position['entry_price'])
            
            # Update highest price
            if current_price > position.get('highest_price', position['entry_price']):
                position['highest_price'] = current_price
            
            # Calculate current profit
            profit_pct = (current_price - position['entry_price']) / position['entry_price']
            
            # Check profit levels
            for level in self.profit_levels:
                level_key = f"level_{level['threshold']}"
                
                if profit_pct >= level['threshold'] and level_key not in self.profit_taken.get(coin, []):
                    # Take partial profit
                    sell_size = position['size'] * level['sell_percent']
                    
                    logger.info(f"\n🎯 PROFIT TAKING for {coin} at +{profit_pct*100:.1f}%")
                    await self.execute_trade(coin, 'SELL', 0.5, size_override=sell_size)
                    
                    # Record profit taken
                    if coin not in self.profit_taken:
                        self.profit_taken[coin] = []
                    self.profit_taken[coin].append(level_key)
            
            # Trailing stop for moon bag
            if profit_pct > 0.50:  # If up more than 50%
                trailing_stop = position['highest_price'] * 0.80  # 20% trailing stop
                if current_price <= trailing_stop:
                    logger.info(f"\n🛑 TRAILING STOP hit for {coin}")
                    await self.execute_trade(coin, 'SELL', 0.5)
    
    async def scan_and_trade(self):
        """Main scanning and trading logic"""
        logger.info(f"\n{'='*60}")
        logger.info(f"🔍 Ultimate Scan at {datetime.now().strftime('%H:%M:%S')}")
        logger.info(f"Balance: ${self.balance:.2f} | Positions: {len(self.active_positions)}")
        
        # Check if trading hours
        if not self.is_trading_hours():
            logger.warning("⏰ Outside trading hours (9 AM - 11 PM EST)")
            return
        else:
            logger.info("✅ Within trading hours")
        
        logger.info(f"{'='*60}")
        
        # Get trending from CoinGecko
        await self.get_coingecko_trending()
        
        # Find opportunities
        opportunities = []
        
        for coin in self.meme_coins:
            # Skip if we have a full position
            if coin in self.active_positions and len(self.profit_taken.get(coin, [])) == 0:
                continue
            
            # Check if we can trade
            can_trade, reason = self.risk_mgr.can_trade(coin)
            if not can_trade:
                continue
            
            # Calculate comprehensive score
            score = await self.calculate_coin_score(coin)
            
            if score > 0.5:  # Only show decent opportunities
                opportunities.append({
                    'coin': coin,
                    'score': score
                })
        
        # Sort by score
        opportunities.sort(key=lambda x: x['score'], reverse=True)
        
        # Display opportunities
        if opportunities:
            logger.info("\n🎯 Opportunities Found:")
            for opp in opportunities[:5]:
                logger.info(f"  {opp['coin']}: Score {opp['score']:.2f}")
        
        # Execute best trade
        if opportunities and opportunities[0]['score'] >= self.min_score:
            best = opportunities[0]
            await self.execute_trade(best['coin'], 'BUY', best['score'])
        
        # Check profit taking
        await self.check_profit_taking()
        
        # Check stop losses
        await self.check_stop_losses()
        
        # Show positions
        if self.active_positions:
            logger.info("\n📈 Active Positions:")
            prices = await self.get_coinbase_prices()
            
            for coin, pos in self.active_positions.items():
                current_price = prices.get(coin, pos['entry_price'])
                pnl = ((current_price - pos['entry_price']) / pos['entry_price']) * 100
                profits_taken = len(self.profit_taken.get(coin, []))
                
                logger.info(f"  {coin}: {pos['size']:.2f} @ ${pos['entry_price']:.8f} "
                           f"(PnL: {pnl:+.1f}%) [Profits taken: {profits_taken}/3]")
    
    async def check_stop_losses(self):
        """Check stop losses"""
        for coin, position in list(self.active_positions.items()):
            exit_signal = self.risk_mgr.update_position(coin, position.get('current_price', position['entry_price']))
            
            if exit_signal != 'HOLD':
                logger.info(f"\n🛑 STOP LOSS: {coin} - {exit_signal}")
                await self.execute_trade(coin, 'SELL', 0.3)
    
    async def run(self):
        """Main loop"""
        logger.info("\n🚀 Starting Ultimate Meme Trader")
        logger.info(f"Features: Reddit + CoinGecko + Google Trends + Smart Profits + Optimal Timing")
        logger.info(f"Trading hours: 9 AM - 11 PM EST")
        logger.info(f"Profit levels: 8%, 15%, 30% + moon bag")
        
        while True:
            try:
                await self.scan_and_trade()
                
                # Performance summary
                if len(self.risk_mgr.daily_trades) > 0:
                    logger.info(f"\n📊 Daily trades: {len(self.risk_mgr.daily_trades)}")
                    logger.info(f"Total PnL: {self.risk_mgr.total_pnl*100:.1f}%")
                
                logger.info("\n⏰ Next scan in 60 seconds...")
                await asyncio.sleep(60)
                
            except KeyboardInterrupt:
                logger.info("\n👋 Shutting down...")
                
                # Close all positions
                for coin in list(self.active_positions.keys()):
                    await self.execute_trade(coin, 'SELL', 0.3)
                
                logger.info(f"\n📊 Final Balance: ${self.balance:.2f}")
                break
                
            except Exception as e:
                logger.error(f"Error: {e}")
                await asyncio.sleep(60)

async def main():
    """Main entry point"""
    import sys
    
    paper_trading = True
    if len(sys.argv) > 1 and sys.argv[1] == 'live':
        response = input("⚠️  LIVE TRADING MODE - Are you sure? (yes/no): ")
        if response.lower() == 'yes':
            paper_trading = False
    
    trader = UltimateMemeTrader(paper_trading=paper_trading)
    await trader.run()

if __name__ == "__main__":
    asyncio.run(main())