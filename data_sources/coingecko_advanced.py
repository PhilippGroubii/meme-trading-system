"""
Advanced CoinGecko integration for comprehensive market data
"""

from pycoingecko import CoinGeckoAPI
import asyncio
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class CoinGeckoAdvanced:
    def __init__(self):
        self.cg = CoinGeckoAPI()
        self.cache = {}
        self.cache_duration = 300  # 5 minutes
        
    async def get_trending_with_data(self):
        """Get trending coins with full market data"""
        try:
            # Get trending
            trending = self.cg.get_search_trending()
            
            detailed_trending = []
            for coin in trending['coins'][:10]:  # Top 10
                item = coin['item']
                coin_id = item['id']
                
                # Get detailed data
                try:
                    data = self.cg.get_coin_by_id(
                        id=coin_id,
                        localization=False,
                        tickers=False,
                        market_data=True,
                        community_data=True,
                        developer_data=False
                    )
                    
                    market_data = data.get('market_data', {})
                    
                    detailed_trending.append({
                        'symbol': item['symbol'].upper(),
                        'name': item['name'],
                        'id': coin_id,
                        'market_cap_rank': item.get('market_cap_rank', 999),
                        'price': market_data.get('current_price', {}).get('usd', 0),
                        'volume_24h': market_data.get('total_volume', {}).get('usd', 0),
                        'price_change_24h': market_data.get('price_change_percentage_24h', 0),
                        'price_change_7d': market_data.get('price_change_percentage_7d', 0),
                        'market_cap': market_data.get('market_cap', {}).get('usd', 0),
                        'twitter_followers': data.get('community_data', {}).get('twitter_followers', 0),
                        'score': self._calculate_opportunity_score(market_data, data.get('community_data', {}))
                    })
                    
                    await asyncio.sleep(1.5)  # Rate limit
                    
                except Exception as e:
                    logger.error(f"Error getting data for {coin_id}: {e}")
                    continue
            
            # Sort by opportunity score
            detailed_trending.sort(key=lambda x: x['score'], reverse=True)
            
            return detailed_trending
            
        except Exception as e:
            logger.error(f"CoinGecko trending error: {e}")
            return []
    
    async def get_new_listings(self):
        """Get recently listed coins"""
        try:
            # Get all coins
            all_coins = self.cg.get_coins_list()
            
            # This is a simplified version - in production you'd track and compare
            # For now, return coins with low market cap rank as proxy for new
            new_coins = []
            
            # Get coins with market data
            markets = self.cg.get_coins_markets(
                vs_currency='usd',
                order='market_cap_desc',
                per_page=250,
                page=1
            )
            
            for coin in markets:
                # Look for signs of new coins
                if (coin.get('market_cap_rank', 999) > 200 and 
                    coin.get('total_volume', 0) > 100000 and
                    coin.get('price_change_percentage_24h', 0) > 10):
                    
                    new_coins.append({
                        'symbol': coin['symbol'].upper(),
                        'name': coin['name'],
                        'id': coin['id'],
                        'price': coin['current_price'],
                        'volume_24h': coin['total_volume'],
                        'price_change_24h': coin['price_change_percentage_24h'],
                        'market_cap': coin.get('market_cap', 0),
                        'is_new': True
                    })
            
            return new_coins[:10]  # Top 10
            
        except Exception as e:
            logger.error(f"CoinGecko new listings error: {e}")
            return []
    
    async def get_top_gainers(self, time_period='24h'):
        """Get top gaining coins"""
        try:
            # Get top coins by market cap
            markets = self.cg.get_coins_markets(
                vs_currency='usd',
                order='percent_change_24h_desc' if time_period == '24h' else 'percent_change_7d_desc',
                per_page=50,
                page=1
            )
            
            gainers = []
            for coin in markets[:20]:  # Top 20 gainers
                if coin.get('price_change_percentage_24h', 0) > 20:  # At least 20% gain
                    gainers.append({
                        'symbol': coin['symbol'].upper(),
                        'name': coin['name'],
                        'id': coin['id'],
                        'price': coin['current_price'],
                        'gain_24h': coin['price_change_percentage_24h'],
                        'volume_24h': coin['total_volume'],
                        'market_cap_rank': coin.get('market_cap_rank', 999)
                    })
            
            return gainers
            
        except Exception as e:
            logger.error(f"CoinGecko gainers error: {e}")
            return []
    
    def _calculate_opportunity_score(self, market_data, community_data):
        """Calculate opportunity score for a coin"""
        score = 0
        
        # Price momentum (up to 30 points)
        price_change_24h = market_data.get('price_change_percentage_24h', 0)
        if price_change_24h > 50:
            score += 30
        elif price_change_24h > 20:
            score += 20
        elif price_change_24h > 10:
            score += 10
        
        # Volume surge (up to 30 points)
        # Note: Would need historical data for proper comparison
        volume = market_data.get('total_volume', {}).get('usd', 0)
        if volume > 10000000:  # $10M
            score += 30
        elif volume > 1000000:  # $1M
            score += 20
        elif volume > 100000:   # $100k
            score += 10
        
        # Social growth (up to 20 points)
        twitter = community_data.get('twitter_followers', 0)
        if twitter > 100000:
            score += 20
        elif twitter > 10000:
            score += 10
        elif twitter > 1000:
            score += 5
        
        # Market cap potential (up to 20 points)
        mcap = market_data.get('market_cap', {}).get('usd', 0)
        if 0 < mcap < 10000000:  # Under $10M
            score += 20
        elif mcap < 100000000:   # Under $100M
            score += 10
        
        return score
    
    async def analyze_coin(self, coin_id):
        """Deep analysis of a specific coin"""
        try:
            # Get full coin data
            data = self.cg.get_coin_by_id(
                id=coin_id,
                localization=False,
                tickers=True,
                market_data=True,
                community_data=True,
                developer_data=True
            )
            
            analysis = {
                'fundamentals': self._analyze_fundamentals(data),
                'technicals': self._analyze_technicals(data),
                'social': self._analyze_social(data),
                'risk_score': self._calculate_risk_score(data),
                'opportunity_score': self._calculate_opportunity_score(
                    data.get('market_data', {}),
                    data.get('community_data', {})
                )
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Coin analysis error: {e}")
            return None
    
    def _analyze_fundamentals(self, data):
        """Analyze fundamental metrics"""
        market_data = data.get('market_data', {})
        
        return {
            'market_cap': market_data.get('market_cap', {}).get('usd', 0),
            'fully_diluted_valuation': market_data.get('fully_diluted_valuation', {}).get('usd', 0),
            'circulating_supply': market_data.get('circulating_supply', 0),
            'total_supply': market_data.get('total_supply', 0),
            'max_supply': market_data.get('max_supply', 0)
        }
    
    def _analyze_technicals(self, data):
        """Analyze technical indicators"""
        market_data = data.get('market_data', {})
        
        return {
            'ath': market_data.get('ath', {}).get('usd', 0),
            'ath_date': market_data.get('ath_date', {}).get('usd', ''),
            'atl': market_data.get('atl', {}).get('usd', 0),
            'current_price': market_data.get('current_price', {}).get('usd', 0),
            'distance_from_ath': market_data.get('ath_change_percentage', {}).get('usd', 0)
        }
    
    def _analyze_social(self, data):
        """Analyze social metrics"""
        community = data.get('community_data', {})
        
        return {
            'twitter_followers': community.get('twitter_followers', 0),
            'reddit_subscribers': community.get('reddit_subscribers', 0),
            'telegram_users': community.get('telegram_channel_user_count', 0),
            'social_score': sum([
                min(community.get('twitter_followers', 0) / 1000, 50),
                min(community.get('reddit_subscribers', 0) / 1000, 30),
                min(community.get('telegram_channel_user_count', 0) / 1000, 20)
            ])
        }
    
    def _calculate_risk_score(self, data):
        """Calculate risk score (0-100, higher = riskier)"""
        risk = 0
        
        market_data = data.get('market_data', {})
        
        # Low market cap = higher risk
        mcap = market_data.get('market_cap', {}).get('usd', 0)
        if mcap < 1000000:  # Under $1M
            risk += 40
        elif mcap < 10000000:  # Under $10M
            risk += 20
        
        # High volatility
        price_change = abs(market_data.get('price_change_percentage_24h', 0))
        if price_change > 50:
            risk += 30
        elif price_change > 20:
            risk += 15
        
        # Low liquidity
        volume = market_data.get('total_volume', {}).get('usd', 0)
        if volume < 100000:
            risk += 30
        elif volume < 1000000:
            risk += 15
        
        return min(risk, 100)