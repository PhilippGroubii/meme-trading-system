# data_sources/coingecko.py
import requests
import time
from typing import Dict, List, Optional, Any
import os

class CoinGeckoAPI:
    def __init__(self, api_key: Optional[str] = None):
        self.base_url = "https://api.coingecko.com/api/v3"
        self.api_key = api_key or os.getenv('COINGECKO_API_KEY')
        self.headers = {}
        if self.api_key:
            self.headers['x-cg-demo-api-key'] = self.api_key
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 1.0  # 1 second between requests for free tier
    
    def _rate_limit(self):
        """Ensure we don't exceed rate limits"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        
        self.last_request_time = time.time()
    
    def _make_request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Make API request with error handling and rate limiting"""
        self._rate_limit()
        
        url = f"{self.base_url}/{endpoint}"
        
        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"CoinGecko API error: {e}")
            return None
    
    def get_trending_coins(self) -> Optional[List[Dict]]:
        """Get trending coins from CoinGecko"""
        data = self._make_request("search/trending")
        if data and 'coins' in data:
            return [coin['item'] for coin in data['coins']]
        return None
    
    def get_coin_data(self, coin_id: str) -> Optional[Dict]:
        """Get detailed coin data"""
        return self._make_request(f"coins/{coin_id}")
    
    def get_price_data(self, coin_ids: List[str], vs_currencies: List[str] = ['usd']) -> Optional[Dict]:
        """Get current prices for multiple coins"""
        params = {
            'ids': ','.join(coin_ids),
            'vs_currencies': ','.join(vs_currencies),
            'include_market_cap': 'true',
            'include_24hr_vol': 'true',
            'include_24hr_change': 'true'
        }
        return self._make_request("simple/price", params)
    
    def get_market_data(self, coin_id: str) -> Optional[Dict]:
        """Get market data for a specific coin"""
        data = self.get_coin_data(coin_id)
        if data and 'market_data' in data:
            return data['market_data']
        return None
    
    def get_meme_coin_indicators(self, coin_id: str) -> Dict[str, float]:
        """Calculate meme coin specific indicators"""
        try:
            data = self.get_coin_data(coin_id)
            if not data:
                return {}
            
            market_data = data.get('market_data', {})
            community = data.get('community_data', {})
            
            # Safe division helper function
            def safe_divide(numerator, denominator, default=0):
                if denominator is None or denominator == 0:
                    return default
                if numerator is None:
                    return default
                return numerator / denominator
            
            # Safe value extraction
            def safe_get(data_dict, key, default=0):
                value = data_dict.get(key, default)
                return value if value is not None else default
            
            # Calculate indicators with safe operations
            indicators = {
                'price_change_24h': safe_get(market_data, 'price_change_percentage_24h', 0),
                'volume_24h_usd': safe_get(market_data.get('total_volume', {}), 'usd', 0),
                'market_cap_usd': safe_get(market_data.get('market_cap', {}), 'usd', 0),
                'circulating_supply': safe_get(market_data, 'circulating_supply', 0),
                
                # Social metrics (handle None values)
                'reddit_subscribers': safe_get(community, 'reddit_subscribers', 0),
                'twitter_followers': safe_get(community, 'twitter_followers', 0),
                'telegram_users': safe_get(community, 'telegram_channel_user_count', 0),
                
                # Calculated ratios
                'volume_market_cap_ratio': safe_divide(
                    safe_get(market_data.get('total_volume', {}), 'usd', 0),
                    safe_get(market_data.get('market_cap', {}), 'usd', 0)
                ),
                
                # Social score (normalized)
                'social_score': (
                    safe_divide(safe_get(community, 'reddit_subscribers', 0), 10000) +
                    safe_divide(safe_get(community, 'twitter_followers', 0), 10000) +
                    safe_divide(safe_get(community, 'telegram_channel_user_count', 0), 1000)
                ),
                
                # Volatility indicators
                'volatility_7d': abs(safe_get(market_data, 'price_change_percentage_7d', 0)),
                'volatility_30d': abs(safe_get(market_data, 'price_change_percentage_30d', 0)),
                
                # Additional meme coin metrics
                'ath_change_percentage': safe_get(market_data, 'ath_change_percentage', 0),
                'atl_change_percentage': safe_get(market_data, 'atl_change_percentage', 0),
                'max_supply': safe_get(market_data, 'max_supply', 0),
                'total_supply': safe_get(market_data, 'total_supply', 0),
            }
            
            # Calculate supply ratio if both values exist
            if indicators['max_supply'] > 0:
                indicators['supply_ratio'] = safe_divide(
                    indicators['circulating_supply'], 
                    indicators['max_supply']
                )
            else:
                indicators['supply_ratio'] = 0
            
            return indicators
            
        except Exception as e:
            print(f"Error calculating meme coin indicators: {e}")
            return {}
    
    def get_historical_data(self, coin_id: str, days: int = 7) -> Optional[Dict]:
        """Get historical price data"""
        params = {
            'vs_currency': 'usd',
            'days': str(days),
            'interval': 'hourly' if days <= 7 else 'daily'
        }
        return self._make_request(f"coins/{coin_id}/market_chart", params)
    
    def search_coins(self, query: str) -> Optional[List[Dict]]:
        """Search for coins by name or symbol"""
        params = {'query': query}
        data = self._make_request("search", params)
        if data and 'coins' in data:
            return data['coins']
        return None
    
    def get_global_data(self) -> Optional[Dict]:
        """Get global cryptocurrency market data"""
        return self._make_request("global")
    
    def get_fear_greed_index(self) -> Optional[Dict]:
        """Get fear and greed index"""
        # Note: This is not directly available from CoinGecko
        # You would need to use alternative.me API for this
        try:
            response = requests.get("https://api.alternative.me/fng/", timeout=10)
            return response.json()
        except:
            return None
    
    def get_top_coins(self, limit: int = 100) -> Optional[List[Dict]]:
        """Get top coins by market cap"""
        params = {
            'vs_currency': 'usd',
            'order': 'market_cap_desc',
            'per_page': limit,
            'page': 1,
            'sparkline': False
        }
        return self._make_request("coins/markets", params)
    
    def get_coin_categories(self) -> Optional[List[Dict]]:
        """Get coin categories"""
        return self._make_request("coins/categories/list")
    
    def get_exchanges(self) -> Optional[List[Dict]]:
        """Get list of exchanges"""
        return self._make_request("exchanges")
    
    def is_meme_coin(self, coin_data: Dict) -> bool:
        """Check if a coin is likely a meme coin based on various indicators"""
        try:
            name = coin_data.get('name', '').lower()
            symbol = coin_data.get('symbol', '').lower()
            description = coin_data.get('description', {}).get('en', '').lower()
            categories = coin_data.get('categories', [])
            
            # Meme coin keywords
            meme_keywords = [
                'meme', 'dog', 'cat', 'moon', 'safe', 'baby', 'mini', 'shiba', 'inu',
                'doge', 'pepe', 'wojak', 'chad', 'based', 'rocket', 'diamond', 'hands',
                'ape', 'banana', 'harambe', 'floki', 'bonk', 'cum', 'ass', 'tits'
            ]
            
            # Check categories
            meme_categories = ['Meme Token', 'Dog-themed', 'Cat-themed']
            
            # Check if any meme keywords are in name or symbol
            has_meme_keyword = any(keyword in name or keyword in symbol for keyword in meme_keywords)
            
            # Check if any meme categories
            has_meme_category = any(category in categories for category in meme_categories)
            
            # Check description for meme indicators
            has_meme_description = any(keyword in description for keyword in meme_keywords[:10])  # Top keywords only
            
            return has_meme_keyword or has_meme_category or has_meme_description
            
        except Exception:
            return False