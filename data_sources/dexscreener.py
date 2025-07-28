# data_sources/dexscreener.py - Fixed with correct API endpoints
import requests
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

@dataclass
class TradingPair:
    address: str
    base_token: Dict
    quote_token: Dict
    price_usd: float
    volume_24h: float
    price_change_24h: float
    liquidity: float
    market_cap: float
    chain_id: str
    dex_id: str
    url: str
    fdv: float = 0
    price_change_1h: float = 0
    price_change_6h: float = 0
    buys_24h: int = 0
    sells_24h: int = 0
    volume_1h: float = 0
    volume_6h: float = 0

class DexScreenerAPI:
    def __init__(self):
        self.base_url = "https://api.dexscreener.com/latest"
        self.last_request_time = 0
        self.min_request_interval = 1.0  # 1 second between requests
    
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
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"DexScreener API error: {e}")
            return None
    
    def get_trending_pairs(self, chain: str = "ethereum") -> Optional[List[TradingPair]]:
        """Get trending trading pairs using search for popular tokens"""
        try:
            # DexScreener doesn't have a direct trending endpoint
            # Instead, we'll search for popular meme tokens to simulate trending
            popular_tokens = ['PEPE', 'SHIB', 'DOGE', 'FLOKI', 'BONK']
            all_pairs = []
            
            for token in popular_tokens:
                pairs = self.search_pairs(token)
                if pairs:
                    # Add top 2 pairs from each token
                    all_pairs.extend(pairs[:2])
                    
                    # Don't overwhelm the API
                    if len(all_pairs) >= 10:
                        break
            
            if all_pairs:
                # Sort by volume to get "trending"
                all_pairs.sort(key=lambda x: x.volume_24h, reverse=True)
                return all_pairs[:10]  # Return top 10
            
            return []
            
        except Exception as e:
            print(f"Error getting trending pairs: {e}")
            return []
    
    def search_pairs(self, query: str) -> Optional[List[TradingPair]]:
        """Search for trading pairs by token name or address"""
        try:
            data = self._make_request(f"dex/search/?q={query}")
            if not data or 'pairs' not in data:
                return []
            
            pairs = []
            for pair_data in data['pairs']:
                if self._is_valid_pair_data(pair_data):
                    pair = self._parse_pair_data(pair_data)
                    if pair:
                        pairs.append(pair)
            
            return pairs
        except Exception as e:
            print(f"Error searching pairs: {e}")
            return []
    
    def get_pair_by_address(self, pair_address: str) -> Optional[TradingPair]:
        """Get specific pair data by address"""
        try:
            data = self._make_request(f"dex/pairs/{pair_address}")
            if not data:
                return None
            
            # Handle both single pair and pairs array response
            pair_data = data.get('pair') or (data.get('pairs', [{}])[0] if data.get('pairs') else {})
            
            if self._is_valid_pair_data(pair_data):
                return self._parse_pair_data(pair_data)
            
            return None
        except Exception as e:
            print(f"Error getting pair by address: {e}")
            return None
    
    def get_token_pairs(self, token_address: str) -> Optional[List[TradingPair]]:
        """Get all pairs for a specific token"""
        try:
            data = self._make_request(f"dex/tokens/{token_address}")
            if not data or 'pairs' not in data:
                return []
            
            pairs = []
            for pair_data in data['pairs']:
                if self._is_valid_pair_data(pair_data):
                    pair = self._parse_pair_data(pair_data)
                    if pair:
                        pairs.append(pair)
            
            return pairs
        except Exception as e:
            print(f"Error getting token pairs: {e}")
            return []
    
    def get_top_pairs_by_chain(self, chain: str, limit: int = 50) -> Optional[List[TradingPair]]:
        """Get top pairs by searching for popular tokens on a specific chain"""
        try:
            # Since DexScreener doesn't have a direct "top pairs by chain" endpoint,
            # we'll search for popular tokens and filter by chain
            popular_searches = ['ETH', 'USDC', 'USDT', 'WETH', 'PEPE', 'SHIB']
            all_pairs = []
            
            for search_term in popular_searches:
                pairs = self.search_pairs(search_term)
                if pairs:
                    # Filter by chain if specified
                    if chain.lower() != 'all':
                        chain_pairs = [p for p in pairs if p.chain_id.lower() == chain.lower()]
                        all_pairs.extend(chain_pairs[:5])  # Top 5 from each search
                    else:
                        all_pairs.extend(pairs[:5])
                
                if len(all_pairs) >= limit:
                    break
            
            # Remove duplicates based on pair address
            seen_addresses = set()
            unique_pairs = []
            for pair in all_pairs:
                if pair.address not in seen_addresses:
                    seen_addresses.add(pair.address)
                    unique_pairs.append(pair)
            
            # Sort by volume
            unique_pairs.sort(key=lambda x: x.volume_24h, reverse=True)
            return unique_pairs[:limit]
            
        except Exception as e:
            print(f"Error getting top pairs for {chain}: {e}")
            return []
    
    def get_popular_tokens(self) -> Optional[List[TradingPair]]:
        """Get popular tokens by searching for well-known symbols"""
        try:
            popular_symbols = [
                'PEPE', 'SHIB', 'DOGE', 'FLOKI', 'BONK', 'WIF', 'POPCAT', 
                'MEW', 'BRETT', 'WOJAK', 'MEMES', 'CHAD', 'TURBO'
            ]
            
            all_pairs = []
            for symbol in popular_symbols:
                pairs = self.search_pairs(symbol)
                if pairs:
                    # Take the highest volume pair for each symbol
                    best_pair = max(pairs, key=lambda x: x.volume_24h)
                    all_pairs.append(best_pair)
            
            # Sort by volume
            all_pairs.sort(key=lambda x: x.volume_24h, reverse=True)
            return all_pairs[:20]  # Return top 20
            
        except Exception as e:
            print(f"Error getting popular tokens: {e}")
            return []
    
    def _is_valid_pair_data(self, pair_data: Dict) -> bool:
        """Check if pair data contains required fields"""
        if not isinstance(pair_data, dict):
            return False
        
        required_fields = ['pairAddress', 'baseToken', 'quoteToken', 'chainId']
        return all(field in pair_data and pair_data[field] is not None for field in required_fields)
    
    def _parse_pair_data(self, pair_data: Dict) -> Optional[TradingPair]:
        """Parse API response into TradingPair object"""
        try:
            # Safely extract numeric values
            def safe_float(value, default=0.0):
                if value is None:
                    return default
                try:
                    return float(value)
                except (ValueError, TypeError):
                    return default
            
            def safe_int(value, default=0):
                if value is None:
                    return default
                try:
                    return int(value)
                except (ValueError, TypeError):
                    return default
            
            # Extract volume data
            volume_data = pair_data.get('volume', {})
            if isinstance(volume_data, (int, float)):
                volume_24h = safe_float(volume_data)
                volume_1h = 0
                volume_6h = 0
            else:
                volume_24h = safe_float(volume_data.get('h24', 0))
                volume_1h = safe_float(volume_data.get('h1', 0))
                volume_6h = safe_float(volume_data.get('h6', 0))
            
            # Extract price change data
            price_change_data = pair_data.get('priceChange', {})
            if isinstance(price_change_data, (int, float)):
                price_change_24h = safe_float(price_change_data)
                price_change_1h = 0
                price_change_6h = 0
            else:
                price_change_24h = safe_float(price_change_data.get('h24', 0))
                price_change_1h = safe_float(price_change_data.get('h1', 0))
                price_change_6h = safe_float(price_change_data.get('h6', 0))
            
            # Extract liquidity data
            liquidity_data = pair_data.get('liquidity', {})
            if isinstance(liquidity_data, (int, float)):
                liquidity = safe_float(liquidity_data)
            else:
                liquidity = safe_float(liquidity_data.get('usd', 0))
            
            # Extract transaction data
            txns_data = pair_data.get('txns', {})
            if isinstance(txns_data, dict):
                h24_data = txns_data.get('h24', {})
                buys_24h = safe_int(h24_data.get('buys', 0))
                sells_24h = safe_int(h24_data.get('sells', 0))
            else:
                buys_24h = 0
                sells_24h = 0
            
            return TradingPair(
                address=pair_data.get('pairAddress', ''),
                base_token=pair_data.get('baseToken', {}),
                quote_token=pair_data.get('quoteToken', {}),
                price_usd=safe_float(pair_data.get('priceUsd')),
                volume_24h=volume_24h,
                price_change_24h=price_change_24h,
                liquidity=liquidity,
                market_cap=safe_float(pair_data.get('marketCap')),
                chain_id=pair_data.get('chainId', ''),
                dex_id=pair_data.get('dexId', ''),
                url=pair_data.get('url', ''),
                fdv=safe_float(pair_data.get('fdv', 0)),
                price_change_1h=price_change_1h,
                price_change_6h=price_change_6h,
                buys_24h=buys_24h,
                sells_24h=sells_24h,
                volume_1h=volume_1h,
                volume_6h=volume_6h
            )
        except Exception as e:
            print(f"Error parsing pair data: {e}")
            return None
    
    def get_meme_coin_metrics(self, pair: TradingPair) -> Dict[str, float]:
        """Calculate meme coin specific metrics from pair data"""
        try:
            metrics = {
                'volume_24h': pair.volume_24h,
                'price_change_24h': pair.price_change_24h,
                'price_change_1h': pair.price_change_1h,
                'price_change_6h': pair.price_change_6h,
                'liquidity_usd': pair.liquidity,
                'market_cap': pair.market_cap,
                'fdv': pair.fdv,
                'volume_to_liquidity_ratio': 0,
                'volatility_score': 0,
                'liquidity_score': 0,
                'activity_score': 0,
                'momentum_score': 0,
                'buys_24h': pair.buys_24h,
                'sells_24h': pair.sells_24h,
                'buy_sell_ratio': 0
            }
            
            # Calculate volume to liquidity ratio
            if pair.liquidity > 0:
                metrics['volume_to_liquidity_ratio'] = pair.volume_24h / pair.liquidity
            
            # Calculate volatility score (higher is more volatile)
            metrics['volatility_score'] = (
                abs(pair.price_change_1h) * 0.5 +
                abs(pair.price_change_6h) * 0.3 +
                abs(pair.price_change_24h) * 0.2
            )
            
            # Calculate liquidity score (higher is better)
            if pair.liquidity >= 1000000:  # $1M+
                metrics['liquidity_score'] = 5
            elif pair.liquidity >= 500000:  # $500k+
                metrics['liquidity_score'] = 4
            elif pair.liquidity >= 100000:  # $100k+
                metrics['liquidity_score'] = 3
            elif pair.liquidity >= 50000:   # $50k+
                metrics['liquidity_score'] = 2
            elif pair.liquidity >= 10000:   # $10k+
                metrics['liquidity_score'] = 1
            else:
                metrics['liquidity_score'] = 0
            
            # Calculate activity score based on transaction volume
            total_txns = pair.buys_24h + pair.sells_24h
            if total_txns >= 1000:
                metrics['activity_score'] = 5
            elif total_txns >= 500:
                metrics['activity_score'] = 4
            elif total_txns >= 100:
                metrics['activity_score'] = 3
            elif total_txns >= 50:
                metrics['activity_score'] = 2
            elif total_txns >= 10:
                metrics['activity_score'] = 1
            else:
                metrics['activity_score'] = 0
            
            # Calculate momentum score
            momentum_weights = [0.5, 0.3, 0.2]  # 1h, 6h, 24h
            price_changes = [pair.price_change_1h, pair.price_change_6h, pair.price_change_24h]
            
            metrics['momentum_score'] = sum(
                change * weight for change, weight in zip(price_changes, momentum_weights)
            )
            
            # Calculate buy/sell ratio
            if pair.sells_24h > 0:
                metrics['buy_sell_ratio'] = pair.buys_24h / pair.sells_24h
            elif pair.buys_24h > 0:
                metrics['buy_sell_ratio'] = 10  # All buys, no sells
            else:
                metrics['buy_sell_ratio'] = 1   # No activity
            
            return metrics
        except Exception as e:
            print(f"Error calculating meme coin metrics: {e}")
            return {}
    
    def filter_meme_coins(self, pairs: List[TradingPair], min_volume: float = 1000, min_liquidity: float = 5000) -> List[TradingPair]:
        """Filter pairs to find potential meme coins"""
        if not pairs:
            return []
        
        meme_indicators = [
            'DOGE', 'SHIB', 'PEPE', 'MEME', 'WOJAK', 'APU', 'BONK',
            'FLOKI', 'BABY', 'MOON', 'SAFE', 'INU', 'ELON', 'DEGEN',
            'SNEK', 'CHAD', 'BASED', 'DIAMOND', 'ROCKET', 'LAMBO',
            'COPE', 'HOPIUM', 'NGMI', 'WAGMI', 'HODL', 'FUD', 'PUMP',
            'WIF', 'POPCAT', 'MEW', 'BRETT', 'TURBO', 'MEMES'
        ]
        
        filtered_pairs = []
        for pair in pairs:
            try:
                base_symbol = pair.base_token.get('symbol', '').upper()
                base_name = pair.base_token.get('name', '').upper()
                
                # Check for meme coin indicators in symbol or name
                is_meme_coin = any(
                    indicator in base_symbol or indicator in base_name
                    for indicator in meme_indicators
                )
                
                # Additional heuristics for meme coins
                has_emoji_or_special = any(char in base_name for char in ['🚀', '💎', '🌙', '🦍', '🔥', '💰'])
                has_meme_pattern = any(pattern in base_name for pattern in ['2.0', 'V2', 'CLASSIC', 'NEW', 'MINI'])
                
                # Filters for viable trading pairs
                has_sufficient_volume = pair.volume_24h >= min_volume
                has_sufficient_liquidity = pair.liquidity >= min_liquidity
                has_price_movement = abs(pair.price_change_24h) > 1  # 1%+ change
                has_recent_activity = pair.buys_24h + pair.sells_24h > 10
                
                # Check if it meets meme coin criteria
                if (is_meme_coin or has_emoji_or_special or has_meme_pattern) and \
                   has_sufficient_volume and has_sufficient_liquidity and \
                   (has_price_movement or has_recent_activity):
                    filtered_pairs.append(pair)
            
            except Exception as e:
                print(f"Error filtering pair {pair.address}: {e}")
                continue
        
        # Sort by volume (descending) and return
        filtered_pairs.sort(key=lambda x: x.volume_24h, reverse=True)
        return filtered_pairs
    
    def get_chain_summary(self, chain: str) -> Dict[str, Any]:
        """Get summary statistics for a specific chain"""
        try:
            pairs = self.get_top_pairs_by_chain(chain, 50)
            if not pairs:
                return {}
            
            total_volume = sum(pair.volume_24h for pair in pairs)
            total_liquidity = sum(pair.liquidity for pair in pairs)
            avg_price_change = sum(pair.price_change_24h for pair in pairs) / len(pairs)
            
            # Count pairs by price change direction
            bullish_pairs = len([p for p in pairs if p.price_change_24h > 0])
            bearish_pairs = len([p for p in pairs if p.price_change_24h < 0])
            
            return {
                'chain': chain,
                'total_pairs': len(pairs),
                'total_volume_24h': total_volume,
                'total_liquidity': total_liquidity,
                'avg_price_change_24h': avg_price_change,
                'bullish_pairs': bullish_pairs,
                'bearish_pairs': bearish_pairs,
                'top_pair_by_volume': pairs[0] if pairs else None
            }
            
        except Exception as e:
            print(f"Error getting chain summary for {chain}: {e}")
            return {}