#!/usr/bin/env python3
"""
Emerging Coin Discovery Engine
Automatically finds new meme coins before they pump
"""

import requests
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import re
import asyncio
import aiohttp
from dataclasses import dataclass
import os

@dataclass
class EmergingCoin:
    symbol: str
    name: str
    contract_address: str
    market_cap: float
    volume_24h: float
    price: float
    age_hours: float
    liquidity: float
    holder_count: int
    social_score: float
    risk_score: float
    opportunity_score: float

class CoinDiscoveryEngine:
    def __init__(self):
        self.discovered_coins = {}
        self.tracking_coins = {}
        self.blacklist = set()  # Scam/rug pull coins
        
        # Discovery sources
        self.sources = {
            'dexscreener': self._scan_dexscreener,
            'dextools': self._scan_dextools,
            'poocoin': self._scan_poocoin,
            'reddit_new': self._scan_reddit_new,
            'twitter_mentions': self._scan_twitter_trending,
            'telegram_alpha': self._scan_telegram_channels,
            'github_new': self._scan_github_releases
        }
        
        # Opportunity criteria
        self.criteria = {
            'max_market_cap': 5_000_000,      # Under $5M mcap
            'min_volume_24h': 50_000,         # At least $50K volume
            'max_age_hours': 168,             # Less than 7 days old
            'min_liquidity': 25_000,          # At least $25K liquidity
            'min_holders': 100,               # At least 100 holders
            'max_risk_score': 7.0,            # Risk score out of 10
            'min_social_score': 3.0           # Social interest score
        }
    
    async def discover_emerging_coins(self) -> List[EmergingCoin]:
        """Main discovery function - scans all sources"""
        print("🔍 SCANNING FOR EMERGING COINS...")
        
        all_coins = []
        
        # Scan all sources simultaneously
        tasks = []
        for source_name, source_func in self.sources.items():
            print(f"   📡 Scanning {source_name}...")
            tasks.append(self._safe_scan(source_name, source_func))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine results
        for result in results:
            if isinstance(result, list):
                all_coins.extend(result)
            elif isinstance(result, Exception):
                print(f"   ⚠️ Source error: {result}")
        
        # Deduplicate and analyze
        unique_coins = self._deduplicate_coins(all_coins)
        analyzed_coins = await self._analyze_coins(unique_coins)
        filtered_coins = self._filter_opportunities(analyzed_coins)
        
        print(f"✅ Discovery complete: {len(filtered_coins)} opportunities found")
        return filtered_coins
    
    async def _safe_scan(self, source_name: str, source_func) -> List[Dict]:
        """Safely execute source scan with error handling"""
        try:
            return await source_func()
        except Exception as e:
            print(f"   ❌ {source_name} failed: {e}")
            return []
    
    async def _scan_dexscreener(self) -> List[Dict]:
        """Scan DexScreener for new tokens"""
        try:
            url = "https://api.dexscreener.com/latest/dex/tokens/trending"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        coins = []
                        for pair in data.get('pairs', [])[:50]:  # Top 50 trending
                            if self._is_meme_coin(pair):
                                coin_data = {
                                    'symbol': pair.get('baseToken', {}).get('symbol', ''),
                                    'name': pair.get('baseToken', {}).get('name', ''),
                                    'contract': pair.get('baseToken', {}).get('address', ''),
                                    'price': float(pair.get('priceUsd', 0)),
                                    'volume_24h': float(pair.get('volume', {}).get('h24', 0)),
                                    'market_cap': float(pair.get('fdv', 0)),
                                    'liquidity': float(pair.get('liquidity', {}).get('usd', 0)),
                                    'age_hours': self._calculate_token_age(pair),
                                    'source': 'dexscreener'
                                }
                                coins.append(coin_data)
                        
                        return coins
        except Exception as e:
            print(f"   DexScreener API failed: {e}, using mock data")
            return self._get_mock_dexscreener_data()
        
        return self._get_mock_dexscreener_data()
    async def _scan_dextools(self) -> List[Dict]:
        """Scan DexTools for new tokens"""
        # Note: This would require DexTools API key
        url = "https://api.dextools.io/v1/ranking/hotpools"
        
        # Mock implementation - replace with real API
        return []
    
    async def _scan_poocoin(self) -> List[Dict]:
        """Scan PooCoin for new BSC tokens"""
        # Would scrape PooCoin's new token listings
        return []
    
    async def _scan_reddit_new(self) -> List[Dict]:
        """Scan Reddit for newly mentioned coins"""
        try:
            # Check if Reddit credentials are available
            if not all([os.getenv('REDDIT_CLIENT_ID'), os.getenv('REDDIT_CLIENT_SECRET')]):
                print("   ⚠️ Reddit credentials not found, using mock data")
                return self._get_mock_reddit_data()
            
            import praw
            
            reddit = praw.Reddit(
                client_id=os.getenv('REDDIT_CLIENT_ID'),
                client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
                user_agent='CoinDiscovery:v1.0'
            )
            
            coins = []
            subreddits = ['CryptoMoonShots', 'SatoshiStreetBets', 'CryptoCurrency']
            
            for sub_name in subreddits:
                subreddit = reddit.subreddit(sub_name)
                
                # Get new posts from last 24 hours
                for post in subreddit.new(limit=50):
                    if time.time() - post.created_utc < 86400:  # 24 hours
                        coin_mentions = self._extract_coin_mentions(post.title + " " + post.selftext)
                        for mention in coin_mentions:
                            coins.append({
                                'symbol': mention['symbol'],
                                'name': mention.get('name', ''),
                                'source': f'reddit_{sub_name}',
                                'social_score': post.score,
                                'mentions': 1,
                                'post_url': f"https://reddit.com{post.permalink}"
                            })
            
            return coins
            
        except Exception as e:
            print(f"   Reddit scan failed: {e}, using mock data")
            return self._get_mock_reddit_data()
    async def _scan_twitter_trending(self) -> List[Dict]:
        """Scan Twitter for trending coin mentions"""
        # Would use Twitter API to find trending hashtags and coin mentions
        return []
    
    async def _scan_telegram_channels(self) -> List[Dict]:
        """Scan Telegram alpha channels for coin calls"""
        # Would monitor crypto alpha channels for new coin mentions
        return []
    
    async def _scan_github_releases(self) -> List[Dict]:
        """Scan GitHub for new token contracts"""
        url = "https://api.github.com/search/repositories"
        params = {
            'q': 'token contract solidity created:>2024-07-20',
            'sort': 'updated',
            'per_page': 30
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    coins = []
                    for repo in data.get('items', []):
                        if self._is_token_repo(repo):
                            coins.append({
                                'name': repo['name'],
                                'source': 'github',
                                'github_url': repo['html_url'],
                                'created': repo['created_at'],
                                'stars': repo['stargazers_count']
                            })
                    
                    return coins
        return []
    
    def _is_meme_coin(self, pair_data: Dict) -> bool:
        """Determine if token is likely a meme coin"""
        name = pair_data.get('baseToken', {}).get('name', '').lower()
        symbol = pair_data.get('baseToken', {}).get('symbol', '').lower()
        
        meme_indicators = [
            'doge', 'shib', 'pepe', 'moon', 'safe', 'baby', 'mini',
            'inu', 'cat', 'frog', 'rocket', 'ape', 'monkey', 'bear',
            'bull', 'chad', 'wojak', 'elon', 'tesla', 'floki'
        ]
        
        return any(indicator in name or indicator in symbol for indicator in meme_indicators)
    
    def _extract_coin_mentions(self, text: str) -> List[Dict]:
        """Extract coin mentions from text"""
        # Look for patterns like $SYMBOL, SYMBOL/USD, etc.
        patterns = [
            r'\$([A-Z]{3,10})',  # $SYMBOL
            r'\b([A-Z]{3,10})(?:/USD|\s+USD)',  # SYMBOL/USD
            r'\b([A-Z]{3,10})\s+(?:coin|token)',  # SYMBOL coin/token
        ]
        
        mentions = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                mentions.append({'symbol': match.upper()})
        
        return mentions
    
    def _calculate_token_age(self, pair_data: Dict) -> float:
        """Calculate token age in hours"""
        created_at = pair_data.get('pairCreatedAt')
        if created_at:
            created_time = datetime.fromtimestamp(created_at / 1000)
            age = datetime.now() - created_time
            return age.total_seconds() / 3600
        return 0
    
    def _is_token_repo(self, repo: Dict) -> bool:
        """Check if GitHub repo is a token project"""
        name = repo['name'].lower()
        description = repo.get('description', '').lower()
        
        token_indicators = ['token', 'coin', 'erc20', 'bep20', 'contract']
        return any(indicator in name or indicator in description for indicator in token_indicators)
    
    def _deduplicate_coins(self, coins: List[Dict]) -> List[Dict]:
        """Remove duplicate coins from different sources"""
        seen = set()
        unique_coins = []
        
        for coin in coins:
            identifier = coin.get('symbol', '') + coin.get('contract', '')
            if identifier not in seen and identifier:
                seen.add(identifier)
                unique_coins.append(coin)
        
        return unique_coins
    
    async def _analyze_coins(self, coins: List[Dict]) -> List[EmergingCoin]:
        """Analyze coins for opportunity and risk"""
        analyzed_coins = []
        
        for coin_data in coins:
            try:
                # Get additional data if needed
                enhanced_data = await self._enhance_coin_data(coin_data)
                
                # Calculate scores
                opportunity_score = self._calculate_opportunity_score(enhanced_data)
                risk_score = self._calculate_risk_score(enhanced_data)
                social_score = self._calculate_social_score(enhanced_data)
                
                coin = EmergingCoin(
                    symbol=enhanced_data.get('symbol', ''),
                    name=enhanced_data.get('name', ''),
                    contract_address=enhanced_data.get('contract', ''),
                    market_cap=enhanced_data.get('market_cap', 0),
                    volume_24h=enhanced_data.get('volume_24h', 0),
                    price=enhanced_data.get('price', 0),
                    age_hours=enhanced_data.get('age_hours', 0),
                    liquidity=enhanced_data.get('liquidity', 0),
                    holder_count=enhanced_data.get('holder_count', 0),
                    social_score=social_score,
                    risk_score=risk_score,
                    opportunity_score=opportunity_score
                )
                
                analyzed_coins.append(coin)
                
            except Exception as e:
                print(f"   ⚠️ Analysis failed for {coin_data.get('symbol', 'unknown')}: {e}")
        
        return analyzed_coins
    
    async def _enhance_coin_data(self, coin_data: Dict) -> Dict:
        """Get additional data about the coin"""
        # This would fetch additional data like holder count, contract info, etc.
        # For now, return the original data with some defaults
        enhanced = coin_data.copy()
        enhanced.setdefault('holder_count', 500)  # Default estimate
        enhanced.setdefault('age_hours', 24)      # Default 1 day
        return enhanced
    
    def _calculate_opportunity_score(self, coin_data: Dict) -> float:
        """Calculate opportunity score (0-10)"""
        score = 0
        
        # Volume score (higher volume = more opportunity)
        volume = coin_data.get('volume_24h', 0)
        if volume > 1_000_000:
            score += 3
        elif volume > 100_000:
            score += 2
        elif volume > 10_000:
            score += 1
        
        # Market cap score (lower mcap = more upside potential)
        mcap = coin_data.get('market_cap', 0)
        if mcap < 100_000:
            score += 3
        elif mcap < 1_000_000:
            score += 2
        elif mcap < 5_000_000:
            score += 1
        
        # Age score (newer = more potential)
        age = coin_data.get('age_hours', 168)
        if age < 24:
            score += 2
        elif age < 72:
            score += 1
        
        # Liquidity score
        liquidity = coin_data.get('liquidity', 0)
        if liquidity > 100_000:
            score += 1
        elif liquidity > 50_000:
            score += 0.5
        
        return min(score, 10)
    
    def _calculate_risk_score(self, coin_data: Dict) -> float:
        """Calculate risk score (0-10, higher = riskier)"""
        risk = 0
        
        # Low liquidity = high risk
        liquidity = coin_data.get('liquidity', 0)
        if liquidity < 10_000:
            risk += 4
        elif liquidity < 50_000:
            risk += 2
        elif liquidity < 100_000:
            risk += 1
        
        # Very new = high risk
        age = coin_data.get('age_hours', 168)
        if age < 6:
            risk += 3
        elif age < 24:
            risk += 1
        
        # Low holder count = high risk
        holders = coin_data.get('holder_count', 0)
        if holders < 50:
            risk += 3
        elif holders < 200:
            risk += 1
        
        # No contract verification = high risk (would check this)
        # risk += 2 if not verified
        
        return min(risk, 10)
    
    def _calculate_social_score(self, coin_data: Dict) -> float:
        """Calculate social interest score (0-10)"""
        score = 0
        
        # Reddit mentions/score
        if coin_data.get('source', '').startswith('reddit'):
            reddit_score = coin_data.get('social_score', 0)
            if reddit_score > 100:
                score += 3
            elif reddit_score > 50:
                score += 2
            elif reddit_score > 10:
                score += 1
        
        # Twitter mentions (would implement)
        # GitHub activity (would implement)
        # Telegram mentions (would implement)
        
        return min(score, 10)
    
    def _filter_opportunities(self, coins: List[EmergingCoin]) -> List[EmergingCoin]:
        """Filter coins based on opportunity criteria"""
        filtered = []
        
        for coin in coins:
            # Apply filters
            if (coin.market_cap <= self.criteria['max_market_cap'] and
                coin.volume_24h >= self.criteria['min_volume_24h'] and
                coin.age_hours <= self.criteria['max_age_hours'] and
                coin.liquidity >= self.criteria['min_liquidity'] and
                coin.holder_count >= self.criteria['min_holders'] and
                coin.risk_score <= self.criteria['max_risk_score'] and
                coin.social_score >= self.criteria['min_social_score'] and
                coin.symbol not in self.blacklist):
                
                filtered.append(coin)
        
        # Sort by opportunity score
        filtered.sort(key=lambda x: x.opportunity_score, reverse=True)
        return filtered[:20]  # Top 20 opportunities
    
    def get_discovery_summary(self, coins: List[EmergingCoin]) -> Dict:
        """Generate discovery summary"""
        if not coins:
            return {'total_found': 0, 'message': 'No opportunities found'}
        
        top_coin = coins[0]
        avg_opportunity = sum(c.opportunity_score for c in coins) / len(coins)
        avg_risk = sum(c.risk_score for c in coins) / len(coins)
        
        return {
            'total_found': len(coins),
            'top_opportunity': {
                'symbol': top_coin.symbol,
                'opportunity_score': top_coin.opportunity_score,
                'market_cap': top_coin.market_cap,
                'age_hours': top_coin.age_hours
            },
            'avg_opportunity_score': avg_opportunity,
            'avg_risk_score': avg_risk,
            'recommendations': self._generate_recommendations(coins[:5])
        }
    
    def _generate_recommendations(self, top_coins: List[EmergingCoin]) -> List[str]:
        """Generate trading recommendations"""
        recommendations = []
        
        for coin in top_coins:
            if coin.opportunity_score >= 7 and coin.risk_score <= 5:
                recommendations.append(f"🚀 HIGH PRIORITY: {coin.symbol} - Opportunity: {coin.opportunity_score:.1f}/10")
            elif coin.opportunity_score >= 5:
                recommendations.append(f"⚡ MONITOR: {coin.symbol} - Potential early entry")
            else:
                recommendations.append(f"👀 WATCH: {coin.symbol} - Developing opportunity")
        
        return recommendations


    def _get_mock_reddit_data(self) -> List[Dict]:
        """Generate mock Reddit data for testing"""
        import random
        mock_coins = ['MOONSHOT', 'ROCKETCOIN', 'MEMEGOD', 'ALPHAINU', 'GEMFINDER']
        
        return [{
            'symbol': coin,
            'name': f"{coin} Token",
            'source': 'reddit_mock',
            'social_score': random.randint(10, 500),
            'mentions': random.randint(1, 50),
            'post_url': f"https://reddit.com/mock/{coin.lower()}"
        } for coin in random.sample(mock_coins, 2)]

    def _get_mock_dexscreener_data(self) -> List[Dict]:
        """Generate mock DexScreener data for testing"""
        import random
        
        mock_tokens = [
            {'symbol': 'TESTPEPE', 'name': 'Test Pepe'},
            {'symbol': 'MOCKDOGE', 'name': 'Mock Doge'},
            {'symbol': 'DEMODOGE', 'name': 'Demo Doge'},
            {'symbol': 'ALPHATEST', 'name': 'Alpha Test Token'},
            {'symbol': 'BETAMOON', 'name': 'Beta Moon Coin'}
        ]
        
        coins = []
        for token in random.sample(mock_tokens, 3):
            coins.append({
                'symbol': token['symbol'],
                'name': token['name'],
                'contract': f"0x{''.join(random.choices('0123456789abcdef', k=40))}",
                'price': random.uniform(0.000001, 0.01),
                'volume_24h': random.uniform(50000, 2000000),
                'market_cap': random.uniform(100000, 5000000),
                'liquidity': random.uniform(25000, 500000),
                'age_hours': random.uniform(1, 168),
                'source': 'dexscreener_mock'
            })
        
        return coins

# Usage example and testing
if __name__ == "__main__":
    async def test_discovery():
        discovery = CoinDiscoveryEngine()
        
        print("🔍 Starting coin discovery...")
        emerging_coins = await discovery.discover_emerging_coins()
        
        if emerging_coins:
            print(f"\n🎯 TOP EMERGING OPPORTUNITIES:")
            print("-" * 50)
            
            for i, coin in enumerate(emerging_coins[:10], 1):
                print(f"{i}. {coin.symbol} ({coin.name})")
                print(f"   💰 Market Cap: ${coin.market_cap:,.0f}")
                print(f"   📊 Volume 24h: ${coin.volume_24h:,.0f}")
                print(f"   ⏰ Age: {coin.age_hours:.1f} hours")
                print(f"   🎯 Opportunity: {coin.opportunity_score:.1f}/10")
                print(f"   ⚠️ Risk: {coin.risk_score:.1f}/10")
                print()
            
            summary = discovery.get_discovery_summary(emerging_coins)
            print("📈 DISCOVERY SUMMARY:")
            for rec in summary['recommendations']:
                print(f"   {rec}")
        
        else:
            print("No emerging opportunities found this scan.")
    
    # Run the test
    asyncio.run(test_discovery())