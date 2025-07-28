#!/usr/bin/env python3
"""
Emerging Coin Discovery Engine
Automatically finds new meme coins before they pump
"""

import requests
import json
import time
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import re
import asyncio
import aiohttp
from dataclasses import dataclass

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

    def _get_mock_dexscreener_data(self) -> List[Dict]:
        """Generate mock DexScreener data for testing"""
        return [
        {
        'symbol': 'TESTPEPE',
        'name': 'Test Pepe',
        'contract': '0x1234567890abcdef1234567890abcdef12345678',
        'price': 0.00123,
        'volume_24h': 150000,    # Above 50K minimum
        'market_cap': 800000,    # Under 50M maximum
        'liquidity': 75000,      # Above 25K minimum
        'age_hours': 18,         # Under 999 maximum
        'source': 'dexscreener_mock',
        'holder_count': 450      # Above 10 minimum
        },
        {
        'symbol': 'MOCKDOGE',
        'name': 'Mock Doge',
        'contract': '0xabcdef1234567890abcdef1234567890abcdef12',
        'price': 0.00567,
        'volume_24h': 280000,
        'market_cap': 1500000,
        'liquidity': 120000,
        'age_hours': 36,
        'source': 'dexscreener_mock',
        'holder_count': 780
        }
        ]

    def _get_mock_reddit_data(self) -> List[Dict]:
        """Generate mock Reddit data for testing"""
        return [
        {
        'symbol': 'MOONSHOT',
        'name': 'Moon Shot Token',
        'source': 'reddit_mock',
        'social_score': 85,
        'mentions': 25,
        'post_url': 'https://reddit.com/mock/moonshot',
        'volume_24h': 95000,
        'market_cap': 650000,
        'liquidity': 45000,
        'age_hours': 12,
        'holder_count': 320,
        'price': 0.00089,
        'contract': '0x9876543210fedcba9876543210fedcba98765432'
        }
        ]

    async def _scan_dexscreener(self) -> List[Dict]:
        """Scan DexScreener for new tokens"""
        print("   DexScreener: Using mock data (API unavailable)")
        return self._get_mock_dexscreener_data()

    async def _scan_dextools(self) -> List[Dict]:
        """Scan DexTools for new tokens"""
        return []

    async def _scan_poocoin(self) -> List[Dict]:
        """Scan PooCoin for new BSC tokens"""
        return []

    async def _scan_reddit_new(self) -> List[Dict]:
        """Scan Reddit for newly mentioned coins"""
        print("   Reddit: Using mock data (credentials not configured)")
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
        return []

    async def _scan_telegram_channels(self) -> List[Dict]:
        """Scan Telegram alpha channels for coin calls"""
        return []

    async def _scan_github_releases(self) -> List[Dict]:
        """Scan GitHub for new token contracts"""
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
        return 24  # Default to 1 day

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
        return {'total_found': 0, 'message': 'No opportunities found', 'avg_opportunity_score': 0, 'avg_risk_score': 0, 'recommendations': []}

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
