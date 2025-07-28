"""
Fast Reddit scanner with timeout for meme coin sentiment
"""

import praw
import re
from collections import defaultdict
from datetime import datetime
import logging
import asyncio
import concurrent.futures
from functools import partial

logger = logging.getLogger(__name__)

class RedditScannerFast:
    def __init__(self):
        # Initialize Reddit
        self.reddit = praw.Reddit(
            client_id="5tjt7lWW_v6V-fKpHyHqjA",
            client_secret="CRvfPwliTJPRfHGtxe88n7MrYb0JcQ",
            user_agent="MemeCoinScanner/1.0"
        )
        
        # Subreddits focused on meme coins
        self.subreddits = [
            'CryptoMoonShots',
            'SatoshiStreetBets', 
            'MemeCoins',
            'CryptoMarkets',
            'CryptoCurrency'
        ]
        
        # Common meme coin terms
        self.meme_terms = {
            'moon', 'rocket', 'gem', 'pump', 'lambo', 'hodl', 
            'diamond hands', 'ape', 'degen', 'chad', 'based',
            'wagmi', 'gm', 'ser', 'fren', 'ngmi', 'wen moon'
        }
        
        # Negative indicators
        self.negative_terms = {
            'scam', 'rug', 'dump', 'dead', 'avoid', 'warning',
            'rugpull', 'honeypot', 'fake', 'sketchy', 'stay away'
        }
        
        logger.info("Fast Reddit scanner initialized")
        
        # Cache for performance
        self.cache = {}
        self.cache_time = None
        self.cache_duration = 300  # 5 minutes
        
        # Thread pool for sync Reddit calls
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)
    
    def _scan_subreddit_sync(self, subreddit_name, limit=25):
        """Synchronous Reddit scanning for use in thread pool"""
        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            mentions = defaultdict(lambda: {'count': 0, 'sentiment': 0, 'posts': []})
            
            # Get hot posts (faster than new)
            for post in subreddit.hot(limit=limit):
                # Skip pinned posts
                if post.stickied:
                    continue
                
                # Process title and selftext
                text = f"{post.title} {post.selftext}".upper()
                
                # Find coin mentions (3-6 letter uppercase)
                potential_coins = re.findall(r'\b[A-Z]{3,6}\b', text)
                
                for coin in potential_coins:
                    # Skip common words
                    if coin in ['THE', 'AND', 'FOR', 'NOT', 'BUY', 'SELL', 'USD', 'USDT']:
                        continue
                    
                    mentions[coin]['count'] += 1
                    
                    # Calculate sentiment
                    sentiment = 0
                    text_lower = text.lower()
                    
                    # Positive signals
                    for term in self.meme_terms:
                        if term in text_lower:
                            sentiment += 1
                    
                    # Negative signals
                    for term in self.negative_terms:
                        if term in text_lower:
                            sentiment -= 2
                    
                    mentions[coin]['sentiment'] += sentiment
                    mentions[coin]['posts'].append({
                        'title': post.title[:100],
                        'score': post.score,
                        'url': f"https://reddit.com{post.permalink}"
                    })
            
            return dict(mentions)
            
        except Exception as e:
            logger.error(f"Error scanning {subreddit_name}: {e}")
            return {}
    
    async def get_hot_coins(self, timeout=30):
        """Get hot coins with timeout"""
        # Check cache first
        if self.cache_time and (datetime.now() - self.cache_time).seconds < self.cache_duration:
            logger.info("Using cached Reddit data")
            return self.cache.get('hot_coins', {})
        
        logger.info("Scanning Reddit for hot coins...")
        
        try:
            # Run Reddit scans in thread pool with timeout
            loop = asyncio.get_event_loop()
            tasks = []
            
            for subreddit in self.subreddits[:3]:  # Only scan top 3 for speed
                task = loop.run_in_executor(
                    self.executor,
                    partial(self._scan_subreddit_sync, subreddit, 15)  # Reduced limit
                )
                tasks.append(task)
            
            # Wait with timeout
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=timeout
            )
            
            # Aggregate results
            all_mentions = defaultdict(lambda: {'count': 0, 'sentiment': 0, 'posts': []})
            
            for result in results:
                if isinstance(result, dict):
                    for coin, data in result.items():
                        all_mentions[coin]['count'] += data['count']
                        all_mentions[coin]['sentiment'] += data['sentiment']
                        all_mentions[coin]['posts'].extend(data['posts'][:2])  # Limit posts
            
            # Filter and score
            hot_coins = {}
            for coin, data in all_mentions.items():
                if data['count'] >= 2:  # At least 2 mentions
                    avg_sentiment = data['sentiment'] / data['count']
                    
                    # Calculate score
                    score = data['count'] * 10 + avg_sentiment * 5
                    
                    hot_coins[coin] = {
                        'mentions': data['count'],
                        'sentiment': avg_sentiment,
                        'score': score,
                        'sample_posts': data['posts'][:3]
                    }
            
            # Sort by score
            hot_coins = dict(sorted(hot_coins.items(), key=lambda x: x[1]['score'], reverse=True)[:20])
            
            # Cache results
            self.cache['hot_coins'] = hot_coins
            self.cache_time = datetime.now()
            
            logger.info(f"Found {len(hot_coins)} hot coins on Reddit")
            return hot_coins
            
        except asyncio.TimeoutError:
            logger.warning(f"Reddit scan timed out after {timeout}s, using partial results")
            # Return whatever we have in cache
            return self.cache.get('hot_coins', {})
        except Exception as e:
            logger.error(f"Reddit scan error: {e}")
            return self.cache.get('hot_coins', {})
    
    async def get_coin_sentiment(self, coin_symbol, timeout=10):
        """Get sentiment for specific coin with timeout"""
        logger.info(f"Getting Reddit sentiment for {coin_symbol}")
        
        try:
            # Quick search in top subreddits
            loop = asyncio.get_event_loop()
            task = loop.run_in_executor(
                self.executor,
                partial(self._search_coin_sync, coin_symbol, limit=10)
            )
            
            result = await asyncio.wait_for(task, timeout=timeout)
            return result
            
        except asyncio.TimeoutError:
            logger.warning(f"Coin sentiment timed out for {coin_symbol}")
            return {'mentions': 0, 'overall': 0, 'positive': 0, 'negative': 0}
        except Exception as e:
            logger.error(f"Error getting sentiment for {coin_symbol}: {e}")
            return {'mentions': 0, 'overall': 0, 'positive': 0, 'negative': 0}
    
    def _search_coin_sync(self, coin_symbol, limit=10):
        """Synchronous coin search"""
        mentions = 0
        positive = 0
        negative = 0
        
        try:
            # Search in top subreddits
            for subreddit_name in self.subreddits[:2]:  # Only top 2 for speed
                subreddit = self.reddit.subreddit(subreddit_name)
                
                # Search for the coin
                for post in subreddit.search(coin_symbol, limit=limit, time_filter='week'):
                    text = f"{post.title} {post.selftext}".lower()
                    
                    if coin_symbol.lower() in text:
                        mentions += 1
                        
                        # Sentiment
                        for term in self.meme_terms:
                            if term in text:
                                positive += 1
                        
                        for term in self.negative_terms:
                            if term in text:
                                negative += 1
            
            # Calculate overall sentiment
            if mentions > 0:
                overall = (positive - negative) / mentions
            else:
                overall = 0
            
            return {
                'mentions': mentions,
                'overall': overall,
                'positive': positive,
                'negative': negative
            }
            
        except Exception as e:
            logger.error(f"Search error for {coin_symbol}: {e}")
            return {'mentions': 0, 'overall': 0, 'positive': 0, 'negative': 0}
    
    def __del__(self):
        """Cleanup thread pool"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)

# For backwards compatibility
RedditScanner = RedditScannerFast