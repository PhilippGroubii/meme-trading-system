"""
Reddit Scanner Component
Scans Reddit for meme coin sentiment and trends
"""

import praw
import os
import asyncio
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import re
from dotenv import load_dotenv
import logging

load_dotenv()
logger = logging.getLogger(__name__)

class RedditScanner:
    def __init__(self):
        """Initialize Reddit scanner"""
        self.reddit = praw.Reddit(
            client_id=os.getenv('REDDIT_CLIENT_ID'),
            client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
            user_agent=os.getenv('REDDIT_USER_AGENT', 'MemeBot/1.0')
        )
        
        # Subreddits to monitor
        self.meme_subreddits = [
            'CryptoMoonShots',
            'SatoshiStreetBets', 
            'CryptoCurrency'
        ]
        
        # Cache settings
        self.cache = {}
        self.cache_duration = 300  # 5 minutes
        
        # Tracking
        self.processed_posts = set()
        self.coin_momentum = defaultdict(list)
        
        logger.info("Reddit scanner initialized")
    
    async def get_hot_coins(self, min_mentions=2):
        """Get currently hot coins from Reddit"""
        # Check cache first
        cache_key = 'hot_coins'
        if cache_key in self.cache:
            cached_time, cached_data = self.cache[cache_key]
            if (datetime.now() - cached_time).seconds < self.cache_duration:
                logger.info("Using cached Reddit data")
                return cached_data
        
        logger.info("Scanning Reddit for hot coins...")
        
        coin_data = defaultdict(lambda: {
            'mentions': 0,
            'score': 0,
            'sentiment': 0,
            'posts': [],
            'top_post': None
        })
        
        # Scan each subreddit
        for subreddit_name in self.meme_subreddits:
            try:
                await self._scan_subreddit(subreddit_name, coin_data)
            except Exception as e:
                logger.error(f"Error scanning r/{subreddit_name}: {e}")
        
        # Filter and format results
        results = {}
        for coin, data in coin_data.items():
            if data['mentions'] >= min_mentions:
                # Calculate momentum
                momentum = self._calculate_momentum(coin, data['mentions'])
                
                results[coin] = {
                    'mentions': data['mentions'],
                    'score': data['score'],
                    'sentiment': data['sentiment'] / max(data['mentions'], 1),
                    'momentum': momentum,
                    'trending': momentum > 1.5,
                    'top_post': data['top_post']
                }
        
        # Sort by score
        sorted_results = dict(sorted(
            results.items(),
            key=lambda x: x[1]['score'],
            reverse=True
        ))
        
        # Cache results
        self.cache[cache_key] = (datetime.now(), sorted_results)
        
        return sorted_results
    
    async def _scan_subreddit(self, subreddit_name, coin_data):
        """Scan individual subreddit"""
        subreddit = self.reddit.subreddit(subreddit_name)
        
        # Scan hot posts
        for post in subreddit.hot(limit=25):
            # Skip if already processed
            if post.id in self.processed_posts:
                continue
            
            self.processed_posts.add(post.id)
            
            # Extract text
            text = f"{post.title} {post.selftext}"
            
            # Find coin mentions
            coins = self._extract_coins(text)
            
            for coin in coins:
                # Update mention data
                coin_data[coin]['mentions'] += 1
                coin_data[coin]['score'] += post.score
                coin_data[coin]['sentiment'] += self._analyze_sentiment(text)
                coin_data[coin]['posts'].append(post.id)
                
                # Track top post
                if not coin_data[coin]['top_post'] or post.score > coin_data[coin]['top_post']['score']:
                    coin_data[coin]['top_post'] = {
                        'title': post.title,
                        'score': post.score,
                        'url': f"https://reddit.com{post.permalink}",
                        'subreddit': subreddit_name
                    }
        
        # Also scan new posts for early signals
        for post in subreddit.new(limit=10):
            if post.id not in self.processed_posts:
                self.processed_posts.add(post.id)
                
                text = f"{post.title} {post.selftext}"
                coins = self._extract_coins(text)
                
                for coin in coins:
                    coin_data[coin]['mentions'] += 0.5  # Half weight for new posts
                    coin_data[coin]['score'] += post.score * 0.5
    
    def _extract_coins(self, text):
        """Extract coin mentions from text"""
        coins = set()
        text_upper = text.upper()
        
        # Pattern 1: $SYMBOL
        dollar_pattern = r'\$([A-Z]{2,10})\b'
        dollar_matches = re.findall(dollar_pattern, text_upper)
        coins.update(dollar_matches)
        
        # Pattern 2: Known coins without $
        known_coins = ['DOGE', 'SHIB', 'PEPE', 'FLOKI', 'BONK', 'WIF', 'PENGU']
        for coin in known_coins:
            if coin in text_upper:
                coins.add(coin)
        
        # Pattern 3: "SYMBOL coin" or "SYMBOL token"
        coin_pattern = r'\b([A-Z]{2,10})\s+(?:coin|token|crypto)\b'
        coin_matches = re.findall(coin_pattern, text_upper)
        coins.update(coin_matches)
        
        return list(coins)
    
    def _analyze_sentiment(self, text):
        """Simple sentiment analysis"""
        text_lower = text.lower()
        
        # Positive indicators
        positive_words = [
            'moon', 'rocket', 'bullish', 'gem', 'pump', 
            'buy', 'hodl', 'profit', 'gains', 'winner',
            '🚀', '💎', '🙌', '📈', '💰'
        ]
        
        # Negative indicators
        negative_words = [
            'dump', 'scam', 'rug', 'bearish', 'sell',
            'crash', 'dead', 'avoid', 'warning', 'careful',
            '📉', '⚠️', '🚫'
        ]
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        # Return sentiment score (-1 to 1)
        if positive_count + negative_count == 0:
            return 0
        
        return (positive_count - negative_count) / (positive_count + negative_count)
    
    def _calculate_momentum(self, coin, current_mentions):
        """Calculate mention momentum"""
        # Add current mentions to history
        self.coin_momentum[coin].append({
            'time': datetime.now(),
            'mentions': current_mentions
        })
        
        # Keep only recent history (last 24 hours)
        cutoff = datetime.now() - timedelta(hours=24)
        self.coin_momentum[coin] = [
            m for m in self.coin_momentum[coin] 
            if m['time'] > cutoff
        ]
        
        # Calculate momentum
        history = self.coin_momentum[coin]
        if len(history) < 2:
            return 1.0
        
        # Compare recent vs older mentions
        mid_point = len(history) // 2
        older_avg = sum(m['mentions'] for m in history[:mid_point]) / mid_point
        recent_avg = sum(m['mentions'] for m in history[mid_point:]) / (len(history) - mid_point)
        
        if older_avg == 0:
            return 2.0 if recent_avg > 0 else 1.0
        
        return recent_avg / older_avg
    
    async def get_coin_sentiment(self, coin_symbol):
        """Get detailed sentiment for specific coin"""
        logger.info(f"Getting Reddit sentiment for {coin_symbol}")
        
        sentiment_data = {
            'mentions': 0,
            'positive': 0,
            'negative': 0,
            'neutral': 0,
            'posts': []
        }
        
        # Search for the coin
        try:
            for submission in self.reddit.subreddit('all').search(
                f'${coin_symbol}', 
                time_filter='day', 
                limit=50
            ):
                # Analyze sentiment
                text = f"{submission.title} {submission.selftext}"
                sentiment = self._analyze_sentiment(text)
                
                sentiment_data['mentions'] += 1
                
                if sentiment > 0.2:
                    sentiment_data['positive'] += 1
                elif sentiment < -0.2:
                    sentiment_data['negative'] += 1
                else:
                    sentiment_data['neutral'] += 1
                
                sentiment_data['posts'].append({
                    'title': submission.title,
                    'score': submission.score,
                    'sentiment': sentiment,
                    'url': f"https://reddit.com{submission.permalink}"
                })
        
        except Exception as e:
            logger.error(f"Error searching for {coin_symbol}: {e}")
        
        # Calculate overall sentiment
        if sentiment_data['mentions'] > 0:
            sentiment_data['overall'] = (
                sentiment_data['positive'] - sentiment_data['negative']
            ) / sentiment_data['mentions']
        else:
            sentiment_data['overall'] = 0
        
        return sentiment_data