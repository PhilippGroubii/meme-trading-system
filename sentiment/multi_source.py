# sentiment/multi_source.py
import requests
import re
import time
import os
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import json

@dataclass
class SentimentResult:
    source: str
    symbol: str
    sentiment_score: float  # -1 to 1 scale
    confidence: float      # 0 to 1 scale
    post_count: int
    timestamp: datetime
    raw_data: Dict = None

class MultiSourceSentiment:
    def __init__(self, config: Dict):
        self.config = config
        
        # Reddit API configuration
        self.reddit_client_id = config.get('reddit', {}).get('client_id')
        self.reddit_client_secret = config.get('reddit', {}).get('client_secret')
        self.reddit_user_agent = config.get('reddit', {}).get('user_agent', 'MemeCoinTrader/1.0')
        
        # Twitter API configuration
        self.twitter_bearer_token = config.get('twitter', {}).get('bearer_token')
        
        # Telegram configuration
        self.telegram_bot_token = config.get('telegram', {}).get('bot_token')
        self.telegram_channels = config.get('telegram', {}).get('channels', [])
        
        # Rate limiting
        self.last_request_times = {}
        self.min_request_interval = 2.0
        
        # Sentiment keywords
        self.bullish_keywords = [
            'moon', 'pump', 'bull', 'buy', 'hodl', 'diamond', 'hands', 'rocket',
            'lambo', 'profit', 'gains', 'rally', 'breakout', 'surge', 'explode',
            'skyrocket', 'bullish', 'long', 'accumulate', 'ape', 'yolo', 'gem'
        ]
        
        self.bearish_keywords = [
            'dump', 'bear', 'sell', 'crash', 'drop', 'fall', 'panic', 'fear',
            'liquidation', 'rekt', 'loss', 'decline', 'correction', 'bearish',
            'short', 'exit', 'scam', 'rug', 'dead', 'worthless', 'shit'
        ]
        
        # Mock data for when APIs aren't available
        self.use_mock_data = config.get('use_mock_data', True)
    
    def _rate_limit(self, source: str):
        """Rate limiting per source"""
        current_time = time.time()
        last_time = self.last_request_times.get(source, 0)
        
        if current_time - last_time < self.min_request_interval:
            time.sleep(self.min_request_interval - (current_time - last_time))
        
        self.last_request_times[source] = time.time()
    
    def _analyze_text_sentiment(self, text: str) -> float:
        """Simple keyword-based sentiment analysis"""
        if not text:
            return 0.0
        
        text_lower = text.lower()
        
        # Count bullish and bearish keywords
        bullish_count = sum(1 for word in self.bullish_keywords if word in text_lower)
        bearish_count = sum(1 for word in self.bearish_keywords if word in text_lower)
        
        total_keywords = bullish_count + bearish_count
        if total_keywords == 0:
            return 0.0  # Neutral
        
        # Calculate sentiment score (-1 to 1)
        sentiment = (bullish_count - bearish_count) / total_keywords
        return max(-1.0, min(1.0, sentiment))
    
    def get_reddit_sentiment(self, symbol: str, subreddits: List[str] = None, limit: int = 100) -> SentimentResult:
        """Get sentiment from Reddit"""
        if not subreddits:
            subreddits = ['CryptoCurrency', 'SatoshiStreetBets', 'CryptoMoonShots', 'altcoin']
        
        try:
            self._rate_limit('reddit')
            
            if self.use_mock_data or not (self.reddit_client_id and self.reddit_client_secret):
                # Return mock data
                import random
                mock_sentiment = random.uniform(-0.3, 0.7)  # Slightly bullish bias
                mock_posts = random.randint(10, 100)
                
                return SentimentResult(
                    source='reddit',
                    symbol=symbol,
                    sentiment_score=mock_sentiment,
                    confidence=0.7,
                    post_count=mock_posts,
                    timestamp=datetime.now(),
                    raw_data={'subreddits': subreddits, 'mock': True}
                )
            
            # Real Reddit API implementation would go here
            # For now, using Reddit's JSON API (limited functionality)
            all_posts = []
            sentiment_scores = []
            
            for subreddit in subreddits:
                try:
                    url = f"https://www.reddit.com/r/{subreddit}/search.json"
                    params = {
                        'q': symbol,
                        'sort': 'new',
                        'limit': limit // len(subreddits),
                        't': 'day'
                    }
                    headers = {'User-Agent': self.reddit_user_agent}
                    
                    response = requests.get(url, params=params, headers=headers, timeout=10)
                    if response.status_code == 200:
                        data = response.json()
                        posts = data.get('data', {}).get('children', [])
                        
                        for post in posts:
                            post_data = post.get('data', {})
                            title = post_data.get('title', '')
                            selftext = post_data.get('selftext', '')
                            
                            # Analyze sentiment of title and text
                            combined_text = f"{title} {selftext}"
                            sentiment = self._analyze_text_sentiment(combined_text)
                            sentiment_scores.append(sentiment)
                            all_posts.append(post_data)
                
                except Exception as e:
                    print(f"Error fetching from r/{subreddit}: {e}")
                    continue
            
            if not sentiment_scores:
                return SentimentResult(
                    source='reddit',
                    symbol=symbol,
                    sentiment_score=0.0,
                    confidence=0.0,
                    post_count=0,
                    timestamp=datetime.now()
                )
            
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
            confidence = min(len(sentiment_scores) / 50.0, 1.0)  # More posts = higher confidence
            
            return SentimentResult(
                source='reddit',
                symbol=symbol,
                sentiment_score=avg_sentiment,
                confidence=confidence,
                post_count=len(sentiment_scores),
                timestamp=datetime.now(),
                raw_data={'subreddits': subreddits, 'posts': len(all_posts)}
            )
        
        except Exception as e:
            print(f"Error getting Reddit sentiment: {e}")
            return SentimentResult(
                source='reddit',
                symbol=symbol,
                sentiment_score=0.0,
                confidence=0.0,
                post_count=0,
                timestamp=datetime.now()
            )
    
    def get_twitter_sentiment(self, symbol: str, limit: int = 100) -> SentimentResult:
        """Get sentiment from Twitter/X"""
        try:
            self._rate_limit('twitter')
            
            if self.use_mock_data or not self.twitter_bearer_token:
                # Return mock data
                import random
                mock_sentiment = random.uniform(-0.2, 0.8)  # Slightly bullish bias
                mock_tweets = random.randint(20, 200)
                
                return SentimentResult(
                    source='twitter',
                    symbol=symbol,
                    sentiment_score=mock_sentiment,
                    confidence=0.6,
                    post_count=mock_tweets,
                    timestamp=datetime.now(),
                    raw_data={'mock': True}
                )
            
            # Real Twitter API implementation would go here
            # This requires Twitter API v2 and proper authentication
            headers = {
                'Authorization': f'Bearer {self.twitter_bearer_token}',
                'Content-Type': 'application/json'
            }
            
            # Search for tweets mentioning the symbol
            query = f"{symbol} crypto -is:retweet lang:en"
            url = "https://api.twitter.com/2/tweets/search/recent"
            params = {
                'query': query,
                'max_results': min(limit, 100),
                'tweet.fields': 'created_at,public_metrics,context_annotations'
            }
            
            response = requests.get(url, headers=headers, params=params, timeout=10)
            
            if response.status_code != 200:
                raise Exception(f"Twitter API error: {response.status_code}")
            
            data = response.json()
            tweets = data.get('data', [])
            
            sentiment_scores = []
            for tweet in tweets:
                text = tweet.get('text', '')
                sentiment = self._analyze_text_sentiment(text)
                sentiment_scores.append(sentiment)
            
            if not sentiment_scores:
                return SentimentResult(
                    source='twitter',
                    symbol=symbol,
                    sentiment_score=0.0,
                    confidence=0.0,
                    post_count=0,
                    timestamp=datetime.now()
                )
            
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
            confidence = min(len(sentiment_scores) / 50.0, 1.0)
            
            return SentimentResult(
                source='twitter',
                symbol=symbol,
                sentiment_score=avg_sentiment,
                confidence=confidence,
                post_count=len(sentiment_scores),
                timestamp=datetime.now(),
                raw_data={'tweets_analyzed': len(tweets)}
            )
        
        except Exception as e:
            print(f"Error getting Twitter sentiment: {e}")
            # Return neutral result on error
            return SentimentResult(
                source='twitter',
                symbol=symbol,
                sentiment_score=0.0,
                confidence=0.0,
                post_count=0,
                timestamp=datetime.now()
            )
    
    def get_telegram_sentiment(self, symbol: str, channels: List[str] = None) -> SentimentResult:
        """Get sentiment from Telegram channels"""
        if not channels:
            channels = self.telegram_channels or ['cryptosignals', 'pumpsignals']
        
        try:
            self._rate_limit('telegram')
            
            if self.use_mock_data or not self.telegram_bot_token:
                # Return mock data
                import random
                mock_sentiment = random.uniform(-0.4, 0.6)
                mock_messages = random.randint(5, 50)
                
                return SentimentResult(
                    source='telegram',
                    symbol=symbol,
                    sentiment_score=mock_sentiment,
                    confidence=0.5,
                    post_count=mock_messages,
                    timestamp=datetime.now(),
                    raw_data={'channels': channels, 'mock': True}
                )
            
            # Real Telegram API implementation would go here
            # This requires bot token and channel access
            sentiment_scores = []
            total_messages = 0
            
            for channel in channels:
                try:
                    # Get channel updates (requires bot to be in channel)
                    url = f"https://api.telegram.org/bot{self.telegram_bot_token}/getUpdates"
                    
                    response = requests.get(url, timeout=10)
                    if response.status_code == 200:
                        data = response.json()
                        updates = data.get('result', [])
                        
                        for update in updates:
                            message = update.get('message', {})
                            text = message.get('text', '')
                            
                            if symbol.lower() in text.lower():
                                sentiment = self._analyze_text_sentiment(text)
                                sentiment_scores.append(sentiment)
                                total_messages += 1
                
                except Exception as e:
                    print(f"Error fetching from Telegram channel {channel}: {e}")
                    continue
            
            if not sentiment_scores:
                return SentimentResult(
                    source='telegram',
                    symbol=symbol,
                    sentiment_score=0.0,
                    confidence=0.0,
                    post_count=0,
                    timestamp=datetime.now()
                )
            
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
            confidence = min(len(sentiment_scores) / 20.0, 1.0)  # Telegram has fewer messages
            
            return SentimentResult(
                source='telegram',
                symbol=symbol,
                sentiment_score=avg_sentiment,
                confidence=confidence,
                post_count=total_messages,
                timestamp=datetime.now(),
                raw_data={'channels': channels}
            )
        
        except Exception as e:
            print(f"Error getting Telegram sentiment: {e}")
            return SentimentResult(
                source='telegram',
                symbol=symbol,
                sentiment_score=0.0,
                confidence=0.0,
                post_count=0,
                timestamp=datetime.now()
            )
    
    def get_combined_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Get combined sentiment from all sources"""
        sentiments = []
        
        # Get sentiment from each source
        reddit_sentiment = self.get_reddit_sentiment(symbol)
        if reddit_sentiment.post_count > 0:
            sentiments.append(reddit_sentiment)
        
        twitter_sentiment = self.get_twitter_sentiment(symbol)
        if twitter_sentiment.post_count > 0:
            sentiments.append(twitter_sentiment)
        
        telegram_sentiment = self.get_telegram_sentiment(symbol)
        if telegram_sentiment.post_count > 0:
            sentiments.append(telegram_sentiment)
        
        if not sentiments:
            return {
                'symbol': symbol,
                'combined_sentiment_score': 0.0,
                'confidence': 0.0,
                'individual_sentiments': [],
                'sources_count': 0,
                'total_posts': 0,
                'timestamp': datetime.now().isoformat()
            }
        
        # Calculate weighted average based on confidence and source reliability
        source_weights = {
            'reddit': 0.4,
            'twitter': 0.4,
            'telegram': 0.2
        }
        
        weighted_sum = 0
        total_weight = 0
        total_posts = 0
        
        for sentiment in sentiments:
            source_weight = source_weights.get(sentiment.source, 0.33)
            confidence_weight = sentiment.confidence
            final_weight = source_weight * confidence_weight
            
            weighted_sum += sentiment.sentiment_score * final_weight
            total_weight += final_weight
            total_posts += sentiment.post_count
        
        combined_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        combined_confidence = total_weight / sum(source_weights.values())
        
        # Convert sentiment results to dictionaries for JSON serialization
        individual_sentiments = []
        for sentiment in sentiments:
            individual_sentiments.append({
                'source': sentiment.source,
                'sentiment_score': sentiment.sentiment_score,
                'confidence': sentiment.confidence,
                'post_count': sentiment.post_count,
                'timestamp': sentiment.timestamp.isoformat()
            })
        
        return {
            'symbol': symbol,
            'combined_sentiment_score': combined_score,
            'confidence': combined_confidence,
            'individual_sentiments': individual_sentiments,
            'sources_count': len(sentiments),
            'total_posts': total_posts,
            'timestamp': datetime.now().isoformat(),
            'sentiment_label': self._get_sentiment_label(combined_score)
        }
    
    def _get_sentiment_label(self, score: float) -> str:
        """Convert sentiment score to human-readable label"""
        if score >= 0.5:
            return "Very Bullish"
        elif score >= 0.2:
            return "Bullish"
        elif score >= -0.2:
            return "Neutral"
        elif score >= -0.5:
            return "Bearish"
        else:
            return "Very Bearish"
    
    def get_trending_sentiment(self, limit: int = 10) -> List[Dict]:
        """Get sentiment for trending cryptocurrencies"""
        # This would typically get trending coins from an API
        # For now, using a predefined list
        trending_symbols = ['BTC', 'ETH', 'DOGE', 'SHIB', 'PEPE', 'BONK', 'FLOKI']
        
        results = []
        for symbol in trending_symbols[:limit]:
            sentiment = self.get_combined_sentiment(symbol)
            results.append(sentiment)
            time.sleep(1)  # Rate limiting
        
        # Sort by sentiment score (most bullish first)
        results.sort(key=lambda x: x['combined_sentiment_score'], reverse=True)
        return results
    
    def analyze_sentiment_trends(self, symbol: str, days: int = 7) -> Dict:
        """Analyze sentiment trends over time (mock implementation)"""
        # This would require storing historical sentiment data
        # For now, return mock trend data
        import random
        
        trend_data = []
        base_sentiment = random.uniform(-0.5, 0.5)
        
        for i in range(days):
            # Simulate daily sentiment with some volatility
            daily_sentiment = base_sentiment + random.uniform(-0.3, 0.3)
            daily_sentiment = max(-1.0, min(1.0, daily_sentiment))
            
            date = (datetime.now() - timedelta(days=days-i-1)).strftime('%Y-%m-%d')
            trend_data.append({
                'date': date,
                'sentiment_score': daily_sentiment,
                'post_count': random.randint(10, 100)
            })
        
        # Calculate trend direction
        recent_avg = sum(d['sentiment_score'] for d in trend_data[-3:]) / 3
        older_avg = sum(d['sentiment_score'] for d in trend_data[:3]) / 3
        trend_direction = 'improving' if recent_avg > older_avg else 'declining'
        
        return {
            'symbol': symbol,
            'trend_data': trend_data,
            'trend_direction': trend_direction,
            'avg_sentiment': sum(d['sentiment_score'] for d in trend_data) / len(trend_data),
            'volatility': max(d['sentiment_score'] for d in trend_data) - min(d['sentiment_score'] for d in trend_data)
        }