"""
Free Twitter sentiment analysis using web scraping and public APIs
"""
import requests
import time
import re
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import pandas as pd
from textblob import TextBlob
import tweepy


class TwitterFreeSentiment:
    def __init__(self, bearer_token: Optional[str] = None):
        """Initialize Twitter sentiment analyzer"""
        self.bearer_token = bearer_token
        self.rate_limit_delay = 1.0
        
        # Initialize Twitter API v2 client if token provided
        if bearer_token:
            self.client = tweepy.Client(bearer_token=bearer_token)
        else:
            self.client = None
            print("No bearer token provided - using alternative methods")
    
    def search_tweets(self, query: str, max_results: int = 100) -> List[Dict]:
        """Search for tweets using Twitter API v2"""
        tweets = []
        
        if not self.client:
            return self.search_tweets_alternative(query, max_results)
        
        try:
            # Search recent tweets
            response = self.client.search_recent_tweets(
                query=query,
                max_results=min(max_results, 100),
                tweet_fields=['created_at', 'public_metrics', 'context_annotations'],
                user_fields=['username', 'public_metrics']
            )
            
            if response.data:
                for tweet in response.data:
                    tweet_data = {
                        'id': tweet.id,
                        'text': tweet.text,
                        'created_at': tweet.created_at,
                        'retweet_count': tweet.public_metrics['retweet_count'],
                        'like_count': tweet.public_metrics['like_count'],
                        'reply_count': tweet.public_metrics['reply_count'],
                        'quote_count': tweet.public_metrics['quote_count']
                    }
                    tweets.append(tweet_data)
            
            time.sleep(self.rate_limit_delay)
            
        except Exception as e:
            print(f"Error searching tweets: {e}")
        
        return tweets
    
    def search_tweets_alternative(self, query: str, max_results: int = 100) -> List[Dict]:
        """Alternative method to get tweet data without official API"""
        # This is a placeholder for alternative methods like:
        # - Web scraping (be careful of ToS)
        # - Third-party APIs
        # - Social media aggregators
        
        print(f"Alternative search for: {query}")
        # For now, return mock data structure
        return []
    
    def analyze_tweet_sentiment(self, text: str) -> Dict:
        """Analyze sentiment of a single tweet"""
        # Clean the text
        cleaned_text = self.clean_tweet_text(text)
        
        # Use TextBlob for sentiment analysis
        blob = TextBlob(cleaned_text)
        
        # Get polarity (-1 to 1) and subjectivity (0 to 1)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Enhanced sentiment analysis
        sentiment_score = self.calculate_enhanced_sentiment(cleaned_text, polarity)
        
        return {
            'text': cleaned_text,
            'polarity': polarity,
            'subjectivity': subjectivity,
            'sentiment_score': sentiment_score,
            'sentiment_label': self.classify_sentiment(sentiment_score)
        }
    
    def clean_tweet_text(self, text: str) -> str:
        """Clean tweet text for sentiment analysis"""
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions and hashtags (but keep the content)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def calculate_enhanced_sentiment(self, text: str, base_polarity: float) -> float:
        """Enhanced sentiment calculation with crypto-specific terms"""
        # Crypto-specific sentiment modifiers
        bullish_terms = [
            'moon', 'rocket', '🚀', 'pump', 'bull', 'buy', 'hodl', 'hold',
            'bullish', 'up', 'rise', 'gain', 'profit', 'green', 'diamond hands',
            'to the moon', 'lambo', 'ath', 'breakout'
        ]
        
        bearish_terms = [
            'dump', 'crash', 'bear', 'sell', 'drop', 'down', 'fall', 'loss',
            'bearish', 'red', 'paper hands', 'rug', 'scam', 'dead', 'rip'
        ]
        
        neutral_terms = [
            'dyor', 'research', 'analysis', 'chart', 'technical', 'support',
            'resistance', 'volume', 'market cap'
        ]
        
        text_lower = text.lower()
        
        # Count sentiment indicators
        bullish_count = sum(1 for term in bullish_terms if term in text_lower)
        bearish_count = sum(1 for term in bearish_terms if term in text_lower)
        
        # Emoji sentiment
        rocket_count = text.count('🚀') + text.count('🌙')
        fire_count = text.count('🔥') + text.count('💎')
        sad_count = text.count('😢') + text.count('💀') + text.count('📉')
        
        # Calculate enhanced sentiment
        sentiment_modifier = 0
        
        if bullish_count > bearish_count:
            sentiment_modifier = 0.3 * (bullish_count - bearish_count)
        elif bearish_count > bullish_count:
            sentiment_modifier = -0.3 * (bearish_count - bullish_count)
        
        # Add emoji influence
        sentiment_modifier += 0.2 * (rocket_count + fire_count - sad_count)
        
        # Combine with base polarity
        enhanced_sentiment = base_polarity + sentiment_modifier
        
        # Normalize to -1 to 1 range
        return max(-1, min(1, enhanced_sentiment))
    
    def classify_sentiment(self, score: float) -> str:
        """Classify sentiment score"""
        if score > 0.5:
            return 'very_bullish'
        elif score > 0.1:
            return 'bullish'
        elif score > -0.1:
            return 'neutral'
        elif score > -0.5:
            return 'bearish'
        else:
            return 'very_bearish'
    
    def get_coin_sentiment(self, coin_symbol: str, coin_name: str) -> Dict:
        """Get sentiment analysis for a specific coin"""
        # Create search queries
        queries = [
            f"${coin_symbol}",
            f"{coin_name}",
            f"{coin_symbol} coin",
            f"{coin_name} crypto"
        ]
        
        all_tweets = []
        
        # Search for each query
        for query in queries:
            tweets = self.search_tweets(query, max_results=25)
            all_tweets.extend(tweets)
            time.sleep(self.rate_limit_delay)
        
        if not all_tweets:
            return {
                'coin_symbol': coin_symbol,
                'coin_name': coin_name,
                'sentiment_score': 0,
                'tweet_count': 0,
                'sentiment_distribution': {},
                'engagement_metrics': {},
                'volume_change': 0
            }
        
        # Analyze sentiment for all tweets
        sentiments = []
        total_engagement = 0
        
        for tweet in all_tweets:
            sentiment_data = self.analyze_tweet_sentiment(tweet['text'])
            sentiments.append(sentiment_data)
            
            # Calculate engagement score
            engagement = (
                tweet.get('like_count', 0) +
                tweet.get('retweet_count', 0) * 2 +
                tweet.get('reply_count', 0) +
                tweet.get('quote_count', 0)
            )
            total_engagement += engagement
        
        # Calculate overall sentiment metrics
        sentiment_scores = [s['sentiment_score'] for s in sentiments]
        weighted_sentiment = self.calculate_weighted_sentiment(sentiments, all_tweets)
        
        # Sentiment distribution
        sentiment_labels = [s['sentiment_label'] for s in sentiments]
        sentiment_distribution = {}
        for label in set(sentiment_labels):
            sentiment_distribution[label] = sentiment_labels.count(label) / len(sentiment_labels)
        
        # Engagement metrics
        avg_engagement = total_engagement / len(all_tweets) if all_tweets else 0
        
        return {
            'coin_symbol': coin_symbol,
            'coin_name': coin_name,
            'sentiment_score': weighted_sentiment,
            'average_sentiment': sum(sentiment_scores) / len(sentiment_scores),
            'tweet_count': len(all_tweets),
            'sentiment_distribution': sentiment_distribution,
            'engagement_metrics': {
                'total_engagement': total_engagement,
                'average_engagement': avg_engagement,
                'high_engagement_tweets': len([t for t in all_tweets 
                                             if (t.get('like_count', 0) + t.get('retweet_count', 0)) > 100])
            },
            'volume_change': self.estimate_volume_change(all_tweets),
            'timestamp': datetime.now()
        }
    
    def calculate_weighted_sentiment(self, sentiments: List[Dict], tweets: List[Dict]) -> float:
        """Calculate engagement-weighted sentiment"""
        if not sentiments or not tweets:
            return 0
        
        weighted_sum = 0
        total_weight = 0
        
        for i, sentiment in enumerate(sentiments):
            if i < len(tweets):
                tweet = tweets[i]
                # Weight by engagement (likes + retweets)
                weight = max(1, tweet.get('like_count', 0) + tweet.get('retweet_count', 0))
                weighted_sum += sentiment['sentiment_score'] * weight
                total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0
    
    def estimate_volume_change(self, tweets: List[Dict]) -> float:
        """Estimate volume change based on tweet timestamps"""
        if len(tweets) < 10:
            return 0
        
        # Sort tweets by creation time
        sorted_tweets = sorted(tweets, key=lambda x: x.get('created_at', datetime.now()))
        
        # Split into two halves
        mid_point = len(sorted_tweets) // 2
        older_half = sorted_tweets[:mid_point]
        newer_half = sorted_tweets[mid_point:]
        
        # Calculate tweet frequency
        if older_half and newer_half:
            older_rate = len(older_half)
            newer_rate = len(newer_half)
            
            if older_rate > 0:
                return ((newer_rate - older_rate) / older_rate) * 100
        
        return 0
    
    def monitor_trending_crypto(self, trending_symbols: List[str]) -> List[Dict]:
        """Monitor sentiment for trending crypto symbols"""
        results = []
        
        for symbol in trending_symbols:
            sentiment_data = self.get_coin_sentiment(symbol, symbol)
            results.append(sentiment_data)
            time.sleep(self.rate_limit_delay * 2)  # Be respectful with rate limits
        
        # Sort by sentiment score
        results.sort(key=lambda x: x['sentiment_score'], reverse=True)
        return results


def main():
    """Test the Twitter sentiment analyzer"""
    # Initialize without API key for testing
    twitter_sentiment = TwitterFreeSentiment()
    
    # Test sentiment analysis on sample text
    sample_tweets = [
        "DOGE to the moon! 🚀🚀🚀 This is going to be huge!",
        "Careful with this pump, looks like it might dump soon...",
        "Just did some technical analysis on $BTC, support levels holding strong",
        "This coin is dead 💀 rug pull incoming",
        "Diamond hands baby! 💎🙌 HODL strong!"
    ]
    
    print("Sample Tweet Sentiment Analysis:")
    for tweet in sample_tweets:
        sentiment = twitter_sentiment.analyze_tweet_sentiment(tweet)
        print(f"Tweet: {tweet[:50]}...")
        print(f"Sentiment: {sentiment['sentiment_label']} ({sentiment['sentiment_score']:.3f})")
        print()


if __name__ == "__main__":
    main()