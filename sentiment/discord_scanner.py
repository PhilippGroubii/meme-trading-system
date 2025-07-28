"""
Discord Scanner for Meme Coin Communities
Monitors Discord servers for meme coin discussions and sentiment
"""

import asyncio
import json
import re
from typing import Dict, List, Optional
from datetime import datetime, timedelta

class DiscordScanner:
    """Discord community sentiment scanner"""
    
    def __init__(self):
        self.sentiment_keywords = {
            'bullish': ['moon', 'pump', 'bullish', 'buy', 'hold', '🚀', '💎', 'diamond hands'],
            'bearish': ['dump', 'sell', 'bearish', 'crash', 'dead', 'rug', 'scam'],
            'neutral': ['stable', 'sideways', 'wait', 'watching', 'hmm']
        }
        
    def analyze_message_sentiment(self, message: str) -> Dict:
        """Analyze sentiment of a Discord message"""
        message_lower = message.lower()
        
        bullish_count = sum(1 for word in self.sentiment_keywords['bullish'] 
                           if word in message_lower)
        bearish_count = sum(1 for word in self.sentiment_keywords['bearish'] 
                           if word in message_lower)
        neutral_count = sum(1 for word in self.sentiment_keywords['neutral'] 
                           if word in message_lower)
        
        total_signals = bullish_count + bearish_count + neutral_count
        
        if total_signals == 0:
            sentiment_score = 0.0
        else:
            sentiment_score = (bullish_count - bearish_count) / total_signals
            
        return {
            'sentiment_score': sentiment_score,
            'bullish_signals': bullish_count,
            'bearish_signals': bearish_count,
            'neutral_signals': neutral_count,
            'confidence': min(total_signals / 3.0, 1.0)  # Max confidence at 3+ signals
        }
    
    def scan_server_messages(self, server_data: List[Dict]) -> Dict:
        """
        Scan Discord server messages for sentiment
        
        Args:
            server_data: List of message dictionaries
            
        Returns:
            Aggregated sentiment analysis
        """
        if not server_data:
            return {
                'overall_sentiment': 0.0,
                'confidence': 0.0,
                'message_count': 0,
                'active_users': 0
            }
            
        sentiment_scores = []
        total_confidence = 0
        active_users = set()
        
        for message_data in server_data:
            message = message_data.get('content', '')
            user_id = message_data.get('user_id', '')
            
            if message and user_id:
                analysis = self.analyze_message_sentiment(message)
                sentiment_scores.append(analysis['sentiment_score'])
                total_confidence += analysis['confidence']
                active_users.add(user_id)
        
        if sentiment_scores:
            overall_sentiment = sum(sentiment_scores) / len(sentiment_scores)
            avg_confidence = total_confidence / len(sentiment_scores)
        else:
            overall_sentiment = 0.0
            avg_confidence = 0.0
            
        return {
            'overall_sentiment': overall_sentiment,
            'confidence': avg_confidence,
            'message_count': len(server_data),
            'active_users': len(active_users),
            'sentiment_distribution': self._calculate_sentiment_distribution(sentiment_scores)
        }
    
    def _calculate_sentiment_distribution(self, scores: List[float]) -> Dict:
        """Calculate distribution of sentiment scores"""
        if not scores:
            return {'bullish': 0, 'neutral': 0, 'bearish': 0}
            
        bullish = len([s for s in scores if s > 0.1])
        bearish = len([s for s in scores if s < -0.1])
        neutral = len(scores) - bullish - bearish
        
        total = len(scores)
        return {
            'bullish': bullish / total,
            'neutral': neutral / total,
            'bearish': bearish / total
        }
    
    def get_server_activity_score(self, message_count: int, active_users: int, 
                                time_window_hours: int = 24) -> float:
        """Calculate server activity score"""
        # Normalize activity based on time window
        messages_per_hour = message_count / time_window_hours
        
        # Score based on activity level
        if messages_per_hour > 50:
            activity_score = 1.0  # Very active
        elif messages_per_hour > 20:
            activity_score = 0.8  # Active
        elif messages_per_hour > 5:
            activity_score = 0.6  # Moderate
        elif messages_per_hour > 1:
            activity_score = 0.4  # Low
        else:
            activity_score = 0.2  # Very low
            
        # Boost score for more unique users
        user_diversity = min(active_users / 10, 1.0)  # Max boost at 10+ users
        
        return min(activity_score * (1 + user_diversity * 0.5), 1.0)

    def simulate_discord_data(self, symbol: str) -> List[Dict]:
        """Simulate Discord messages for testing"""
        import random
        
        sample_messages = [
            f"{symbol} is going to the moon! 🚀🚀🚀",
            f"Just bought more {symbol}, diamond hands 💎",
            f"{symbol} looks like it might dump soon",
            f"Holding my {symbol} bag, not selling",
            f"{symbol} chart looking bullish",
            f"When {symbol} pump?",
            f"{symbol} is dead, moving to other coins",
            f"Still bullish on {symbol} long term",
            f"{symbol} to $1 soon",
            f"Bought the {symbol} dip again"
        ]
        
        messages = []
        for i in range(random.randint(10, 50)):
            messages.append({
                'content': random.choice(sample_messages),
                'user_id': f"user_{random.randint(1, 20)}",
                'timestamp': datetime.now() - timedelta(minutes=random.randint(1, 1440))
            })
            
        return messages


# Example usage and testing
if __name__ == "__main__":
    scanner = DiscordScanner()
    
    # Test with simulated data
    test_symbol = "DOGE"
    simulated_messages = scanner.simulate_discord_data(test_symbol)
    
    print(f"Analyzing {len(simulated_messages)} Discord messages for {test_symbol}")
    
    analysis = scanner.scan_server_messages(simulated_messages)
    
    print(f"Overall Sentiment: {analysis['overall_sentiment']:.2f}")
    print(f"Confidence: {analysis['confidence']:.2f}")
    print(f"Message Count: {analysis['message_count']}")
    print(f"Active Users: {analysis['active_users']}")
    print(f"Distribution: {analysis['sentiment_distribution']}")
    
    activity_score = scanner.get_server_activity_score(
        analysis['message_count'], 
        analysis['active_users']
    )
    print(f"Activity Score: {activity_score:.2f}")