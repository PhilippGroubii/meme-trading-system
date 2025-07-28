# File: test_twitter_sentiment.py
from sentiment.twitter_free import TwitterFreeSentiment

def test_twitter_sentiment():
    # Initialize (without API key for testing)
    twitter = TwitterFreeSentiment()
    
    # Test texts with various crypto scenarios
    test_texts = [
        "DOGE to the moon! 🚀🚀🚀 Diamond hands! 💎🙌",
        "This pump looks suspicious... might dump soon 📉",
        "HODL strong! Bull market is here! 📈💪",
        "Paper hands selling at the bottom again... 😢",
        "Rug pull incoming! This coin is dead 💀",
        "Yeah right, totally going to moon 🙄",  # Sarcastic
        "DYOR before investing. Technical analysis shows support at $0.08",
        "LFG! 🔥 New ATH incoming! 🚀🌙💯",
        "Seems legit... definitely not a scam 🤔",  # Sarcastic
        "Buy the dip! Accumulation phase! 💰"
    ]
    
    print("=== Testing Individual Tweet Analysis ===")
    for i, text in enumerate(test_texts, 1):
        analysis = twitter.analyze_tweet_sentiment(text)
        print(f"\n{i}. Text: {text}")
        print(f"   Sentiment: {analysis['sentiment_label']} ({analysis['sentiment_score']:.3f})")
        print(f"   Confidence: {analysis['confidence']:.3f}")
        
        # Show component breakdown
        components = analysis['components']
        print(f"   Components: Emoji: {components['emoji']:.2f}, "
              f"Crypto: {components['crypto_terms']:.2f}, "
              f"Sarcasm: {components['sarcasm']:.2f}")
    
    # Test batch analysis
    print(f"\n=== Testing Batch Analysis ===")
    batch_results = twitter.batch_analyze(test_texts)
    
    sentiments = [r['sentiment_score'] for r in batch_results]
    avg_sentiment = sum(sentiments) / len(sentiments)
    print(f"Average Sentiment: {avg_sentiment:.3f}")
    
    # Count sentiment distribution
    labels = [r['sentiment_label'] for r in batch_results]
    from collections import Counter
    distribution = Counter(labels)
    
    print("Sentiment Distribution:")
    for label, count in distribution.items():
        print(f"  {label}: {count} ({count/len(labels)*100:.1f}%)")

if __name__ == "__main__":
    test_twitter_sentiment()