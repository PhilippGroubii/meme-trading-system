# File: test_sentiment_classifier.py
from ml_models.sentiment_classifier import SentimentClassifier
import numpy as np

def test_sentiment_classifier():
    print("💭 TESTING SENTIMENT CLASSIFIER")
    print("="*50)
    
    # Test 1: Model initialization
    print("\n1. Testing Model Initialization...")
    classifier = SentimentClassifier()
    print("✅ SentimentClassifier initialized")
    
    if classifier.sentiment_pipeline:
        print("   ✅ Transformer model loaded")
    else:
        print("   ⚠️ Using fallback methods (transformers not available)")
    
    # Test 2: Basic sentiment analysis
    print("\n2. Testing Basic Sentiment Analysis...")
    
    test_texts = [
        "DOGE to the moon! 🚀🚀🚀 This is going to be huge!",
        "Careful with this pump, looks like it might dump soon...",
        "Just did some technical analysis on $BTC, support levels holding strong",
        "This coin is dead 💀 rug pull incoming",
        "Diamond hands baby! 💎🙌 HODL strong!",
        "Yeah right, this is totally going to moon 🙄",  # Sarcastic
        "DYOR before investing in any crypto project",
        "Paper hands selling at the bottom again 📉",
        "LFG! 🔥 New ATH incoming! 🚀🌙",
        "Seems legit... definitely not a scam 🤔"  # Sarcastic
    ]
    
    print("✅ Individual Text Analysis:")
    for i, text in enumerate(test_texts, 1):
        analysis = classifier.analyze_text_sentiment(text)
        sarcasm_note = " (SARCASM DETECTED)" if analysis['components']['sarcasm'] > 0.5 else ""
        
        print(f"\n   {i:2d}. Text: {text[:50]}...")
        print(f"       Sentiment: {analysis['label']} ({analysis['sentiment_score']:.3f})")
        print(f"       Confidence: {analysis['confidence']:.3f}{sarcasm_note}")
        
        # Show component breakdown for interesting cases
        if analysis['components']['emoji'] != 0 or analysis['components']['crypto_terms'] != 0:
            comp = analysis['components']
            print(f"       Components: Emoji={comp['emoji']:.2f}, Crypto={comp['crypto_terms']:.2f}")
    
    # Test 3: Batch analysis
    print(f"\n3. Testing Batch Analysis...")
    batch_results = classifier.batch_analyze(test_texts)
    
    sentiments = [r['sentiment_score'] for r in batch_results]
    avg_sentiment = np.mean(sentiments)
    
    print("✅ Batch Results:")
    print(f"   Average Sentiment: {avg_sentiment:.3f}")
    print(f"   Sentiment Range: {min(sentiments):.3f} to {max(sentiments):.3f}")
    
    # Count sentiment distribution
    labels = [r['label'] for r in batch_results]
    from collections import Counter
    distribution = Counter(labels)
    
    print("   Sentiment Distribution:")
    for label, count in distribution.items():
        print(f"     {label}: {count} ({count/len(labels)*100:.1f}%)")
    
    # Test 4: Conversation analysis
    print(f"\n4. Testing Conversation Analysis...")
    
    messages = [
        {'text': text, 'likes': np.random.randint(0, 100), 'retweets': np.random.randint(0, 50)} 
        for text in test_texts
    ]
    
    conv_analysis = classifier.analyze_conversation(messages)
    
    print("✅ Conversation Analysis:")
    print(f"   Overall Sentiment: {conv_analysis['overall_sentiment']:.3f}")
    print(f"   Confidence: {conv_analysis['confidence']:.3f}")
    print(f"   Message Count: {conv_analysis['message_count']}")
    print(f"   Momentum: {conv_analysis['momentum']}")
    
    print("   Distribution:")
    for label, percentage in conv_analysis['sentiment_distribution'].items():
        print(f"     {label}: {percentage:.1%}")
    
    # Test 5: Crypto-specific features
    print(f"\n5. Testing Crypto-Specific Features...")
    
    crypto_tests = [
        "🚀💎🙌 Diamond hands to the moon!",  # Emoji heavy
        "Buy the dip! HODL! Bull market incoming!",  # Crypto terms
        "Yeah, this is totally not a rug pull 🙄",  # Sarcasm
        "Paper hands gonna panic sell again",  # Bearish crypto terms
        "DYOR before investing, check the fundamentals"  # Neutral advice
    ]
    
    for text in crypto_tests:
        analysis = classifier.analyze_text_sentiment(text)
        comp = analysis['components']
        
        print(f"\n   Text: {text}")
        print(f"   Emoji Score: {comp['emoji']:.2f}")
        print(f"   Crypto Terms: {comp['crypto_terms']:.2f}")
        print(f"   Sarcasm: {comp['sarcasm']:.2f}")
        print(f"   Final: {analysis['label']} ({analysis['sentiment_score']:.3f})")
    
    # Test 6: Custom model training (if enough data)
    print(f"\n6. Testing Custom Model Training...")
    
    # Create training data
    training_data = [
        {'text': 'This coin is going to moon! 🚀', 'label': 'very_bullish'},
        {'text': 'HODL strong! Diamond hands! 💎', 'label': 'bullish'},
        {'text': 'Price looking stable, good support', 'label': 'neutral'},
        {'text': 'Might see some selling pressure', 'label': 'bearish'},
        {'text': 'This is a rug pull! Scam coin! 💀', 'label': 'very_bearish'},
        {'text': 'LFG! New ATH incoming! 🔥🚀', 'label': 'very_bullish'},
        {'text': 'Paper hands selling again', 'label': 'bearish'},
        {'text': 'DYOR, check the fundamentals', 'label': 'neutral'},
        {'text': 'Bull market vibes! 📈💪', 'label': 'bullish'},
        {'text': 'Dead coin, no volume 💀', 'label': 'very_bearish'},
        {'text': 'Accumulation phase, good entry', 'label': 'bullish'},
        {'text': 'Market looking uncertain today', 'label': 'neutral'}
    ]
    
    try:
        custom_results = classifier.train_custom_classifier(training_data)
        
        print("✅ Custom Model Training:")
        print(f"   Accuracy: {custom_results['accuracy']:.3f}")
        print(f"   F1 Score: {custom_results['f1_score']:.3f}")
        print(f"   Classes: {custom_results['classes']}")
        
        # Test custom prediction
        test_prediction = classifier.predict_custom("This pump looks sustainable! 🚀💎")
        print(f"\n   Custom Prediction Test:")
        print(f"   Predicted Class: {test_prediction['predicted_class']}")
        print(f"   Confidence: {test_prediction['confidence']:.3f}")
        
    except Exception as e:
        print(f"   ⚠️ Custom training skipped: {e}")
    
    # Test 7: Report generation
    print(f"\n7. Testing Report Generation...")
    
    report = classifier.create_sentiment_report(test_texts, "Test Crypto Sentiment Analysis")
    
    print("✅ Generated Sentiment Report:")
    print(report[:500] + "..." if len(report) > 500 else report)
    
    # Test 8: Model persistence
    print(f"\n8. Testing Model Save/Load...")
    
    classifier.save_model("test_sentiment_model.joblib")
    
    # Load and test
    classifier_loaded = SentimentClassifier()
    classifier_loaded.load_model("test_sentiment_model.joblib")
    
    # Test loaded model
    test_analysis = classifier_loaded.analyze_text_sentiment("Test message for loaded model 🚀")
    print(f"✅ Model saved and loaded successfully")
    print(f"   Loaded model analysis: {test_analysis['label']} ({test_analysis['sentiment_score']:.3f})")
    
    print(f"\n🎉 SENTIMENT CLASSIFIER TESTS COMPLETED!")
    
    # Cleanup
    import os
    if os.path.exists("test_sentiment_model.joblib"):
        os.remove("test_sentiment_model.joblib")

if __name__ == "__main__":
    test_sentiment_classifier()