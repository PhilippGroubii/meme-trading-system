"""
Advanced sentiment classifier using Hugging Face transformers with sarcasm detection and emoji analysis
"""
import numpy as np
import pandas as pd
import re
import json
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
import joblib
import warnings
warnings.filterwarnings('ignore')

# Try to import transformers, fallback to basic NLP
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Transformers not available, using basic sentiment analysis")

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False


class SentimentClassifier:
    def __init__(self, model_name: str = 'cardiffnlp/twitter-roberta-base-sentiment-latest'):
        """Initialize advanced sentiment classifier"""
        self.model_name = model_name
        self.sentiment_pipeline = None
        self.custom_model = None
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        
        # Emoji sentiment mappings
        self.emoji_sentiment = {
            # Very Positive
            '🚀': 0.9, '🌙': 0.8, '💎': 0.8, '🔥': 0.7, '🎉': 0.7,
            '💪': 0.6, '✨': 0.6, '🙌': 0.7, '💯': 0.8, '⚡': 0.6,
            
            # Positive
            '😊': 0.5, '😃': 0.5, '😍': 0.6, '🤑': 0.7, '💰': 0.6,
            '📈': 0.7, '🟢': 0.5, '✅': 0.5, '👍': 0.4, '❤️': 0.4,
            
            # Negative
            '😢': -0.5, '😭': -0.6, '😞': -0.4, '🤔': -0.2, '😬': -0.3,
            '📉': -0.7, '🔴': -0.5, '❌': -0.5, '👎': -0.4, '💔': -0.6,
            
            # Very Negative
            '💀': -0.8, '🤮': -0.7, '😡': -0.7, '🤬': -0.8, '👎': -0.5,
            '🗿': -0.3  # Often used sarcastically
        }
        
        # Crypto-specific sentiment terms
        self.crypto_sentiment_terms = {
            # Very Bullish
            'moon': 0.9, 'rocket': 0.8, 'lambo': 0.8, 'ath': 0.7, 'breakout': 0.6,
            'pump': 0.7, 'bull': 0.6, 'diamond hands': 0.8, 'hodl': 0.6, 'hold': 0.4,
            'buy the dip': 0.7, 'accumulate': 0.5, 'bullish': 0.6, 'moonshot': 0.9,
            
            # Moderately Bullish
            'long': 0.4, 'entry': 0.3, 'support': 0.3, 'bounce': 0.4, 'reversal': 0.3,
            'oversold': 0.4, 'undervalued': 0.5, 'gem': 0.6, 'hidden gem': 0.7,
            
            # Bearish
            'dump': -0.7, 'crash': -0.8, 'bear': -0.6, 'bearish': -0.6, 'sell': -0.5,
            'exit': -0.4, 'resistance': -0.3, 'overbought': -0.4, 'correction': -0.3,
            
            # Very Bearish
            'rug': -0.9, 'scam': -0.9, 'dead': -0.8, 'rip': -0.7, 'dump it': -0.8,
            'paper hands': -0.6, 'panic sell': -0.7, 'capitulation': -0.8, 'rekt': -0.8,
            
            # Neutral/Informational
            'dyor': 0.0, 'research': 0.1, 'analysis': 0.0, 'chart': 0.0, 'ta': 0.0,
            'volume': 0.0, 'market cap': 0.0, 'liquidity': 0.0
        }
        
        # Sarcasm indicators
        self.sarcasm_indicators = [
            'yeah right', 'sure thing', 'totally', 'absolutely not', 'what could go wrong',
            'this is fine', 'nothing to see here', 'perfectly normal', 'very stable',
            'definitely not', 'trust me bro', 'seems legit', 'what a surprise'
        ]
        
        # Initialize transformer model if available
        if TRANSFORMERS_AVAILABLE:
            self._initialize_transformer_model()
    
    def _initialize_transformer_model(self):
        """Initialize Hugging Face transformer model"""
        try:
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                tokenizer=self.model_name,
                return_all_scores=True
            )
            print(f"Initialized transformer model: {self.model_name}")
        except Exception as e:
            print(f"Failed to initialize transformer model: {e}")
            self.sentiment_pipeline = None
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for sentiment analysis"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Keep important punctuation and emojis
        text = re.sub(r'[^\w\s\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF!?.,]', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_emoji_sentiment(self, text: str) -> float:
        """Extract sentiment from emojis"""
        emoji_scores = []
        
        for emoji, score in self.emoji_sentiment.items():
            count = text.count(emoji)
            if count > 0:
                # Weight by frequency but with diminishing returns
                weighted_score = score * min(count, 3) * (0.7 ** (count - 1))
                emoji_scores.append(weighted_score)
        
        if emoji_scores:
            return np.mean(emoji_scores)
        return 0.0
    
    def extract_crypto_sentiment(self, text: str) -> float:
        """Extract sentiment from crypto-specific terms"""
        text_lower = text.lower()
        term_scores = []
        
        for term, score in self.crypto_sentiment_terms.items():
            if term in text_lower:
                # Count occurrences
                count = text_lower.count(term)
                weighted_score = score * min(count, 2)  # Max 2x weighting
                term_scores.append(weighted_score)
        
        if term_scores:
            return np.mean(term_scores)
        return 0.0
    
    def detect_sarcasm(self, text: str) -> float:
        """Detect sarcasm and return sarcasm score (0-1)"""
        text_lower = text.lower()
        sarcasm_score = 0.0
        
        # Check for sarcasm indicators
        for indicator in self.sarcasm_indicators:
            if indicator in text_lower:
                sarcasm_score += 0.3
        
        # Check for contradictory patterns
        positive_words = ['good', 'great', 'amazing', 'perfect', 'wonderful']
        negative_context = ['not', 'never', 'hardly', 'barely', 'totally not']
        
        has_positive = any(word in text_lower for word in positive_words)
        has_negative_context = any(word in text_lower for word in negative_context)
        
        if has_positive and has_negative_context:
            sarcasm_score += 0.4
        
        # Excessive punctuation can indicate sarcasm
        if text.count('!') > 2 or text.count('?') > 2:
            sarcasm_score += 0.2
        
        # Mixed emoji sentiment can indicate sarcasm
        positive_emojis = sum(1 for emoji in self.emoji_sentiment 
                            if emoji in text and self.emoji_sentiment[emoji] > 0.5)
        negative_emojis = sum(1 for emoji in self.emoji_sentiment 
                            if emoji in text and self.emoji_sentiment[emoji] < -0.5)
        
        if positive_emojis > 0 and negative_emojis > 0:
            sarcasm_score += 0.3
        
        return min(1.0, sarcasm_score)
    
    def analyze_text_sentiment(self, text: str) -> Dict:
        """Analyze sentiment of text using multiple methods"""
        preprocessed_text = self.preprocess_text(text)
        
        if not preprocessed_text:
            return {
                'sentiment_score': 0.0,
                'confidence': 0.0,
                'label': 'neutral',
                'method': 'empty_text'
            }
        
        # Method 1: Transformer model (if available)
        transformer_sentiment = self._get_transformer_sentiment(preprocessed_text)
        
        # Method 2: TextBlob (if available)
        textblob_sentiment = self._get_textblob_sentiment(preprocessed_text)
        
        # Method 3: Emoji sentiment
        emoji_sentiment = self.extract_emoji_sentiment(text)
        
        # Method 4: Crypto-specific sentiment
        crypto_sentiment = self.extract_crypto_sentiment(text)
        
        # Method 5: Sarcasm detection
        sarcasm_score = self.detect_sarcasm(text)
        
        # Combine sentiments with weights
        sentiments = []
        weights = []
        
        if transformer_sentiment['confidence'] > 0.5:
            sentiments.append(transformer_sentiment['score'])
            weights.append(0.4)
        
        if textblob_sentiment['confidence'] > 0.3:
            sentiments.append(textblob_sentiment['score'])
            weights.append(0.2)
        
        if abs(emoji_sentiment) > 0.1:
            sentiments.append(emoji_sentiment)
            weights.append(0.2)
        
        if abs(crypto_sentiment) > 0.1:
            sentiments.append(crypto_sentiment)
            weights.append(0.2)
        
        # Calculate weighted sentiment
        if sentiments and weights:
            final_sentiment = np.average(sentiments, weights=weights)
            confidence = sum(weights) / sum([0.4, 0.2, 0.2, 0.2])  # Normalize to max possible weight
        else:
            final_sentiment = 0.0
            confidence = 0.0
        
        # Apply sarcasm adjustment
        if sarcasm_score > 0.5:
            final_sentiment *= -1  # Flip sentiment if sarcastic
            confidence *= (1 - sarcasm_score * 0.5)  # Reduce confidence
        
        # Classify sentiment
        if final_sentiment > 0.3:
            label = 'very_bullish'
        elif final_sentiment > 0.1:
            label = 'bullish'
        elif final_sentiment > -0.1:
            label = 'neutral'
        elif final_sentiment > -0.3:
            label = 'bearish'
        else:
            label = 'very_bearish'
        
        return {
            'sentiment_score': final_sentiment,
            'confidence': confidence,
            'label': label,
            'components': {
                'transformer': transformer_sentiment,
                'textblob': textblob_sentiment,
                'emoji': emoji_sentiment,
                'crypto_terms': crypto_sentiment,
                'sarcasm': sarcasm_score
            },
            'method': 'ensemble'
        }
    
    def _get_transformer_sentiment(self, text: str) -> Dict:
        """Get sentiment using transformer model"""
        if not self.sentiment_pipeline or not text:
            return {'score': 0.0, 'confidence': 0.0}
        
        try:
            results = self.sentiment_pipeline(text)
            
            # Convert to our scale (-1 to 1)
            if isinstance(results[0], list):
                results = results[0]
            
            score = 0.0
            confidence = 0.0
            
            for result in results:
                label = result['label'].lower()
                prob = result['score']
                
                if 'positive' in label or 'pos' in label:
                    score += prob
                elif 'negative' in label or 'neg' in label:
                    score -= prob
                
                confidence = max(confidence, prob)
            
            return {'score': score, 'confidence': confidence}
            
        except Exception as e:
            print(f"Transformer sentiment error: {e}")
            return {'score': 0.0, 'confidence': 0.0}
    
    def _get_textblob_sentiment(self, text: str) -> Dict:
        """Get sentiment using TextBlob"""
        if not TEXTBLOB_AVAILABLE or not text:
            return {'score': 0.0, 'confidence': 0.0}
        
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            # Confidence based on subjectivity (subjective text is more confident)
            confidence = subjectivity * 0.8
            
            return {'score': polarity, 'confidence': confidence}
            
        except Exception as e:
            print(f"TextBlob sentiment error: {e}")
            return {'score': 0.0, 'confidence': 0.0}
    
    def batch_analyze(self, texts: List[str]) -> List[Dict]:
        """Analyze sentiment for multiple texts"""
        results = []
        
        for text in texts:
            result = self.analyze_text_sentiment(text)
            results.append(result)
        
        return results
    
    def analyze_conversation(self, messages: List[Dict]) -> Dict:
        """Analyze sentiment of entire conversation"""
        if not messages:
            return {'overall_sentiment': 0.0, 'confidence': 0.0, 'message_count': 0}
        
        sentiments = []
        confidences = []
        weights = []
        
        for msg in messages:
            text = msg.get('text', '')
            timestamp = msg.get('timestamp', None)
            likes = msg.get('likes', 0)
            retweets = msg.get('retweets', 0)
            
            if not text:
                continue
            
            analysis = self.analyze_text_sentiment(text)
            
            # Weight by engagement and recency
            weight = 1.0
            if likes > 0 or retweets > 0:
                engagement_weight = min(3.0, 1 + (likes + retweets * 2) / 100)
                weight *= engagement_weight
            
            # More recent messages get higher weight
            if timestamp and len(messages) > 1:
                recency_weight = 1.0 + 0.5 * (messages.index(msg) / len(messages))
                weight *= recency_weight
            
            sentiments.append(analysis['sentiment_score'])
            confidences.append(analysis['confidence'])
            weights.append(weight)
        
        if not sentiments:
            return {'overall_sentiment': 0.0, 'confidence': 0.0, 'message_count': 0}
        
        # Calculate weighted sentiment
        overall_sentiment = np.average(sentiments, weights=weights)
        overall_confidence = np.average(confidences, weights=weights)
        
        # Sentiment distribution
        sentiment_labels = []
        for sentiment in sentiments:
            if sentiment > 0.3:
                sentiment_labels.append('very_bullish')
            elif sentiment > 0.1:
                sentiment_labels.append('bullish')
            elif sentiment > -0.1:
                sentiment_labels.append('neutral')
            elif sentiment > -0.3:
                sentiment_labels.append('bearish')
            else:
                sentiment_labels.append('very_bearish')
        
        label_counts = {}
        for label in sentiment_labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        
        # Convert to percentages
        total = len(sentiment_labels)
        sentiment_distribution = {label: count/total for label, count in label_counts.items()}
        
        return {
            'overall_sentiment': overall_sentiment,
            'confidence': overall_confidence,
            'message_count': len(sentiments),
            'sentiment_distribution': sentiment_distribution,
            'individual_sentiments': sentiments,
            'momentum': self._calculate_sentiment_momentum(sentiments)
        }
    
    def _calculate_sentiment_momentum(self, sentiments: List[float]) -> str:
        """Calculate sentiment momentum"""
        if len(sentiments) < 3:
            return 'insufficient_data'
        
        # Compare recent vs older sentiments
        recent = sentiments[-3:]
        older = sentiments[:-3] if len(sentiments) > 6 else sentiments[:-3]
        
        if not older:
            return 'insufficient_data'
        
        recent_avg = np.mean(recent)
        older_avg = np.mean(older)
        
        if recent_avg > older_avg + 0.1:
            return 'improving'
        elif recent_avg < older_avg - 0.1:
            return 'declining'
        else:
            return 'stable'
    
    def train_custom_classifier(self, training_data: List[Dict]) -> Dict:
        """Train custom sentiment classifier on domain-specific data"""
        if not training_data:
            raise ValueError("Training data cannot be empty")
        
        texts = []
        labels = []
        
        for item in training_data:
            if 'text' in item and 'label' in item:
                texts.append(item['text'])
                labels.append(item['label'])
        
        if len(texts) < 10:
            raise ValueError("Need at least 10 training examples")
        
        # Extract features from texts
        features = []
        for text in texts:
            analysis = self.analyze_text_sentiment(text)
            feature_vector = [
                analysis['sentiment_score'],
                analysis['confidence'],
                analysis['components']['emoji'],
                analysis['components']['crypto_terms'],
                analysis['components']['sarcasm'],
                len(text.split()),  # Text length
                text.count('!'),    # Exclamation marks
                text.count('?'),    # Question marks
                text.count('$'),    # Dollar signs
                len([c for c in text if c.isupper()]) / max(1, len(text))  # Caps ratio
            ]
            features.append(feature_vector)
        
        # Encode labels
        encoded_labels = self.label_encoder.fit_transform(labels)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels
        )
        
        # Train simple classifier
        from sklearn.ensemble import RandomForestClassifier
        self.custom_model = RandomForestClassifier(n_estimators=50, random_state=42)
        self.custom_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.custom_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        self.is_trained = True
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'classes': self.label_encoder.classes_.tolist()
        }
    
    def predict_custom(self, text: str) -> Dict:
        """Make prediction using custom trained model"""
        if not self.is_trained or not self.custom_model:
            raise ValueError("Custom model must be trained before making predictions")
        
        # Extract features
        analysis = self.analyze_text_sentiment(text)
        feature_vector = [
            analysis['sentiment_score'],
            analysis['confidence'],
            analysis['components']['emoji'],
            analysis['components']['crypto_terms'],
            analysis['components']['sarcasm'],
            len(text.split()),
            text.count('!'),
            text.count('?'),
            text.count('$'),
            len([c for c in text if c.isupper()]) / max(1, len(text))
        ]
        
        # Predict
        prediction = self.custom_model.predict([feature_vector])[0]
        probabilities = self.custom_model.predict_proba([feature_vector])[0]
        
        # Get class name
        predicted_class = self.label_encoder.inverse_transform([prediction])[0]
        
        # Class probabilities
        class_probs = {}
        for i, class_name in enumerate(self.label_encoder.classes_):
            class_probs[class_name] = probabilities[i]
        
        return {
            'predicted_class': predicted_class,
            'confidence': max(probabilities),
            'class_probabilities': class_probs,
            'base_analysis': analysis
        }
    
    def create_sentiment_report(self, texts: List[str], title: str = "Sentiment Analysis Report") -> str:
        """Create comprehensive sentiment analysis report"""
        if not texts:
            return "No texts provided for analysis"
        
        # Analyze all texts
        analyses = self.batch_analyze(texts)
        
        # Calculate summary statistics
        sentiments = [a['sentiment_score'] for a in analyses]
        confidences = [a['confidence'] for a in analyses]
        
        avg_sentiment = np.mean(sentiments)
        sentiment_std = np.std(sentiments)
        avg_confidence = np.mean(confidences)
        
        # Count labels
        label_counts = {}
        for analysis in analyses:
            label = analysis['label']
            label_counts[label] = label_counts.get(label, 0) + 1
        
        # Create report
        report = f"""
{title}
{'='*60}

SUMMARY STATISTICS:
Average Sentiment: {avg_sentiment:.3f} (-1.0 = Very Bearish, +1.0 = Very Bullish)
Sentiment Volatility: {sentiment_std:.3f}
Average Confidence: {avg_confidence:.3f}
Total Messages: {len(texts)}

SENTIMENT DISTRIBUTION:
"""
        
        for label, count in sorted(label_counts.items()):
            percentage = (count / len(texts)) * 100
            report += f"{label.replace('_', ' ').title()}: {count} messages ({percentage:.1f}%)\n"
        
        report += f"""
SENTIMENT CLASSIFICATION:
"""
        
        if avg_sentiment > 0.3:
            overall_sentiment = "VERY BULLISH 🚀"
        elif avg_sentiment > 0.1:
            overall_sentiment = "BULLISH 📈"
        elif avg_sentiment > -0.1:
            overall_sentiment = "NEUTRAL 🔄"
        elif avg_sentiment > -0.3:
            overall_sentiment = "BEARISH 📉"
        else:
            overall_sentiment = "VERY BEARISH 💀"
        
        report += f"Overall Market Sentiment: {overall_sentiment}\n"
        
        # Risk assessment
        if sentiment_std > 0.4:
            risk_level = "HIGH - Very volatile sentiment"
        elif sentiment_std > 0.25:
            risk_level = "MEDIUM - Moderate sentiment swings"
        else:
            risk_level = "LOW - Stable sentiment"
        
        report += f"Sentiment Risk Level: {risk_level}\n"
        
        # Confidence assessment
        if avg_confidence > 0.7:
            confidence_level = "HIGH - Strong signal confidence"
        elif avg_confidence > 0.5:
            confidence_level = "MEDIUM - Moderate signal confidence"
        else:
            confidence_level = "LOW - Weak signal confidence"
        
        report += f"Analysis Confidence: {confidence_level}\n"
        
        # Sample messages
        report += f"""
SAMPLE ANALYSES:
"""
        
        # Show most bullish, bearish, and neutral messages
        sorted_analyses = sorted(analyses, key=lambda x: x['sentiment_score'])
        
        if len(sorted_analyses) >= 3:
            # Most bearish
            bearish = sorted_analyses[0]
            report += f"Most Bearish ({bearish['sentiment_score']:.2f}): {texts[analyses.index(bearish)][:100]}...\n\n"
            
            # Most neutral
            neutral_idx = len(sorted_analyses) // 2
            neutral = sorted_analyses[neutral_idx]
            report += f"Most Neutral ({neutral['sentiment_score']:.2f}): {texts[analyses.index(neutral)][:100]}...\n\n"
            
            # Most bullish
            bullish = sorted_analyses[-1]
            report += f"Most Bullish ({bullish['sentiment_score']:.2f}): {texts[analyses.index(bullish)][:100]}...\n\n"
        
        report += f"Generated at: {pd.Timestamp.now()}\n"
        
        return report
    
    def save_model(self, filepath: str):
        """Save trained model"""
        model_data = {
            'custom_model': self.custom_model,
            'label_encoder': self.label_encoder,
            'is_trained': self.is_trained,
            'model_name': self.model_name,
            'emoji_sentiment': self.emoji_sentiment,
            'crypto_sentiment_terms': self.crypto_sentiment_terms,
            'sarcasm_indicators': self.sarcasm_indicators
        }
        
        joblib.dump(model_data, filepath)
        print(f"Sentiment classifier saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model"""
        model_data = joblib.load(filepath)
        
        self.custom_model = model_data.get('custom_model')
        self.label_encoder = model_data.get('label_encoder', LabelEncoder())
        self.is_trained = model_data.get('is_trained', False)
        self.model_name = model_data.get('model_name', self.model_name)
        self.emoji_sentiment = model_data.get('emoji_sentiment', self.emoji_sentiment)
        self.crypto_sentiment_terms = model_data.get('crypto_sentiment_terms', self.crypto_sentiment_terms)
        self.sarcasm_indicators = model_data.get('sarcasm_indicators', self.sarcasm_indicators)
        
        print(f"Sentiment classifier loaded from {filepath}")


def main():
    """Test the sentiment classifier"""
    # Initialize classifier
    classifier = SentimentClassifier()
    
    # Test texts
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
    
    print("Testing Sentiment Classifier...")
    print("="*50)
    
    # Analyze individual texts
    for i, text in enumerate(test_texts, 1):
        analysis = classifier.analyze_text_sentiment(text)
        print(f"\n{i}. Text: {text}")
        print(f"   Sentiment: {analysis['label']} (Score: {analysis['sentiment_score']:.3f})")
        print(f"   Confidence: {analysis['confidence']:.3f}")
        if analysis['components']['sarcasm'] > 0.3:
            print(f"   Sarcasm Detected: {analysis['components']['sarcasm']:.2f}")
    
    # Batch analysis
    print(f"\n\nBatch Analysis:")
    print("="*50)
    batch_results = classifier.batch_analyze(test_texts)
    
    sentiments = [r['sentiment_score'] for r in batch_results]
    avg_sentiment = np.mean(sentiments)
    print(f"Average Sentiment: {avg_sentiment:.3f}")
    
    # Conversation analysis
    messages = [{'text': text, 'likes': np.random.randint(0, 100), 'retweets': np.random.randint(0, 50)} 
                for text in test_texts]
    
    conv_analysis = classifier.analyze_conversation(messages)
    print(f"\nConversation Analysis:")
    print(f"Overall Sentiment: {conv_analysis['overall_sentiment']:.3f}")
    print(f"Confidence: {conv_analysis['confidence']:.3f}")
    print(f"Momentum: {conv_analysis['momentum']}")
    print(f"Distribution: {conv_analysis['sentiment_distribution']}")
    
    # Generate report
    print(f"\n\nSentiment Report:")
    print("="*50)
    report = classifier.create_sentiment_report(test_texts, "Test Crypto Sentiment Analysis")
    print(report)


if __name__ == "__main__":
    main()