"""
Multi-source sentiment aggregation for meme coin analysis
"""
import asyncio
from typing import Dict, List, Optional
import pandas as pd
from datetime import datetime, timedelta
import numpy as np


class MultiSourceSentiment:
    def __init__(self):
        """Initialize multi-source sentiment analyzer"""
        self.sources = {
            'twitter': None,
            'reddit': None,
            'discord': None,
            'telegram': None,
            'google_trends': None
        }
        self.weights = {
            'twitter': 0.30,
            'reddit': 0.25,
            'discord': 0.20,
            'telegram': 0.15,
            'google_trends': 0.10
        }
    
    def register_source(self, source_name: str, source_instance):
        """Register a sentiment data source"""
        if source_name in self.sources:
            self.sources[source_name] = source_instance
            print(f"Registered {source_name} sentiment source")
        else:
            print(f"Unknown source: {source_name}")
    
    def calculate_composite_sentiment(self, sentiments: Dict[str, float]) -> Dict:
        """Calculate weighted composite sentiment score"""
        total_weight = 0
        weighted_sum = 0
        source_contributions = {}
        
        for source, sentiment in sentiments.items():
            if source in self.weights and sentiment is not None:
                weight = self.weights[source]
                weighted_sum += sentiment * weight
                total_weight += weight
                source_contributions[source] = {
                    'sentiment': sentiment,
                    'weight': weight,
                    'contribution': sentiment * weight
                }
        
        if total_weight == 0:
            return {
                'composite_score': 0,
                'confidence': 0,
                'sources_used': 0,
                'source_breakdown': {}
            }
        
        composite_score = weighted_sum / total_weight
        confidence = total_weight  # Higher when more sources available
        
        return {
            'composite_score': composite_score,
            'confidence': confidence,
            'sources_used': len(source_contributions),
            'source_breakdown': source_contributions,
            'timestamp': datetime.now()
        }
    
    def analyze_coin_sentiment(self, coin_symbol: str, coin_name: str) -> Dict:
        """Analyze sentiment for a specific coin across all sources"""
        sentiments = {}
        raw_data = {}
        
        # Twitter sentiment
        if self.sources['twitter']:
            try:
                twitter_data = self.sources['twitter'].get_coin_sentiment(coin_symbol, coin_name)
                sentiments['twitter'] = twitter_data.get('sentiment_score', None)
                raw_data['twitter'] = twitter_data
            except Exception as e:
                print(f"Twitter sentiment error: {e}")
                sentiments['twitter'] = None
        
        # Reddit sentiment
        if self.sources['reddit']:
            try:
                reddit_data = self.sources['reddit'].get_coin_sentiment(coin_symbol, coin_name)
                sentiments['reddit'] = reddit_data.get('sentiment_score', None)
                raw_data['reddit'] = reddit_data
            except Exception as e:
                print(f"Reddit sentiment error: {e}")
                sentiments['reddit'] = None
        
        # Discord sentiment
        if self.sources['discord']:
            try:
                discord_data = self.sources['discord'].get_coin_sentiment(coin_symbol, coin_name)
                sentiments['discord'] = discord_data.get('sentiment_score', None)
                raw_data['discord'] = discord_data
            except Exception as e:
                print(f"Discord sentiment error: {e}")
                sentiments['discord'] = None
        
        # Google Trends (convert trend score to sentiment)
        if self.sources['google_trends']:
            try:
                trends_data = self.sources['google_trends'].analyze_meme_coin_hype(coin_name, coin_symbol)
                # Convert trend score (0-100) to sentiment (-1 to 1)
                trend_score = trends_data.get('trend_score', 0)
                trend_sentiment = (trend_score / 100) * 2 - 1  # Scale to -1 to 1
                sentiments['google_trends'] = trend_sentiment
                raw_data['google_trends'] = trends_data
            except Exception as e:
                print(f"Google Trends sentiment error: {e}")
                sentiments['google_trends'] = None
        
        # Calculate composite sentiment
        composite = self.calculate_composite_sentiment(sentiments)
        
        # Add additional analysis
        analysis = {
            'coin_symbol': coin_symbol,
            'coin_name': coin_name,
            'composite_sentiment': composite,
            'raw_sentiments': sentiments,
            'raw_data': raw_data,
            'sentiment_classification': self.classify_sentiment(composite['composite_score']),
            'momentum_indicator': self.calculate_momentum(raw_data),
            'risk_assessment': self.assess_sentiment_risk(composite, raw_data)
        }
        
        return analysis
    
    def classify_sentiment(self, score: float) -> str:
        """Classify sentiment score into categories"""
        if score > 0.6:
            return 'extremely_bullish'
        elif score > 0.3:
            return 'bullish'
        elif score > 0.1:
            return 'slightly_bullish'
        elif score > -0.1:
            return 'neutral'
        elif score > -0.3:
            return 'slightly_bearish'
        elif score > -0.6:
            return 'bearish'
        else:
            return 'extremely_bearish'
    
    def calculate_momentum(self, raw_data: Dict) -> Dict:
        """Calculate sentiment momentum indicators"""
        momentum = {
            'direction': 'neutral',
            'strength': 0,
            'acceleration': 0
        }
        
        # Twitter momentum
        if 'twitter' in raw_data and raw_data['twitter']:
            twitter_data = raw_data['twitter']
            if 'volume_change' in twitter_data:
                momentum['twitter_volume_change'] = twitter_data['volume_change']
        
        # Reddit momentum
        if 'reddit' in raw_data and raw_data['reddit']:
            reddit_data = raw_data['reddit']
            if 'post_frequency_change' in reddit_data:
                momentum['reddit_activity_change'] = reddit_data['post_frequency_change']
        
        # Google Trends momentum
        if 'google_trends' in raw_data and raw_data['google_trends']:
            trends_data = raw_data['google_trends']
            if 'momentum' in trends_data:
                momentum['search_momentum'] = trends_data['momentum']
                momentum['search_change'] = trends_data.get('search_volume_change', 0)
        
        return momentum
    
    def assess_sentiment_risk(self, composite: Dict, raw_data: Dict) -> Dict:
        """Assess risk based on sentiment patterns"""
        risk_assessment = {
            'overall_risk': 'medium',
            'risk_factors': [],
            'risk_score': 0.5,  # 0 = low risk, 1 = high risk
            'warnings': []
        }
        
        risk_score = 0
        
        # Low confidence risk
        if composite['confidence'] < 0.5:
            risk_score += 0.2
            risk_assessment['risk_factors'].append('low_data_confidence')
        
        # Extreme sentiment risk
        if abs(composite['composite_score']) > 0.8:
            risk_score += 0.3
            risk_assessment['risk_factors'].append('extreme_sentiment')
            risk_assessment['warnings'].append('Extreme sentiment may indicate bubble or crash')
        
        # Sentiment divergence risk
        source_sentiments = [data.get('sentiment_score', 0) for data in raw_data.values() 
                           if isinstance(data, dict) and 'sentiment_score' in data]
        if len(source_sentiments) > 1:
            sentiment_std = np.std(source_sentiments)
            if sentiment_std > 0.5:
                risk_score += 0.2
                risk_assessment['risk_factors'].append('sentiment_divergence')
                risk_assessment['warnings'].append('Mixed signals across platforms')
        
        # Hype vs substance risk
        if 'google_trends' in raw_data and raw_data['google_trends']:
            hype_level = raw_data['google_trends'].get('hype_level', 'low')
            if hype_level in ['extreme', 'high'] and composite['composite_score'] > 0.5:
                risk_score += 0.2
                risk_assessment['risk_factors'].append('hype_bubble_risk')
                risk_assessment['warnings'].append('High hype may indicate unsustainable pump')
        
        # Set overall risk level
        risk_assessment['risk_score'] = min(1.0, risk_score)
        
        if risk_score > 0.7:
            risk_assessment['overall_risk'] = 'high'
        elif risk_score > 0.4:
            risk_assessment['overall_risk'] = 'medium'
        else:
            risk_assessment['overall_risk'] = 'low'
        
        return risk_assessment
    
    def batch_analyze_coins(self, coin_list: List[tuple]) -> List[Dict]:
        """Analyze sentiment for multiple coins"""
        results = []
        
        for coin_symbol, coin_name in coin_list:
            try:
                analysis = self.analyze_coin_sentiment(coin_symbol, coin_name)
                results.append(analysis)
            except Exception as e:
                print(f"Error analyzing {coin_symbol}: {e}")
                continue
        
        # Sort by composite sentiment score
        results.sort(key=lambda x: x['composite_sentiment']['composite_score'], reverse=True)
        return results
    
    def generate_sentiment_report(self, analysis: Dict) -> str:
        """Generate human-readable sentiment report"""
        composite = analysis['composite_sentiment']
        classification = analysis['sentiment_classification']
        risk = analysis['risk_assessment']
        
        report = f"""
SENTIMENT ANALYSIS REPORT: {analysis['coin_name']} ({analysis['coin_symbol']})
{'='*60}

OVERALL SENTIMENT: {classification.upper()}
Composite Score: {composite['composite_score']:.3f} (-1.0 to 1.0)
Confidence Level: {composite['confidence']:.2f}
Sources Analyzed: {composite['sources_used']}

RISK ASSESSMENT: {risk['overall_risk'].upper()}
Risk Score: {risk['risk_score']:.2f} (0.0 = low, 1.0 = high)
Risk Factors: {', '.join(risk['risk_factors']) if risk['risk_factors'] else 'None'}

SOURCE BREAKDOWN:
"""
        
        for source, data in composite['source_breakdown'].items():
            report += f"  {source.title()}: {data['sentiment']:.3f} (weight: {data['weight']:.2f})\n"
        
        if risk['warnings']:
            report += f"\nWARNINGS:\n"
            for warning in risk['warnings']:
                report += f"  ⚠️  {warning}\n"
        
        return report


def main():
    """Test the multi-source sentiment analyzer"""
    # This would be used with actual sentiment sources
    sentiment_analyzer = MultiSourceSentiment()
    
    # Mock sentiment data for demonstration
    mock_sentiments = {
        'twitter': 0.7,
        'reddit': 0.5,
        'discord': 0.8,
        'google_trends': 0.6
    }
    
    composite = sentiment_analyzer.calculate_composite_sentiment(mock_sentiments)
    print("Mock Composite Sentiment Analysis:")
    print(f"Score: {composite['composite_score']:.3f}")
    print(f"Confidence: {composite['confidence']:.2f}")
    print(f"Sources Used: {composite['sources_used']}")


if __name__ == "__main__":
    main()