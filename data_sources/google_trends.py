"""
Google Trends integration for cryptocurrency sentiment analysis
"""
import time
from typing import Dict, List, Optional, Tuple
import pandas as pd
from pytrends.request import TrendReq


class GoogleTrendsDataSource:
    def __init__(self, hl: str = 'en-US', tz: int = 360):
        """Initialize Google Trends client"""
        self.pytrends = TrendReq(hl=hl, tz=tz)
        self.rate_limit_delay = 2.0  # seconds between requests
        
    def get_crypto_interest(self, keywords: List[str], timeframe: str = 'today 7-d') -> pd.DataFrame:
        """Get Google Trends interest for crypto keywords"""
        try:
            time.sleep(self.rate_limit_delay)
            self.pytrends.build_payload(keywords, cat=0, timeframe=timeframe, geo='', gprop='')
            interest_df = self.pytrends.interest_over_time()
            
            if not interest_df.empty:
                interest_df = interest_df.drop('isPartial', axis=1, errors='ignore')
            
            return interest_df
            
        except Exception as e:
            print(f"Error fetching trends for {keywords}: {e}")
            return pd.DataFrame()
    
    def get_regional_interest(self, keyword: str) -> pd.DataFrame:
        """Get regional interest breakdown for a keyword"""
        try:
            time.sleep(self.rate_limit_delay)
            self.pytrends.build_payload([keyword], cat=0, timeframe='today 7-d', geo='', gprop='')
            regional_df = self.pytrends.interest_by_region(resolution='COUNTRY')
            return regional_df.sort_values(by=keyword, ascending=False)
            
        except Exception as e:
            print(f"Error fetching regional data for {keyword}: {e}")
            return pd.DataFrame()
    
    def get_related_queries(self, keyword: str) -> Dict:
        """Get related queries for a keyword"""
        try:
            time.sleep(self.rate_limit_delay)
            self.pytrends.build_payload([keyword], cat=0, timeframe='today 7-d', geo='', gprop='')
            related_queries = self.pytrends.related_queries()
            return related_queries.get(keyword, {})
            
        except Exception as e:
            print(f"Error fetching related queries for {keyword}: {e}")
            return {}
    
    def get_trending_searches(self, country: str = 'united_states') -> List[str]:
        """Get current trending searches"""
        try:
            time.sleep(self.rate_limit_delay)
            trending_searches = self.pytrends.trending_searches(pn=country)
            return trending_searches[0].tolist() if not trending_searches.empty else []
            
        except Exception as e:
            print(f"Error fetching trending searches: {e}")
            return []
    
    def analyze_meme_coin_hype(self, coin_name: str, coin_symbol: str) -> Dict:
        """Analyze Google Trends hype for a meme coin"""
        keywords = [coin_name, coin_symbol, f"{coin_name} crypto", f"{coin_symbol} coin"]
        
        # Filter out None values and limit to 5 keywords (Google Trends limit)
        keywords = [k for k in keywords if k][:5]
        
        analysis = {
            'coin_name': coin_name,
            'coin_symbol': coin_symbol,
            'trend_score': 0,
            'momentum': 'neutral',
            'regional_strength': {},
            'related_topics': [],
            'hype_level': 'low',
            'search_volume_change': 0
        }
        
        try:
            # Get interest over time
            interest_df = self.get_crypto_interest(keywords)
            
            if not interest_df.empty:
                # Calculate trend score (average of all keywords)
                recent_scores = []
                for keyword in keywords:
                    if keyword in interest_df.columns:
                        recent_values = interest_df[keyword].tail(3).values
                        if len(recent_values) > 0:
                            recent_scores.append(recent_values.mean())
                
                if recent_scores:
                    analysis['trend_score'] = sum(recent_scores) / len(recent_scores)
                
                # Calculate momentum (comparing recent vs older data)
                if len(interest_df) >= 6:
                    recent_avg = interest_df.tail(3).mean().mean()
                    older_avg = interest_df.head(3).mean().mean()
                    
                    if recent_avg > older_avg * 1.5:
                        analysis['momentum'] = 'bullish'
                        analysis['search_volume_change'] = ((recent_avg - older_avg) / older_avg) * 100
                    elif recent_avg < older_avg * 0.75:
                        analysis['momentum'] = 'bearish'
                        analysis['search_volume_change'] = ((recent_avg - older_avg) / older_avg) * 100
            
            # Get regional interest for main coin name
            regional_df = self.get_regional_interest(coin_name)
            if not regional_df.empty:
                analysis['regional_strength'] = regional_df.head(5).to_dict()[coin_name]
            
            # Get related queries
            related = self.get_related_queries(coin_name)
            if 'rising' in related and related['rising'] is not None:
                analysis['related_topics'] = related['rising']['query'].head(5).tolist()
            
            # Determine hype level
            if analysis['trend_score'] > 75:
                analysis['hype_level'] = 'extreme'
            elif analysis['trend_score'] > 50:
                analysis['hype_level'] = 'high'
            elif analysis['trend_score'] > 25:
                analysis['hype_level'] = 'medium'
            else:
                analysis['hype_level'] = 'low'
                
        except Exception as e:
            print(f"Error in meme coin analysis: {e}")
        
        return analysis
    
    def monitor_crypto_trends(self, coin_list: List[Tuple[str, str]]) -> List[Dict]:
        """Monitor trends for multiple coins"""
        results = []
        
        for coin_name, coin_symbol in coin_list:
            analysis = self.analyze_meme_coin_hype(coin_name, coin_symbol)
            results.append(analysis)
            
            # Rate limiting
            time.sleep(self.rate_limit_delay)
        
        # Sort by trend score
        results.sort(key=lambda x: x['trend_score'], reverse=True)
        return results
    
    def detect_emerging_trends(self, base_keywords: List[str]) -> List[str]:
        """Detect emerging crypto trends"""
        emerging = []
        
        try:
            # Get trending searches
            trending = self.get_trending_searches()
            
            # Filter for crypto-related terms
            crypto_terms = ['coin', 'crypto', 'token', 'blockchain', 'defi', 'nft']
            
            for trend in trending:
                trend_lower = trend.lower()
                if any(term in trend_lower for term in crypto_terms):
                    emerging.append(trend)
            
            # Check for rising related queries from base keywords
            for keyword in base_keywords:
                related = self.get_related_queries(keyword)
                if 'rising' in related and related['rising'] is not None:
                    rising_queries = related['rising']['query'].head(3).tolist()
                    emerging.extend(rising_queries)
                
                time.sleep(self.rate_limit_delay)
        
        except Exception as e:
            print(f"Error detecting emerging trends: {e}")
        
        return list(set(emerging))  # Remove duplicates
    
    def get_crypto_category_trends(self) -> Dict:
        """Get trends for different crypto categories"""
        categories = {
            'meme_coins': ['dogecoin', 'shiba inu', 'pepe coin'],
            'defi': ['uniswap', 'compound', 'aave'],
            'layer1': ['ethereum', 'solana', 'cardano'],
            'nft': ['opensea', 'nft marketplace', 'digital art']
        }
        
        category_trends = {}
        
        for category, keywords in categories.items():
            try:
                interest_df = self.get_crypto_interest(keywords)
                if not interest_df.empty:
                    # Calculate average trend score for category
                    recent_avg = interest_df.tail(3).mean().mean()
                    category_trends[category] = {
                        'score': recent_avg,
                        'top_keyword': interest_df.tail(1).mean().idxmax()
                    }
                
                time.sleep(self.rate_limit_delay)
                
            except Exception as e:
                print(f"Error analyzing {category}: {e}")
                category_trends[category] = {'score': 0, 'top_keyword': ''}
        
        return category_trends


def main():
    """Test the Google Trends data source"""
    trends_source = GoogleTrendsDataSource()
    
    # Test meme coin analysis
    print("Analyzing meme coin trends...")
    analysis = trends_source.analyze_meme_coin_hype("Dogecoin", "DOGE")
    print(f"DOGE Analysis:")
    print(f"  Trend Score: {analysis['trend_score']:.1f}")
    print(f"  Hype Level: {analysis['hype_level']}")
    print(f"  Momentum: {analysis['momentum']}")
    
    # Test category trends
    print("\nAnalyzing crypto category trends...")
    category_trends = trends_source.get_crypto_category_trends()
    for category, data in category_trends.items():
        print(f"{category}: {data['score']:.1f} (top: {data['top_keyword']})")


if __name__ == "__main__":
    main()