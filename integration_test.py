#!/usr/bin/env python3
"""
Live Integration Test with Real Data Sources
Tests system with actual APIs and data feeds
"""

import time
import asyncio
from datetime import datetime
import json

# Test with real APIs (use your actual API keys)
class LiveIntegrationTest:
    """Test system with live data sources"""
    
    def __init__(self):
        self.test_results = {}
        
    async def test_real_data_sources(self):
        """Test with real API endpoints"""
        print("🌐 Testing Live Data Sources...")
        
        # Test CoinGecko API
        try:
            from data_sources.coingecko import CoinGeckoAPI
            cg = CoinGeckoAPI()
            doge_data = cg.get_coin_data('dogecoin')
            
            if doge_data and 'current_price' in doge_data:
                print(f"✅ CoinGecko: DOGE price ${doge_data['current_price']}")
                self.test_results['coingecko'] = True
            else:
                print("❌ CoinGecko: No data returned")
                self.test_results['coingecko'] = False
                
        except Exception as e:
            print(f"❌ CoinGecko error: {e}")
            self.test_results['coingecko'] = False
            
        # Test Reddit API
        try:
            from sentiment.reddit_scanner import RedditScanner
            reddit = RedditScanner()
            posts = reddit.scan_subreddit('dogecoin', limit=5)
            
            if posts and len(posts) > 0:
                print(f"✅ Reddit: Found {len(posts)} posts")
                self.test_results['reddit'] = True
            else:
                print("❌ Reddit: No posts found")
                self.test_results['reddit'] = False
                
        except Exception as e:
            print(f"❌ Reddit error: {e}")
            self.test_results['reddit'] = False
            
        # Add more real API tests as needed...
        
    def test_file_structure(self):
        """Verify all required files exist"""
        print("📁 Testing File Structure...")
        
        required_files = [
            'sentiment/reddit_scanner.py',
            'sentiment/multi_source.py', 
            'data_sources/coingecko.py',
            'ml_models/price_predictor.py',
            'trading_strategies/kelly_criterion.py',
            'risk_management/advanced_risk.py',
            'utils/logger.py'
        ]
        
        missing_files = []
        for file_path in required_files:
            try:
                with open(file_path, 'r') as f:
                    pass
                print(f"✅ Found: {file_path}")
            except FileNotFoundError:
                print(f"❌ Missing: {file_path}")
                missing_files.append(file_path)
                
        self.test_results['file_structure'] = len(missing_files) == 0
        return len(missing_files) == 0

if __name__ == "__main__":
    tester = LiveIntegrationTest()
    
    print("🧪 LIVE INTEGRATION TEST")
    print("=" * 40)
    
    # Test file structure first
    files_ok = tester.test_file_structure()
    
    if files_ok:
        # Test with real APIs
        asyncio.run(tester.test_real_data_sources())
    else:
        print("❌ Fix file structure before testing live APIs")
        
    print("\n📊 Integration Test Complete!")