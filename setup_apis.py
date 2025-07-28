#!/usr/bin/env python3
"""
API Setup Script
Get your trading system connected to live data sources
"""

import os
import sys

def create_env_file():
    """Create .env file for API keys"""
    
    env_content = """# Meme Coin Trading System API Keys
# Get these from the respective platforms

# Reddit API (https://www.reddit.com/prefs/apps)
REDDIT_CLIENT_ID=Kx2A9-tnaN7UVR-ilUyvzg
REDDIT_CLIENT_SECRET=Fu_LR9CWJV8pIg-xDXB9XYfcJHI-tw
REDDIT_USER_AGENT=SentimentBot/1.0 by u/PhilippGroubii

# CoinGecko API (https://www.coingecko.com/en/api)
COINGECKO_API_KEY=CG-ae874mhL57ELErsighWwF3mP

# Twitter API (https://developer.twitter.com/)
TWITTER_BEARER_TOKEN=AAAAAAAAAAAAAAAAAAAAAOeDzgEAAAAAJABOyNmVFl7rLHSYyDoNsZ8L7Rw%3DFVzMDY1hl6Q9mStcXe1EO13PSsEcYiGWAkIXUyHDgMTgvtGQO9

# Discord Bot (https://discord.com/developers/applications)
DISCORD_BOT_TOKEN=7057ac3e3697aa208e696a809e73382496caaaab8616a119748ac5d93e3ac7e3

# Trading Settings
PAPER_TRADING=true
STARTING_CAPITAL=10000
MAX_POSITION_SIZE=0.20
"""
    
    with open('.env', 'w') as f:
        f.write(env_content)
    
    print("✅ Created .env file")
    print("📝 Edit .env file with your actual API keys")

def install_required_packages():
    """Install additional packages needed for live trading"""
    
    packages = [
        'python-dotenv',  # For environment variables
        'asyncpraw',      # Async Reddit API
        'tweepy',         # Twitter API
        'discord.py',     # Discord API
        'pandas-ta',      # Technical analysis
        'schedule',       # Task scheduling
    ]
    
    print("📦 Installing required packages...")
    
    for package in packages:
        try:
            import subprocess
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"✅ Installed {package}")
        except Exception as e:
            print(f"❌ Failed to install {package}: {e}")

def create_api_test_script():
    """Create script to test API connections"""
    
    test_script = '''#!/usr/bin/env python3
"""
API Connection Test
Test all your API connections before live trading
"""

import os
from dotenv import load_dotenv
load_dotenv()

def test_reddit_api():
    """Test Reddit API connection"""
    try:
        import praw
        
        reddit = praw.Reddit(
            client_id=os.getenv('REDDIT_CLIENT_ID'),
            client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
            user_agent=os.getenv('REDDIT_USER_AGENT', 'MemeTrader:v1.0')
        )
        
        # Test by getting hot posts from dogecoin subreddit
        subreddit = reddit.subreddit('dogecoin')
        posts = list(subreddit.hot(limit=5))
        
        print(f"✅ Reddit API: Found {len(posts)} posts")
        return True
        
    except Exception as e:
        print(f"❌ Reddit API: {e}")
        return False

def test_coingecko_api():
    """Test CoinGecko API connection"""
    try:
        import requests
        
        api_key = os.getenv('COINGECKO_API_KEY')
        
        if api_key:
            headers = {'x-cg-demo-api-key': api_key}
            url = 'https://api.coingecko.com/api/v3/simple/price'
        else:
            headers = {}
            url = 'https://api.coingecko.com/api/v3/simple/price'
        
        params = {
            'ids': 'dogecoin',
            'vs_currencies': 'usd',
            'include_24hr_change': 'true'
        }
        
        response = requests.get(url, headers=headers, params=params)
        data = response.json()
        
        if 'dogecoin' in data:
            price = data['dogecoin']['usd']
            print(f"✅ CoinGecko API: DOGE price ${price}")
            return True
        else:
            print(f"❌ CoinGecko API: Unexpected response")
            return False
            
    except Exception as e:
        print(f"❌ CoinGecko API: {e}")
        return False

def test_twitter_api():
    """Test Twitter API connection"""
    try:
        import tweepy
        
        bearer_token = os.getenv('TWITTER_BEARER_TOKEN')
        
        if not bearer_token:
            print("⚠️ Twitter API: No bearer token provided (optional)")
            return True
        
        client = tweepy.Client(bearer_token=bearer_token)
        
        # Test by searching for recent dogecoin tweets
        tweets = client.search_recent_tweets(
            query='dogecoin', 
            max_results=10
        )
        
        if tweets.data:
            print(f"✅ Twitter API: Found {len(tweets.data)} tweets")
            return True
        else:
            print(f"❌ Twitter API: No tweets found")
            return False
            
    except Exception as e:
        print(f"❌ Twitter API: {e}")
        return False

if __name__ == "__main__":
    print("🔍 TESTING API CONNECTIONS")
    print("=" * 40)
    
    results = []
    
    results.append(test_reddit_api())
    results.append(test_coingecko_api())
    results.append(test_twitter_api())
    
    success_rate = sum(results) / len(results)
    
    print(f"\\n📊 API Test Results: {success_rate:.0%} successful")
    
    if success_rate >= 0.5:
        print("✅ Ready for live data collection!")
    else:
        print("❌ Please fix API keys before proceeding")
        print("💡 Edit the .env file with your actual API keys")
'''
    
    with open('test_apis.py', 'w') as f:
        f.write(test_script)
    
    print("✅ Created test_apis.py")

def print_api_instructions():
    """Print instructions for getting API keys"""
    
    instructions = """
🔑 API KEY SETUP INSTRUCTIONS
================================

1. REDDIT API (Required for sentiment analysis):
   • Go to: https://www.reddit.com/prefs/apps
   • Click "Create App"
   • Choose "script" type
   • Note down client_id and client_secret

2. COINGECKO API (Required for price data):
   • Go to: https://www.coingecko.com/en/api
   • Sign up for free account
   • Get your API key from dashboard
   • Free tier: 30 calls/minute (sufficient for testing)

3. TWITTER API (Optional, for enhanced sentiment):
   • Go to: https://developer.twitter.com/
   • Apply for developer account
   • Create new app and get Bearer Token
   • Essential tier is free (500k tweets/month)

4. DISCORD API (Optional, for community monitoring):
   • Go to: https://discord.com/developers/applications
   • Create new application
   • Go to "Bot" section and create bot
   • Copy bot token

📝 NEXT STEPS:
1. Get at least Reddit + CoinGecko APIs
2. Edit .env file with your keys
3. Run: python test_apis.py
4. Start paper trading: python paper_trading.py
"""
    
    print(instructions)

def main():
    """Main setup function"""
    
    print("🚀 MEME COIN TRADING SYSTEM - API SETUP")
    print("=" * 50)
    
    # Check if .env already exists
    if os.path.exists('.env'):
        overwrite = input("📁 .env file exists. Overwrite? (y/n): ").lower()
        if overwrite != 'y':
            print("⏭️ Skipping .env creation")
        else:
            create_env_file()
    else:
        create_env_file()
    
    # Install packages
    install_packages = input("📦 Install required packages? (y/n): ").lower()
    if install_packages == 'y':
        install_required_packages()
    
    # Create test script
    create_api_test_script()
    
    # Print instructions
    print_api_instructions()
    
    print("\n✅ API SETUP COMPLETE!")
    print("\n🎯 Next steps:")
    print("1. Edit .env file with your API keys")
    print("2. Run: python test_apis.py")
    print("3. Start paper trading: python paper_trading.py")

if __name__ == "__main__":
    main()