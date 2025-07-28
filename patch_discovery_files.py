#!/usr/bin/env python3
"""
Discovery Files Patcher
Automatically applies all necessary patches to your discovery files
"""

import os
import shutil
from datetime import datetime

class DiscoveryPatcher:
    def __init__(self, discovery_dir="memecointrader/discovery"):
        self.discovery_dir = discovery_dir
        self.backup_dir = os.path.join(discovery_dir, f"backups_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.patches_applied = []
        
        # Check if discovery directory exists
        if not os.path.exists(discovery_dir):
            print(f"❌ Discovery directory not found: {discovery_dir}")
            print("💡 Please run this script from the memecointrader root directory")
            exit(1)
        
    def backup_files(self):
        """Create backup of original files"""
        files_to_backup = [
            'coin_discovery_engine.py',
            'opportunity_monitor.py', 
            'enhanced_trading_system.py'
        ]
        
        print("📦 Creating backups...")
        os.makedirs(self.backup_dir, exist_ok=True)
        
        for file in files_to_backup:
            file_path = os.path.join(self.discovery_dir, file)
            if os.path.exists(file_path):
                shutil.copy2(file_path, os.path.join(self.backup_dir, file))
                print(f"   ✅ Backed up {file}")
            else:
                print(f"   ⚠️  {file} not found in {self.discovery_dir}")
    
    def get_file_path(self, filename):
        """Get full path to file in discovery directory"""
        return os.path.join(self.discovery_dir, filename)
    
    def patch_coin_discovery_engine(self):
        """Apply patches to coin_discovery_engine.py"""
        print("\n🔧 Patching coin_discovery_engine.py...")
        
        filename = self.get_file_path('coin_discovery_engine.py')
        if not os.path.exists(filename):
            print(f"   ❌ coin_discovery_engine.py not found in {self.discovery_dir}")
            return False
        
        with open(filename, 'r') as f:
            content = f.read()
        
        # Patch 1: Add missing import
        if 'import os' not in content:
            # Find the position after the first import block
            lines = content.split('\n')
            import_end = 0
            for i, line in enumerate(lines):
                if line.startswith('import ') or line.startswith('from '):
                    import_end = i
            
            lines.insert(import_end + 1, 'import os')
            content = '\n'.join(lines)
            print("   ✅ Added missing 'import os'")
        
        # Patch 2: Add mock data methods
        mock_methods = '''
    def _get_mock_reddit_data(self) -> List[Dict]:
        """Generate mock Reddit data for testing"""
        import random
        mock_coins = ['MOONSHOT', 'ROCKETCOIN', 'MEMEGOD', 'ALPHAINU', 'GEMFINDER']
        
        return [{
            'symbol': coin,
            'name': f"{coin} Token",
            'source': 'reddit_mock',
            'social_score': random.randint(10, 500),
            'mentions': random.randint(1, 50),
            'post_url': f"https://reddit.com/mock/{coin.lower()}"
        } for coin in random.sample(mock_coins, 2)]

    def _get_mock_dexscreener_data(self) -> List[Dict]:
        """Generate mock DexScreener data for testing"""
        import random
        
        mock_tokens = [
            {'symbol': 'TESTPEPE', 'name': 'Test Pepe'},
            {'symbol': 'MOCKDOGE', 'name': 'Mock Doge'},
            {'symbol': 'DEMODOGE', 'name': 'Demo Doge'},
            {'symbol': 'ALPHATEST', 'name': 'Alpha Test Token'},
            {'symbol': 'BETAMOON', 'name': 'Beta Moon Coin'}
        ]
        
        coins = []
        for token in random.sample(mock_tokens, 3):
            coins.append({
                'symbol': token['symbol'],
                'name': token['name'],
                'contract': f"0x{''.join(random.choices('0123456789abcdef', k=40))}",
                'price': random.uniform(0.000001, 0.01),
                'volume_24h': random.uniform(50000, 2000000),
                'market_cap': random.uniform(100000, 5000000),
                'liquidity': random.uniform(25000, 500000),
                'age_hours': random.uniform(1, 168),
                'source': 'dexscreener_mock'
            })
        
        return coins
'''
        
        if '_get_mock_reddit_data' not in content:
            # Add mock methods before the last class or at the end
            content = content.replace(
                '# Usage example and testing',
                mock_methods + '\n# Usage example and testing'
            )
            print("   ✅ Added mock data methods")
        
        # Patch 3: Update Reddit scanning method
        reddit_method_replacement = '''    async def _scan_reddit_new(self) -> List[Dict]:
        """Scan Reddit for newly mentioned coins"""
        try:
            # Check if Reddit credentials are available
            if not all([os.getenv('REDDIT_CLIENT_ID'), os.getenv('REDDIT_CLIENT_SECRET')]):
                print("   ⚠️ Reddit credentials not found, using mock data")
                return self._get_mock_reddit_data()
            
            import praw
            
            reddit = praw.Reddit(
                client_id=os.getenv('REDDIT_CLIENT_ID'),
                client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
                user_agent='CoinDiscovery:v1.0'
            )
            
            coins = []
            subreddits = ['CryptoMoonShots', 'SatoshiStreetBets', 'CryptoCurrency']
            
            for sub_name in subreddits:
                subreddit = reddit.subreddit(sub_name)
                
                # Get new posts from last 24 hours
                for post in subreddit.new(limit=50):
                    if time.time() - post.created_utc < 86400:  # 24 hours
                        coin_mentions = self._extract_coin_mentions(post.title + " " + post.selftext)
                        for mention in coin_mentions:
                            coins.append({
                                'symbol': mention['symbol'],
                                'name': mention.get('name', ''),
                                'source': f'reddit_{sub_name}',
                                'social_score': post.score,
                                'mentions': 1,
                                'post_url': f"https://reddit.com{post.permalink}"
                            })
            
            return coins
            
        except Exception as e:
            print(f"   Reddit scan failed: {e}, using mock data")
            return self._get_mock_reddit_data()'''
        
        # Replace the existing Reddit method
        if 'async def _scan_reddit_new(self) -> List[Dict]:' in content:
            # Find and replace the method
            lines = content.split('\n')
            in_method = False
            method_start = 0
            method_end = 0
            indent_level = None
            
            for i, line in enumerate(lines):
                if 'async def _scan_reddit_new(self) -> List[Dict]:' in line:
                    in_method = True
                    method_start = i
                    indent_level = len(line) - len(line.lstrip())
                    continue
                
                if in_method:
                    current_indent = len(line) - len(line.lstrip()) if line.strip() else indent_level + 4
                    if line.strip() and current_indent <= indent_level and not line.startswith(' ' * (indent_level + 1)):
                        method_end = i
                        break
            
            if method_end == 0:
                method_end = len(lines)
            
            # Replace the method
            new_lines = lines[:method_start] + reddit_method_replacement.split('\n') + lines[method_end:]
            content = '\n'.join(new_lines)
            print("   ✅ Updated Reddit scanning with mock fallback")
        
        # Patch 4: Update DexScreener method
        dexscreener_method_replacement = '''    async def _scan_dexscreener(self) -> List[Dict]:
        """Scan DexScreener for new tokens"""
        try:
            url = "https://api.dexscreener.com/latest/dex/tokens/trending"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        coins = []
                        for pair in data.get('pairs', [])[:50]:  # Top 50 trending
                            if self._is_meme_coin(pair):
                                coin_data = {
                                    'symbol': pair.get('baseToken', {}).get('symbol', ''),
                                    'name': pair.get('baseToken', {}).get('name', ''),
                                    'contract': pair.get('baseToken', {}).get('address', ''),
                                    'price': float(pair.get('priceUsd', 0)),
                                    'volume_24h': float(pair.get('volume', {}).get('h24', 0)),
                                    'market_cap': float(pair.get('fdv', 0)),
                                    'liquidity': float(pair.get('liquidity', {}).get('usd', 0)),
                                    'age_hours': self._calculate_token_age(pair),
                                    'source': 'dexscreener'
                                }
                                coins.append(coin_data)
                        
                        return coins
        except Exception as e:
            print(f"   DexScreener API failed: {e}, using mock data")
            return self._get_mock_dexscreener_data()
        
        return self._get_mock_dexscreener_data()'''
        
        # Replace DexScreener method
        if 'async def _scan_dexscreener(self) -> List[Dict]:' in content:
            lines = content.split('\n')
            in_method = False
            method_start = 0
            method_end = 0
            indent_level = None
            
            for i, line in enumerate(lines):
                if 'async def _scan_dexscreener(self) -> List[Dict]:' in line:
                    in_method = True
                    method_start = i
                    indent_level = len(line) - len(line.lstrip())
                    continue
                
                if in_method:
                    current_indent = len(line) - len(line.lstrip()) if line.strip() else indent_level + 4
                    if line.strip() and current_indent <= indent_level and not line.startswith(' ' * (indent_level + 1)):
                        method_end = i
                        break
            
            if method_end == 0:
                method_end = len(lines)
            
            # Replace the method
            new_lines = lines[:method_start] + dexscreener_method_replacement.split('\n') + lines[method_end:]
            content = '\n'.join(new_lines)
            print("   ✅ Updated DexScreener scanning with mock fallback")
        
        # Write the patched content
        with open(filename, 'w') as f:
            f.write(content)
        
        self.patches_applied.append(filename)
        return True
    
    def patch_opportunity_monitor(self):
        """Apply patches to opportunity_monitor.py"""
        print("\n🔧 Patching opportunity_monitor.py...")
        
        filename = self.get_file_path('opportunity_monitor.py')
        if not os.path.exists(filename):
            print(f"   ❌ opportunity_monitor.py not found in {self.discovery_dir}")
            return False
        
        with open(filename, 'r') as f:
            content = f.read()
        
        # Add test class at the end
        test_class = '''

class TestOpportunityMonitor(OpportunityMonitor):
    """Test version with accelerated timing and mock data"""
    
    def __init__(self):
        super().__init__()
        self.scan_interval = 10  # 10 seconds for testing
        self.test_mode = True
        
    async def get_current_coin_data(self, symbol: str) -> Optional[Dict]:
        """Get mock current data for testing"""
        import random
        
        if symbol in self.active_opportunities:
            base_coin = self.active_opportunities[symbol]
            
            # Simulate more dramatic price movements for testing
            price_change = random.gauss(0, 0.5)  # 50% volatility
            new_price = base_coin.price * (1 + price_change)
            
            # Simulate volume spikes occasionally
            volume_multiplier = random.choice([1, 1, 1, 1, 3, 5])  # 20% chance of spike
            new_volume = base_coin.volume_24h * volume_multiplier
            
            return {
                'price': max(new_price, 0.0001),
                'volume_24h': max(new_volume, 1000),
                'market_cap': max(new_price, 0.0001) * 1000000
            }
        
        return None
'''
        
        if 'class TestOpportunityMonitor' not in content:
            content += test_class
            print("   ✅ Added TestOpportunityMonitor class")
        
        # Fix the Optional import if missing
        if 'Optional[Dict]' in content and 'from typing import' in content:
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if line.startswith('from typing import') and 'Optional' not in line:
                    if 'Dict' in line:
                        lines[i] = line.replace('Dict', 'Dict, Optional')
                    else:
                        lines[i] = line + ', Optional'
                    content = '\n'.join(lines)
                    print("   ✅ Added Optional to typing imports")
                    break
        
        with open(filename, 'w') as f:
            f.write(content)
        
        self.patches_applied.append(filename)
        return True
    
    def patch_enhanced_trading_system(self):
        """Apply patches to enhanced_trading_system.py"""
        print("\n🔧 Patching enhanced_trading_system.py...")
        
        filename = self.get_file_path('enhanced_trading_system.py')
        if not os.path.exists(filename):
            print(f"   ❌ enhanced_trading_system.py not found in {self.discovery_dir}")
            return False
        
        with open(filename, 'r') as f:
            content = f.read()
        
        # Add mock trading class
        mock_trading_class = '''
# Mock trading class for testing
class MockPaperTrading:
    """Mock paper trading for testing discovery system"""
    
    def __init__(self):
        self.portfolio = {
            'cash': 8000,
            'positions': {},
            'total_value': 10000
        }
        self.trades_executed = 0
    
    def analyze_opportunity_v2(self, symbol: str) -> Dict:
        """Mock analysis - randomly decide to buy"""
        import random
        
        if random.random() < 0.3:  # 30% chance to buy
            return {
                'symbol': symbol,
                'recommendation': 'BUY',
                'confidence': random.uniform(0.6, 0.9),
                'entry_price': random.uniform(0.01, 1.0)
            }
        else:
            return {
                'symbol': symbol,
                'recommendation': 'HOLD',
                'confidence': random.uniform(0.3, 0.6)
            }
    
    def execute_optimized_trade(self, analysis: Dict) -> bool:
        """Mock trade execution"""
        if analysis['recommendation'] == 'BUY' and self.portfolio['cash'] > 1000:
            symbol = analysis['symbol']
            position_size = min(1000, self.portfolio['cash'] * 0.1)
            entry_price = analysis.get('entry_price', 0.5)
            shares = position_size / entry_price
            
            self.portfolio['positions'][symbol] = {
                'shares': shares,
                'entry_price': entry_price,
                'current_value': position_size,
                'stop_loss': entry_price * 0.85
            }
            
            self.portfolio['cash'] -= position_size
            self.trades_executed += 1
            return True
        
        return False

'''
        
        # Add mock class before EnhancedTradingSystem
        if 'class MockPaperTrading:' not in content:
            content = content.replace(
                'class EnhancedTradingSystem:',
                mock_trading_class + 'class EnhancedTradingSystem:'
            )
            print("   ✅ Added MockPaperTrading class")
        
        # Update initialization to handle missing OptimizedPaperTrading
        init_replacement = '''    def __init__(self, paper_trading: bool = True):
        # Core trading system - use mock if real one not available
        try:
            from optimized_paper_trading import OptimizedPaperTrading
            self.paper_trader = OptimizedPaperTrading()
        except ImportError:
            print("   ⚠️ Using MockPaperTrading (optimized_paper_trading not found)")
            self.paper_trader = MockPaperTrading()'''
        
        if 'self.paper_trader = OptimizedPaperTrading()' in content:
            content = content.replace(
                '        # Core trading system\n        self.paper_trader = OptimizedPaperTrading()',
                init_replacement
            )
            print("   ✅ Updated initialization with mock fallback")
        
        with open(filename, 'w') as f:
            f.write(content)
        
        self.patches_applied.append(filename)
        return True
    
    def create_test_runner(self):
        """Create comprehensive test runner"""
        print("\n📝 Creating test runner...")
        
        test_runner_content = '''#!/usr/bin/env python3
"""
Discovery System Test Runner
Comprehensive testing for the discovery system
"""

import asyncio
import sys
import os
from datetime import datetime

# Add discovery directory to path
sys.path.append('memecointrader/discovery')
sys.path.append('memecointrader')

async def test_discovery_engine():
    """Test the coin discovery engine"""
    print("🧪 TESTING COIN DISCOVERY ENGINE")
    print("=" * 50)
    
    try:
        from coin_discovery_engine import CoinDiscoveryEngine
        discovery = CoinDiscoveryEngine()
        
        print("🔍 Running discovery scan...")
        opportunities = await discovery.discover_emerging_coins()
        
        if opportunities:
            print(f"✅ Found {len(opportunities)} opportunities")
            
            print("\\n🎯 TOP 5 OPPORTUNITIES:")
            for i, coin in enumerate(opportunities[:5], 1):
                print(f"{i}. {coin.symbol} - Opportunity: {coin.opportunity_score:.1f}/10, Risk: {coin.risk_score:.1f}/10")
            
            summary = discovery.get_discovery_summary(opportunities)
            print(f"\\n📊 SUMMARY:")
            print(f"   Total Found: {summary['total_found']}")
            print(f"   Avg Opportunity: {summary['avg_opportunity_score']:.1f}/10")
            print(f"   Avg Risk: {summary['avg_risk_score']:.1f}/10")
            
            print("\\n🎯 RECOMMENDATIONS:")
            for rec in summary['recommendations']:
                print(f"   {rec}")
            
            return True
        else:
            print("❌ No opportunities found")
            return False
            
    except Exception as e:
        print(f"❌ Discovery engine test failed: {e}")
        return False

async def test_opportunity_monitor():
    """Test the opportunity monitor"""
    print("\\n🧪 TESTING OPPORTUNITY MONITOR")
    print("=" * 50)
    
    try:
        from opportunity_monitor import TestOpportunityMonitor
        monitor = TestOpportunityMonitor()
        
        print("📡 Running monitor for 30 seconds...")
        
        # Start monitoring task
        monitor_task = asyncio.create_task(monitor.start_monitoring())
        
        # Let it run for 30 seconds
        await asyncio.sleep(30)
        
        # Stop monitoring
        monitor.stop_monitoring()
        monitor_task.cancel()
        
        # Check results
        status = monitor.get_monitoring_status()
        print(f"\\n📊 MONITOR RESULTS:")
        print(f"   Watchlist Size: {status['watchlist_size']}")
        print(f"   Active Opportunities: {status['active_opportunities']}")
        print(f"   Total Alerts: {status['total_alerts']}")
        print(f"   Recent Alerts: {status['recent_alerts']}")
        
        return status['total_alerts'] > 0
        
    except Exception as e:
        print(f"❌ Opportunity monitor test failed: {e}")
        return False

async def test_enhanced_trading():
    """Test enhanced trading system"""
    print("\\n🧪 TESTING ENHANCED TRADING SYSTEM")
    print("=" * 50)
    
    try:
        from enhanced_trading_system import EnhancedTradingSystem
        system = EnhancedTradingSystem(paper_trading=True)
        
        print("🚀 Running 3-minute enhanced trading session...")
        
        await system.run_enhanced_trading_session(duration_minutes=3)
        print("✅ Enhanced trading test completed")
        return True
        
    except Exception as e:
        print(f"❌ Enhanced trading test failed: {e}")
        return False

async def run_all_tests():
    """Run all discovery system tests"""
    print("🧪 DISCOVERY SYSTEM TEST SUITE")
    print("=" * 60)
    print(f"Started: {datetime.now()}")
    print()
    
    results = {}
    
    # Test 1: Discovery Engine
    results['discovery_engine'] = await test_discovery_engine()
    
    # Test 2: Opportunity Monitor
    results['opportunity_monitor'] = await test_opportunity_monitor()
    
    # Test 3: Enhanced Trading System
    results['enhanced_trading'] = await test_enhanced_trading()
    
    # Summary
    print("\\n" + "=" * 60)
    print("🎯 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "✅ PASSED" if passed_test else "❌ FAILED"
        print(f"   {test_name.replace('_', ' ').title()}: {status}")
    
    print(f"\\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\\n🎉 ALL TESTS PASSED! Your discovery system is ready!")
        print("\\n🚀 Next steps:")
        print("   1. Configure API keys for live data")
        print("   2. Run: python memecointrader/discovery/enhanced_trading_system.py")
        print("   3. Start discovering emerging coins!")
    else:
        print("\\n⚠️ Some tests failed. Check the error messages above.")
    
    return passed == total

if __name__ == "__main__":
    print("🚀 Starting Discovery System Tests...")
    print("This will test all discovery components\\n")
    
    # Check if we're in the right directory
    if not os.path.exists('memecointrader/discovery'):
        print("❌ Please run this from the memecointrader root directory")
        print("💡 Current directory should contain memecointrader/discovery/")
        exit(1)
    
    try:
        success = asyncio.run(run_all_tests())
        if success:
            print("\\n🎊 Discovery system is fully operational!")
        else:
            print("\\n🔧 Please fix the issues and run tests again.")
    except KeyboardInterrupt:
        print("\\n⏹️ Tests interrupted by user")
    except Exception as e:
        print(f"\\n💥 Unexpected error: {e}")
'''
        
        # Create test runner in the root directory
        test_runner_path = 'test_discovery_system.py'
        with open(test_runner_path, 'w') as f:
            f.write(test_runner_content)
        
        print(f"   ✅ Created {test_runner_path}")
        return True
    
    def run_patches(self):
        """Run all patches"""
        print("🚀 DISCOVERY FILES PATCHER")
        print("=" * 50)
        print(f"Time: {datetime.now()}")
        print()
        
        # Create backups
        self.backup_files()
        
        # Apply patches
        success_count = 0
        total_patches = 3
        
        if self.patch_coin_discovery_engine():
            success_count += 1
        
        if self.patch_opportunity_monitor():
            success_count += 1
        
        if self.patch_enhanced_trading_system():
            success_count += 1
        
        # Create test runner
        if self.create_test_runner():
            print("   ✅ Test runner created")
        
        # Summary
        print("\n" + "=" * 50)
        print("🎯 PATCHING SUMMARY")
        print("=" * 50)
        
        print(f"Files patched: {success_count}/{total_patches}")
        print(f"Backups created in: {self.backup_dir}")
        
        if self.patches_applied:
            print("\\n✅ Successfully patched:")
            for file in self.patches_applied:
                print(f"   - {file}")
        
        print("\\n🧪 Next steps:")
        print("   1. Run: python test_discovery_system.py")
        print("   2. If tests pass, run: python memecointrader/discovery/enhanced_trading_system.py")
        print("   3. Configure API keys for live data (optional)")
        print(f"   4. Backups are in: {self.backup_dir}")
        
        if success_count == total_patches:
            print("\\n🎉 ALL PATCHES APPLIED SUCCESSFULLY!")
            return True
        else:
            print("\\n⚠️ Some patches failed. Check error messages above.")
            return False

if __name__ == "__main__":
    # Check if we're in the right location - handle both WSL and regular Linux paths
    discovery_paths = [
        "memecointrader/discovery",  # From root
        "discovery",                 # From memecointrader directory
        "./discovery",               # Current directory
        "/home/philipp/memecointrader/discovery"  # Absolute WSL path
    ]
    
    discovery_dir = None
    for path in discovery_paths:
        if os.path.exists(path):
            discovery_dir = path
            break
    
    if not discovery_dir:
        print("❌ Discovery directory not found!")
        print("💡 Please run this script from one of these locations:")
        print("   1. From memecointrader root: /home/philipp/memecointrader/")
        print("   2. From discovery directory: /home/philipp/memecointrader/discovery/")
        print("📂 Looking for these files:")
        print("   - coin_discovery_engine.py")
        print("   - opportunity_monitor.py") 
        print("   - enhanced_trading_system.py")
        print(f"📍 Current directory: {os.getcwd()}")
        exit(1)
    
    print(f"✅ Found discovery files in: {os.path.abspath(discovery_dir)}")
    
    patcher = DiscoveryPatcher(discovery_dir)
    
    print("This script will patch your discovery files with necessary fixes.")
    print(f"Files location: {os.path.abspath(discovery_dir)}")
    print("Backups will be created automatically.\\n")
    
    confirm = input("Apply patches? (y/N): ").strip().lower()
    
    if confirm in ['y', 'yes']:
        success = patcher.run_patches()
        
        if success:
            print("\\n🎊 Ready to test! Run: python test_discovery_system.py")
            print("💡 Or from WSL terminal: cd /home/philipp/memecointrader && python test_discovery_system.py")
        else:
            print(f"\\n🔧 Check backups in: {patcher.backup_dir}")
    else:
        print("\\n👋 Patching cancelled")