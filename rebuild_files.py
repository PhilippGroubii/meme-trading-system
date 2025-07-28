#!/usr/bin/env python3
"""
Rebuild Discovery Files from Scratch
Create completely clean, working files
"""

def create_clean_coin_discovery():
    """Create a clean coin_discovery_engine.py"""
    print("🔧 Creating clean coin_discovery_engine.py...")
    
    clean_content = '''#!/usr/bin/env python3
"""
Emerging Coin Discovery Engine
"""

import requests
import json
import time
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import re
import asyncio
import aiohttp
from dataclasses import dataclass

@dataclass
class EmergingCoin:
    symbol: str
    name: str
    contract_address: str
    market_cap: float
    volume_24h: float
    price: float
    age_hours: float
    liquidity: float
    holder_count: int
    social_score: float
    risk_score: float
    opportunity_score: float

class CoinDiscoveryEngine:
    def __init__(self):
        self.discovered_coins = {}
        self.tracking_coins = {}
        self.blacklist = set()
        
        # Discovery sources
        self.sources = {
            'dexscreener': self._scan_dexscreener,
            'reddit_new': self._scan_reddit_new,
        }
        
        # Relaxed criteria for testing
        self.criteria = {
            'max_market_cap': 50_000_000,
            'min_volume_24h': 10_000,
            'max_age_hours': 999,
            'min_liquidity': 1_000,
            'min_holders': 10,
            'max_risk_score': 10.0,
            'min_social_score': 0.1
        }
    
    async def discover_emerging_coins(self) -> List[EmergingCoin]:
        """Main discovery function"""
        print("🔍 SCANNING FOR EMERGING COINS...")
        
        all_coins = []
        
        # Scan all sources
        for source_name, source_func in self.sources.items():
            print(f"   📡 Scanning {source_name}...")
            try:
                result = await source_func()
                all_coins.extend(result)
            except Exception as e:
                print(f"   ❌ {source_name} failed: {e}")
        
        # Process results
        unique_coins = self._deduplicate_coins(all_coins)
        analyzed_coins = await self._analyze_coins(unique_coins)
        filtered_coins = self._filter_opportunities(analyzed_coins)
        
        print(f"✅ Discovery complete: {len(filtered_coins)} opportunities found")
        return filtered_coins
    
    async def _scan_dexscreener(self) -> List[Dict]:
        """Scan DexScreener - returns mock data"""
        print("   Using mock DexScreener data")
        return [
            {
                'symbol': 'TESTPEPE',
                'name': 'Test Pepe',
                'contract': '0x1234567890abcdef1234567890abcdef12345678',
                'price': 0.00123,
                'volume_24h': 150000,
                'market_cap': 800000,
                'liquidity': 75000,
                'age_hours': 18,
                'source': 'dexscreener_mock',
                'holder_count': 450
            },
            {
                'symbol': 'MOCKDOGE',
                'name': 'Mock Doge',
                'contract': '0xabcdef1234567890abcdef1234567890abcdef12',
                'price': 0.00567,
                'volume_24h': 280000,
                'market_cap': 1500000,
                'liquidity': 120000,
                'age_hours': 36,
                'source': 'dexscreener_mock',
                'holder_count': 780
            }
        ]
    
    async def _scan_reddit_new(self) -> List[Dict]:
        """Scan Reddit - returns mock data"""
        print("   Using mock Reddit data")
        return [
            {
                'symbol': 'MOONSHOT',
                'name': 'Moon Shot Token',
                'source': 'reddit_mock',
                'social_score': 85,
                'mentions': 25,
                'volume_24h': 95000,
                'market_cap': 650000,
                'liquidity': 45000,
                'age_hours': 12,
                'holder_count': 320,
                'price': 0.00089,
                'contract': '0x9876543210fedcba9876543210fedcba98765432'
            }
        ]
    
    def _deduplicate_coins(self, coins: List[Dict]) -> List[Dict]:
        """Remove duplicates"""
        seen = set()
        unique_coins = []
        
        for coin in coins:
            identifier = coin.get('symbol', '') + coin.get('contract', '')
            if identifier not in seen and identifier:
                seen.add(identifier)
                unique_coins.append(coin)
        
        return unique_coins
    
    async def _analyze_coins(self, coins: List[Dict]) -> List[EmergingCoin]:
        """Analyze coins"""
        analyzed_coins = []
        
        for coin_data in coins:
            try:
                enhanced_data = await self._enhance_coin_data(coin_data)
                
                opportunity_score = self._calculate_opportunity_score(enhanced_data)
                risk_score = self._calculate_risk_score(enhanced_data)
                social_score = self._calculate_social_score(enhanced_data)
                
                coin = EmergingCoin(
                    symbol=enhanced_data.get('symbol', ''),
                    name=enhanced_data.get('name', ''),
                    contract_address=enhanced_data.get('contract', ''),
                    market_cap=enhanced_data.get('market_cap', 0),
                    volume_24h=enhanced_data.get('volume_24h', 0),
                    price=enhanced_data.get('price', 0),
                    age_hours=enhanced_data.get('age_hours', 0),
                    liquidity=enhanced_data.get('liquidity', 0),
                    holder_count=enhanced_data.get('holder_count', 0),
                    social_score=social_score,
                    risk_score=risk_score,
                    opportunity_score=opportunity_score
                )
                
                analyzed_coins.append(coin)
                
            except Exception as e:
                print(f"   ⚠️ Analysis failed for {coin_data.get('symbol', 'unknown')}: {e}")
        
        return analyzed_coins
    
    async def _enhance_coin_data(self, coin_data: Dict) -> Dict:
        """Enhance coin data"""
        enhanced = coin_data.copy()
        enhanced.setdefault('holder_count', 500)
        enhanced.setdefault('age_hours', 24)
        return enhanced
    
    def _calculate_opportunity_score(self, coin_data: Dict) -> float:
        """Calculate opportunity score"""
        score = 5.0  # Base score
        
        volume = coin_data.get('volume_24h', 0)
        if volume > 100000:
            score += 2
        elif volume > 50000:
            score += 1
        
        mcap = coin_data.get('market_cap', 0)
        if mcap < 1000000:
            score += 2
        
        return min(score, 10)
    
    def _calculate_risk_score(self, coin_data: Dict) -> float:
        """Calculate risk score"""
        risk = 3.0  # Base risk
        
        liquidity = coin_data.get('liquidity', 0)
        if liquidity < 50000:
            risk += 1
        
        return min(risk, 10)
    
    def _calculate_social_score(self, coin_data: Dict) -> float:
        """Calculate social score"""
        score = 2.0  # Base score
        
        if coin_data.get('source', '').startswith('reddit'):
            reddit_score = coin_data.get('social_score', 0)
            if reddit_score > 50:
                score += 3
        
        return min(score, 10)
    
    def _filter_opportunities(self, coins: List[EmergingCoin]) -> List[EmergingCoin]:
        """Filter opportunities"""
        filtered = []
        
        for coin in coins:
            if (coin.market_cap <= self.criteria['max_market_cap'] and
                coin.volume_24h >= self.criteria['min_volume_24h'] and
                coin.age_hours <= self.criteria['max_age_hours'] and
                coin.liquidity >= self.criteria['min_liquidity'] and
                coin.holder_count >= self.criteria['min_holders'] and
                coin.risk_score <= self.criteria['max_risk_score'] and
                coin.social_score >= self.criteria['min_social_score'] and
                coin.symbol not in self.blacklist):
                
                filtered.append(coin)
        
        filtered.sort(key=lambda x: x.opportunity_score, reverse=True)
        return filtered[:10]
    
    def get_discovery_summary(self, coins: List[EmergingCoin]) -> Dict:
        """Generate summary"""
        if not coins:
            return {
                'total_found': 0,
                'message': 'No opportunities found',
                'avg_opportunity_score': 0,
                'avg_risk_score': 0,
                'recommendations': []
            }
        
        top_coin = coins[0]
        avg_opportunity = sum(c.opportunity_score for c in coins) / len(coins)
        avg_risk = sum(c.risk_score for c in coins) / len(coins)
        
        return {
            'total_found': len(coins),
            'top_opportunity': {
                'symbol': top_coin.symbol,
                'opportunity_score': top_coin.opportunity_score,
                'market_cap': top_coin.market_cap,
                'age_hours': top_coin.age_hours
            },
            'avg_opportunity_score': avg_opportunity,
            'avg_risk_score': avg_risk,
            'recommendations': self._generate_recommendations(coins[:3])
        }
    
    def _generate_recommendations(self, top_coins: List[EmergingCoin]) -> List[str]:
        """Generate recommendations"""
        recommendations = []
        
        for coin in top_coins:
            if coin.opportunity_score >= 7:
                recommendations.append(f"🚀 HIGH PRIORITY: {coin.symbol} - Score: {coin.opportunity_score:.1f}/10")
            elif coin.opportunity_score >= 5:
                recommendations.append(f"⚡ MONITOR: {coin.symbol} - Potential entry")
            else:
                recommendations.append(f"👀 WATCH: {coin.symbol} - Developing")
        
        return recommendations

if __name__ == "__main__":
    async def test_discovery():
        discovery = CoinDiscoveryEngine()
        
        print("🔍 Testing discovery...")
        opportunities = await discovery.discover_emerging_coins()
        
        if opportunities:
            print(f"\\nFound {len(opportunities)} opportunities:")
            for coin in opportunities:
                print(f"  {coin.symbol} - Score: {coin.opportunity_score:.1f}/10")
        else:
            print("No opportunities found")
    
    asyncio.run(test_discovery())
'''
    
    with open('discovery/coin_discovery_engine.py', 'w') as f:
        f.write(clean_content)
    
    print("   ✅ Created clean coin_discovery_engine.py")

def create_clean_opportunity_monitor():
    """Create a clean opportunity_monitor.py"""
    print("🔧 Creating clean opportunity_monitor.py...")
    
    clean_content = '''#!/usr/bin/env python3
"""
Opportunity Monitor
"""

import asyncio
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Set, Optional
import logging

from coin_discovery_engine import CoinDiscoveryEngine, EmergingCoin

class OpportunityMonitor:
    def __init__(self):
        self.discovery_engine = CoinDiscoveryEngine()
        self.monitoring = True
        self.scan_interval = 60
        
        self.watchlist: Set[str] = set()
        self.active_opportunities: Dict[str, EmergingCoin] = {}
        self.alert_history: List[Dict] = []
        
        self.alert_criteria = {
            'min_opportunity_score': 6.0,
            'max_risk_score': 6.0,
            'volume_spike_threshold': 3.0,
            'price_spike_threshold': 0.5,
        }
        
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler()]
        )
        self.logger = logging.getLogger(__name__)
    
    async def start_monitoring(self):
        """Start monitoring"""
        self.logger.info("🚀 Starting Opportunity Monitor")
        
        scan_count = 0
        
        while self.monitoring:
            try:
                scan_count += 1
                self.logger.info(f"📡 SCAN #{scan_count} - {datetime.now()}")
                
                opportunities = await self.discovery_engine.discover_emerging_coins()
                
                new_alerts = await self.process_opportunities(opportunities)
                
                if new_alerts:
                    await self.send_alerts(new_alerts)
                
                self.update_monitoring_state(opportunities)
                
                self.log_scan_summary(opportunities, new_alerts)
                
                await asyncio.sleep(self.scan_interval)
                
            except Exception as e:
                self.logger.error(f"❌ Monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def process_opportunities(self, opportunities: List[EmergingCoin]) -> List[Dict]:
        """Process opportunities"""
        alerts = []
        
        for coin in opportunities:
            if (coin.opportunity_score >= self.alert_criteria['min_opportunity_score'] and
                coin.risk_score <= self.alert_criteria['max_risk_score']):
                
                if not self.recently_alerted(coin.symbol):
                    alert = self.create_opportunity_alert(coin)
                    alerts.append(alert)
                    
                    self.watchlist.add(coin.symbol)
                    self.active_opportunities[coin.symbol] = coin
        
        return alerts
    
    def create_opportunity_alert(self, coin: EmergingCoin) -> Dict:
        """Create alert"""
        return {
            'type': 'NEW_OPPORTUNITY',
            'symbol': coin.symbol,
            'name': coin.name,
            'opportunity_score': coin.opportunity_score,
            'risk_score': coin.risk_score,
            'market_cap': coin.market_cap,
            'volume_24h': coin.volume_24h,
            'age_hours': coin.age_hours,
            'message': f"🚀 NEW OPPORTUNITY: {coin.symbol} - Score: {coin.opportunity_score:.1f}/10",
            'priority': 'HIGH' if coin.opportunity_score >= 8 else 'MEDIUM',
            'timestamp': datetime.now(),
            'recommended_action': self.get_recommended_action(coin)
        }
    
    def get_recommended_action(self, coin: EmergingCoin) -> str:
        """Get recommendation"""
        if coin.opportunity_score >= 8 and coin.risk_score <= 4:
            return 'STRONG_BUY'
        elif coin.opportunity_score >= 6 and coin.risk_score <= 6:
            return 'CONSIDER_ENTRY'
        elif coin.opportunity_score >= 4:
            return 'MONITOR_CLOSELY'
        else:
            return 'WATCH_ONLY'
    
    def recently_alerted(self, symbol: str, hours: int = 24) -> bool:
        """Check recent alerts"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        for alert in self.alert_history:
            if (alert.get('symbol') == symbol and 
                alert.get('timestamp', datetime.min) > cutoff_time):
                return True
        
        return False
    
    async def send_alerts(self, alerts: List[Dict]):
        """Send alerts"""
        for alert in alerts:
            self.logger.info(f"🚨 ALERT: {alert['message']}")
            self.alert_history.append(alert)
            await self.send_console_alert(alert)
    
    async def send_console_alert(self, alert: Dict):
        """Console alert"""
        priority_emoji = {
            'HIGH': '⚡',
            'MEDIUM': '📊',
            'LOW': '👀'
        }
        
        emoji = priority_emoji.get(alert.get('priority', 'MEDIUM'), '📊')
        print(f"\\n{emoji} {alert['message']}")
        
        if alert.get('recommended_action'):
            print(f"   🎯 Recommendation: {alert['recommended_action']}")
    
    def update_monitoring_state(self, opportunities: List[EmergingCoin]):
        """Update state"""
        cutoff_time = datetime.now() - timedelta(days=7)
        
        symbols_to_remove = []
        for symbol, coin in self.active_opportunities.items():
            coin_age = datetime.now() - timedelta(hours=coin.age_hours)
            if coin_age < cutoff_time:
                symbols_to_remove.append(symbol)
        
        for symbol in symbols_to_remove:
            self.watchlist.discard(symbol)
            self.active_opportunities.pop(symbol, None)
        
        if len(self.alert_history) > 100:
            self.alert_history = self.alert_history[-100:]
    
    def log_scan_summary(self, opportunities: List[EmergingCoin], alerts: List[Dict]):
        """Log summary"""
        summary = f"📡 Scan Summary: {len(opportunities)} opportunities, {len(alerts)} alerts"
        
        if opportunities:
            top_opportunity = max(opportunities, key=lambda x: x.opportunity_score)
            summary += f", Top: {top_opportunity.symbol} ({top_opportunity.opportunity_score:.1f}/10)"
        
        summary += f", Watchlist: {len(self.watchlist)} coins"
        
        self.logger.info(summary)
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring = False
        self.logger.info("🛑 Stopping opportunity monitor")
    
    def get_monitoring_status(self) -> Dict:
        """Get status"""
        return {
            'monitoring': self.monitoring,
            'watchlist_size': len(self.watchlist),
            'active_opportunities': len(self.active_opportunities),
            'total_alerts': len(self.alert_history),
            'recent_alerts': len([a for a in self.alert_history 
                                if a.get('timestamp', datetime.min) > 
                                datetime.now() - timedelta(hours=24)])
        }

class TestOpportunityMonitor(OpportunityMonitor):
    """Test version"""
    
    def __init__(self):
        super().__init__()
        self.scan_interval = 10
        self.test_mode = True

if __name__ == "__main__":
    async def test_monitor():
        monitor = TestOpportunityMonitor()
        
        print("🚀 Testing monitor for 30 seconds...")
        
        monitor_task = asyncio.create_task(monitor.start_monitoring())
        
        await asyncio.sleep(30)
        
        monitor.stop_monitoring()
        try:
            monitor_task.cancel()
            await monitor_task
        except asyncio.CancelledError:
            pass
        
        status = monitor.get_monitoring_status()
        print(f"Monitor results: {status}")
    
    asyncio.run(test_monitor())
'''
    
    with open('discovery/opportunity_monitor.py', 'w') as f:
        f.write(clean_content)
    
    print("   ✅ Created clean opportunity_monitor.py")

def main():
    print("🔧 REBUILDING DISCOVERY FILES FROM SCRATCH")
    print("=" * 50)
    
    import os
    if not os.path.exists('discovery'):
        print("❌ Run from /home/philipp/memecointrader")
        return
    
    # Backup broken files
    import shutil
    import datetime
    backup_suffix = datetime.datetime.now().strftime("_%Y%m%d_%H%M%S_broken")
    
    if os.path.exists('discovery/coin_discovery_engine.py'):
        shutil.copy('discovery/coin_discovery_engine.py', f'discovery/coin_discovery_engine{backup_suffix}.py')
        
    if os.path.exists('discovery/opportunity_monitor.py'):
        shutil.copy('discovery/opportunity_monitor.py', f'discovery/opportunity_monitor{backup_suffix}.py')
    
    # Create clean files
    create_clean_coin_discovery()
    create_clean_opportunity_monitor()
    
    print("\\n✅ ALL FILES REBUILT!")
    print("\\n🧪 Test everything:")
    print("   python test_discovery.py")
    print("\\n🚀 Run enhanced trading:")
    print("   python discovery/enhanced_trading_system.py")

if __name__ == "__main__":
    main()
