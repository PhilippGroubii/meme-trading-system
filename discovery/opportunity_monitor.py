#!/usr/bin/env python3
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
        print(f"\n{emoji} {alert['message']}")
        
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
