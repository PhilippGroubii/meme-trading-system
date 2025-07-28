#!/usr/bin/env python3
"""
24/7 Opportunity Monitor
Continuously scans for new coins and trading opportunities
"""

import asyncio
import time
import json
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Set
import logging

from coin_discovery_engine import CoinDiscoveryEngine, EmergingCoin

class OpportunityMonitor:
    def __init__(self):
        self.discovery_engine = CoinDiscoveryEngine()
        self.monitoring = True
        self.scan_interval = 300  # 5 minutes

        # Tracking state
        self.watchlist: Set[str] = set()
        self.active_opportunities: Dict[str, EmergingCoin] = {}
        self.alert_history: List[Dict] = []

        # Alert thresholds
        self.alert_criteria = {
        'min_opportunity_score': 6.0,
        'max_risk_score': 6.0,
        'volume_spike_threshold': 3.0,  # 3x volume increase
        'price_spike_threshold': 0.5,   # 50% price increase
        }

        # Setup logging
        self.setup_logging()

    def setup_logging(self):
        """Setup logging for monitoring"""
        logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
        logging.FileHandler('logs/opportunity_monitor.log'),
        logging.StreamHandler()
        ]
        )
        self.logger = logging.getLogger(__name__)

    async def start_monitoring(self):
        """Start continuous monitoring"""
        self.logger.info("🚀 Starting 24/7 Opportunity Monitor")

        scan_count = 0

        while self.monitoring:
        try:
        scan_count += 1
        self.logger.info(f"📡 SCAN #{scan_count} - {datetime.now()}")

        # Discover new opportunities
        opportunities = await self.discovery_engine.discover_emerging_coins()

        # Process discoveries
        new_alerts = await self.process_opportunities(opportunities)

        # Monitor existing watchlist
        watchlist_alerts = await self.monitor_watchlist()

        # Send alerts
        all_alerts = new_alerts + watchlist_alerts
        if all_alerts:
        await self.send_alerts(all_alerts)

        # Update state
        self.update_monitoring_state(opportunities)

        # Log summary
        self.log_scan_summary(opportunities, all_alerts)

        # Wait for next scan
        await asyncio.sleep(self.scan_interval)

        except Exception as e:
        self.logger.error(f"❌ Monitoring error: {e}")
        await asyncio.sleep(60)  # Wait 1 minute on error

    async def process_opportunities(self, opportunities: List[EmergingCoin]) -> List[Dict]:
        """Process new opportunities and generate alerts"""
        alerts = []

        for coin in opportunities:
        # Check if this is a new high-priority opportunity
        if (coin.opportunity_score >= self.alert_criteria['min_opportunity_score'] and
        coin.risk_score <= self.alert_criteria['max_risk_score']):

        # Check if we haven't alerted on this coin recently
        if not self.recently_alerted(coin.symbol):
        alert = self.create_opportunity_alert(coin)
        alerts.append(alert)

        # Add to watchlist for monitoring
        self.watchlist.add(coin.symbol)
        self.active_opportunities[coin.symbol] = coin

        return alerts

    async def monitor_watchlist(self) -> List[Dict]:
        """Monitor coins in watchlist for significant changes"""
        alerts = []

        for symbol in list(self.watchlist):
        try:
        # Get current data for watchlist coin
        current_data = await self.get_current_coin_data(symbol)

        if current_data and symbol in self.active_opportunities:
        previous_coin = self.active_opportunities[symbol]

        # Check for volume spike
        if (current_data.get('volume_24h', 0) >
        previous_coin.volume_24h * self.alert_criteria['volume_spike_threshold']):

        alert = self.create_volume_spike_alert(symbol, current_data, previous_coin)
        alerts.append(alert)

        # Check for price spike
        price_change = ((current_data.get('price', 0) - previous_coin.price) /
        previous_coin.price if previous_coin.price > 0 else 0)

        if price_change >= self.alert_criteria['price_spike_threshold']:
        alert = self.create_price_spike_alert(symbol, current_data, price_change)
        alerts.append(alert)

        # Update tracking data
        if current_data:
        self.active_opportunities[symbol] = self.update_coin_from_data(
        previous_coin, current_data
        )

        except Exception as e:
        self.logger.warning(f"⚠️ Failed to monitor {symbol}: {e}")

        return alerts

    def create_opportunity_alert(self, coin: EmergingCoin) -> Dict:
        """Create new opportunity alert"""
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

    def create_volume_spike_alert(self, symbol: str, current_data: Dict,
        previous_coin: EmergingCoin) -> Dict:
        """Create volume spike alert"""
        volume_multiplier = current_data['volume_24h'] / previous_coin.volume_24h

        return {
        'type': 'VOLUME_SPIKE',
        'symbol': symbol,
        'volume_increase': volume_multiplier,
        'current_volume': current_data['volume_24h'],
        'previous_volume': previous_coin.volume_24h,
        'message': f"📈 VOLUME SPIKE: {symbol} - {volume_multiplier:.1f}x increase!",
        'priority': 'HIGH',
        'timestamp': datetime.now(),
        'recommended_action': 'MONITOR_CLOSELY'
        }

    def create_price_spike_alert(self, symbol: str, current_data: Dict,
        price_change: float) -> Dict:
        """Create price spike alert"""
        return {
        'type': 'PRICE_SPIKE',
        'symbol': symbol,
        'price_change': price_change,
        'current_price': current_data['price'],
        'message': f"🚀 PRICE SPIKE: {symbol} - +{price_change:.1%} pump!",
        'priority': 'URGENT',
        'timestamp': datetime.now(),
        'recommended_action': 'CONSIDER_ENTRY' if price_change < 1.0 else 'WAIT_FOR_DIP'
        }

    def get_recommended_action(self, coin: EmergingCoin) -> str:
        """Get recommended action for a coin"""
        if coin.opportunity_score >= 8 and coin.risk_score <= 4:
        return 'STRONG_BUY'
        elif coin.opportunity_score >= 6 and coin.risk_score <= 6:
        return 'CONSIDER_ENTRY'
        elif coin.opportunity_score >= 4:
        return 'MONITOR_CLOSELY'
        else:
        return 'WATCH_ONLY'

    async def get_current_coin_data(self, symbol: str) -> Optional[Dict]:
        """Get current data for a specific coin"""
        try:
        # This would make API calls to get current price/volume data
        # For now, simulate some data
import random

if symbol in self.active_opportunities:
base_coin = self.active_opportunities[symbol]

# Simulate price movement
price_change = random.gauss(0, 0.3)  # 30% volatility
new_price = base_coin.price * (1 + price_change)

# Simulate volume change
volume_change = random.gauss(0, 0.5)  # 50% volume volatility
new_volume = base_coin.volume_24h * (1 + volume_change)

return {
'price': max(new_price, 0.0001),
'volume_24h': max(new_volume, 1000),
'market_cap': max(new_price, 0.0001) * 1000000  # Rough estimate
}

except Exception as e:
self.logger.error(f"Failed to get data for {symbol}: {e}")

return None

def update_coin_from_data(self, coin: EmergingCoin, data: Dict) -> EmergingCoin:
"""Update coin object with new data"""
coin.price = data.get('price', coin.price)
coin.volume_24h = data.get('volume_24h', coin.volume_24h)
coin.market_cap = data.get('market_cap', coin.market_cap)
return coin

def recently_alerted(self, symbol: str, hours: int = 24) -> bool:
"""Check if we recently alerted on this symbol"""
cutoff_time = datetime.now() - timedelta(hours=hours)

for alert in self.alert_history:
if (alert.get('symbol') == symbol and
alert.get('timestamp', datetime.min) > cutoff_time):
return True

return False

async def send_alerts(self, alerts: List[Dict]):
"""Send alerts via various channels"""
for alert in alerts:
# Log alert
self.logger.info(f"🚨 ALERT: {alert['message']}")

# Add to history
self.alert_history.append(alert)

# Send via different channels
await self.send_console_alert(alert)
await self.send_file_alert(alert)
# await self.send_email_alert(alert)  # If configured
# await self.send_telegram_alert(alert)  # If configured
# await self.send_discord_alert(alert)  # If configured

async def send_console_alert(self, alert: Dict):
"""Send alert to console"""
priority_emoji = {
'URGENT': '🚨',
'HIGH': '⚡',
'MEDIUM': '📊',
'LOW': '👀'
}

emoji = priority_emoji.get(alert.get('priority', 'MEDIUM'), '📊')
print(f"\n{emoji} {alert['message']}")

if alert.get('recommended_action'):
print(f"   🎯 Recommendation: {alert['recommended_action']}")

if alert.get('type') == 'NEW_OPPORTUNITY':
print(f"   💰 Market Cap: ${alert['market_cap']:,.0f}")
print(f"   📊 Volume: ${alert['volume_24h']:,.0f}")
print(f"   ⏰ Age: {alert['age_hours']:.1f} hours")

async def send_file_alert(self, alert: Dict):
"""Save alert to file"""
try:
with open('logs/alerts.jsonl', 'a') as f:
alert_copy = alert.copy()
alert_copy['timestamp'] = alert_copy['timestamp'].isoformat()
f.write(json.dumps(alert_copy) + '\n')
except Exception as e:
self.logger.error(f"Failed to save alert: {e}")

def update_monitoring_state(self, opportunities: List[EmergingCoin]):
"""Update monitoring state with new data"""
# Remove old opportunities from watchlist (older than 7 days)
cutoff_time = datetime.now() - timedelta(days=7)

symbols_to_remove = []
for symbol, coin in self.active_opportunities.items():
coin_age = datetime.now() - timedelta(hours=coin.age_hours)
if coin_age < cutoff_time:
symbols_to_remove.append(symbol)

for symbol in symbols_to_remove:
self.watchlist.discard(symbol)
self.active_opportunities.pop(symbol, None)

# Clean old alert history (keep last 100)
if len(self.alert_history) > 100:
self.alert_history = self.alert_history[-100:]

def log_scan_summary(self, opportunities: List[EmergingCoin], alerts: List[Dict]):
"""Log summary of scan results"""
summary = f"📡 Scan Summary: {len(opportunities)} opportunities, {len(alerts)} alerts"

if opportunities:
top_opportunity = max(opportunities, key=lambda x: x.opportunity_score)
summary += f", Top: {top_opportunity.symbol} ({top_opportunity.opportunity_score:.1f}/10)"

summary += f", Watchlist: {len(self.watchlist)} coins"

self.logger.info(summary)

def stop_monitoring(self):
"""Stop the monitoring loop"""
self.monitoring = False
self.logger.info("🛑 Stopping opportunity monitor")

def get_monitoring_status(self) -> Dict:
"""Get current monitoring status"""
return {
'monitoring': self.monitoring,
'watchlist_size': len(self.watchlist),
'active_opportunities': len(self.active_opportunities),
'total_alerts': len(self.alert_history),
'recent_alerts': len([a for a in self.alert_history
if a.get('timestamp', datetime.min) >
datetime.now() - timedelta(hours=24)])
}

# Integration with trading system
class AutoTrader:
    """Automated trader that acts on discovered opportunities"""

    def __init__(self, paper_trading: bool = True):
        self.paper_trading = paper_trading
        self.monitor = OpportunityMonitor()
        self.active_trades = {}

    async def start_auto_trading(self):
        """Start automated trading on discovered opportunities"""
        print("🤖 Starting Auto-Trader with Opportunity Discovery")

        # Start monitoring in background
        monitor_task = asyncio.create_task(self.monitor.start_monitoring())

        # Process alerts as they come in
        await self.process_trading_signals()

    async def process_trading_signals(self):
        """Process trading signals from opportunity monitor"""
        while True:
        try:
        # Check for new high-priority alerts
        recent_alerts = [a for a in self.monitor.alert_history
        if a.get('timestamp', datetime.min) >
        datetime.now() - timedelta(minutes=5)]

        for alert in recent_alerts:
        if alert.get('recommended_action') == 'STRONG_BUY':
        await self.execute_auto_trade(alert)

        await asyncio.sleep(30)  # Check every 30 seconds

        except Exception as e:
        print(f"Auto-trading error: {e}")
        await asyncio.sleep(60)

    async def execute_auto_trade(self, alert: Dict):
        """Execute automated trade based on alert"""
        symbol = alert['symbol']

        if symbol not in self.active_trades:
        print(f"🤖 AUTO-TRADING: Executing trade on {symbol}")

        # This would integrate with your existing trading system
        # For now, just log the action
        trade_record = {
        'symbol': symbol,
        'action': 'AUTO_BUY',
        'timestamp': datetime.now(),
        'reason': alert['message'],
        'paper_trading': self.paper_trading
        }

        self.active_trades[symbol] = trade_record
        print(f"   ✅ {symbol} added to auto-trading portfolio")

        # Example usage
        if __name__ == "__main__":
    async def run_monitor():
        monitor = OpportunityMonitor()

        print("🚀 Starting Opportunity Monitor Demo")
        print("Press Ctrl+C to stop")

        try:
        await monitor.start_monitoring()
        except KeyboardInterrupt:
        monitor.stop_monitoring()
        print("\n👋 Monitor stopped")

        # Run the monitor
        asyncio.run(run_monitor())

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
