# utils/alerts.py
import smtplib
import logging
import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import requests
import time

@dataclass
class Alert:
    title: str
    message: str
    level: str  # 'info', 'warning', 'critical'
    timestamp: datetime
    data: Dict = None
    source: str = "TradingBot"
    category: str = "general"
    
    def to_dict(self) -> Dict:
        """Convert alert to dictionary for serialization"""
        alert_dict = asdict(self)
        alert_dict['timestamp'] = self.timestamp.isoformat()
        return alert_dict

class AlertSystem:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Email configuration
        self.smtp_server = config.get('email', {}).get('smtp_server', 'smtp.gmail.com')
        self.smtp_port = config.get('email', {}).get('smtp_port', 587)
        self.email_user = config.get('email', {}).get('user')
        self.email_password = config.get('email', {}).get('password')
        self.recipients = config.get('email', {}).get('recipients', [])
        
        # Discord webhook
        self.discord_webhook = config.get('discord', {}).get('webhook_url')
        
        # Telegram bot
        self.telegram_token = config.get('telegram', {}).get('bot_token')
        self.telegram_chat_id = config.get('telegram', {}).get('chat_id')
        
        # Slack webhook
        self.slack_webhook = config.get('slack', {}).get('webhook_url')
        
        # Alert levels that trigger notifications
        self.notification_levels = config.get('notification_levels', ['warning', 'critical'])
        
        # Rate limiting
        self.last_alert_times = {}
        self.min_alert_interval = config.get('min_alert_interval', 300)  # 5 minutes
        
        # Alert categories and their settings
        self.category_settings = config.get('categories', {
            'trade': {'min_interval': 60, 'max_per_hour': 10},
            'price': {'min_interval': 300, 'max_per_hour': 6},
            'system': {'min_interval': 600, 'max_per_hour': 3},
            'error': {'min_interval': 180, 'max_per_hour': 8}
        })
        
        # Alert history for analytics
        self.alert_history = []
        self.max_history = config.get('max_alert_history', 1000)
        
        # Delivery retry settings
        self.max_retries = config.get('max_retries', 3)
        self.retry_delay = config.get('retry_delay', 30)  # seconds
    
    def should_send_alert(self, alert_key: str, category: str = "general") -> bool:
        """Check if enough time has passed since last alert of this type"""
        now = datetime.now()
        last_time = self.last_alert_times.get(alert_key)
        
        # Get category-specific interval
        category_config = self.category_settings.get(category, {})
        min_interval = category_config.get('min_interval', self.min_alert_interval)
        
        if last_time is None:
            self.last_alert_times[alert_key] = now
            return True
        
        time_diff = (now - last_time).total_seconds()
        if time_diff >= min_interval:
            self.last_alert_times[alert_key] = now
            return True
        
        return False
    
    def check_rate_limits(self, category: str) -> bool:
        """Check if we're within rate limits for this category"""
        category_config = self.category_settings.get(category, {})
        max_per_hour = category_config.get('max_per_hour', 20)
        
        # Count alerts in the last hour for this category
        one_hour_ago = datetime.now() - timedelta(hours=1)
        recent_alerts = [
            alert for alert in self.alert_history 
            if alert.timestamp > one_hour_ago and alert.category == category
        ]
        
        return len(recent_alerts) < max_per_hour
    
    def send_alert(self, alert: Alert) -> bool:
        """Send alert through all configured channels"""
        try:
            # Check if we should send this alert
            if alert.level not in self.notification_levels:
                self.logger.info(f"Alert level '{alert.level}' not in notification levels")
                return False
            
            alert_key = f"{alert.category}_{alert.title}_{alert.level}"
            
            if not self.should_send_alert(alert_key, alert.category):
                self.logger.info(f"Rate limited: {alert_key}")
                return False
            
            if not self.check_rate_limits(alert.category):
                self.logger.warning(f"Rate limit exceeded for category: {alert.category}")
                return False
            
            # Add to history
            self.alert_history.append(alert)
            if len(self.alert_history) > self.max_history:
                self.alert_history.pop(0)
            
            # Log the alert
            self.logger.info(f"Sending alert: {alert.title} - {alert.message}")
            
            # Send through configured channels
            success_count = 0
            total_channels = 0
            
            if self.email_user and self.recipients:
                total_channels += 1
                if self._send_email_alert(alert):
                    success_count += 1
            
            if self.discord_webhook:
                total_channels += 1
                if self._send_discord_alert(alert):
                    success_count += 1
            
            if self.telegram_token and self.telegram_chat_id:
                total_channels += 1
                if self._send_telegram_alert(alert):
                    success_count += 1
            
            if self.slack_webhook:
                total_channels += 1
                if self._send_slack_alert(alert):
                    success_count += 1
            
            # Consider successful if at least one channel worked
            success = success_count > 0
            self.logger.info(f"Alert sent to {success_count}/{total_channels} channels")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error sending alert: {e}")
            return False
    
    def _send_email_alert(self, alert: Alert) -> bool:
        """Send alert via email with retry logic"""
        for attempt in range(self.max_retries):
            try:
                msg = MIMEMultipart()
                msg['From'] = self.email_user
                msg['To'] = ', '.join(self.recipients)
                msg['Subject'] = f"🚨 Trading Alert: {alert.title}"
                
                # Create HTML and text versions
                html_body = self._create_email_html(alert)
                text_body = self._create_email_text(alert)
                
                msg.attach(MIMEText(text_body, 'plain'))
                msg.attach(MIMEText(html_body, 'html'))
                
                # Add JSON attachment with alert data if present
                if alert.data:
                    json_data = json.dumps(alert.data, indent=2, default=str)
                    attachment = MIMEBase('application', 'json')
                    attachment.set_payload(json_data.encode('utf-8'))
                    encoders.encode_base64(attachment)
                    attachment.add_header(
                        'Content-Disposition',
                        f'attachment; filename="alert_data_{int(alert.timestamp.timestamp())}.json"'
                    )
                    msg.attach(attachment)
                
                server = smtplib.SMTP(self.smtp_server, self.smtp_port)
                server.starttls()
                server.login(self.email_user, self.email_password)
                server.send_message(msg)
                server.quit()
                
                self.logger.info("Email alert sent successfully")
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to send email alert (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
        
        return False
    
    def _send_discord_alert(self, alert: Alert) -> bool:
        """Send alert via Discord webhook"""
        for attempt in range(self.max_retries):
            try:
                color_map = {
                    'info': 0x3498db,      # Blue
                    'warning': 0xf39c12,   # Orange  
                    'critical': 0xe74c3c,  # Red
                    'success': 0x2ecc71    # Green
                }
                
                embed = {
                    "title": f"🚨 {alert.title}",
                    "description": alert.message,
                    "color": color_map.get(alert.level, 0x95a5a6),
                    "timestamp": alert.timestamp.isoformat(),
                    "footer": {
                        "text": f"Trading Bot • {alert.category.title()} Alert"
                    },
                    "fields": [
                        {
                            "name": "Level",
                            "value": alert.level.upper(),
                            "inline": True
                        },
                        {
                            "name": "Category",
                            "value": alert.category.title(),
                            "inline": True
                        }
                    ]
                }
                
                # Add data fields if present
                if alert.data:
                    for key, value in alert.data.items():
                        if len(embed["fields"]) < 25:  # Discord limit
                            embed["fields"].append({
                                "name": key.replace('_', ' ').title(),
                                "value": str(value)[:1024],  # Discord field value limit
                                "inline": True
                            })
                
                payload = {
                    "embeds": [embed],
                    "username": "Trading Bot",
                    "avatar_url": "https://cdn-icons-png.flaticon.com/512/2830/2830284.png"
                }
                
                response = requests.post(self.discord_webhook, json=payload, timeout=10)
                response.raise_for_status()
                
                self.logger.info("Discord alert sent successfully")
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to send Discord alert (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
        
        return False
    
    def _send_telegram_alert(self, alert: Alert) -> bool:
        """Send alert via Telegram"""
        for attempt in range(self.max_retries):
            try:
                # Format message with Markdown
                level_emoji = {
                    'info': 'ℹ️',
                    'warning': '⚠️',
                    'critical': '🚨',
                    'success': '✅'
                }
                
                message = f"{level_emoji.get(alert.level, '📢')} *{alert.title}*\n\n"
                message += f"*Level:* {alert.level.upper()}\n"
                message += f"*Category:* {alert.category.title()}\n"
                message += f"*Time:* {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                message += f"{alert.message}\n"
                
                if alert.data:
                    message += "\n*Additional Data:*\n"
                    for key, value in list(alert.data.items())[:10]:  # Limit to avoid message length issues
                        message += f"• *{key.replace('_', ' ').title()}:* {value}\n"
                
                # Telegram message length limit
                if len(message) > 4096:
                    message = message[:4090] + "..."
                
                url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
                payload = {
                    "chat_id": self.telegram_chat_id,
                    "text": message,
                    "parse_mode": "Markdown",
                    "disable_web_page_preview": True
                }
                
                response = requests.post(url, json=payload, timeout=10)
                response.raise_for_status()
                
                self.logger.info("Telegram alert sent successfully")
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to send Telegram alert (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
        
        return False
    
    def _send_slack_alert(self, alert: Alert) -> bool:
        """Send alert via Slack webhook"""
        for attempt in range(self.max_retries):
            try:
                color_map = {
                    'info': '#3498db',
                    'warning': '#f39c12',
                    'critical': '#e74c3c',
                    'success': '#2ecc71'
                }
                
                fields = [
                    {"title": "Level", "value": alert.level.upper(), "short": True},
                    {"title": "Category", "value": alert.category.title(), "short": True},
                    {"title": "Time", "value": alert.timestamp.strftime('%Y-%m-%d %H:%M:%S'), "short": True}
                ]
                
                if alert.data:
                    for key, value in list(alert.data.items())[:5]:  # Limit fields
                        fields.append({
                            "title": key.replace('_', ' ').title(),
                            "value": str(value),
                            "short": True
                        })
                
                payload = {
                    "attachments": [{
                        "color": color_map.get(alert.level, '#95a5a6'),
                        "title": f"🚨 {alert.title}",
                        "text": alert.message,
                        "fields": fields,
                        "footer": "Trading Bot",
                        "ts": int(alert.timestamp.timestamp())
                    }]
                }
                
                response = requests.post(self.slack_webhook, json=payload, timeout=10)
                response.raise_for_status()
                
                self.logger.info("Slack alert sent successfully")
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to send Slack alert (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
        
        return False
    
    def _create_email_html(self, alert: Alert) -> str:
        """Create HTML email body"""
        color_map = {
            'info': '#3498db',
            'warning': '#f39c12',
            'critical': '#e74c3c',
            'success': '#2ecc71'
        }
        
        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
                .alert-container {{ max-width: 600px; margin: 0 auto; border: 2px solid {color_map.get(alert.level, '#95a5a6')}; border-radius: 8px; }}
                .alert-header {{ background-color: {color_map.get(alert.level, '#95a5a6')}; color: white; padding: 15px; }}
                .alert-body {{ padding: 20px; }}
                .alert-data {{ background-color: #f8f9fa; padding: 15px; border-radius: 4px; margin-top: 15px; }}
                .data-item {{ margin: 5px 0; }}
                .footer {{ text-align: center; color: #6c757d; font-size: 12px; margin-top: 20px; }}
            </style>
        </head>
        <body>
            <div class="alert-container">
                <div class="alert-header">
                    <h2>🚨 {alert.title}</h2>
                    <p><strong>Level:</strong> {alert.level.upper()} | <strong>Category:</strong> {alert.category.title()}</p>
                </div>
                <div class="alert-body">
                    <p><strong>Time:</strong> {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
                    <p><strong>Message:</strong></p>
                    <p>{alert.message}</p>
        """
        
        if alert.data:
            html += '<div class="alert-data"><h4>Additional Data:</h4>'
            for key, value in alert.data.items():
                html += f'<div class="data-item"><strong>{key.replace("_", " ").title()}:</strong> {value}</div>'
            html += '</div>'
        
        html += """
                </div>
                <div class="footer">
                    <p>This alert was generated by the Trading Bot system.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _create_email_text(self, alert: Alert) -> str:
        """Create plain text email body"""
        text = f"""
TRADING ALERT: {alert.title}

Level: {alert.level.upper()}
Category: {alert.category.title()}
Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}

Message:
{alert.message}
"""
        
        if alert.data:
            text += "\nAdditional Data:\n"
            for key, value in alert.data.items():
                text += f"- {key.replace('_', ' ').title()}: {value}\n"
        
        text += "\n---\nThis alert was generated by the Trading Bot system."
        
        return text
    
    # Convenience methods for common alert types
    def price_alert(self, symbol: str, current_price: float, threshold: float, direction: str, data: Dict = None):
        """Send price-based alert"""
        alert = Alert(
            title=f"Price Alert: {symbol}",
            message=f"{symbol} price {direction} ${threshold:.6f}. Current: ${current_price:.6f}",
            level="warning",
            timestamp=datetime.now(),
            category="price",
            data={
                "symbol": symbol,
                "current_price": current_price,
                "threshold": threshold,
                "direction": direction,
                **(data or {})
            }
        )
        return self.send_alert(alert)
    
    def volume_alert(self, symbol: str, volume_change: float, data: Dict = None):
        """Send volume spike alert"""
        alert = Alert(
            title=f"Volume Spike: {symbol}",
            message=f"{symbol} volume increased by {volume_change:.2f}%",
            level="info",
            timestamp=datetime.now(),
            category="trade",
            data={
                "symbol": symbol,
                "volume_change": volume_change,
                **(data or {})
            }
        )
        return self.send_alert(alert)
    
    def trade_alert(self, trade_data: Dict):
        """Send trade execution alert"""
        alert = Alert(
            title=f"Trade Executed: {trade_data.get('symbol', 'Unknown')}",
            message=f"Executed {trade_data.get('side', 'unknown')} order for {trade_data.get('amount', 0)} {trade_data.get('symbol', '')} at ${trade_data.get('price', 0):.6f}",
            level="info",
            timestamp=datetime.now(),
            category="trade",
            data=trade_data
        )
        return self.send_alert(alert)
    
    def error_alert(self, error_type: str, error_message: str, data: Dict = None):
        """Send system error alert"""
        alert = Alert(
            title=f"System Error: {error_type}",
            message=error_message,
            level="critical",
            timestamp=datetime.now(),
            category="error",
            data={
                "error_type": error_type,
                "error_message": error_message,
                **(data or {})
            }
        )
        return self.send_alert(alert)
    
    def system_alert(self, title: str, message: str, level: str = "info", data: Dict = None):
        """Send general system alert"""
        alert = Alert(
            title=title,
            message=message,
            level=level,
            timestamp=datetime.now(),
            category="system",
            data=data
        )
        return self.send_alert(alert)
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get statistics about sent alerts"""
        if not self.alert_history:
            return {}
        
        now = datetime.now()
        last_hour = now - timedelta(hours=1)
        last_day = now - timedelta(days=1)
        
        # Count by time periods
        alerts_last_hour = [a for a in self.alert_history if a.timestamp > last_hour]
        alerts_last_day = [a for a in self.alert_history if a.timestamp > last_day]
        
        # Count by level
        level_counts = {}
        category_counts = {}
        
        for alert in self.alert_history:
            level_counts[alert.level] = level_counts.get(alert.level, 0) + 1
            category_counts[alert.category] = category_counts.get(alert.category, 0) + 1
        
        return {
            'total_alerts': len(self.alert_history),
            'alerts_last_hour': len(alerts_last_hour),
            'alerts_last_day': len(alerts_last_day),
            'by_level': level_counts,
            'by_category': category_counts,
            'oldest_alert': self.alert_history[0].timestamp.isoformat() if self.alert_history else None,
            'newest_alert': self.alert_history[-1].timestamp.isoformat() if self.alert_history else None
        }