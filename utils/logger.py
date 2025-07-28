# utils/logger.py
import logging
import os
import json
import gzip
import shutil
from datetime import datetime, timedelta
from typing import Dict, Optional, Any
from pathlib import Path

class TradingLogger:
    def __init__(self, log_dir: str = "logs", enable_console: bool = True, log_level: str = "INFO", max_file_size: int = 10485760, backup_count: int = 5):
        self.log_dir = log_dir
        self.enable_console = enable_console
        self.log_level = getattr(logging, log_level.upper())
        self.max_file_size = max_file_size  # 10MB default
        self.backup_count = backup_count
        
        # Create logs directory
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize loggers for different components
        self.loggers = {
            'main': self._setup_logger('trading_main', 'main.log'),
            'data': self._setup_logger('trading_data', 'data.log'),
            'trades': self._setup_logger('trading_trades', 'trades.log'),
            'sentiment': self._setup_logger('trading_sentiment', 'sentiment.log'),
            'analysis': self._setup_logger('trading_analysis', 'analysis.log'),
            'alerts': self._setup_logger('trading_alerts', 'alerts.log'),
            'errors': self._setup_logger('trading_errors', 'errors.log'),
            'performance': self._setup_logger('trading_performance', 'performance.log'),
            'api': self._setup_logger('trading_api', 'api.log')
        }
        
        # Structured log files
        self.trade_log_file = os.path.join(log_dir, 'trades.jsonl')
        self.performance_log_file = os.path.join(log_dir, 'performance.jsonl')
        self.alert_log_file = os.path.join(log_dir, 'alerts.jsonl')
        
        # Session tracking
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.start_time = datetime.now()
    
    def _setup_logger(self, logger_name: str, log_file: str) -> logging.Logger:
        """Setup individual logger with file and console handlers"""
        logger = logging.getLogger(logger_name)
        logger.setLevel(self.log_level)
        
        # Remove existing handlers to avoid duplicates
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # File handler with rotation
        file_path = os.path.join(self.log_dir, log_file)
        file_handler = logging.handlers.RotatingFileHandler(
            file_path, 
            maxBytes=self.max_file_size, 
            backupCount=self.backup_count
        )
        file_handler.setLevel(self.log_level)
        
        # Console handler
        console_handler = None
        if self.enable_console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(self.log_level)
        
        # Custom formatter with colors for console
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
        )
        
        if self.enable_console:
            console_formatter = ColoredFormatter(
                '%(asctime)s - %(name)s - %(log_color)s%(levelname)s%(reset)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
        
        file_handler.setFormatter(file_formatter)
        
        # Add handlers
        logger.addHandler(file_handler)
        if console_handler:
            logger.addHandler(console_handler)
        
        return logger
    
    def log_trade(self, trade_data: Dict):
        """Log trade in both human-readable and structured format"""
        # Enhanced trade logging
        trade_summary = (
            f"Trade executed - Symbol: {trade_data.get('symbol')}, "
            f"Side: {trade_data.get('side')}, "
            f"Amount: {trade_data.get('amount')}, "
            f"Price: ${trade_data.get('price', 0):.6f}, "
            f"Value: ${trade_data.get('value', 0):.2f}, "
            f"Fee: ${trade_data.get('fee', 0):.4f}"
        )
        
        self.loggers['trades'].info(trade_summary)
        
        # Structured log with additional metadata
        trade_entry = {
            'timestamp': datetime.now().isoformat(),
            'session_id': self.session_id,
            'type': 'trade',
            'event': 'trade_executed',
            **trade_data
        }
        
        self._write_jsonl(self.trade_log_file, trade_entry)
    
    def log_data_fetch(self, source: str, symbol: str, success: bool, data_points: int = 0, response_time: float = 0, error: str = None):
        """Log data fetching activities with enhanced metrics"""
        if success:
            message = f"✓ Data fetch successful - Source: {source}, Symbol: {symbol}, Points: {data_points}, Time: {response_time:.3f}s"
            self.loggers['data'].info(message)
        else:
            message = f"✗ Data fetch failed - Source: {source}, Symbol: {symbol}, Error: {error or 'Unknown'}"
            self.loggers['data'].error(message)
        
        # Log API performance
        self.loggers['api'].info(f"{source} - {symbol} - Success: {success} - Time: {response_time:.3f}s")
    
    def log_sentiment_analysis(self, symbol: str, sentiment_data: Dict):
        """Log enhanced sentiment analysis results"""
        sentiment_score = sentiment_data.get('combined_sentiment_score', 0)
        confidence = sentiment_data.get('confidence', 0)
        sources = sentiment_data.get('sources_count', 0)
        total_posts = sentiment_data.get('total_posts', 0)
        
        message = (
            f"Sentiment analysis - Symbol: {symbol}, "
            f"Score: {sentiment_score:.3f}, "
            f"Confidence: {confidence:.3f}, "
            f"Sources: {sources}, "
            f"Posts: {total_posts}"
        )
        
        self.loggers['sentiment'].info(message)
        
        # Log detailed sentiment data
        sentiment_entry = {
            'timestamp': datetime.now().isoformat(),
            'session_id': self.session_id,
            'type': 'sentiment',
            'symbol': symbol,
            **sentiment_data
        }
        
        self._write_jsonl(os.path.join(self.log_dir, 'sentiment_detailed.jsonl'), sentiment_entry)
    
    def log_technical_analysis(self, symbol: str, indicators: Dict, signals: Dict = None):
        """Log technical analysis results with signals"""
        # Key indicators summary
        key_indicators = {
            'rsi': indicators.get('rsi', 0),
            'macd': indicators.get('macd', 0),
            'bb_position': indicators.get('bb_position', 0.5),
            'volume_ratio': indicators.get('volume_ratio', 1.0),
            'volatility': indicators.get('volatility', 0)
        }
        
        indicator_summary = ', '.join([f"{k}={v:.3f}" for k, v in key_indicators.items()])
        
        message = f"Technical analysis - Symbol: {symbol}, {indicator_summary}"
        if signals:
            overall_signal = signals.get('overall_signal', 'HOLD')
            confidence = signals.get('confidence', 0)
            message += f", Signal: {overall_signal} (conf: {confidence:.2f})"
        
        self.loggers['analysis'].info(message)
        
        # Detailed technical analysis log
        analysis_entry = {
            'timestamp': datetime.now().isoformat(),
            'session_id': self.session_id,
            'type': 'technical_analysis',
            'symbol': symbol,
            'indicators': indicators,
            'signals': signals or {}
        }
        
        self._write_jsonl(os.path.join(self.log_dir, 'technical_detailed.jsonl'), analysis_entry)
    
    def log_alert(self, alert_type: str, message: str, level: str = "INFO", data: Dict = None):
        """Log alerts with structured data"""
        log_level = getattr(logging, level.upper())
        alert_message = f"{alert_type}: {message}"
        
        self.loggers['alerts'].log(log_level, alert_message)
        
        # Structured alert log
        alert_entry = {
            'timestamp': datetime.now().isoformat(),
            'session_id': self.session_id,
            'type': 'alert',
            'alert_type': alert_type,
            'level': level,
            'message': message,
            'data': data or {}
        }
        
        self._write_jsonl(self.alert_log_file, alert_entry)
    
    def log_error(self, error_type: str, error_message: str, exception: Optional[Exception] = None, context: Dict = None):
        """Log errors with context and exception details"""
        error_msg = f"{error_type}: {error_message}"
        if exception:
            error_msg += f" - Exception: {str(exception)}"
        
        self.loggers['errors'].error(error_msg, exc_info=exception is not None)
        
        # Structured error log
        error_entry = {
            'timestamp': datetime.now().isoformat(),
            'session_id': self.session_id,
            'type': 'error',
            'error_type': error_type,
            'error_message': error_message,
            'exception': str(exception) if exception else None,
            'context': context or {}
        }
        
        self._write_jsonl(os.path.join(self.log_dir, 'errors_detailed.jsonl'), error_entry)
    
    def log_startup(self, config: Dict):
        """Log system startup information"""
        self.loggers['main'].info("=" * 50)
        self.loggers['main'].info("🚀 TRADING SYSTEM STARTING")
        self.loggers['main'].info("=" * 50)
        self.loggers['main'].info(f"Session ID: {self.session_id}")
        self.loggers['main'].info(f"Start Time: {self.start_time}")
        self.loggers['main'].info(f"Log Level: {logging.getLevelName(self.log_level)}")
        self.loggers['main'].info(f"Log Directory: {self.log_dir}")
        
        # Log sanitized config (remove sensitive data)
        safe_config = self._sanitize_config(config)
        self.loggers['main'].info(f"Configuration loaded: {len(safe_config)} settings")
        
        startup_entry = {
            'timestamp': datetime.now().isoformat(),
            'session_id': self.session_id,
            'type': 'startup',
            'config': safe_config
        }
        
        self._write_jsonl(os.path.join(self.log_dir, 'system_events.jsonl'), startup_entry)
    
    def log_shutdown(self, stats: Dict = None):
        """Log system shutdown with session statistics"""
        uptime = datetime.now() - self.start_time
        
        self.loggers['main'].info("=" * 50)
        self.loggers['main'].info("🛑 TRADING SYSTEM SHUTTING DOWN")
        self.loggers['main'].info("=" * 50)
        self.loggers['main'].info(f"Session ID: {self.session_id}")
        self.loggers['main'].info(f"Uptime: {uptime}")
        
        if stats:
            self.loggers['main'].info("Session Statistics:")
            for key, value in stats.items():
                self.loggers['main'].info(f"  {key}: {value}")
        
        shutdown_entry = {
            'timestamp': datetime.now().isoformat(),
            'session_id': self.session_id,
            'type': 'shutdown',
            'uptime_seconds': uptime.total_seconds(),
            'stats': stats or {}
        }
        
        self._write_jsonl(os.path.join(self.log_dir, 'system_events.jsonl'), shutdown_entry)
    
    def log_performance(self, metrics: Dict):
        """Log performance metrics"""
        metrics_str = ', '.join([f"{k}={v}" for k, v in metrics.items()])
        self.loggers['performance'].info(f"Performance metrics: {metrics_str}")
        
        # Structured performance log
        perf_entry = {
            'timestamp': datetime.now().isoformat(),
            'session_id': self.session_id,
            'type': 'performance',
            **metrics
        }
        
        self._write_jsonl(self.performance_log_file, perf_entry)
    
    def log_portfolio_update(self, portfolio_data: Dict):
        """Log portfolio updates"""
        total_value = portfolio_data.get('total_value', 0)
        pnl = portfolio_data.get('unrealized_pnl', 0)
        positions = portfolio_data.get('positions', {})
        
        message = f"Portfolio update - Value: ${total_value:.2f}, PnL: ${pnl:.2f}, Positions: {len(positions)}"
        self.loggers['main'].info(message)
        
        portfolio_entry = {
            'timestamp': datetime.now().isoformat(),
            'session_id': self.session_id,
            'type': 'portfolio_update',
            **portfolio_data
        }
        
        self._write_jsonl(os.path.join(self.log_dir, 'portfolio.jsonl'), portfolio_entry)
    
    def get_logger(self, component: str) -> logging.Logger:
        """Get logger for specific component"""
        return self.loggers.get(component, self.loggers['main'])
    
    def set_log_level(self, level: str):
        """Change log level for all loggers"""
        new_level = getattr(logging, level.upper())
        self.log_level = new_level
        
        for logger in self.loggers.values():
            logger.setLevel(new_level)
            for handler in logger.handlers:
                handler.setLevel(new_level)
    
    def _write_jsonl(self, file_path: str, data: Dict):
        """Write data to JSONL file"""
        try:
            with open(file_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(data, default=str) + '\n')
        except Exception as e:
            self.loggers['errors'].error(f"Failed to write to {file_path}: {e}")
    
    def _sanitize_config(self, config: Dict) -> Dict:
        """Remove sensitive information from config for logging"""
        safe_config = {}
        sensitive_keys = ['api_key', 'secret', 'password', 'token', 'webhook']
        
        for key, value in config.items():
            if any(sensitive in key.lower() for sensitive in sensitive_keys):
                safe_config[key] = "***"
            elif isinstance(value, dict):
                safe_config[key] = self._sanitize_config(value)
            else:
                safe_config[key] = value
        
        return safe_config
    
    def compress_old_logs(self, days_old: int = 7):
        """Compress log files older than specified days"""
        cutoff_date = datetime.now() - timedelta(days=days_old)
        
        for log_file in Path(self.log_dir).glob("*.log*"):
            if log_file.stat().st_mtime < cutoff_date.timestamp():
                if not log_file.name.endswith('.gz'):
                    compressed_name = f"{log_file}.gz"
                    with open(log_file, 'rb') as f_in:
                        with gzip.open(compressed_name, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    log_file.unlink()
                    self.loggers['main'].info(f"Compressed old log: {compressed_name}")
    
    def get_session_stats(self) -> Dict:
        """Get statistics for current session"""
        uptime = datetime.now() - self.start_time
        
        # Count log entries by type
        stats = {
            'session_id': self.session_id,
            'uptime_seconds': uptime.total_seconds(),
            'uptime_formatted': str(uptime),
            'log_files_created': len(list(Path(self.log_dir).glob("*.log"))),
        }
        
        # Count entries in structured logs
        for log_file in [self.trade_log_file, self.performance_log_file, self.alert_log_file]:
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    count = sum(1 for _ in f)
                stats[f"{Path(log_file).stem}_entries"] = count
        
        return stats

class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output"""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    
    RESET = '\033[0m'
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, '')
        record.log_color = log_color
        record.reset = self.RESET
        return super().format(record)

# Import the rotating file handler
import logging.handlers