"""
Database utilities for meme coin trading system
"""
import sqlite3
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import json
import pickle
from pathlib import Path


class DatabaseManager:
    def __init__(self, db_path: str = "meme_trading.db"):
        """Initialize database manager"""
        self.db_path = db_path
        self.conn = None
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize database with required tables"""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.execute("PRAGMA foreign_keys = ON")
        
        # Trade execution methods
    def insert_trade(self, coin_symbol: str, timestamp: datetime, action: str,
                    quantity: float, price: float, fees: float = 0,
                    exchange: str = None, strategy: str = None,
                    signal_id: int = None, notes: str = None) -> int:
        """Insert trade record"""
        coin_id = self.get_coin_id(coin_symbol)
        if not coin_id:
            raise ValueError(f"Coin {coin_symbol} not found in database")
        
        total_value = quantity * price
        
        cursor = self.conn.execute(
            """INSERT INTO trades
               (coin_id, timestamp, action, quantity, price, total_value, fees, exchange, strategy, signal_id, notes)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (coin_id, timestamp, action.upper(), quantity, price, total_value, 
             fees, exchange, strategy, signal_id, notes)
        )
        self.conn.commit()
        
        # Update portfolio
        self._update_portfolio(coin_id, action, quantity, price)
        
        return cursor.lastrowid
    
    def _update_portfolio(self, coin_id: int, action: str, quantity: float, price: float):
        """Update portfolio holdings after trade"""
        # Get current position
        cursor = self.conn.execute(
            "SELECT quantity, average_price, total_invested FROM portfolio WHERE coin_id = ?",
            (coin_id,)
        )
        result = cursor.fetchone()
        
        if result:
            current_qty, avg_price, total_invested = result
        else:
            current_qty, avg_price, total_invested = 0, 0, 0
        
        if action.upper() == 'BUY':
            new_total_invested = total_invested + (quantity * price)
            new_quantity = current_qty + quantity
            new_avg_price = new_total_invested / new_quantity if new_quantity > 0 else 0
        elif action.upper() == 'SELL':
            # Calculate profit/loss for the sold portion
            sold_cost = (quantity * avg_price) if avg_price > 0 else 0
            new_total_invested = max(0, total_invested - sold_cost)
            new_quantity = max(0, current_qty - quantity)
            new_avg_price = avg_price  # Keep same average price for remaining holdings
        else:
            return  # Unknown action
        
        # Update or insert portfolio record
        self.conn.execute(
            """INSERT OR REPLACE INTO portfolio
               (coin_id, quantity, average_price, total_invested, last_updated)
               VALUES (?, ?, ?, ?, ?)""",
            (coin_id, new_quantity, new_avg_price, new_total_invested, datetime.now())
        )
        self.conn.commit()
    
    def get_trades(self, coin_symbol: str = None, start_date: datetime = None,
                  end_date: datetime = None, action: str = None) -> pd.DataFrame:
        """Get trade history"""
        query = """SELECT t.*, c.symbol 
                   FROM trades t
                   JOIN coins c ON t.coin_id = c.id"""
        params = []
        conditions = []
        
        if coin_symbol:
            conditions.append("c.symbol = ?")
            params.append(coin_symbol.upper())
        
        if start_date:
            conditions.append("t.timestamp >= ?")
            params.append(start_date)
        
        if end_date:
            conditions.append("t.timestamp <= ?")
            params.append(end_date)
        
        if action:
            conditions.append("t.action = ?")
            params.append(action.upper())
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY t.timestamp DESC"
        
        return pd.read_sql_query(query, self.conn, params=params, parse_dates=['timestamp'])
    
    # Portfolio methods
    def get_portfolio(self) -> pd.DataFrame:
        """Get current portfolio holdings"""
        query = """SELECT p.*, c.symbol, c.name
                   FROM portfolio p
                   JOIN coins c ON p.coin_id = c.id
                   WHERE p.quantity > 0
                   ORDER BY p.total_invested DESC"""
        
        return pd.read_sql_query(query, self.conn, parse_dates=['last_updated'])
    
    def get_portfolio_value(self, coin_prices: Dict[str, float] = None) -> Dict:
        """Calculate current portfolio value"""
        portfolio = self.get_portfolio()
        
        if portfolio.empty:
            return {'total_value': 0, 'total_invested': 0, 'unrealized_pnl': 0, 'holdings': []}
        
        total_value = 0
        total_invested = portfolio['total_invested'].sum()
        holdings = []
        
        for _, holding in portfolio.iterrows():
            symbol = holding['symbol']
            quantity = holding['quantity']
            avg_price = holding['average_price']
            invested = holding['total_invested']
            
            # Get current price
            if coin_prices and symbol in coin_prices:
                current_price = coin_prices[symbol]
            else:
                # Get latest price from database
                latest_price = self._get_latest_price(symbol)
                current_price = latest_price if latest_price else avg_price
            
            current_value = quantity * current_price
            unrealized_pnl = current_value - invested
            pnl_percentage = (unrealized_pnl / invested * 100) if invested > 0 else 0
            
            total_value += current_value
            
            holdings.append({
                'symbol': symbol,
                'quantity': quantity,
                'average_price': avg_price,
                'current_price': current_price,
                'invested': invested,
                'current_value': current_value,
                'unrealized_pnl': unrealized_pnl,
                'pnl_percentage': pnl_percentage
            })
        
        return {
            'total_value': total_value,
            'total_invested': total_invested,
            'unrealized_pnl': total_value - total_invested,
            'unrealized_pnl_percentage': ((total_value - total_invested) / total_invested * 100) if total_invested > 0 else 0,
            'holdings': holdings
        }
    
    def _get_latest_price(self, symbol: str) -> Optional[float]:
        """Get latest price for a symbol"""
        coin_id = self.get_coin_id(symbol)
        if not coin_id:
            return None
        
        cursor = self.conn.execute(
            "SELECT close_price FROM price_data WHERE coin_id = ? ORDER BY timestamp DESC LIMIT 1",
            (coin_id,)
        )
        result = cursor.fetchone()
        return result[0] if result else None
    
    # Performance tracking methods
    def update_daily_performance(self, date: datetime.date, portfolio_value: float,
                               daily_pnl: float, total_pnl: float):
        """Update daily performance metrics"""
        # Calculate additional metrics from trades
        trades_today = self.get_trades(
            start_date=datetime.combine(date, datetime.min.time()),
            end_date=datetime.combine(date, datetime.max.time())
        )
        
        total_trades = len(trades_today)
        winning_trades = 0
        losing_trades = 0
        largest_win = 0
        largest_loss = 0
        
        # Calculate trade outcomes (simplified)
        for _, trade in trades_today.iterrows():
            if trade['action'] == 'SELL':
                # This is a simplified P&L calculation
                # In reality, you'd need to match with corresponding BUY orders
                pnl = trade['total_value'] - (trade['quantity'] * trade.get('cost_basis', trade['price']))
                
                if pnl > 0:
                    winning_trades += 1
                    largest_win = max(largest_win, pnl)
                else:
                    losing_trades += 1
                    largest_loss = min(largest_loss, pnl)
        
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        self.conn.execute(
            """INSERT OR REPLACE INTO performance_metrics
               (date, total_portfolio_value, daily_pnl, total_pnl, win_rate, 
                total_trades, winning_trades, losing_trades, largest_win, largest_loss)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (date, portfolio_value, daily_pnl, total_pnl, win_rate,
             total_trades, winning_trades, losing_trades, largest_win, largest_loss)
        )
        self.conn.commit()
    
    def get_performance_metrics(self, start_date: datetime = None,
                              end_date: datetime = None) -> pd.DataFrame:
        """Get performance metrics"""
        query = "SELECT * FROM performance_metrics"
        params = []
        conditions = []
        
        if start_date:
            conditions.append("date >= ?")
            params.append(start_date.date())
        
        if end_date:
            conditions.append("date <= ?")
            params.append(end_date.date())
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY date"
        
        return pd.read_sql_query(query, self.conn, params=params, parse_dates=['date'])
    
    # Model predictions methods
    def insert_model_prediction(self, coin_symbol: str, timestamp: datetime,
                              model_name: str, prediction_type: str,
                              predicted_value: float = None, predicted_class: str = None,
                              confidence: float = None, horizon_hours: int = 1,
                              features_used: List[str] = None):
        """Insert model prediction"""
        coin_id = self.get_coin_id(coin_symbol)
        if not coin_id:
            raise ValueError(f"Coin {coin_symbol} not found in database")
        
        features_json = json.dumps(features_used) if features_used else None
        
        self.conn.execute(
            """INSERT INTO model_predictions
               (coin_id, timestamp, model_name, prediction_type, predicted_value, 
                predicted_class, confidence, horizon_hours, features_used)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (coin_id, timestamp, model_name, prediction_type, predicted_value,
             predicted_class, confidence, horizon_hours, features_json)
        )
        self.conn.commit()
    
    def update_prediction_actual(self, prediction_id: int, actual_value: float):
        """Update prediction with actual value for evaluation"""
        self.conn.execute(
            "UPDATE model_predictions SET actual_value = ? WHERE id = ?",
            (actual_value, prediction_id)
        )
        self.conn.commit()
    
    def get_model_predictions(self, coin_symbol: str = None, model_name: str = None,
                            start_date: datetime = None, end_date: datetime = None) -> pd.DataFrame:
        """Get model predictions"""
        query = """SELECT mp.*, c.symbol 
                   FROM model_predictions mp
                   JOIN coins c ON mp.coin_id = c.id"""
        params = []
        conditions = []
        
        if coin_symbol:
            conditions.append("c.symbol = ?")
            params.append(coin_symbol.upper())
        
        if model_name:
            conditions.append("mp.model_name = ?")
            params.append(model_name)
        
        if start_date:
            conditions.append("mp.timestamp >= ?")
            params.append(start_date)
        
        if end_date:
            conditions.append("mp.timestamp <= ?")
            params.append(end_date)
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY mp.timestamp DESC"
        
        return pd.read_sql_query(query, self.conn, params=params, parse_dates=['timestamp'])
    
    # Alert methods
    def insert_alert(self, coin_symbol: str, alert_type: str, condition_type: str,
                    threshold_value: float, current_value: float, message: str,
                    priority: str = 'medium'):
        """Insert alert"""
        coin_id = self.get_coin_id(coin_symbol) if coin_symbol else None
        
        self.conn.execute(
            """INSERT INTO alerts
               (coin_id, alert_type, condition_type, threshold_value, current_value, message, priority)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (coin_id, alert_type, condition_type, threshold_value, current_value, message, priority)
        )
        self.conn.commit()
    
    def trigger_alert(self, alert_id: int):
        """Mark alert as triggered"""
        self.conn.execute(
            "UPDATE alerts SET status = 'triggered', triggered_at = ? WHERE id = ?",
            (datetime.now(), alert_id)
        )
        self.conn.commit()
    
    def get_active_alerts(self) -> pd.DataFrame:
        """Get active alerts"""
        query = """SELECT a.*, c.symbol 
                   FROM alerts a
                   LEFT JOIN coins c ON a.coin_id = c.id
                   WHERE a.status = 'active'
                   ORDER BY a.priority DESC, a.created_at DESC"""
        
        return pd.read_sql_query(query, self.conn, parse_dates=['created_at', 'triggered_at'])
    
    # Utility methods
    def backup_database(self, backup_path: str):
        """Create database backup"""
        backup_conn = sqlite3.connect(backup_path)
        self.conn.backup(backup_conn)
        backup_conn.close()
        print(f"Database backed up to {backup_path}")
    
    def get_database_stats(self) -> Dict:
        """Get database statistics"""
        stats = {}
        
        tables = ['coins', 'price_data', 'sentiment_data', 'trading_signals', 
                 'trades', 'portfolio', 'performance_metrics', 'model_predictions', 'alerts']
        
        for table in tables:
            cursor = self.conn.execute(f"SELECT COUNT(*) FROM {table}")
            stats[f'{table}_count'] = cursor.fetchone()[0]
        
        # Additional stats
        cursor = self.conn.execute("SELECT COUNT(DISTINCT coin_id) FROM price_data")
        stats['coins_with_price_data'] = cursor.fetchone()[0]
        
        cursor = self.conn.execute("SELECT MIN(timestamp), MAX(timestamp) FROM price_data")
        result = cursor.fetchone()
        stats['price_data_date_range'] = result if result[0] else (None, None)
        
        return stats
    
    def clean_old_data(self, days_to_keep: int = 90):
        """Clean old data beyond retention period"""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        # Clean old price data (keep more recent data)
        self.conn.execute(
            "DELETE FROM price_data WHERE timestamp < ?", (cutoff_date,)
        )
        
        # Clean old sentiment data
        self.conn.execute(
            "DELETE FROM sentiment_data WHERE timestamp < ?", (cutoff_date,)
        )
        
        # Clean old predictions
        self.conn.execute(
            "DELETE FROM model_predictions WHERE timestamp < ?", (cutoff_date,)
        )
        
        # Clean triggered alerts older than 30 days
        alert_cutoff = datetime.now() - timedelta(days=30)
        self.conn.execute(
            "DELETE FROM alerts WHERE status = 'triggered' AND triggered_at < ?", 
            (alert_cutoff,)
        )
        
        self.conn.commit()
        print(f"Cleaned data older than {days_to_keep} days")
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            print("Database connection closed")


def main():
    """Test the database manager"""
    # Initialize database
    db = DatabaseManager("test_meme_trading.db")
    
    print("Testing Database Manager...")
    print("="*50)
    
    # Add test coins
    doge_id = db.add_coin("DOGE", "Dogecoin", "dogecoin")
    pepe_id = db.add_coin("PEPE", "Pepe", "pepe")
    
    # Insert test price data
    dates = pd.date_range('2024-01-01', periods=10, freq='H')
    price_data = pd.DataFrame({
        'close': np.random.uniform(0.08, 0.12, 10),
        'volume': np.random.uniform(1000000, 5000000, 10),
        'open': np.random.uniform(0.08, 0.12, 10),
        'high': np.random.uniform(0.10, 0.15, 10),
        'low': np.random.uniform(0.06, 0.10, 10)
    }, index=dates)
    
    db.insert_price_data("DOGE", price_data)
    
    # Insert test sentiment data
    db.insert_sentiment_data("DOGE", datetime.now(), 0.7, 0.8, "twitter", "twitter", 100)
    
    # Insert test trading signal
    signal_id = db.insert_trading_signal(
        "DOGE", datetime.now(), "BUY_SIGNAL", 0.85, "BUY", 0.095, 
        {"rsi": 30, "macd": "bullish"}, "momentum"
    )
    
    # Insert test trade
    trade_id = db.insert_trade(
        "DOGE", datetime.now(), "BUY", 1000, 0.095, 0.5, "binance", "momentum", signal_id
    )
    
    # Get portfolio
    portfolio = db.get_portfolio_value()
    print(f"Portfolio Value: ${portfolio['total_value']:.2f}")
    print(f"Total Invested: ${portfolio['total_invested']:.2f}")
    print(f"Unrealized P&L: ${portfolio['unrealized_pnl']:.2f}")
    
    # Get database stats
    stats = db.get_database_stats()
    print(f"\nDatabase Stats:")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Close database
    db.close()
    
    # Clean up test database
    import os
    if os.path.exists("test_meme_trading.db"):
        os.remove("test_meme_trading.db")
    
    print("Database test completed successfully!")


if __name__ == "__main__":
    # Create tables for testing
    db = DatabaseManager("test.db")
    print("Database tables created successfully!")
    db.close()
    
    def _create_tables(self):
        """Create all required tables"""
        # Coins table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS coins (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                coingecko_id TEXT,
                contract_address TEXT,
                blockchain TEXT DEFAULT 'ethereum',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT 1
            )
        """)
        
        # Price data table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS price_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                coin_id INTEGER NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                open_price REAL,
                high_price REAL,
                low_price REAL,
                close_price REAL NOT NULL,
                volume REAL,
                market_cap REAL,
                source TEXT DEFAULT 'coingecko',
                FOREIGN KEY (coin_id) REFERENCES coins (id),
                UNIQUE(coin_id, timestamp, source)
            )
        """)
        
        # Sentiment data table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS sentiment_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                coin_id INTEGER NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                sentiment_score REAL NOT NULL,
                confidence REAL,
                source TEXT NOT NULL,
                platform TEXT,
                message_count INTEGER DEFAULT 0,
                raw_data TEXT,
                FOREIGN KEY (coin_id) REFERENCES coins (id)
            )
        """)
        
        # Trading signals table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS trading_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                coin_id INTEGER NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                signal_type TEXT NOT NULL,
                signal_strength REAL NOT NULL,
                action TEXT NOT NULL,
                price REAL,
                indicators TEXT,
                strategy TEXT,
                confidence REAL,
                FOREIGN KEY (coin_id) REFERENCES coins (id)
            )
        """)
        
        # Trades table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                coin_id INTEGER NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                action TEXT NOT NULL,
                quantity REAL NOT NULL,
                price REAL NOT NULL,
                total_value REAL NOT NULL,
                fees REAL DEFAULT 0,
                exchange TEXT,
                strategy TEXT,
                signal_id INTEGER,
                status TEXT DEFAULT 'completed',
                notes TEXT,
                FOREIGN KEY (coin_id) REFERENCES coins (id),
                FOREIGN KEY (signal_id) REFERENCES trading_signals (id)
            )
        """)
        
        # Portfolio table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS portfolio (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                coin_id INTEGER NOT NULL,
                quantity REAL NOT NULL,
                average_price REAL NOT NULL,
                total_invested REAL NOT NULL,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (coin_id) REFERENCES coins (id),
                UNIQUE(coin_id)
            )
        """)
        
        # Performance metrics table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE NOT NULL UNIQUE,
                total_portfolio_value REAL,
                daily_pnl REAL,
                total_pnl REAL,
                win_rate REAL,
                total_trades INTEGER DEFAULT 0,
                winning_trades INTEGER DEFAULT 0,
                losing_trades INTEGER DEFAULT 0,
                largest_win REAL DEFAULT 0,
                largest_loss REAL DEFAULT 0,
                sharpe_ratio REAL,
                max_drawdown REAL
            )
        """)
        
        # Model predictions table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS model_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                coin_id INTEGER NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                model_name TEXT NOT NULL,
                prediction_type TEXT NOT NULL,
                predicted_value REAL,
                predicted_class TEXT,
                confidence REAL,
                actual_value REAL,
                horizon_hours INTEGER DEFAULT 1,
                features_used TEXT,
                FOREIGN KEY (coin_id) REFERENCES coins (id)
            )
        """)
        
        # Alerts table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                coin_id INTEGER,
                alert_type TEXT NOT NULL,
                condition_type TEXT NOT NULL,
                threshold_value REAL,
                current_value REAL,
                message TEXT NOT NULL,
                priority TEXT DEFAULT 'medium',
                status TEXT DEFAULT 'active',
                triggered_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (coin_id) REFERENCES coins (id)
            )
        """)
        
        # Create indexes for better performance
        self._create_indexes()
        
        self.conn.commit()
    
    def _create_indexes(self):
        """Create database indexes"""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_price_data_coin_timestamp ON price_data (coin_id, timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_sentiment_data_coin_timestamp ON sentiment_data (coin_id, timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_trading_signals_coin_timestamp ON trading_signals (coin_id, timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_trades_coin_timestamp ON trades (coin_id, timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_model_predictions_coin_timestamp ON model_predictions (coin_id, timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_coins_symbol ON coins (symbol)",
            "CREATE INDEX IF NOT EXISTS idx_performance_date ON performance_metrics (date)"
        ]
        
        for index_sql in indexes:
            self.conn.execute(index_sql)
    
    # Coin management methods
    def add_coin(self, symbol: str, name: str, coingecko_id: str = None, 
                 contract_address: str = None, blockchain: str = 'ethereum') -> int:
        """Add a new coin to the database"""
        try:
            cursor = self.conn.execute(
                """INSERT INTO coins (symbol, name, coingecko_id, contract_address, blockchain)
                   VALUES (?, ?, ?, ?, ?)""",
                (symbol.upper(), name, coingecko_id, contract_address, blockchain)
            )
            self.conn.commit()
            print(f"Added coin: {symbol} ({name})")
            return cursor.lastrowid
        except sqlite3.IntegrityError:
            # Coin already exists, return existing ID
            cursor = self.conn.execute("SELECT id FROM coins WHERE symbol = ?", (symbol.upper(),))
            return cursor.fetchone()[0]
    
    def get_coin_id(self, symbol: str) -> Optional[int]:
        """Get coin ID by symbol"""
        cursor = self.conn.execute("SELECT id FROM coins WHERE symbol = ?", (symbol.upper(),))
        result = cursor.fetchone()
        return result[0] if result else None
    
    def get_coin_info(self, symbol: str) -> Optional[Dict]:
        """Get coin information"""
        cursor = self.conn.execute(
            "SELECT * FROM coins WHERE symbol = ?", (symbol.upper(),)
        )
        result = cursor.fetchone()
        if result:
            columns = [description[0] for description in cursor.description]
            return dict(zip(columns, result))
        return None
    
    def list_coins(self, active_only: bool = True) -> List[Dict]:
        """List all coins in database"""
        query = "SELECT * FROM coins"
        if active_only:
            query += " WHERE is_active = 1"
        
        cursor = self.conn.execute(query)
        columns = [description[0] for description in cursor.description]
        
        return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    # Price data methods
    def insert_price_data(self, coin_symbol: str, data: pd.DataFrame, source: str = 'coingecko'):
        """Insert price data for a coin"""
        coin_id = self.get_coin_id(coin_symbol)
        if not coin_id:
            raise ValueError(f"Coin {coin_symbol} not found in database")
        
        # Prepare data for insertion
        records = []
        for index, row in data.iterrows():
            timestamp = index if isinstance(index, datetime) else pd.to_datetime(index)
            
            record = (
                coin_id,
                timestamp,
                row.get('open'),
                row.get('high'),
                row.get('low'),
                row.get('close'),
                row.get('volume'),
                row.get('market_cap'),
                source
            )
            records.append(record)
        
        # Insert with conflict resolution
        self.conn.executemany(
            """INSERT OR REPLACE INTO price_data 
               (coin_id, timestamp, open_price, high_price, low_price, close_price, volume, market_cap, source)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            records
        )
        self.conn.commit()
        print(f"Inserted {len(records)} price records for {coin_symbol}")
    
    def get_price_data(self, coin_symbol: str, start_date: datetime = None, 
                      end_date: datetime = None, source: str = None) -> pd.DataFrame:
        """Get price data for a coin"""
        coin_id = self.get_coin_id(coin_symbol)
        if not coin_id:
            return pd.DataFrame()
        
        query = """SELECT timestamp, open_price, high_price, low_price, close_price, volume, market_cap
                   FROM price_data WHERE coin_id = ?"""
        params = [coin_id]
        
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date)
        
        if source:
            query += " AND source = ?"
            params.append(source)
        
        query += " ORDER BY timestamp"
        
        df = pd.read_sql_query(query, self.conn, params=params, parse_dates=['timestamp'])
        
        if not df.empty:
            df.set_index('timestamp', inplace=True)
            df.columns = ['open', 'high', 'low', 'close', 'volume', 'market_cap']
        
        return df
    
    # Sentiment data methods
    def insert_sentiment_data(self, coin_symbol: str, timestamp: datetime, 
                            sentiment_score: float, confidence: float = None,
                            source: str = 'multi_source', platform: str = None,
                            message_count: int = 0, raw_data: Dict = None):
        """Insert sentiment data"""
        coin_id = self.get_coin_id(coin_symbol)
        if not coin_id:
            raise ValueError(f"Coin {coin_symbol} not found in database")
        
        raw_data_json = json.dumps(raw_data) if raw_data else None
        
        self.conn.execute(
            """INSERT OR REPLACE INTO sentiment_data
               (coin_id, timestamp, sentiment_score, confidence, source, platform, message_count, raw_data)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (coin_id, timestamp, sentiment_score, confidence, source, platform, message_count, raw_data_json)
        )
        self.conn.commit()
    
    def get_sentiment_data(self, coin_symbol: str, start_date: datetime = None,
                          end_date: datetime = None) -> pd.DataFrame:
        """Get sentiment data for a coin"""
        coin_id = self.get_coin_id(coin_symbol)
        if not coin_id:
            return pd.DataFrame()
        
        query = """SELECT timestamp, sentiment_score, confidence, source, platform, message_count
                   FROM sentiment_data WHERE coin_id = ?"""
        params = [coin_id]
        
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date)
        
        query += " ORDER BY timestamp"
        
        return pd.read_sql_query(query, self.conn, params=params, parse_dates=['timestamp'])
    
    # Trading signals methods
    def insert_trading_signal(self, coin_symbol: str, timestamp: datetime,
                            signal_type: str, signal_strength: float, action: str,
                            price: float = None, indicators: Dict = None,
                            strategy: str = None, confidence: float = None) -> int:
        """Insert trading signal"""
        coin_id = self.get_coin_id(coin_symbol)
        if not coin_id:
            raise ValueError(f"Coin {coin_symbol} not found in database")
        
        indicators_json = json.dumps(indicators) if indicators else None
        
        cursor = self.conn.execute(
            """INSERT INTO trading_signals
               (coin_id, timestamp, signal_type, signal_strength, action, price, indicators, strategy, confidence)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (coin_id, timestamp, signal_type, signal_strength, action, price, 
             indicators_json, strategy, confidence)
        )
        self.conn.commit()
        return cursor.lastrowid
    
    def get_trading_signals(self, coin_symbol: str = None, start_date: datetime = None,
                          end_date: datetime = None, action: str = None) -> pd.DataFrame:
        """Get trading signals"""
        query = """SELECT ts.*, c.symbol 
                   FROM trading_signals ts
                   JOIN coins c ON ts.coin_id = c.id"""
        params = []
        conditions = []
        
        if coin_symbol:
            conditions.append("c.symbol = ?")
            params.append(coin_symbol.upper())
        
        if start_date:
            conditions.append("ts.timestamp >= ?")
            params.append(start_date)
        
        if end_date:
            conditions.append("ts.timestamp <= ?")
            params.append(end_date)
        
        if action:
            conditions.append("ts.action = ?")
            params.append(action.upper())
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY ts.timestamp DESC"
        
        return pd.read_sql_query(query, self.conn, params=params, parse_dates=['timestamp'])