"""
Backtesting system for meme coin trading strategies
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class BacktestConfig:
    """Backtesting configuration"""
    initial_capital: float = 10000.0
    commission: float = 0.001  # 0.1% commission
    slippage: float = 0.0005   # 0.05% slippage
    max_positions: int = 5     # Maximum concurrent positions
    position_size: float = 0.2 # 20% of capital per position
    stop_loss: float = 0.1     # 10% stop loss
    take_profit: float = 0.3   # 30% take profit
    rebalance_frequency: str = '1D'  # Rebalancing frequency


@dataclass
class Trade:
    """Trade record"""
    symbol: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: float
    side: str  # 'long' or 'short'
    pnl: float
    pnl_pct: float
    commission: float
    slippage: float
    reason: str  # Exit reason: 'signal', 'stop_loss', 'take_profit', 'end_of_period'


class BacktestEngine:
    def __init__(self, config: BacktestConfig = None):
        """Initialize backtesting engine"""
        self.config = config or BacktestConfig()
        self.trades = []
        self.portfolio_history = []
        self.positions = {}  # Current positions
        self.cash = self.config.initial_capital
        self.total_value = self.config.initial_capital
        
    def run_backtest(self, data: Dict[str, pd.DataFrame], 
                    strategy_func: Callable, 
                    start_date: datetime = None, 
                    end_date: datetime = None) -> Dict:
        """Run backtest with given data and strategy"""
        
        print(f"Starting backtest...")
        print(f"Initial capital: ${self.config.initial_capital:,.2f}")
        
        # Prepare data
        if start_date or end_date:
            data = self._filter_data_by_date(data, start_date, end_date)
        
        # Get common time index
        time_index = self._get_common_time_index(data)
        
        if len(time_index) == 0:
            raise ValueError("No common time index found in data")
        
        # Initialize portfolio tracking
        self.portfolio_history = []
        self.trades = []
        self.positions = {}
        self.cash = self.config.initial_capital
        
        # Main backtesting loop
        for i, current_time in enumerate(time_index):
            # Get current market data
            current_data = self._get_current_data(data, current_time, lookback=50)
            
            if not current_data:
                continue
            
            # Generate trading signals
            signals = strategy_func(current_data, current_time)
            
            # Process signals
            self._process_signals(signals, current_data, current_time)
            
            # Check stop losses and take profits
            self._check_exit_conditions(current_data, current_time)
            
            # Update portfolio value
            self._update_portfolio_value(current_data, current_time)
            
            # Record portfolio state
            self._record_portfolio_state(current_time)
        
        # Close all remaining positions
        self._close_all_positions(data, time_index[-1])
        
        # Calculate performance metrics
        performance = self._calculate_performance_metrics()
        
        print(f"Backtest completed!")
        print(f"Final portfolio value: ${self.total_value:,.2f}")
        print(f"Total return: {((self.total_value / self.config.initial_capital) - 1) * 100:.2f}%")
        print(f"Total trades: {len(self.trades)}")
        
        return {
            'performance': performance,
            'trades': self.trades,
            'portfolio_history': pd.DataFrame(self.portfolio_history),
            'config': self.config
        }
    
    def _filter_data_by_date(self, data: Dict[str, pd.DataFrame], 
                            start_date: datetime, end_date: datetime) -> Dict[str, pd.DataFrame]:
        """Filter data by date range"""
        filtered_data = {}
        
        for symbol, df in data.items():
            mask = pd.Series(True, index=df.index)
            
            if start_date:
                mask &= (df.index >= start_date)
            if end_date:
                mask &= (df.index <= end_date)
            
            filtered_data[symbol] = df[mask]
        
        return filtered_data
    
    def _get_common_time_index(self, data: Dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
        """Get common time index across all symbols"""
        if not data:
            return pd.DatetimeIndex([])
        
        # Start with the first symbol's index
        common_index = data[list(data.keys())[0]].index
        
        # Find intersection with all other symbols
        for symbol, df in data.items():
            common_index = common_index.intersection(df.index)
        
        return common_index.sort_values()
    
    def _get_current_data(self, data: Dict[str, pd.DataFrame], 
                         current_time: datetime, lookback: int = 50) -> Dict[str, pd.DataFrame]:
        """Get current and historical data up to current time"""
        current_data = {}
        
        for symbol, df in data.items():
            # Get data up to current time
            historical_data = df[df.index <= current_time]
            
            if len(historical_data) > 0:
                # Limit to lookback period
                current_data[symbol] = historical_data.tail(lookback)
        
        return current_data
    
    def _process_signals(self, signals: List[Dict], current_data: Dict[str, pd.DataFrame], 
                        current_time: datetime):
        """Process trading signals"""
        for signal in signals:
            symbol = signal.get('symbol')
            action = signal.get('action', '').upper()
            strength = signal.get('strength', 0)
            
            if not symbol or symbol not in current_data:
                continue
            
            current_price = current_data[symbol]['close'].iloc[-1]
            
            if action == 'BUY' and self._can_open_position():
                self._open_long_position(symbol, current_price, current_time, strength)
            elif action == 'SELL' and symbol in self.positions:
                self._close_position(symbol, current_price, current_time, 'signal')
    
    def _can_open_position(self) -> bool:
        """Check if we can open a new position"""
        return len(self.positions) < self.config.max_positions and self.cash > 100
    
    def _open_long_position(self, symbol: str, price: float, timestamp: datetime, strength: float):
        """Open a long position"""
        # Calculate position size
        position_value = min(
            self.cash * self.config.position_size,
            self.cash * 0.9  # Leave some cash
        )
        
        if position_value < 100:  # Minimum position size
            return
        
        # Apply slippage
        entry_price = price * (1 + self.config.slippage)
        
        # Calculate quantity
        quantity = position_value / entry_price
        
        # Calculate commission
        commission = position_value * self.config.commission
        
        # Update cash
        total_cost = position_value + commission
        self.cash -= total_cost
        
        # Store position
        self.positions[symbol] = {
            'symbol': symbol,
            'entry_time': timestamp,
            'entry_price': entry_price,
            'quantity': quantity,
            'side': 'long',
            'commission': commission,
            'strength': strength
        }
        
        print(f"Opened long position: {symbol} @ ${entry_price:.6f}, Qty: {quantity:.2f}")
    
    def _close_position(self, symbol: str, price: float, timestamp: datetime, reason: str):
        """Close a position"""
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        
        # Apply slippage
        exit_price = price * (1 - self.config.slippage) if position['side'] == 'long' else price * (1 + self.config.slippage)
        
        # Calculate P&L
        if position['side'] == 'long':
            pnl = (exit_price - position['entry_price']) * position['quantity']
        else:  # short
            pnl = (position['entry_price'] - exit_price) * position['quantity']
        
        # Calculate commission for exit
        exit_commission = (exit_price * position['quantity']) * self.config.commission
        
        # Net P&L after commissions
        net_pnl = pnl - position['commission'] - exit_commission
        
        # Update cash
        position_value = exit_price * position['quantity']
        self.cash += position_value - exit_commission
        
        # Calculate percentage return
        invested_amount = position['entry_price'] * position['quantity'] + position['commission']
        pnl_pct = (net_pnl / invested_amount) * 100 if invested_amount > 0 else 0
        
        # Create trade record
        trade = Trade(
            symbol=symbol,
            entry_time=position['entry_time'],
            exit_time=timestamp,
            entry_price=position['entry_price'],
            exit_price=exit_price,
            quantity=position['quantity'],
            side=position['side'],
            pnl=net_pnl,
            pnl_pct=pnl_pct,
            commission=position['commission'] + exit_commission,
            slippage=(position['entry_price'] - price) * position['quantity'] if position['side'] == 'long' else 0,
            reason=reason
        )
        
        self.trades.append(trade)
        
        # Remove position
        del self.positions[symbol]
        
        print(f"Closed {position['side']} position: {symbol} @ ${exit_price:.6f}, P&L: ${net_pnl:.2f} ({pnl_pct:.2f}%)")
    
    def _check_exit_conditions(self, current_data: Dict[str, pd.DataFrame], current_time: datetime):
        """Check stop loss and take profit conditions"""
        positions_to_close = []
        
        for symbol, position in self.positions.items():
            if symbol not in current_data:
                continue
            
            current_price = current_data[symbol]['close'].iloc[-1]
            entry_price = position['entry_price']
            
            if position['side'] == 'long':
                # Stop loss
                if current_price <= entry_price * (1 - self.config.stop_loss):
                    positions_to_close.append((symbol, current_price, 'stop_loss'))
                # Take profit
                elif current_price >= entry_price * (1 + self.config.take_profit):
                    positions_to_close.append((symbol, current_price, 'take_profit'))
            else:  # short position
                # Stop loss for short
                if current_price >= entry_price * (1 + self.config.stop_loss):
                    positions_to_close.append((symbol, current_price, 'stop_loss'))
                # Take profit for short
                elif current_price <= entry_price * (1 - self.config.take_profit):
                    positions_to_close.append((symbol, current_price, 'take_profit'))
        
        # Close positions that hit exit conditions
        for symbol, price, reason in positions_to_close:
            self._close_position(symbol, price, current_time, reason)
    
    def _update_portfolio_value(self, current_data: Dict[str, pd.DataFrame], current_time: datetime):
        """Update total portfolio value"""
        positions_value = 0
        
        for symbol, position in self.positions.items():
            if symbol in current_data:
                current_price = current_data[symbol]['close'].iloc[-1]
                position_value = current_price * position['quantity']
                positions_value += position_value
        
        self.total_value = self.cash + positions_value
    
    def _record_portfolio_state(self, timestamp: datetime):
        """Record current portfolio state"""
        positions_value = self.total_value - self.cash
        
        state = {
            'timestamp': timestamp,
            'total_value': self.total_value,
            'cash': self.cash,
            'positions_value': positions_value,
            'num_positions': len(self.positions),
            'returns': (self.total_value / self.config.initial_capital) - 1
        }
        
        self.portfolio_history.append(state)
    
    def _close_all_positions(self, data: Dict[str, pd.DataFrame], final_time: datetime):
        """Close all remaining positions at the end of backtest"""
        for symbol in list(self.positions.keys()):
            if symbol in data:
                final_price = data[symbol]['close'].iloc[-1]
                self._close_position(symbol, final_price, final_time, 'end_of_period')
    
    def _calculate_performance_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics"""
        if not self.trades or not self.portfolio_history:
            return {}
        
        # Convert to DataFrames
        trades_df = pd.DataFrame([
            {
                'symbol': t.symbol,
                'entry_time': t.entry_time,
                'exit_time': t.exit_time,
                'pnl': t.pnl,
                'pnl_pct': t.pnl_pct,
                'duration_hours': (t.exit_time - t.entry_time).total_seconds() / 3600,
                'side': t.side,
                'reason': t.reason
            }
            for t in self.trades
        ])
        
        portfolio_df = pd.DataFrame(self.portfolio_history)
        portfolio_df.set_index('timestamp', inplace=True)
        
        # Basic metrics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] <= 0])
        
        # Returns
        total_return = (self.total_value / self.config.initial_capital) - 1
        
        # Calculate daily returns for advanced metrics
        daily_returns = portfolio_df['returns'].resample('1D').last().pct_change().dropna()
        
        # Risk metrics
        volatility = daily_returns.std() * np.sqrt(252)  # Annualized
        
        # Sharpe ratio (assuming 0% risk-free rate)
        sharpe_ratio = (daily_returns.mean() * 252) / (daily_returns.std() * np.sqrt(252)) if daily_returns.std() > 0 else 0
        
        # Maximum drawdown
        cumulative_returns = (1 + daily_returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()
        
        # Win rate and profit factor
        win_rate = (winning_trades / total_trades) if total_trades > 0 else 0
        
        gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Average trade metrics
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
        avg_trade = trades_df['pnl'].mean()
        
        # Best and worst trades
        best_trade = trades_df['pnl'].max() if total_trades > 0 else 0
        worst_trade = trades_df['pnl'].min() if total_trades > 0 else 0
        
        # Trade duration
        avg_trade_duration = trades_df['duration_hours'].mean() if total_trades > 0 else 0
        
        # Exit reason breakdown
        exit_reasons = trades_df['reason'].value_counts().to_dict() if total_trades > 0 else {}
        
        return {
            # Overall performance
            'initial_capital': self.config.initial_capital,
            'final_value': self.total_value,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            
            # Risk metrics
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown * 100,
            
            # Trade statistics
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'win_rate_pct': win_rate * 100,
            
            # Profit metrics
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'net_profit': gross_profit + gross_loss,  # gross_loss is negative
            'profit_factor': profit_factor,
            
            # Trade averages
            'avg_trade': avg_trade,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'best_trade': best_trade,
            'worst_trade': worst_trade,
            
            # Duration
            'avg_trade_duration_hours': avg_trade_duration,
            
            # Exit analysis
            'exit_reasons': exit_reasons,
            
            # Portfolio composition
            'final_cash': self.cash,
            'final_positions_value': self.total_value - self.cash
        }
    
    def plot_results(self, results: Dict, save_path: str = None):
        """Plot backtest results"""
        portfolio_df = results['portfolio_history']
        trades_df = pd.DataFrame([
            {
                'entry_time': t.entry_time,
                'exit_time': t.exit_time,
                'pnl': t.pnl,
                'symbol': t.symbol,
                'reason': t.reason
            }
            for t in results['trades']
        ])
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Portfolio value over time
        axes[0, 0].plot(portfolio_df.index, portfolio_df['total_value'])
        axes[0, 0].set_title('Portfolio Value Over Time')
        axes[0, 0].set_ylabel('Portfolio Value ($)')
        axes[0, 0].grid(True)
        
        # Drawdown
        returns = portfolio_df['returns']
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - rolling_max) / rolling_max
        
        axes[0, 1].fill_between(portfolio_df.index, drawdowns, 0, alpha=0.7, color='red')
        axes[0, 1].set_title('Drawdown')
        axes[0, 1].set_ylabel('Drawdown (%)')
        axes[0, 1].grid(True)
        
        # Trade P&L distribution
        if not trades_df.empty:
            axes[1, 0].hist(trades_df['pnl'], bins=30, alpha=0.7, edgecolor='black')
            axes[1, 0].axvline(x=0, color='red', linestyle='--')
            axes[1, 0].set_title('Trade P&L Distribution')
            axes[1, 0].set_xlabel('P&L ($)')
            axes[1, 0].set_ylabel('Frequency')
            
            # Cumulative P&L
            trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()
            axes[1, 1].plot(trades_df['exit_time'], trades_df['cumulative_pnl'])
            axes[1, 1].set_title('Cumulative P&L')
            axes[1, 1].set_ylabel('Cumulative P&L ($)')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def generate_report(self, results: Dict) -> str:
        """Generate detailed backtest report"""
        performance = results['performance']
        
        report = f"""
BACKTEST PERFORMANCE REPORT
{'='*60}

STRATEGY OVERVIEW:
Initial Capital: ${performance['initial_capital']:,.2f}
Final Portfolio Value: ${performance['final_value']:,.2f}
Total Return: {performance['total_return_pct']:.2f}%
Backtest Period: {len(results['portfolio_history'])} periods

PERFORMANCE METRICS:
Total Return: {performance['total_return_pct']:.2f}%
Annualized Volatility: {performance['volatility']*100:.2f}%
Sharpe Ratio: {performance['sharpe_ratio']:.3f}
Maximum Drawdown: {performance['max_drawdown_pct']:.2f}%

TRADING STATISTICS:
Total Trades: {performance['total_trades']}
Winning Trades: {performance['winning_trades']}
Losing Trades: {performance['losing_trades']}
Win Rate: {performance['win_rate_pct']:.1f}%

PROFIT ANALYSIS:
Gross Profit: ${performance['gross_profit']:,.2f}
Gross Loss: ${performance['gross_loss']:,.2f}
Net Profit: ${performance['net_profit']:,.2f}
Profit Factor: {performance['profit_factor']:.2f}

TRADE AVERAGES:
Average Trade: ${performance['avg_trade']:.2f}
Average Winner: ${performance['avg_win']:.2f}
Average Loser: ${performance['avg_loss']:.2f}
Best Trade: ${performance['best_trade']:.2f}
Worst Trade: ${performance['worst_trade']:.2f}
Average Trade Duration: {performance['avg_trade_duration_hours']:.1f} hours

EXIT ANALYSIS:
"""
        
        for reason, count in performance['exit_reasons'].items():
            percentage = (count / performance['total_trades']) * 100
            report += f"{reason.replace('_', ' ').title()}: {count} ({percentage:.1f}%)\n"
        
        report += f"""
FINAL PORTFOLIO:
Cash: ${performance['final_cash']:,.2f}
Positions Value: ${performance['final_positions_value']:,.2f}

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        return report


def example_strategy(data: Dict[str, pd.DataFrame], current_time: datetime) -> List[Dict]:
    """Example momentum strategy"""
    signals = []
    
    for symbol, df in data.items():
        if len(df) < 20:
            continue
        
        # Simple momentum strategy
        current_price = df['close'].iloc[-1]
        sma_5 = df['close'].rolling(5).mean().iloc[-1]
        sma_20 = df['close'].rolling(20).mean().iloc[-1]
        rsi = calculate_rsi(df['close']).iloc[-1]
        
        # Buy signal: Price above SMA5, SMA5 above SMA20, RSI not overbought
        if current_price > sma_5 and sma_5 > sma_20 and rsi < 70:
            signals.append({
                'symbol': symbol,
                'action': 'BUY',
                'strength': 0.8,
                'price': current_price
            })
        
        # Sell signal: Price below SMA5 or RSI overbought
        elif current_price < sma_5 or rsi > 80:
            signals.append({
                'symbol': symbol,
                'action': 'SELL',
                'strength': 0.6,
                'price': current_price
            })
    
    return signals


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def main():
    """Test the backtesting system"""
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=1000, freq='H')
    
    # Create sample price data for multiple symbols
    symbols = ['DOGE', 'PEPE', 'SHIB']
    data = {}
    
    for symbol in symbols:
        # Generate realistic price movements
        returns = np.random.normal(0.0005, 0.02, 1000)
        
        # Add some trend and volatility clustering
        trend = np.sin(np.arange(1000) * 0.01) * 0.001
        volatility = np.abs(np.sin(np.arange(1000) * 0.05)) * 0.01 + 0.01
        
        returns = returns + trend + np.random.normal(0, volatility)
        prices = 100 * np.exp(np.cumsum(returns))
        
        # Ensure positive prices
        prices = np.maximum(prices, 0.01)
        
        volume = np.random.lognormal(10, 1, 1000)
        
        data[symbol] = pd.DataFrame({
            'close': prices,
            'volume': volume,
            'open': prices * (1 + np.random.normal(0, 0.001, 1000)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.01, 1000))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.01, 1000)))
        }, index=dates)
    
    # Configure backtest
    config = BacktestConfig(
        initial_capital=10000,
        commission=0.001,
        position_size=0.3,
        stop_loss=0.15,
        take_profit=0.4
    )
    
    # Run backtest
    engine = BacktestEngine(config)
    results = engine.run_backtest(
        data=data,
        strategy_func=example_strategy,
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 2, 1)
    )
    
    # Generate and print report
    report = engine.generate_report(results)
    print(report)
    
    # Plot results
    try:
        engine.plot_results(results)
    except Exception as e:
        print(f"Could not plot results: {e}")


if __name__ == "__main__":
    main()