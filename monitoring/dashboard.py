"""
Real-time Trading Dashboard
Monitor your system performance, P&L, and market conditions
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
import time

class TradingDashboard:
    def __init__(self):
        self.setup_page()
        
    def setup_page(self):
        """Configure Streamlit page"""
        st.set_page_config(
            page_title="🚀 Meme Coin Trading Dashboard",
            page_icon="🚀",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS
        st.markdown("""
        <style>
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
        }
        .profit-positive { color: #00ff00; }
        .profit-negative { color: #ff0000; }
        .big-number {
            font-size: 2rem;
            font-weight: bold;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def create_portfolio_overview(self):
        """Create portfolio overview section"""
        st.header("💰 Portfolio Overview")
        
        # Mock data - replace with real portfolio data
        portfolio_data = {
            'total_value': 15750.00,
            'starting_value': 10000.00,
            'cash': 3250.00,
            'positions_value': 12500.00,
            'daily_pnl': 450.00,
            'total_pnl': 5750.00,
            'win_rate': 0.73,
            'active_positions': 3
        }
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Portfolio Value",
                f"${portfolio_data['total_value']:,.2f}",
                f"${portfolio_data['daily_pnl']:+,.2f}"
            )
        
        with col2:
            total_return = (portfolio_data['total_value'] - portfolio_data['starting_value']) / portfolio_data['starting_value']
            st.metric(
                "Total Return",
                f"{total_return:.1%}",
                f"${portfolio_data['total_pnl']:+,.2f}"
            )
        
        with col3:
            st.metric(
                "Win Rate",
                f"{portfolio_data['win_rate']:.1%}",
                "Last 30 trades"
            )
        
        with col4:
            st.metric(
                "Active Positions",
                portfolio_data['active_positions'],
                "3 coins"
            )
        
        # Portfolio allocation pie chart
        col1, col2 = st.columns(2)
        
        with col1:
            # Cash vs Positions
            allocation_data = {
                'Type': ['Cash', 'Positions'],
                'Value': [portfolio_data['cash'], portfolio_data['positions_value']]
            }
            
            fig_allocation = px.pie(
                pd.DataFrame(allocation_data),
                values='Value',
                names='Type',
                title="Portfolio Allocation"
            )
            st.plotly_chart(fig_allocation, use_container_width=True)
        
        with col2:
            # Position breakdown
            positions_data = {
                'Symbol': ['DOGE', 'SHIB', 'PEPE'],
                'Value': [5000, 4500, 3000],
                'P&L': [750, -200, 450]
            }
            
            fig_positions = px.bar(
                pd.DataFrame(positions_data),
                x='Symbol',
                y='Value',
                color='P&L',
                title="Current Positions",
                color_continuous_scale='RdYlGn'
            )
            st.plotly_chart(fig_positions, use_container_width=True)
    
    def create_performance_charts(self):
        """Create performance tracking charts"""
        st.header("📈 Performance Analysis")
        
        # Generate mock performance data
        dates = pd.date_range(start='2024-01-01', end=datetime.now(), freq='D')
        np.random.seed(42)
        
        # Portfolio value over time
        portfolio_values = [10000]
        for i in range(1, len(dates)):
            daily_return = np.random.normal(0.02, 0.08)  # 2% avg, 8% volatility
            new_value = portfolio_values[-1] * (1 + daily_return)
            portfolio_values.append(new_value)
        
        performance_df = pd.DataFrame({
            'Date': dates,
            'Portfolio_Value': portfolio_values
        })
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Portfolio Value Over Time', 'Daily Returns', 
                          'Monthly Performance', 'Drawdown'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Portfolio value
        fig.add_trace(
            go.Scatter(
                x=performance_df['Date'],
                y=performance_df['Portfolio_Value'],
                name='Portfolio Value',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        # Daily returns
        daily_returns = performance_df['Portfolio_Value'].pct_change().dropna()
        fig.add_trace(
            go.Histogram(
                x=daily_returns,
                name='Daily Returns',
                nbinsx=50,
                marker_color='green'
            ),
            row=1, col=2
        )
        
        # Monthly performance
        monthly_returns = performance_df.set_index('Date')['Portfolio_Value'].resample('M').last().pct_change().dropna()
        fig.add_trace(
            go.Bar(
                x=monthly_returns.index,
                y=monthly_returns.values,
                name='Monthly Returns',
                marker_color=['green' if x > 0 else 'red' for x in monthly_returns.values]
            ),
            row=2, col=1
        )
        
        # Drawdown
        rolling_max = performance_df['Portfolio_Value'].expanding().max()
        drawdown = (performance_df['Portfolio_Value'] - rolling_max) / rolling_max
        fig.add_trace(
            go.Scatter(
                x=performance_df['Date'],
                y=drawdown,
                name='Drawdown',
                fill='tonexty',
                line=dict(color='red', width=1)
            ),
            row=2, col=2
        )
        
        fig.update_layout(height=800, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    def create_trading_activity(self):
        """Create trading activity section"""
        st.header("📊 Trading Activity")
        
        # Recent trades table
        st.subheader("Recent Trades")
        
        trades_data = {
            'Timestamp': ['2024-07-26 10:30', '2024-07-26 11:45', '2024-07-26 14:20'],
            'Symbol': ['DOGE', 'SHIB', 'PEPE'],
            'Action': ['BUY', 'SELL_PROFIT_1', 'BUY'],
            'Quantity': ['37,610', '125,000', '21,768'],
            'Price': ['$0.053178', '$0.000028', '$0.091877'],
            'Value': ['$2,000.00', '$3,500.00', '$2,000.00'],
            'P&L': ['-', '+$875.00', '-']
        }
        
        trades_df = pd.DataFrame(trades_data)
        st.dataframe(trades_df, use_container_width=True)
        
        # Trading statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Today's Stats")
            st.write("🎯 Trades Executed: 8")
            st.write("💰 Profit Taken: $1,250")
            st.write("📈 Positions Opened: 3")
            st.write("📉 Positions Closed: 2")
            st.write("⏱️ Avg Hold Time: 4.2 hours")
        
        with col2:
            st.subheader("Signal Quality")
            st.write("🔍 Scans Performed: 24")
            st.write("✅ Buy Signals: 5")
            st.write("⚠️ Wait Signals: 19")
            st.write("🎯 Signal Accuracy: 87%")
            st.write("📊 Avg Confidence: 0.73")
    
    def create_market_overview(self):
        """Create market overview section"""
        st.header("🌍 Market Overview")
        
        # Watchlist with current prices
        watchlist_data = {
            'Symbol': ['DOGE', 'SHIB', 'PEPE', 'BONK', 'FLOKI'],
            'Price': ['$0.237265', '$0.000024', '$0.000008', '$0.000035', '$0.000156'],
            'Change_24h': ['+5.2%', '-2.1%', '+12.7%', '+8.9%', '-0.8%'],
            'Volume': ['2.1B', '847M', '456M', '234M', '123M'],
            'Sentiment': ['0.72', '-0.23', '0.85', '0.45', '0.12'],
            'Signal': ['BUY', 'WAIT', 'STRONG_BUY', 'BUY', 'WAIT']
        }
        
        watchlist_df = pd.DataFrame(watchlist_data)
        
        # Color code the signals
        def color_signals(val):
            if val == 'STRONG_BUY':
                return 'background-color: #90EE90'
            elif val == 'BUY':
                return 'background-color: #98FB98'
            elif val == 'WAIT':
                return 'background-color: #FFE4B5'
            return ''
        
        styled_df = watchlist_df.style.applymap(color_signals, subset=['Signal'])
        st.dataframe(styled_df, use_container_width=True)
        
        # Market sentiment gauge
        col1, col2 = st.columns(2)
        
        with col1:
            # Overall market sentiment
            market_sentiment = 0.65
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = market_sentiment,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Market Sentiment"},
                gauge = {
                    'axis': {'range': [None, 1]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 0.3], 'color': "lightgray"},
                        {'range': [0.3, 0.7], 'color': "gray"},
                        {'range': [0.7, 1], 'color': "lightgreen"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 0.9
                    }
                }
            ))
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        with col2:
            # Fear & Greed Index simulation
            fear_greed = 75
            fig_fear_greed = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = fear_greed,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Fear & Greed Index"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "orange"},
                    'steps': [
                        {'range': [0, 25], 'color': "red"},
                        {'range': [25, 50], 'color': "orange"},
                        {'range': [50, 75], 'color': "yellow"},
                        {'range': [75, 100], 'color': "green"}
                    ]
                }
            ))
            st.plotly_chart(fig_fear_greed, use_container_width=True)
    
    def create_system_health(self):
        """Create system health monitoring"""
        st.header("🔧 System Health")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("API Status", "🟢 Online", "All APIs operational")
        
        with col2:
            st.metric("Data Latency", "125ms", "Real-time feeds")
        
        with col3:
            st.metric("ML Models", "🟢 Active", "Predictions running")
        
        with col4:
            st.metric("Risk Level", "🟡 Normal", "2.3% portfolio risk")
        
        # System logs
        st.subheader("Recent System Events")
        logs_data = {
            'Time': ['16:45:30', '16:44:15', '16:42:20', '16:40:10'],
            'Level': ['INFO', 'SUCCESS', 'INFO', 'WARNING'],
            'Message': [
                'DOGE sentiment analysis completed (score: 0.72)',
                'Profit taken on SHIB position (+$875)',
                'Portfolio rebalancing executed',
                'High correlation detected between DOGE and SHIB'
            ]
        }
        
        logs_df = pd.DataFrame(logs_data)
        st.dataframe(logs_df, use_container_width=True)
    
    def run_dashboard(self):
        """Run the main dashboard"""
        st.title("🚀 Meme Coin Trading System Dashboard")
        st.markdown("Real-time monitoring of your automated trading system")
        
        # Sidebar controls
        st.sidebar.header("🎛️ Controls")
        
        auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
        refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 5, 60, 30)
        
        if st.sidebar.button("🛑 Emergency Stop"):
            st.sidebar.error("Trading halted - Manual intervention required")
        
        if st.sidebar.button("📊 Generate Report"):
            st.sidebar.success("Performance report generated!")
        
        # Main dashboard content
        self.create_portfolio_overview()
        self.create_performance_charts()
        self.create_trading_activity()
        self.create_market_overview()
        self.create_system_health()
        
        # Auto refresh
        if auto_refresh:
            time.sleep(refresh_interval)
            st.rerun()

# Run the dashboard
if __name__ == "__main__":
    dashboard = TradingDashboard()
    dashboard.run_dashboard()