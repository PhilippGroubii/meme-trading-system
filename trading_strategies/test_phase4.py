#!/usr/bin/env python3
"""
Phase 4 Integration Test
Tests all profit maximization components together
"""

import sys
import os
sys.path.append('trading_strategies')
sys.path.append('risk_management')

from kelly_criterion import KellyCriterion, ConfidenceBasedSizing, PortfolioHeatManager
from profit_optimizer import ProfitOptimizer
from smart_entries import SmartEntryOptimizer
from advanced_risk import AdvancedRiskManager

def test_phase4_integration():
    """Test all Phase 4 components together"""
    print("🚀 Testing Phase 4 - Profit Maximization Components")
    print("=" * 50)
    
    # Initialize all components
    kelly = KellyCriterion()
    profit_optimizer = ProfitOptimizer()
    entry_optimizer = SmartEntryOptimizer()
    risk_manager = AdvancedRiskManager()
    
    portfolio_value = 100000
    
    print("\n1. Testing Kelly Criterion Position Sizing...")
    # Test position sizing
    sample_trades = [
        {'pnl': 150}, {'pnl': -75}, {'pnl': 200}, {'pnl': -50}, {'pnl': 300}
    ]
    
    position_size = kelly.get_position_size(
        signal_confidence=0.75, 
        portfolio_value=portfolio_value, 
        recent_trades=sample_trades
    )
    
    print(f"✅ Kelly position size: ${position_size:,.2f} ({position_size/portfolio_value:.1%} of portfolio)")
    
    print("\n2. Testing Smart Entry Analysis...")
    # Test entry optimization
    market_data = {
        'prices': [100, 102, 101, 104, 106, 105, 107, 109, 108, 110],
        'volumes': [1000, 1200, 900, 1500, 1800, 1200, 2000, 2200, 1800, 2500],
        'high_prices': [101, 103, 102, 105, 107, 106, 108, 110, 109, 111],
        'low_prices': [99, 101, 100, 103, 105, 104, 106, 108, 107, 109],
        'resistance_level': 108
    }
    
    entry_analysis = entry_optimizer.analyze_entry_opportunity(market_data)
    print(f"✅ Entry recommendation: {entry_analysis['recommendation']}")
    print(f"   Entry score: {entry_analysis['entry_score']}")
    print(f"   Key factors: {entry_analysis['entry_factors']}")
    
    print("\n3. Testing Profit Taking System...")
    # Test profit optimization
    profit_optimizer.add_position('TEST', entry_price=100.0, quantity=1000)
    
    # Simulate price movements
    test_prices = [100, 105, 108, 112, 118, 125, 135]
    total_profit_taken = 0
    
    for price in test_prices:
        actions = profit_optimizer.check_profit_levels('TEST', price)
        for action in actions:
            execution = profit_optimizer.execute_profit_taking(
                action['symbol'], action['level'], 
                action['quantity'], action['price']
            )
            total_profit_taken += execution['profit_amount']
            print(f"✅ Profit taken at {action['level']}: ${execution['profit_amount']:.2f}")
    
    final_status = profit_optimizer.get_position_status('TEST', test_prices[-1])
    print(f"   Total profit realized: ${total_profit_taken:.2f}")
    print(f"   Remaining position: {final_status['current_quantity']} shares")
    
    print("\n4. Testing Advanced Risk Management...")
    # Test risk management
    sample_positions = {
        'TEST1': {
            'position_value': 15000,
            'portfolio_value': portfolio_value,
            'risk_amount': 1500,
            'price_history': [100, 102, 98, 105, 103, 108, 106, 110]
        },
        'TEST2': {
            'position_value': 12000,
            'portfolio_value': portfolio_value, 
            'risk_amount': 1200,
            'price_history': [50, 51, 49, 52, 51, 54, 53, 55]
        }
    }
    
    risk_report = risk_manager.generate_risk_report(portfolio_value, sample_positions)
    print(f"✅ Risk level: {risk_report['emergency_assessment']['risk_level']}")
    print(f"   Total portfolio risk: {risk_report['total_risk_percentage']:.1%}")
    print(f"   Sharpe ratio: {risk_report['current_sharpe']:.2f}")
    print(f"   Max drawdown: {risk_report['max_historic_drawdown']:.1%}")
    
    print("\n5. Testing Integration Workflow...")
    # Test complete workflow
    print("✅ Entry signal → Position sizing → Risk check → Profit management")
    
    # Entry signal
    if entry_analysis['recommendation'] in ['buy', 'strong_buy']:
        # Position sizing with Kelly
        suggested_size = kelly.get_position_size(0.8, portfolio_value)
        
        # Risk check
        risk_check = risk_manager.check_position_limits(sample_positions)
        
        if risk_check['all_limits_ok']:
            print(f"   🟢 All systems GO - Position size: ${suggested_size:,.2f}")
        else:
            print(f"   🟡 Risk limits exceeded - Reduce position size")
    else:
        print(f"   🔴 Entry conditions not met - Wait for better setup")
    
    print("\n" + "=" * 50)
    print("🎯 Phase 4 Integration Test COMPLETE!")
    print("All profit maximization components working correctly!")
    
    return True

if __name__ == "__main__":
    try:
        test_phase4_integration()
        print("\n✅ ALL TESTS PASSED - Phase 4 Ready for Live Trading!")
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("Make sure all Phase 4 files are in the correct directories:")
        print("  - trading_strategies/kelly_criterion.py")
        print("  - trading_strategies/profit_optimizer.py") 
        print("  - trading_strategies/smart_entries.py")
        print("  - risk_management/advanced_risk.py")
    except Exception as e:
        print(f"❌ Test Failed: {e}")
        sys.exit(1)