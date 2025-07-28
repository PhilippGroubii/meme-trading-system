#!/usr/bin/env python3
"""
Method Checker - Shows available methods in your classes
"""

import sys
import inspect
sys.path.extend(['sentiment', 'data_sources', 'ml_models', 'trading_strategies', 'risk_management'])

def check_class_methods(module_name, class_name):
    """Check what methods are available in a class"""
    try:
        module = __import__(module_name)
        cls = getattr(module, class_name)
        
        print(f"\n📋 {class_name} Methods:")
        methods = [method for method in dir(cls) if not method.startswith('_')]
        for method in methods:
            if callable(getattr(cls, method)):
                print(f"   ✅ {method}")
        
        return methods
    except Exception as e:
        print(f"❌ Error checking {class_name}: {e}")
        return []

if __name__ == "__main__":
    print("🔍 CHECKING AVAILABLE METHODS IN YOUR CLASSES")
    print("=" * 50)
    
    # Check key classes
    check_class_methods('reddit_scanner', 'RedditScanner')
    check_class_methods('price_predictor', 'PricePredictor') 
    check_class_methods('lifecycle_detector', 'LifecycleDetector')
    check_class_methods('kelly_criterion', 'KellyCriterion')
    check_class_methods('profit_optimizer', 'ProfitOptimizer')
    check_class_methods('smart_entries', 'SmartEntryOptimizer')