#!/usr/bin/env python3
"""
Phase 3 Testing Suite Runner
Runs all Phase 3 ML model and integration tests
"""

import sys
import time
import traceback
from datetime import datetime

def run_test_suite():
    """Run all Phase 3 tests in sequence"""
    
    print("🚀 PHASE 3 COMPLETE TEST SUITE")
    print("="*60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Test modules to run
    test_modules = [
        ("Price Predictor Basic", "test_price_predictor"),
        ("Price Predictor Advanced", "test_price_predictor_advanced"), 
        ("Lifecycle Detector", "test_lifecycle_detector"),
        ("Sentiment Classifier", "test_sentiment_classifier"),
        ("Feature Engineer", "test_feature_engineer"),
        ("Database System", "test_database"),
        ("Backtesting Engine", "test_backtest"),
        ("Complete Integration", "test_phase3_integration")
    ]
    
    results = {}
    total_start_time = time.time()
    
    for test_name, module_name in test_modules:
        print(f"\n{'='*60}")
        print(f"🧪 RUNNING: {test_name}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            # Import and run the test
            if module_name == "test_price_predictor":
                from test_price_predictor import test_price_predictor
                test_price_predictor()
                
            elif module_name == "test_price_predictor_advanced":
                from test_price_predictor_advanced import test_advanced_scenarios
                test_advanced_scenarios()
                
            elif module_name == "test_lifecycle_detector":
                from test_lifecycle_detector import test_lifecycle_detector
                test_lifecycle_detector()
                
            elif module_name == "test_sentiment_classifier":
                from test_sentiment_classifier import test_sentiment_classifier
                test_sentiment_classifier()
                
            elif module_name == "test_feature_engineer":
                from test_feature_engineer import test_feature_engineer
                test_feature_engineer()
                
            elif module_name == "test_database":
                from test_database import test_database
                test_database()
                
            elif module_name == "test_backtest":
                from test_backtest import test_backtesting
                test_backtesting()
                
            elif module_name == "test_phase3_integration":
                from test_phase3_integration import test_phase3_integration
                test_phase3_integration()
            
            test_time = time.time() - start_time
            results[test_name] = {
                'status': 'PASSED',
                'time': test_time,
                'error': None
            }
            
            print(f"\n✅ {test_name} PASSED in {test_time:.2f}s")
            
        except ImportError as e:
            test_time = time.time() - start_time
            results[test_name] = {
                'status': 'SKIPPED',
                'time': test_time,
                'error': f"Import error: {e}"
            }
            print(f"\n⚠️ {test_name} SKIPPED: {e}")
            
        except Exception as e:
            test_time = time.time() - start_time
            results[test_name] = {
                'status': 'FAILED',
                'time': test_time,
                'error': str(e)
            }
            print(f"\n❌ {test_name} FAILED: {e}")
            print("Traceback:")
            traceback.print_exc()
    
    # Final Results Summary
    total_time = time.time() - total_start_time
    
    print(f"\n{'='*60}")
    print("🏁 PHASE 3 TEST SUITE COMPLETE")
    print(f"{'='*60}")
    
    passed = sum(1 for r in results.values() if r['status'] == 'PASSED')
    failed = sum(1 for r in results.values() if r['status'] == 'FAILED')
    skipped = sum(1 for r in results.values() if r['status'] == 'SKIPPED')
    
    print(f"\n📊 SUMMARY:")
    print(f"   Total Tests: {len(results)}")
    print(f"   ✅ Passed: {passed}")
    print(f"   ❌ Failed: {failed}")
    print(f"   ⚠️ Skipped: {skipped}")
    print(f"   ⏱️ Total Time: {total_time:.2f}s")
    
    print(f"\n📋 DETAILED RESULTS:")
    for test_name, result in results.items():
        status_emoji = {
            'PASSED': '✅',
            'FAILED': '❌', 
            'SKIPPED': '⚠️'
        }[result['status']]
        
        print(f"   {status_emoji} {test_name}: {result['status']} ({result['time']:.2f}s)")
        if result['error']:
            print(f"      Error: {result['error']}")
    
    # Grade the overall suite
    if failed == 0 and passed >= len(results) * 0.8:
        grade = "🏆 EXCELLENT"
        message = "Phase 3 is ready for production!"
    elif failed <= 1 and passed >= len(results) * 0.6:
        grade = "✅ GOOD"
        message = "Phase 3 is mostly working, minor fixes needed."
    elif failed <= 2:
        grade = "⚠️ FAIR" 
        message = "Phase 3 has some issues that need attention."
    else:
        grade = "❌ NEEDS WORK"
        message = "Phase 3 requires significant fixes before proceeding."
    
    print(f"\n🎯 OVERALL GRADE: {grade}")
    print(f"   {message}")
    
    if passed >= len(results) * 0.8:
        print(f"\n🚀 Ready for Phase 4: Profit Optimization!")
        print(f"   Next steps:")
        print(f"   1. Kelly Criterion position sizing")
        print(f"   2. Multi-level profit taking")
        print(f"   3. VWAP entry optimization")
        print(f"   4. Advanced risk management")
    else:
        print(f"\n🔧 Recommended next steps:")
        print(f"   1. Fix failing tests")
        print(f"   2. Install missing dependencies")
        print(f"   3. Check ML model implementations")
        print(f"   4. Verify database connections")
    
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return results

if __name__ == "__main__":
    try:
        results = run_test_suite()
        
        # Exit with appropriate code
        failed_count = sum(1 for r in results.values() if r['status'] == 'FAILED')
        sys.exit(failed_count)  # 0 if all passed, number of failures otherwise
        
    except KeyboardInterrupt:
        print(f"\n\n⏹️ Test suite interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n💥 Test suite crashed: {e}")
        traceback.print_exc()
        sys.exit(1)