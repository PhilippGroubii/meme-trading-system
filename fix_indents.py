#!/usr/bin/env python3
"""
Fix Indentation Issues
"""

def fix_coin_discovery_indents():
    """Fix indentation in coin_discovery_engine.py"""
    print("🔧 Fixing indentation in coin_discovery_engine.py...")
    
    with open('discovery/coin_discovery_engine.py', 'r') as f:
        content = f.read()
    
    # Split into lines
    lines = content.split('\n')
    fixed_lines = []
    in_class = False
    in_method = False
    
    for line in lines:
        stripped = line.strip()
        
        # Skip empty lines
        if not stripped:
            fixed_lines.append('')
            continue
        
        # Comments and docstrings
        if stripped.startswith('#') or stripped.startswith('"""') or stripped.startswith("'''"):
            if in_method:
                fixed_lines.append('        ' + stripped)
            elif in_class:
                fixed_lines.append('    ' + stripped)
            else:
                fixed_lines.append(stripped)
            continue
        
        # Class definitions
        if stripped.startswith('class '):
            fixed_lines.append(stripped)
            in_class = True
            in_method = False
            continue
        
        # Method/function definitions
        if stripped.startswith('def ') or stripped.startswith('async def '):
            if in_class:
                fixed_lines.append('    ' + stripped)
                in_method = True
            else:
                fixed_lines.append(stripped)
                in_method = False
            continue
        
        # Method content
        if in_method:
            fixed_lines.append('        ' + stripped)
        elif in_class:
            fixed_lines.append('    ' + stripped)
        else:
            fixed_lines.append(stripped)
    
    # Write fixed content
    with open('discovery/coin_discovery_engine.py', 'w') as f:
        f.write('\n'.join(fixed_lines))
    
    print("   ✅ Fixed coin_discovery_engine.py")

def fix_opportunity_monitor_indents():
    """Fix indentation in opportunity_monitor.py"""
    print("🔧 Fixing indentation in opportunity_monitor.py...")
    
    with open('discovery/opportunity_monitor.py', 'r') as f:
        content = f.read()
    
    # Split into lines
    lines = content.split('\n')
    fixed_lines = []
    in_class = False
    in_method = False
    
    for line in lines:
        stripped = line.strip()
        
        # Skip empty lines
        if not stripped:
            fixed_lines.append('')
            continue
        
        # Comments and docstrings
        if stripped.startswith('#') or stripped.startswith('"""') or stripped.startswith("'''"):
            if in_method:
                fixed_lines.append('        ' + stripped)
            elif in_class:
                fixed_lines.append('    ' + stripped)
            else:
                fixed_lines.append(stripped)
            continue
        
        # Imports and module level code
        if stripped.startswith('import ') or stripped.startswith('from '):
            fixed_lines.append(stripped)
            in_class = False
            in_method = False
            continue
        
        # Class definitions
        if stripped.startswith('class '):
            fixed_lines.append(stripped)
            in_class = True
            in_method = False
            continue
        
        # Method/function definitions
        if stripped.startswith('def ') or stripped.startswith('async def '):
            if in_class:
                fixed_lines.append('    ' + stripped)
                in_method = True
            else:
                fixed_lines.append(stripped)
                in_method = False
            continue
        
        # Method content
        if in_method:
            fixed_lines.append('        ' + stripped)
        elif in_class:
            fixed_lines.append('    ' + stripped)
        else:
            fixed_lines.append(stripped)
    
    # Write fixed content
    with open('discovery/opportunity_monitor.py', 'w') as f:
        f.write('\n'.join(fixed_lines))
    
    print("   ✅ Fixed opportunity_monitor.py")

def main():
    print("🔧 FIXING ALL INDENTATION ISSUES")
    print("=" * 40)
    
    import os
    if not os.path.exists('discovery'):
        print("❌ Run from /home/philipp/memecointrader")
        return
    
    fix_coin_discovery_indents()
    fix_opportunity_monitor_indents()
    
    print("\n✅ ALL INDENTATION FIXED!")
    print("\n🧪 Test now:")
    print("   python test_discovery.py")
    print("\n🚀 Run enhanced trading:")
    print("   python discovery/enhanced_trading_system.py")

if __name__ == "__main__":
    main()
