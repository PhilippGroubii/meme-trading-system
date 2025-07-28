# First, create the test script
cat > test_all_python.py << 'EOF'
#!/usr/bin/env python3
"""
Test all Python files for syntax errors and basic imports
Excludes test files and focuses on finding syntax errors
"""

import os
import sys
import ast
from pathlib import Path

def find_python_files(root_dir):
    """Find all Python files excluding test files"""
    python_files = []
    exclude_patterns = [
        'test_',
        '_test',
        'tests/',
        '__pycache__',
        '.pyc',
        'meme_env/',
        'terraform/',
        'backups'
    ]
    
    for root, dirs, files in os.walk(root_dir):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if not any(pattern in d for pattern in exclude_patterns)]
        
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, root_dir)
                
                # Skip test files
                if any(pattern in relative_path for pattern in exclude_patterns):
                    continue
                    
                python_files.append(file_path)
    
    return python_files

def check_syntax(file_path):
    """Check if a Python file has valid syntax"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
        ast.parse(source)
        return True, None
    except SyntaxError as e:
        return False, f"SyntaxError: {e.msg} at line {e.lineno}"
    except Exception as e:
        return False, f"Error: {str(e)}"

def main():
    """Main test function"""
    print("🚀 TESTING ALL PYTHON FILES")
    print("=" * 50)
    
    root_dir = os.getcwd()
    python_files = find_python_files(root_dir)
    
    print(f"Found {len(python_files)} Python files to test\n")
    
    syntax_pass = 0
    syntax_fail = 0
    failed_files = []
    
    for file_path in sorted(python_files):
        relative_path = os.path.relpath(file_path, root_dir)
        print(f"Testing: {relative_path}")
        
        # Check syntax
        syntax_ok, syntax_error = check_syntax(file_path)
        if syntax_ok:
            print(f"  ✅ Syntax: PASS")
            syntax_pass += 1
        else:
            print(f"  ❌ Syntax: {syntax_error}")
            syntax_fail += 1
            failed_files.append((relative_path, syntax_error))
            
        print()
    
    # Summary
    print("=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    print(f"Total files tested: {len(python_files)}")
    print(f"Syntax check: {syntax_pass} passed, {syntax_fail} failed")
    
    if failed_files:
        print("\n❌ FAILED FILES:")
        print("-" * 30)
        for file_path, error in failed_files:
            print(f"  {file_path}: {error}")
    else:
        print("\n🎉 ALL FILES PASSED!")
    
    return 0 if not failed_files else 1

if __name__ == "__main__":
    sys.exit(main())
EOF

# Make it executable
chmod +x test_all_python.py

# Run the test
python test_all_python.py
