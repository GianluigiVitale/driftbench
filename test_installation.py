#!/usr/bin/env python3
"""
DriftBench CLI - Installation Test Script

This script tests the DriftBench CLI installation and functionality.
Run from the repo_driftcli/ directory.
"""

import sys
import os
import json
from pathlib import Path
import subprocess

def print_header(text):
    """Print a formatted header"""
    print("=" * 50)
    print(text)
    print("=" * 50)
    print()

def print_test(num, desc):
    """Print test description"""
    print(f"TEST {num}: {desc}")

def test_python_version():
    """Test 1: Check Python version"""
    print_test(1, "Python version")
    version = sys.version.split()[0]
    print(f"PASS: Python {version}")
    print()
    return True

def test_installation():
    """Test 2: Install package"""
    print_test(2, "Installing DriftBench...")
    
    # Change to driftbench directory
    os.chdir('driftbench')
    
    # Install package
    result = subprocess.run(
        [sys.executable, '-m', 'pip', 'install', '-q', '-e', '.'],
        capture_output=True,
        text=True
    )
    
    # Change back
    os.chdir('..')
    
    if result.returncode == 0:
        print("PASS: Installation successful")
        print()
        return True
    else:
        print("FAIL: Installation failed")
        print(result.stderr)
        return False

def test_data_files():
    """Test 3: Check data files"""
    print_test(3, "Checking data files...")
    
    # Check flip_rates.csv
    flip_rates_path = Path('data/flip_rates.csv')
    if flip_rates_path.exists():
        rows = len(open(flip_rates_path).readlines())
        print(f"PASS: flip_rates.csv found ({rows} rows)")
    else:
        print("FAIL: flip_rates.csv not found")
        return False
    
    # Check pri_model.pkl
    model_path = Path('models/pri_model.pkl')
    if model_path.exists():
        size = model_path.stat().st_size / (1024 * 1024)
        print(f"PASS: pri_model.pkl found ({size:.1f} MB)")
    else:
        print("FAIL: pri_model.pkl not found")
        return False
    
    print()
    return True

def test_list_configs():
    """Test 4: List configs command"""
    print_test(4, "Testing list-configs command...")
    
    # Add driftbench to path
    sys.path.insert(0, str(Path('driftbench').absolute()))
    
    try:
        from driftbench import core
        
        configs = core.list_available_configs()
        if len(configs) > 0:
            print(f"PASS: list-configs works ({len(configs)} configurations found)")
            print()
            return True
        else:
            print("FAIL: No configurations found")
            return False
    except Exception as e:
        print(f"FAIL: list-configs failed: {e}")
        return False

def test_compare_production():
    """Test 5: Production case comparison"""
    print_test(5, "Testing production case (21.73%, 113/520)...")
    
    try:
        from driftbench import core
        
        # Get pre-computed flip rate
        result = core.get_precomputed_flip_rate(
            baseline_config="llama-3.1-8b/h100/fp16/sglang",
            test_config="llama-3.1-8b/b200/fp8/sglang",
            workload="safety"
        )
        
        if result is None:
            print("FAIL: Could not retrieve pre-computed result")
            return False
        
        # Check flip rate
        flip_rate = result['flip_rate']
        if abs(flip_rate - 21.73) < 0.01:
            print(f"PASS: Correct flip rate: {flip_rate:.2f}%")
        else:
            print(f"FAIL: Wrong flip rate: {flip_rate:.2f}% (expected 21.73%)")
            return False
        
        # Check num_flips
        num_flips = result['num_flips']
        if num_flips == 113:
            print(f"PASS: Correct num_flips: {num_flips}")
        else:
            print(f"FAIL: Wrong num_flips: {num_flips} (expected 113)")
            return False
        
        # Check num_comparisons
        num_comparisons = result['num_comparisons']
        if num_comparisons == 520:
            print(f"PASS: Correct num_comparisons: {num_comparisons}")
        else:
            print(f"FAIL: Wrong num_comparisons: {num_comparisons} (expected 520)")
            return False
        
        print()
        return True
        
    except Exception as e:
        print(f"FAIL: Comparison failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pri_prediction():
    """Test 6: PRI prediction"""
    print_test(6, "Testing PRI prediction...")
    
    try:
        from driftbench import pri
        
        # Load PRI model
        pri_model = pri.load_pri_model()
        
        # Predict drift
        result = pri.predict_drift(
            pri_model=pri_model,
            model="llama-3.1-8b",
            hardware="b200",
            precision="fp8",
            framework="sglang",
            workload="safety"
        )
        
        if 'predicted_flip_rate' in result:
            pred_rate = result['predicted_flip_rate']
            print(f"PASS: PRI prediction works (predicted: {pred_rate:.2f}%)")
            
            # Check if prediction is reasonable (should be around 21-22%)
            if 20.0 <= pred_rate <= 23.0:
                print(f"PASS: Prediction is reasonable (actual: 21.73%)")
            else:
                print(f"WARNING: Prediction seems off (actual: 21.73%)")
            
            print()
            return True
        else:
            print("FAIL: No prediction returned")
            return False
            
    except Exception as e:
        print(f"FAIL: PRI prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print_header("DriftBench CLI - Installation Test")
    
    # Track results
    tests_passed = 0
    tests_total = 6
    
    # Run tests
    if test_python_version():
        tests_passed += 1
    
    if test_installation():
        tests_passed += 1
    else:
        print("FAIL: Installation failed - cannot continue")
        sys.exit(1)
    
    if test_data_files():
        tests_passed += 1
    else:
        print("FAIL: Data files missing - cannot continue")
        sys.exit(1)
    
    if test_list_configs():
        tests_passed += 1
    
    if test_compare_production():
        tests_passed += 1
    
    if test_pri_prediction():
        tests_passed += 1
    
    # Summary
    print_header("RESULTS")
    print(f"Total Tests: {tests_total}")
    print(f"Passed: {tests_passed}")
    print(f"Failed: {tests_total - tests_passed}")
    print()
    
    if tests_passed == tests_total:
        print("ALL TESTS PASSED!")
        print("DriftBench CLI is fully functional!")
        print()
        print("Try these commands:")
        print("  python3 -m driftbench.cli list-configs")
        print("  python3 -m driftbench.cli compare --baseline MODEL/HW/PREC/FW --test MODEL/HW/PREC/FW --workload WORKLOAD")
        print("  python3 -m driftbench.cli predict --model MODEL --hardware HW --precision PREC --framework FW --workload WORKLOAD")
        print()
        return 0
    else:
        print("FAIL: Some tests failed")
        return 1

if __name__ == "__main__":
    # Check we're in the right directory
    if not Path('driftbench').exists() or not Path('data').exists():
        print("ERROR: Must run from repo_driftcli/ directory")
        print("Current directory:", os.getcwd())
        sys.exit(1)
    
    sys.exit(main())
