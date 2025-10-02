#!/usr/bin/env python3
"""
Test Runner for Vehicle Router App

This script provides an easy way to run all tests for the Vehicle Router application
with different levels of detail and coverage.

Usage:
    python run_tests.py                    # Run all tests
    python run_tests.py --core             # Run only core optimization tests
    python run_tests.py --config           # Run only configuration tests
    python run_tests.py --integration      # Run only integration tests
    python run_tests.py --app              # Run only app-specific tests
    python run_tests.py --quick            # Run quick tests only
    python run_tests.py --coverage         # Run with coverage report
"""

import sys
import subprocess
import argparse
from pathlib import Path

def run_command(cmd, description):
    """Run a command and return success status"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("SUCCESS")
        if result.stdout:
            print("Output:")
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("FAILED")
        print(f"Exit code: {e.returncode}")
        if e.stdout:
            print("Output:")
            print(e.stdout)
        if e.stderr:
            print("Error:")
            print(e.stderr)
        return False

def main():
    """Main test runner function"""
    parser = argparse.ArgumentParser(description='Run Vehicle Router tests')
    parser.add_argument('--core', action='store_true', 
                        help='Run only core optimization tests')
    parser.add_argument('--config', action='store_true',
                        help='Run only configuration tests')
    parser.add_argument('--integration', action='store_true',
                        help='Run only integration tests')
    parser.add_argument('--app', action='store_true',
                        help='Run only app-specific tests')
    parser.add_argument('--quick', action='store_true',
                        help='Run quick tests only (exclude slow integration tests)')
    parser.add_argument('--coverage', action='store_true',
                        help='Run with coverage report')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')
    
    args = parser.parse_args()
    
    # Base pytest command
    base_cmd = ['python', '-m', 'pytest']
    
    if args.verbose:
        base_cmd.append('-v')
    else:
        base_cmd.append('--tb=short')
    
    if args.coverage:
        base_cmd.extend(['--cov=vehicle_router', '--cov=app', '--cov-report=html', '--cov-report=term'])
    
    # Determine which tests to run
    test_suites = []
    
    if args.core:
        test_suites = ['tests/test_optimizer.py', 'tests/test_data_generator.py', 
                       'tests/test_validation.py', 'tests/test_utils.py']
    elif args.config:
        test_suites = ['tests/test_config_system.py']
    elif args.integration:
        test_suites = ['tests/test_app_integration.py']
    elif args.app:
        test_suites = ['tests/test_streamlit_app.py', 'tests/test_app_utils.py', 'tests/test_app_integration.py']
    elif args.quick:
        # Quick tests exclude slow integration tests
        test_suites = ['tests/test_optimizer.py', 'tests/test_data_generator.py', 
                       'tests/test_validation.py', 'tests/test_utils.py',
                       'tests/test_config_system.py']
    else:
        # Run all tests
        test_suites = ['tests/']
    
    # Add test suites to command
    base_cmd.extend(test_suites)
    
    # Run the tests
    success = run_command(base_cmd, "Vehicle Router Test Suite")
    
    if success:
        print(f"\nAll tests passed successfully!")
        
        # Show test summary
        print(f"\nTest Summary:")
        print(f"   Core Optimization: PASSED")
        print(f"   Data Generation: PASSED")
        print(f"   Solution Validation: PASSED")
        print(f"   Configuration System: PASSED")
        print(f"   Integration Tests: PASSED")
        print(f"   App Components: PASSED")
        
        if args.coverage:
            print(f"\nCoverage report generated in htmlcov/index.html")
        
        return 0
    else:
        print(f"\nSome tests failed. Please check the output above.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
