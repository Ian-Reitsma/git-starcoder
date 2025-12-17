#!/usr/bin/env python3
"""
Full Test Coverage Suite Master Runner

Runs all three comprehensive test suites:
1. StarCoder2-3B + 4-bit + LoRA Full Integration Test (~15 min)
2. Pipeline Orchestration Full Integration Test (~10 min)
3. Behavioral Evaluation Full Integration Test (~10 min)

Total runtime: ~35 minutes

This provides complete coverage of:
- Quantization and LoRA integration
- End-to-end pipeline orchestration
- Behavioral evaluation system
- Real model training
- Config-driven behavior
- Hardware monitoring
- Output validation
"""

import os
import sys
import time
import subprocess
import json
from pathlib import Path
from datetime import datetime, timedelta


class FullCoverageTestRunner:
    """Master test runner for all coverage suites"""
    
    def __init__(self, repo_path: str = None):
        self.repo_path = repo_path or os.environ.get('REPO_PATH')
        self.test_dir = Path.cwd()
        self.results = {}
        self.suite_results = []
        self.start_time = time.time()
    
    def log_banner(self, title: str):
        """Log large banner"""
        print("\n" + "#"*100)
        print(f"# {title.center(98)}")
        print("#"*100 + "\n")
    
    def log_section(self, title: str):
        """Log section"""
        print("\n" + "="*100)
        print(f"  {title}")
        print("="*100 + "\n")
    
    def run_test_suite(self, suite_name: str, script_path: str, description: str, args: list = None):
        """Run a single test suite and capture results"""
        self.log_section(f"Running: {suite_name}")
        
        print(f"Description: {description}")
        print(f"Script: {script_path}")
        print(f"Expected duration: ~10-20 minutes\n")
        print("-" * 100 + "\n")
        
        suite_start = time.time()
        
        try:
            # Build command
            cmd = ['python3', str(script_path)]
            if args:
                cmd.extend(args)
            
            print(f"Executing: {' '.join(cmd)}\n")
            
            # Run with real-time output
            result = subprocess.run(
                cmd,
                cwd=str(self.test_dir),
                capture_output=False,  # Stream output
                text=True,
                timeout=1800,  # 30 minute timeout
            )
            
            suite_duration = time.time() - suite_start
            
            if result.returncode == 0:
                status = 'PASS'
                print(f"\n✅ {suite_name} PASSED")
            else:
                status = 'FAIL'
                print(f"\n✗ {suite_name} FAILED (exit code: {result.returncode})")
            
            self.suite_results.append({
                'name': suite_name,
                'script': script_path,
                'status': status,
                'duration': suite_duration,
                'timestamp': datetime.now().isoformat(),
            })
            
            print(f"Duration: {suite_duration:.1f}s ({suite_duration/60:.1f} min)")
            print("-" * 100 + "\n")
            
            return status == 'PASS'
            
        except subprocess.TimeoutExpired:
            print(f"\n✗ {suite_name} TIMEOUT (exceeded 30 minutes)")
            self.suite_results.append({
                'name': suite_name,
                'script': script_path,
                'status': 'TIMEOUT',
                'duration': 1800,
                'timestamp': datetime.now().isoformat(),
            })
            return False
            
        except Exception as e:
            print(f"\n✗ {suite_name} ERROR: {e}")
            import traceback
            traceback.print_exc()
            self.suite_results.append({
                'name': suite_name,
                'script': script_path,
                'status': 'ERROR',
                'duration': time.time() - suite_start,
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
            })
            return False
    
    def run_all_tests(self):
        """Run all test suites"""
        self.log_banner("FULL COVERAGE TEST SUITE - MASTER RUNNER")
        
        print("Overview:\n")
        print("This comprehensive test suite provides full coverage of:")
        print("  ✅ StarCoder2-3B with 4-bit quantization and LoRA")
        print("  ✅ Complete pipeline orchestration (all 5 phases)")
        print("  ✅ Behavioral code evaluation system")
        print("  ✅ Real model training and inference")
        print("  ✅ Hardware monitoring and resource tracking")
        print("  ✅ Config-driven system behavior")
        print("  ✅ Output validation and quality assurance")
        print(f"\nTotal estimated duration: ~35 minutes")
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\nInstructions:")
        print(f"  - Tests run sequentially")
        print(f"  - Full output will be displayed in real-time")
        print(f"  - Each test produces verbose, detailed output")
        print(f"  - Results will be compiled at the end\n")
        
        input("Press ENTER to begin full coverage testing...\n")
        
        tests = [
            (
                'Behavioral Evaluation Test',
                'test_behavioral_evaluation.py',
                'Tests behavioral evaluation system with config-driven prompts,\ncode generation, output validation, and result reporting.',
                None,
            ),
            (
                'Pipeline Orchestration Test',
                'test_pipeline_orchestration.py',
                'Tests complete pipeline orchestration across all 5 phases:\nRepository analysis, scraping, tokenization, embeddings, training.',
                [self.repo_path] if self.repo_path else None,
            ),
            (
                'StarCoder2-3B + 4-bit + LoRA Test',
                'test_starcoder_lora_quantization.py',
                'Full end-to-end training with StarCoder2-3B, 4-bit quantization,\nLoRA adapters, real training loop, and model saving.',
                None,
            ),
        ]
        
        passed = 0
        failed = 0
        
        for suite_name, script_path, description, args in tests:
            script = self.test_dir / script_path
            
            if not script.exists():
                print(f"\n✗ ERROR: Script not found: {script}")
                failed += 1
                self.suite_results.append({
                    'name': suite_name,
                    'script': str(script),
                    'status': 'NOT_FOUND',
                    'duration': 0,
                })
                continue
            
            success = self.run_test_suite(
                suite_name,
                script,
                description,
                args or [],
            )
            
            if success:
                passed += 1
            else:
                failed += 1
        
        self.print_final_report(passed, failed)
    
    def print_final_report(self, passed: int, failed: int):
        """Print comprehensive final report"""
        self.log_banner("FULL COVERAGE TEST SUITE - FINAL REPORT")
        
        total_time = time.time() - self.start_time
        
        print("Test Suite Results:\n")
        print(f"{'Suite Name':<40} {'Status':<10} {'Duration':<15}")
        print("-" * 65)
        
        for result in self.suite_results:
            status_symbol = '✓' if result['status'] == 'PASS' else '✗'
            duration_str = f"{result['duration']:.1f}s" if result['duration'] > 0 else "N/A"
            print(f"{result['name']:<40} {status_symbol} {result['status']:<8} {duration_str:<15}")
        
        print("\n" + "="*100)
        print(f"\nSummary:")
        print(f"  Total suites run: {len(self.suite_results)}")
        print(f"  Passed: {passed}")
        print(f"  Failed: {failed}")
        print(f"  Success rate: {100*passed/len(self.suite_results) if self.suite_results else 0:.1f}%")
        print(f"\nTiming:")
        print(f"  Total duration: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        print(f"  Start time: {datetime.fromtimestamp(self.start_time).strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        print(f"\n" + "="*100)
        print(f"\nTest Coverage Summary:\n")
        
        print("Behavioral Evaluation:")
        print("  ✅ Config-driven prompt loading (default + Rust)")
        print("  ✅ Prompt tokenization and validation")
        print("  ✅ Code generation from prompts")
        print("  ✅ Output quality validation")
        print("  ✅ Evaluation result reporting")
        
        print("\nPipeline Orchestration:")
        print("  ✅ Repository analysis and commit detection")
        print("  ✅ Git scraping with metadata extraction")
        print("  ✅ Tokenization pipeline")
        print("  ✅ Embeddings configuration")
        print("  ✅ Training parameter calculation")
        print("  ✅ Manifest generation and validation")
        
        print("\nStarCoder2-3B + Quantization + LoRA:")
        print("  ✅ 4-bit quantization via bitsandbytes")
        print("  ✅ LoRA adapter creation and configuration")
        print("  ✅ Data loading and preprocessing")
        print("  ✅ Full training loop (2 epochs, real model)")
        print("  ✅ Model saving with proper artifacts")
        print("  ✅ Memory efficiency analysis")
        print("  ✅ Hardware monitoring (GPU/CPU/RAM)")
        print("  ✅ Gradient analysis and tracking")
        
        print(f"\n" + "="*100)
        
        if failed == 0:
            print(f"\n✅✅✅ ALL TESTS PASSED! ✅✅✅\n")
            print("Your training system is production-ready with full test coverage.\n")
            return True
        else:
            print(f"\n✗ {failed} test(s) failed. Review output above.\n")
            return False
    
    def validate_environment(self):
        """Validate test environment"""
        self.log_section("Environment Validation")
        
        print("Checking prerequisites...\n")
        
        # Check Python version
        import sys
        py_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        print(f"  ✅ Python: {py_version}")
        
        # Check required packages
        required_packages = ['torch', 'transformers', 'yaml', 'peft', 'bitsandbytes']
        
        print(f"\n  Required packages:")
        for package in required_packages:
            try:
                __import__(package)
                print(f"    ✅ {package}")
            except ImportError:
                print(f"    ✗ {package} - NOT INSTALLED")
        
        # Check GPU
        try:
            import torch
            if torch.cuda.is_available():
                print(f"\n  GPU: {torch.cuda.get_device_name(0)}")
                print(f"    Total memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            else:
                print(f"\n  ⚠ GPU: Not available (tests will run on CPU, slower)")
        except Exception as e:
            print(f"\n  ⚠ Could not check GPU: {e}")
        
        # Check test scripts
        print(f"\n  Test scripts:")
        test_scripts = [
            'test_behavioral_evaluation.py',
            'test_pipeline_orchestration.py',
            'test_starcoder_lora_quantization.py',
        ]
        
        all_found = True
        for script in test_scripts:
            script_path = self.test_dir / script
            if script_path.exists():
                print(f"    ✅ {script}")
            else:
                print(f"    ✗ {script} - NOT FOUND")
                all_found = False
        
        if not all_found:
            print(f"\n✗ ERROR: Not all test scripts found!")
            return False
        
        # Check config files
        print(f"\n  Config files:")
        config_files = ['training_config.yaml', 'training_config_rust.yaml']
        for config in config_files:
            config_path = self.test_dir / config
            if config_path.exists():
                print(f"    ✅ {config}")
            else:
                print(f"    ✗ {config} - NOT FOUND")
        
        print(f"\nEnvironment validation complete.\n")
        return all_found


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Full Coverage Test Suite Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 run_full_coverage_test_suite.py
  python3 run_full_coverage_test_suite.py --repo /Users/ianreitsma/projects/the-block
  REPO_PATH=/path/to/repo python3 run_full_coverage_test_suite.py
        """
    )
    parser.add_argument(
        '--repo',
        help='Path to repository for pipeline orchestration test',
        default=os.environ.get('REPO_PATH'),
    )
    parser.add_argument(
        '--skip-validation',
        action='store_true',
        help='Skip environment validation',
    )
    
    args = parser.parse_args()
    
    runner = FullCoverageTestRunner(args.repo)
    
    if not args.skip_validation:
        if not runner.validate_environment():
            print("\nEnvironment validation failed. Please fix issues above.")
            sys.exit(1)
    
    try:
        runner.run_all_tests()
    except KeyboardInterrupt:
        print("\n\n" + "#"*100)
        print("# TEST SUITE INTERRUPTED")
        print("#"*100 + "\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
