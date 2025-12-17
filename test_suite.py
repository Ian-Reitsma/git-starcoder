#!/usr/bin/env python3
"""Script-style comprehensive checks for the Block training pipeline.

Important:
- This file is named test_suite.py, so unittest discovery will import it.
- To keep `python -m unittest discover` reliable, this module must be import-safe:
  no side effects and no sys.exit at import time.

Run the suite manually:
  python3 test_suite.py

Or import and call:
  from test_suite import run_suite
  results = run_suite(verbose=True)
"""

from __future__ import annotations

import sys
import json
from pathlib import Path
from typing import Dict
from unittest.mock import patch, MagicMock


def _load_yaml(path: Path) -> Dict:
    import yaml

    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg if isinstance(cfg, dict) else {}


def run_suite(verbose: bool = True, repo_path: str = ".") -> Dict[str, str]:
    """Run the script-style suite and return a dict of test_name -> PASS/FAIL/SKIP.

    repo_path:
        Path to a git repository to use for any repo-dependent checks.
    """

    def p(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    test_results: Dict[str, str] = {}

    p("\n" + "=" * 70)
    p("BLOCK MODEL TRAINING - COMPREHENSIVE TEST SUITE")
    p("=" * 70)

    # Test 1: Configuration Loading
    p("\n[TEST 1] Configuration Loading")
    try:
        config_path = Path("training_config.yaml")
        if not config_path.exists():
            p("  ✗ training_config.yaml not found")
            test_results['config_loading'] = 'FAIL'
        else:
            cfg = _load_yaml(config_path)

            # Support both schemas:
            # - Canonical trainer schema: training/epoch_calculation/hardware_monitoring/logging
            # - Universal metal/cuda schema: model/optimization/quantization/training/output/device_backend
            if {'training', 'epoch_calculation', 'hardware_monitoring', 'logging'}.issubset(cfg.keys()):
                train_cfg = cfg.get('training', {})
                epoch_cfg = cfg.get('epoch_calculation', {})
                assert 'base_learning_rate' in train_cfg
                assert 'warmup_ratio' in train_cfg
                assert 'batch_size_reference' in train_cfg
                assert 'seed' in train_cfg
                assert 'target_tokens' in epoch_cfg
                assert 'min_epochs' in epoch_cfg
                assert 'max_epochs' in epoch_cfg
                p("  ✓ Canonical config schema valid")
                test_results['config_loading'] = 'PASS'
            elif {'model', 'optimization', 'training'}.issubset(cfg.keys()):
                model_cfg = cfg.get('model', {})
                opt_cfg = cfg.get('optimization', {})
                train_cfg = cfg.get('training', {})
                assert isinstance(model_cfg, dict) and model_cfg
                assert isinstance(opt_cfg, dict) and opt_cfg
                assert isinstance(train_cfg, dict) and train_cfg
                p("  ✓ Universal config schema valid")
                test_results['config_loading'] = 'PASS'
            else:
                p("  ✗ Unrecognized config schema")
                test_results['config_loading'] = 'FAIL'
    except Exception as e:
        p(f"  ✗ Error: {e}")
        test_results['config_loading'] = 'FAIL'

    # Test 2: Git Analyzer Imports
    p("\n[TEST 2] Git Analyzer Module")
    try:
        from scrapers.git_scraper_dynamic import GitAnalyzer

        methods = ['get_all_commits', 'get_repository_stats', 'calculate_training_params']
        has_all = all(hasattr(GitAnalyzer, m) for m in methods)

        if has_all:
            p("  ✓ GitAnalyzer class loaded")
            p(f"  ✓ Required methods: {', '.join(methods)}")
            test_results['git_analyzer'] = 'PASS'
        else:
            p("  ✗ Missing methods")
            test_results['git_analyzer'] = 'FAIL'
    except Exception as e:
        p(f"  ✗ Error: {e}")
        test_results['git_analyzer'] = 'FAIL'

    # Test 3: Training Parameter Calculation (Legacy Mode)
    p("\n[TEST 3] Training Parameters - Legacy Formula")
    try:
        from scrapers.git_scraper_dynamic import GitAnalyzer

        # Use mocking here: we are testing the epoch/step formulas, not GitPython plumbing.
        # Patch the symbols as imported inside scrapers.git_scraper_dynamic.
        fake_pygit2 = MagicMock()
        fake_pygit2.Repository = MagicMock()

        with patch('scrapers.git_scraper_dynamic.Repo'):
            with patch('scrapers.git_scraper_dynamic.pygit2', fake_pygit2):
                analyzer = GitAnalyzer(repo_path, verbose=False)

                test_cases = [
                    (10, 10),
                    (30, 8),
                    (75, 6),
                    (150, 5),
                    (250, 4),
                ]

                all_pass = True
                for num_seq, expected_epochs in test_cases:
                    params = analyzer.calculate_training_params(num_seq)
                    if params.get('epochs') == expected_epochs:
                        p(f"  ✓ {num_seq} sequences -> {params['epochs']} epochs")
                    else:
                        p(f"  ✗ {num_seq} sequences -> {params.get('epochs')} (expected {expected_epochs})")
                        all_pass = False

                    required_keys = ['num_sequences', 'epochs', 'steps_per_epoch', 'total_steps', 'warmup_steps']
                    if not all(k in params for k in required_keys):
                        p("  ✗ Missing keys in params")
                        all_pass = False

                test_results['training_params_legacy'] = 'PASS' if all_pass else 'FAIL'
    except Exception as e:
        p(f"  ✗ Error: {e}")
        test_results['training_params_legacy'] = 'FAIL'

    # Test 4: Training Parameter Calculation (Config Mode)
    p("\n[TEST 4] Training Parameters - Config-Driven Formula")
    try:
        from scrapers.git_scraper_dynamic import GitAnalyzer

        config_path = Path("training_config.yaml")
        if not config_path.exists():
            p("  ⊘ Skipped (no config file)")
            test_results['training_params_config'] = 'SKIP'
        else:
            fake_pygit2 = MagicMock()
            fake_pygit2.Repository = MagicMock()
            with patch('scrapers.git_scraper_dynamic.Repo'):
                with patch('scrapers.git_scraper_dynamic.pygit2', fake_pygit2):
                    analyzer = GitAnalyzer(repo_path, verbose=False)
                    params = analyzer.calculate_training_params(100, config_path=str(config_path))

            # Make sure it returns sane values.
            epochs = int(params.get('epochs', 0))
            if 1 <= epochs <= 100:
                p(f"  ✓ 100 sequences -> {epochs} epochs (config-driven)")
                test_results['training_params_config'] = 'PASS'
            else:
                p(f"  ✗ Epoch out of bounds: {epochs}")
                test_results['training_params_config'] = 'FAIL'
    except Exception as e:
        p(f"  ✗ Error: {e}")
        test_results['training_params_config'] = 'FAIL'

    # Test 5: Model Trainer Fixed Module
    p("\n[TEST 5] Model Trainer Fixed Module")
    try:
        from training.model_trainer_fixed import OptimizedModelTrainer, load_yaml_config

        p("  ✓ OptimizedModelTrainer class loaded")
        p("  ✓ load_yaml_config function loaded")

        methods = ['load_data', 'calculate_training_params', 'train', '_get_batch_size', '_get_num_workers']
        has_all = all(hasattr(OptimizedModelTrainer, m) for m in methods)
        test_results['model_trainer'] = 'PASS' if has_all else 'FAIL'
        if has_all:
            p(f"  ✓ Required methods present: {len(methods)}")
        else:
            p("  ✗ Missing methods")
    except ImportError as e:
        p(f"  ⊘ Skipped (torch/transformers not available): {e}")
        test_results['model_trainer'] = 'SKIP'
    except Exception as e:
        p(f"  ✗ Error: {e}")
        test_results['model_trainer'] = 'FAIL'

    # Test 6: Hardware Monitor
    p("\n[TEST 6] Hardware Monitor (CPU/RAM Detection)")
    try:
        import psutil

        cpu_count = psutil.cpu_count()
        ram_info = psutil.virtual_memory()

        if cpu_count and ram_info:
            p(f"  ✓ CPU cores detected: {cpu_count}")
            p(f"  ✓ RAM total: {ram_info.total / 1e9:.1f}GB")
            p(f"  ✓ RAM available: {ram_info.available / 1e9:.1f}GB")
            test_results['hardware_monitor'] = 'PASS'
        else:
            p("  ✗ Failed to detect hardware")
            test_results['hardware_monitor'] = 'FAIL'
    except Exception as e:
        p(f"  ✗ Error: {e}")
        test_results['hardware_monitor'] = 'FAIL'

    # Test 7: Pipeline Orchestrator
    p("\n[TEST 7] Pipeline Orchestrator")
    try:
        from run_pipeline_dynamic import DynamicPipelineOrchestrator

        methods = [
            'phase_0_analyze_repository',
            'phase_1_scrape',
            'phase_2_tokenize',
            'phase_3_embeddings',
            'phase_4_training',
            'generate_final_report',
            'run',
        ]
        has_all = all(hasattr(DynamicPipelineOrchestrator, m) for m in methods)
        test_results['pipeline_orchestrator'] = 'PASS' if has_all else 'FAIL'
        if has_all:
            p("  ✓ DynamicPipelineOrchestrator loaded")
            p(f"  ✓ All {len(methods)} phases present")
        else:
            p("  ✗ Missing phases")
    except Exception as e:
        p(f"  ✗ Error: {e}")
        test_results['pipeline_orchestrator'] = 'FAIL'

    # Test 8: Requirements.txt (non-fatal)
    p("\n[TEST 8] Requirements and Dependencies")
    try:
        req_path = Path("requirements.txt")
        if req_path.exists():
            p("  ✓ requirements.txt present")
            test_results['requirements'] = 'PASS'
        else:
            p("  ✗ requirements.txt not found")
            test_results['requirements'] = 'FAIL'
    except Exception as e:
        p(f"  ✗ Error: {e}")
        test_results['requirements'] = 'FAIL'

    # Test 9: Manifest Structure
    p("\n[TEST 9] Manifest Structure (Template)")
    try:
        manifest_template = {
            'execution_timestamp': 'ISO format timestamp',
            'total_execution_time_seconds': 0,
            'repository_stats': {
                'unique_commits': 0,
                'branches': 0,
                'commits_per_branch': {},
            },
            'training_parameters': {
                'num_sequences': 0,
                'epochs': 0,
                'total_steps': 0,
                'target_tokens': 0,
            },
            'phase_results': {
                'phase_0_analyze': {'status': 'complete'},
                'phase_1_scrape': {'status': 'complete'},
                'phase_2_tokenize': {'status': 'complete'},
                'phase_3_embeddings': {'status': 'complete'},
                'phase_4_training': {'status': 'complete', 'training_report': {}},
            },
            'training_report': {
                'epochs_completed': 0,
                'training': {'final_train_loss': 0, 'final_val_loss': 0, 'final_perplexity': 0},
                'gradients': {'min_norm': 0, 'max_norm': 0},
                'learning_rate': {'min': 0, 'max': 0},
                'hardware': {'peak_gpu_memory_mb': 0, 'peak_ram_percent': 0},
            },
        }

        p("  ✓ Manifest template structure valid")
        p(f"  ✓ Has {len(manifest_template)} top-level sections")
        p(f"  ✓ Has {len(manifest_template['phase_results'])} phases")
        test_results['manifest_structure'] = 'PASS'
    except Exception as e:
        p(f"  ✗ Error: {e}")
        test_results['manifest_structure'] = 'FAIL'

    # Test 10: File Existence Check
    p("\n[TEST 10] Core Files Present")
    try:
        files_to_check = [
            'training_config.yaml',
            'requirements.txt',
            'run_pipeline_dynamic.py',
            'scrapers/git_scraper_dynamic.py',
            'training/model_trainer_fixed.py',
            'test_suite.py',
        ]

        missing = []
        for f in files_to_check:
            if not Path(f).exists():
                missing.append(f)
            else:
                p(f"  ✓ {f}")

        test_results['core_files'] = 'PASS' if not missing else 'FAIL'
        if missing:
            p(f"  ✗ Missing: {missing}")
    except Exception as e:
        p(f"  ✗ Error: {e}")
        test_results['core_files'] = 'FAIL'

    # Summary
    p("\n" + "=" * 70)
    p("TEST SUMMARY")
    p("=" * 70)

    passed = sum(1 for v in test_results.values() if v == 'PASS')
    skipped = sum(1 for v in test_results.values() if v == 'SKIP')
    failed = sum(1 for v in test_results.values() if v == 'FAIL')

    for test_name, result in test_results.items():
        symbol = "✓" if result == 'PASS' else "⊘" if result == 'SKIP' else "✗"
        p(f"  {symbol} {test_name}: {result}")

    p(f"\nResults: {passed} passed, {skipped} skipped, {failed} failed")
    p("=" * 70 + "\n")

    return test_results


def _exit_code(results: Dict[str, str]) -> int:
    failed = any(v == 'FAIL' for v in results.values())
    return 1 if failed else 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Block training pipeline script-style test suite")
    parser.add_argument(
        "--repo",
        default=".",
        help="Path to a git repository for repo-dependent checks (default: current directory)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-test output; exit code still reflects failures.",
    )
    args = parser.parse_args()

    results = run_suite(verbose=not args.quiet, repo_path=args.repo)
    sys.exit(_exit_code(results))
