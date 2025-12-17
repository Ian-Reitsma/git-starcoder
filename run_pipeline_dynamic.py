#!/usr/bin/env python3
"""
Dynamic Pipeline Orchestrator

Automatically:
- Detects all commits across all branches
- Calculates true commit counts
- Determines optimal training parameters
- Runs comprehensive training with detailed stats
- Generates detailed reports
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Set
import yaml

from scrapers.file_snapshot_scraper import (
    DEFAULT_INCLUDE_EXTENSIONS,
    DEFAULT_EXCLUDE_DIRS,
    DEFAULT_MAX_FILE_BYTES,
    iter_files as snapshot_iter_files,
)
import subprocess

try:
    from scrapers.git_scraper_dynamic import GitAnalyzer
    from tqdm import tqdm
except ImportError:
    print("Install: pip install GitPython pygit2 tqdm")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DynamicPipelineOrchestrator:
    """Orchestrates the complete pipeline with dynamic analysis"""
    
    def __init__(
        self,
        repo_path: str,
        base_dir: str = None,
        verbose: bool = False,
        force: bool = False,
        sequence_length: int = 2048,  # LONG-CONTEXT: Increased from 512
        overlap: int = 512,  # LONG-CONTEXT: Increased from 128 for better context blending
        epoch_overrides: Dict[str, Any] = None,
        force_epochs: Optional[int] = None,
        long_context_mode: bool = True,  # NEW: Enable long-context optimizations
    ):
        self.repo_path = Path(repo_path)
        self.project_root = Path(__file__).resolve().parent
        self.base_dir = self._resolve_base_dir(base_dir)
        self.verbose = verbose
        self.force = force
        self.config_path = self.project_root / "training_config.yaml"
        self.training_cfg = self._load_training_config()
        self.tokenizer_model = self._extract_tokenizer_name()
        self.tokenizer_trust_remote = self.training_cfg.get("model", {}).get("trust_remote_code", False)
        self.max_sequences_limit = self.training_cfg.get("training", {}).get("max_sequences_limit")
        self.sequence_length = sequence_length
        self.overlap = overlap
        self.epoch_overrides = epoch_overrides or {}
        self.force_epochs = force_epochs
        
        # Create directories
        for dir_name in ['data', 'embeddings', 'models', 'training']:
            (self.base_dir / dir_name).mkdir(parents=True, exist_ok=True)
        
        self.results = {}
        self.stats = {}
        self.start_time = time.time()
        self.sequence_files = {}
        self.sequence_counts = {}
        self.sequence_metadata = {}
        self.training_source_label = "unknown"
    
    def _log(self, msg: str, level: str = 'info'):
        if self.verbose or level in ['error', 'warning']:
            getattr(logger, level)(msg)

    def _load_training_config(self) -> Dict[str, Any]:
        if self.config_path.exists():
            try:
                with open(self.config_path, "r") as f:
                    return yaml.safe_load(f) or {}
            except Exception as exc:
                logger.warning("Failed to read %s (%s); falling back to defaults", self.config_path, exc)
        return {}

    def _apply_epoch_override(self, training_params: Dict[str, Any]) -> None:
        if self.force_epochs:
            steps_per_epoch = training_params.get("steps_per_epoch", 0)
            training_params["epochs"] = self.force_epochs
            training_params["total_steps"] = steps_per_epoch * self.force_epochs

    def _extract_tokenizer_name(self) -> str:
        model_cfg = self.training_cfg.get("model", {})
        return model_cfg.get("tokenizer_name") or model_cfg.get("name") or "gpt2"

    def _resolve_base_dir(self, base_dir: Optional[str]) -> Path:
        """
        Normalize the base directory for outputs/scripts.
        Falls back to the project root (directory containing this file)
        when the user passes an invalid path (e.g., duplicate folder names).
        """
        project_root = self.project_root
        if not base_dir:
            return project_root

        candidate = Path(base_dir)
        if not candidate.is_absolute():
            candidate = (project_root.parent / candidate).resolve()

        if candidate.exists():
            return candidate

        logger.warning(
            "Base directory %s does not exist; defaulting to %s",
            candidate,
            project_root,
        )
        return project_root

    def _analyze_git_history_output(self, path: Path) -> Dict[str, Any]:
        """Return basic stats from an existing git_history_rich.jsonl file."""
        if not path.exists():
            return {}
        size_mb = path.stat().st_size / 1e6
        lines = 0
        with path.open("r", encoding="utf-8") as f:
            for _ in f:
                lines += 1
        return {'size_mb': size_mb, 'lines': lines}

    def _collect_snapshot_statistics(self, output_file: Path) -> Dict[str, Any]:
        """Read snapshot JSONL to compute chunk, coverage, and missing file stats."""
        chunk_count = 0
        unique_files: Set[str] = set()
        if not output_file.exists():
            return {
                'chunk_count': 0,
                'unique_files': 0,
                'expected_files': 0,
                'coverage_ratio': 1.0,
                'missing_files': [],
            }

        with output_file.open("r", encoding="utf-8") as snapshot_file:
            for line in snapshot_file:
                line = line.strip()
                if not line:
                    continue
                chunk_count += 1
                try:
                    record = json.loads(line)
                    file_path = record.get('file_path')
                    if file_path:
                        unique_files.add(file_path)
                except json.JSONDecodeError:
                    continue

        expected_files: Set[str] = set()
        for path in snapshot_iter_files(
            self.repo_path,
            DEFAULT_INCLUDE_EXTENSIONS,
            DEFAULT_EXCLUDE_DIRS,
            DEFAULT_MAX_FILE_BYTES,
        ):
            expected_files.add(path.relative_to(self.repo_path).as_posix())

        missing_files = sorted(expected_files - unique_files)
        coverage_ratio = (len(unique_files) / len(expected_files)) if expected_files else 1.0
        return {
            'chunk_count': chunk_count,
            'unique_files': len(unique_files),
            'expected_files': len(expected_files),
            'coverage_ratio': coverage_ratio,
            'missing_files': missing_files[:20],
        }

    def _log_snapshot_statistics(self, stats: Dict[str, Any], elapsed: float, output_file: Path):
        logger.info(f"\n✓ PHASE 1B COMPLETE ({self.format_time(elapsed)})")
        logger.info(f"  Snapshot chunks: {stats['chunk_count']}")
        logger.info(f"  Unique files covered: {stats['unique_files']}")
        logger.info(f"  Coverage ratio: {stats['coverage_ratio']:.2%}")
        if stats.get('missing_files'):
            logger.warning("  Missing files (first 5): %s", stats['missing_files'][:5])

    def _load_sequence_summary(self, path: Path) -> Optional[Dict[str, Any]]:
        """Load basic sequence stats from a token sequence JSON file."""
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                return None

        sequences = data.get("token_sequences", [])
        total_tokens = data.get("total_tokens")
        if total_tokens is None:
            total_tokens = sum(len(seq) for seq in sequences if isinstance(seq, list))

        sequence_length = data.get("sequence_length")
        if sequence_length is None and sequences:
            sequence_length = len(sequences[0])

        return {
            "num_sequences": data.get("num_sequences", len(sequences)),
            "sequence_length": sequence_length or 0,
            "total_tokens": total_tokens,
            "metadata": data.get("metadata", {}),
        }
    
    def phase_0_analyze_repository(self) -> Dict:
        """
        Phase 0: Analyze repository to determine all commits and parameters
        """
        logger.info("\n" + "#" * 70)
        logger.info("# PHASE 0: REPOSITORY ANALYSIS")
        logger.info("#" * 70)
        
        analyzer = GitAnalyzer(str(self.repo_path), verbose=self.verbose)
        stats, all_commits = analyzer.get_repository_stats()
        
        # Calculate training parameters
        estimated_sequences = max(1, stats['unique_commits'] // 6)
        training_params = analyzer.calculate_training_params(
            estimated_sequences,
            config_path=str(self.config_path),
            overrides=self.epoch_overrides,
        )
        self._apply_epoch_override(training_params)
        
        self.stats['repository'] = stats
        self.stats['training_params'] = training_params
        self.stats['estimated_sequences'] = estimated_sequences
        
        logger.info("\n" + "-"*70)
        logger.info("ANALYSIS SUMMARY")
        logger.info("-"*70)
        logger.info(f"\nRepository Analysis:")
        logger.info(f"  ✓ Total unique commits: {stats['unique_commits']}")
        logger.info(f"  ✓ Commits across branches: {stats['total_commits_across_branches']}")
        logger.info(f"  ✓ Branches: {stats['branches']}")
        logger.info(f"  ✓ Unique authors: {stats['unique_authors']}")
        if stats['time_span_days'] > 0:
            logger.info(f"  ✓ Repository age: {stats['time_span_days']:.0f} days")
            logger.info(f"  ✓ Commit velocity: {stats['commits_per_day']:.2f} commits/day")
        
        logger.info(f"\nEstimated Processing:")
        logger.info(f"  ✓ Token sequences to generate: {estimated_sequences}")
        logger.info(f"  ✓ Training epochs: {training_params['epochs']}")
        logger.info(f"  ✓ Steps per epoch: {training_params['steps_per_epoch']}")
        logger.info(f"  ✓ Total training steps: {training_params['total_steps']}")
        logger.info(f"  ✓ Estimated training time: {training_params['estimated_time_minutes']:.1f}m ({training_params['estimated_time_hours']:.2f}h)")
        logger.info("-"*70 + "\n")
        
        self.results['phase_0_analyze'] = {
            'status': 'complete',
            'repository_stats': stats,
            'training_parameters': training_params,
            'timestamp': datetime.now().isoformat(),
        }
        
        return stats
    
    def phase_1_scrape(self, stats: Dict) -> bool:
        """
        Phase 1: Rich Git scraping (using actual commit counts)
        """
        logger.info("\n" + "#" * 70)
        logger.info(f"# PHASE 1: GIT SCRAPING ({stats['unique_commits']} COMMITS)")
        logger.info("#" * 70)
        
        output_jsonl = self.base_dir / "data" / "git_history_rich.jsonl"
        output_json = self.base_dir / "data" / "git_history_rich.json"
        
        logger.info(f"\nWhat's happening:")
        logger.info(f"  • Analyzing {stats['unique_commits']} unique commits")
        logger.info(f"  • Processing all {stats['branches']} branches")
        logger.info(f"  • Extracting 30+ metadata fields per commit")
        logger.info(f"  • Computing complexity scores")
        logger.info(f"  • Tracking temporal patterns\n")
        
        cmd = [
            "python3",
            str(self.project_root / "scrapers" / "git_scraper_rich.py"),
            "--repo", str(self.repo_path),
            "--output", str(output_jsonl),
            "--output-json", str(output_json),
            "--stats",
        ]
        
        if self.verbose:
            cmd.append("--verbose")
        
        if not self.force and output_jsonl.exists():
            existing = self._analyze_git_history_output(output_jsonl)
            logger.info("\n➤ PHASE 1 SKIPPED (existing git history file)")
            logger.info(f"  Output: {output_jsonl.name}")
            logger.info(f"    Size: {existing.get('size_mb', 0):.1f} MB")
            logger.info(f"    Commits processed: {existing.get('lines', 0)}")
            self.results['phase_1_scrape'] = {
                'status': 'skipped',
                'output_file': str(output_jsonl),
                'size_mb': existing.get('size_mb'),
                'commits_processed': existing.get('lines'),
                'timestamp': datetime.now().isoformat(),
            }
            return True
        start = time.time()
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            elapsed = time.time() - start
            
            if output_jsonl.exists():
                size_mb = output_jsonl.stat().st_size / 1e6
                lines = sum(1 for _ in open(output_jsonl))
                
                logger.info(f"\n✓ PHASE 1 COMPLETE ({self.format_time(elapsed)})")
                logger.info(f"  Output: {output_jsonl.name}")
                logger.info(f"    Size: {size_mb:.1f} MB")
                logger.info(f"    Commits processed: {lines}")
                logger.info(f"    Rate: {lines/elapsed:.0f} commits/second\n")
                
                self.results['phase_1_scrape'] = {
                    'status': 'complete',
                    'output_file': str(output_jsonl),
                    'size_mb': size_mb,
                    'commits_processed': lines,
                    'duration_seconds': elapsed,
                    'timestamp': datetime.now().isoformat(),
                }
                return True
        except subprocess.CalledProcessError as e:
            logger.error(f"\n✗ PHASE 1 FAILED")
            logger.error(f"  {e.stderr}")
            self.results['phase_1_scrape'] = {'status': 'failed', 'error': str(e)}
            return False

    def phase_1b_snapshot_files(self) -> bool:
        """
        Phase 1b: Snapshot current working tree files (current branch)
        """
        logger.info("\n" + "#" * 70)
        logger.info("# PHASE 1B: FILE SNAPSHOT COVERAGE")
        logger.info("#" * 70)

        output_file = self.base_dir / "data" / "file_snapshots.jsonl"
        cmd = [
            "python3",
            str(self.project_root / "scrapers" / "file_snapshot_scraper.py"),
            "--repo", str(self.repo_path),
            "--output", str(output_file),
            "--detect-branch",
        ]

        logger.info("\nCapturing current workspace files (docs + configs + crates)\n")
        if not self.force and output_file.exists():
            stats = self._collect_snapshot_statistics(output_file)
            elapsed = 0.0
            logger.info("\n➤ PHASE 1B SKIPPED (existing snapshot)")
            self._log_snapshot_statistics(stats, elapsed, output_file)
            self.results['phase_1b_snapshot'] = {
                'status': 'skipped',
                'output_file': str(output_file),
                'chunks': stats['chunk_count'],
                'unique_files': stats['unique_files'],
                'expected_files': stats['expected_files'],
                'coverage_ratio': stats['coverage_ratio'],
                'missing_files': stats['missing_files'],
                'duration_seconds': elapsed,
                'timestamp': datetime.now().isoformat(),
            }
            return True

        start = time.time()
        try:
            subprocess.run(cmd, check=True, capture_output=not self.verbose, text=True)
            elapsed = time.time() - start
            stats = self._collect_snapshot_statistics(output_file)
            self._log_snapshot_statistics(stats, elapsed, output_file)
            self.results['phase_1b_snapshot'] = {
                'status': 'complete',
                'output_file': str(output_file),
                'chunks': stats['chunk_count'],
                'unique_files': stats['unique_files'],
                'expected_files': stats['expected_files'],
                'coverage_ratio': stats['coverage_ratio'],
                'missing_files': stats['missing_files'],
                'duration_seconds': elapsed,
                'timestamp': datetime.now().isoformat(),
            }
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"\n✗ PHASE 1B FAILED")
            logger.error(f"  {e.stderr}")
            self.results['phase_1b_snapshot'] = {'status': 'failed', 'error': str(e)}
            return False
    
    def phase_2_tokenize(self) -> bool:
        """
        Phase 2: Tokenization
        """
        logger.info("\n" + "#" * 70)
        logger.info("# PHASE 2: TOKENIZATION")
        logger.info("#" * 70)
        
        input_file = self.base_dir / "data" / "git_history_rich.jsonl"
        output_file = self.base_dir / "data" / "token_sequences_rich.json"
        
        logger.info(f"\nWhat's happening:")
        logger.info(f"  • Converting commits to 512-token sequences")
        logger.info(f"  • Applying semantic markers")
        logger.info(f"  • Creating 128-token overlap for continuity")
        logger.info(f"  • Maintaining chronological order\n")

        sequence_length = self.sequence_length
        overlap = self.overlap

        cmd = [
            "python3",
            str(self.project_root / "tokenizers" / "git_tokenizer_rich.py"),
            "--input", str(input_file),
            "--sequences", str(output_file),
            "--sequence-length", str(sequence_length),
            "--overlap", str(overlap),
            "--include-diff-text",
            "--stats",
            "--model", self.tokenizer_model,
        ]
        if self.tokenizer_trust_remote:
            cmd.append("--trust-remote-code")
        
        if self.verbose:
            cmd.append("--verbose")
        
        if not self.force and output_file.exists():
            summary = self._load_sequence_summary(output_file)
            if summary:
                num_sequences = summary['num_sequences']
                tokens_per_sequence = summary['sequence_length'] or sequence_length
                logger.info("\n➤ PHASE 2 SKIPPED (existing commit sequences)")
                logger.info(f"  Output: {output_file.name}")
                logger.info(f"    Sequences created: {num_sequences}")
                logger.info(f"    Tokens per sequence: {tokens_per_sequence}")
                logger.info(f"    Total tokens: {summary['total_tokens'] or (num_sequences * tokens_per_sequence):,}")
                temp_analyzer = GitAnalyzer(str(self.repo_path), verbose=False)
                previous_params = self.stats.get('training_params', {})
                updated_params = temp_analyzer.calculate_training_params(
                    num_sequences,
                    config_path=str(self.config_path),
                    overrides=self.epoch_overrides,
                )
                self._apply_epoch_override(updated_params)
                self.stats['training_params'] = updated_params
                logger.info(
                    "  Updated epochs: %s (was %s)\n",
                    updated_params.get('epochs'),
                    previous_params.get('epochs', 'unknown')
                )
                self.sequence_files['commits'] = output_file
                self.sequence_counts['commits'] = num_sequences
                self.sequence_metadata['commits'] = {
                    'path': str(output_file),
                    'num_sequences': num_sequences,
                    'sequence_length': tokens_per_sequence,
                    'total_tokens': summary['total_tokens'],
                }
                self.results['phase_2_tokenize'] = {
                    'status': 'skipped',
                    'output_file': str(output_file),
                    'num_sequences': num_sequences,
                    'total_tokens': summary['total_tokens'],
                    'duration_seconds': 0.0,
                    'timestamp': datetime.now().isoformat(),
                }
                return True
        start = time.time()
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            elapsed = time.time() - start
            
            if output_file.exists():
                with open(output_file) as f:
                    sequences_obj = json.load(f)
                
                if isinstance(sequences_obj, dict):
                    token_sequences = sequences_obj.get("token_sequences", [])
                    reported_length = sequences_obj.get("sequence_length")
                else:
                    token_sequences = sequences_obj
                    reported_length = None

                num_sequences = len(token_sequences)
                tokens_per_sequence = reported_length or sequence_length
                logger.info(f"\n✓ PHASE 2 COMPLETE ({self.format_time(elapsed)})")
                logger.info(f"  Output: {output_file.name}")
                logger.info(f"    Sequences created: {num_sequences}")
                logger.info(f"    Tokens per sequence: {tokens_per_sequence}")
                logger.info(f"    Total tokens: {num_sequences * tokens_per_sequence:,}")
                self.sequence_metadata['commits'] = {
                    'path': str(output_file),
                    'num_sequences': num_sequences,
                    'sequence_length': tokens_per_sequence,
                    'total_tokens': num_sequences * tokens_per_sequence,
                }
                
                # RE-CALCULATE TRAINING PARAMS WITH ACTUAL SEQUENCE COUNT
                logger.info(f"\n  Re-computing training parameters based on ACTUAL sequences...")
                from scrapers.git_scraper_dynamic import GitAnalyzer
                temp_analyzer = GitAnalyzer(str(self.repo_path), verbose=False)
                updated_params = temp_analyzer.calculate_training_params(
                    num_sequences,
                    config_path=str(self.config_path),
                    overrides=self.epoch_overrides,
                )
                self._apply_epoch_override(updated_params)
                previous_params = self.stats.get('training_params', {})
                self.stats['training_params'] = updated_params
                logger.info(
                    "  Updated epochs: %s (was %s)\n",
                    updated_params.get('epochs'),
                    previous_params.get('epochs', 'unknown')
                )

                self.sequence_files['commits'] = output_file
                self.sequence_counts['commits'] = num_sequences
                
                self.results['phase_2_tokenize'] = {
                    'status': 'complete',
                    'output_file': str(output_file),
                    'num_sequences': num_sequences,
                    'total_tokens': num_sequences * tokens_per_sequence,
                    'duration_seconds': elapsed,
                    'timestamp': datetime.now().isoformat(),
                }
                return True
        except subprocess.CalledProcessError as e:
            logger.error(f"\n✗ PHASE 2 FAILED")
            logger.error(f"  {e.stderr}")
            self.results['phase_2_tokenize'] = {'status': 'failed', 'error': str(e)}
            return False

    def phase_2b_tokenize_snapshots(self) -> bool:
        """
        Phase 2B: Tokenize file snapshot chunks
        """
        input_file = self.base_dir / "data" / "file_snapshots.jsonl"
        output_file = self.base_dir / "data" / "token_sequences_snapshot.json"
        if not input_file.exists():
            logger.warning("Snapshot input %s missing; skipping snapshot tokenization", input_file)
            self.results['phase_2b_snapshot_tokenize'] = {
                'status': 'skipped',
                'reason': 'snapshot jsonl missing',
            }
            return True

        logger.info("\n" + "#" * 70)
        logger.info("# PHASE 2B: SNAPSHOT TOKENIZATION")
        logger.info("#" * 70)
        logger.info("\n  • Tiling current workspace files into sequences\n")

        cmd = [
            "python3",
            str(self.project_root / "tokenizers" / "file_snapshot_tokenizer.py"),
            "--input", str(input_file),
            "--output", str(output_file),
            "--model", self.tokenizer_model,
            "--sequence-length", str(self.sequence_length),
            "--overlap", str(self.overlap),
            "--stats",
        ]

        if self.tokenizer_trust_remote:
            cmd.append("--trust-remote-code")

        if self.verbose:
            cmd.append("--verbose")

        if not self.force and output_file.exists():
            summary = self._load_sequence_summary(output_file)
            if summary:
                num_sequences = summary['num_sequences']
                seq_length = summary['sequence_length']
                logger.info("\n➤ PHASE 2B SKIPPED (existing snapshot sequences)")
                logger.info(f"  Output: {output_file.name}")
                logger.info(f"    Snapshot sequences: {num_sequences}")
                self.sequence_files['snapshots'] = output_file
                self.sequence_counts['snapshots'] = num_sequences
                self.sequence_metadata['snapshots'] = {
                    'path': str(output_file),
                    'num_sequences': num_sequences,
                    'sequence_length': seq_length,
                    'total_tokens': summary['total_tokens'],
                }
                self.results['phase_2b_snapshot_tokenize'] = {
                    'status': 'skipped',
                    'output_file': str(output_file),
                    'num_sequences': num_sequences,
                    'duration_seconds': 0.0,
                    'timestamp': datetime.now().isoformat(),
                }
                return True
        start = time.time()
        try:
            subprocess.run(cmd, check=True, capture_output=not self.verbose, text=True)
            elapsed = time.time() - start
            with open(output_file, 'r') as f:
                snapshot_obj = json.load(f)
            token_sequences = snapshot_obj.get("token_sequences", [])
            num_sequences = len(token_sequences)
            seq_length = snapshot_obj.get("sequence_length") or (len(token_sequences[0]) if token_sequences else 0)
            total_tokens = snapshot_obj.get("total_tokens")
            if total_tokens is None:
                total_tokens = sum(len(seq) for seq in token_sequences if isinstance(seq, list))
            self.sequence_files['snapshots'] = output_file
            self.sequence_counts['snapshots'] = num_sequences
            self.sequence_metadata['snapshots'] = {
                'path': str(output_file),
                'num_sequences': num_sequences,
                'sequence_length': seq_length,
                'total_tokens': total_tokens,
            }
            logger.info(f"\n✓ PHASE 2B COMPLETE ({self.format_time(elapsed)})")
            logger.info(f"  Snapshot sequences: {num_sequences}")
            self.results['phase_2b_snapshot_tokenize'] = {
                'status': 'complete',
                'output_file': str(output_file),
                'num_sequences': num_sequences,
                'duration_seconds': elapsed,
                'timestamp': datetime.now().isoformat(),
            }
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"\n✗ PHASE 2B FAILED")
            logger.error(f"  {e.stderr}")
            self.results['phase_2b_snapshot_tokenize'] = {'status': 'failed', 'error': str(e)}
            return False

    def phase_2c_merge_sequences(self) -> bool:
        """
        Phase 2C: Merge commit + snapshot sequences into unified dataset
        """
        combined_output = self.base_dir / "data" / "token_sequences_full.json"
        sources = []
        for label in ['commits', 'snapshots']:
            path = self.sequence_files.get(label)
            if path and Path(path).exists():
                sources.append((label, Path(path)))

        if not sources:
            logger.warning("No sequence sources to merge; skipping Phase 2C")
            self.results['phase_2c_merge'] = {'status': 'skipped', 'reason': 'no sequence sources'}
            return False

        temp_analyzer = GitAnalyzer(str(self.repo_path), verbose=False)
        trimmed_path = self.base_dir / "data" / "token_sequences_full_trimmed.json"
        combined_exists = combined_output.exists()
        trimmed_exists = trimmed_path.exists()
        should_skip = (
            not self.force
            and combined_exists
            and (not self.max_sequences_limit or trimmed_exists)
        )

        if should_skip:
            combined_summary = self._load_sequence_summary(combined_output)
            can_skip = bool(combined_summary)
            training_params = None
            if combined_summary and self.max_sequences_limit and trimmed_exists:
                training_summary = self._load_sequence_summary(trimmed_path)
                if not training_summary:
                    can_skip = False
            if can_skip and combined_summary:
                self.sequence_files['combined'] = combined_output
                self.sequence_counts['combined'] = combined_summary['num_sequences']
                self.sequence_metadata['combined'] = {
                    'path': str(combined_output),
                    'num_sequences': combined_summary['num_sequences'],
                    'sequence_length': combined_summary['sequence_length'],
                    'total_tokens': combined_summary['total_tokens'],
                }

                if self.max_sequences_limit and trimmed_exists:
                    training_summary = self._load_sequence_summary(trimmed_path)
                    trimmed_meta = {
                        'path': str(trimmed_path),
                        'num_sequences': training_summary['num_sequences'],
                        'sequence_length': training_summary['sequence_length'],
                        'total_tokens': training_summary['total_tokens'],
                        'trimmed': True,
                        'max_sequences_limit': self.max_sequences_limit,
                        'source': 'combined',
                    }
                    self.sequence_files['training'] = trimmed_path
                    self.sequence_counts['training'] = training_summary['num_sequences']
                    self.sequence_metadata['training'] = trimmed_meta
                    training_params = temp_analyzer.calculate_training_params(
                        training_summary['num_sequences'],
                        config_path=str(self.config_path),
                        overrides=self.epoch_overrides,
                    )
                    self._apply_epoch_override(training_params)
                else:
                    trimmed_meta = {
                        'path': str(combined_output),
                        'num_sequences': combined_summary['num_sequences'],
                        'sequence_length': combined_summary['sequence_length'],
                        'total_tokens': combined_summary['total_tokens'],
                        'trimmed': False,
                        'max_sequences_limit': self.max_sequences_limit,
                        'source': 'combined',
                    }
                    self.sequence_files['training'] = combined_output
                    self.sequence_counts['training'] = combined_summary['num_sequences']
                    self.sequence_metadata['training'] = trimmed_meta
                    training_params = temp_analyzer.calculate_training_params(
                        combined_summary['num_sequences'],
                        config_path=str(self.config_path),
                        overrides=self.epoch_overrides,
                    )
                    self._apply_epoch_override(training_params)

                if training_params:
                    self.stats['training_params'] = training_params
                logger.info("\n➤ PHASE 2C SKIPPED (existing merged sequences)")
                logger.info(f"  Combined sequences: {combined_summary['num_sequences']}")
                self.results['phase_2c_merge'] = {
                    'status': 'skipped',
                    'output_file': str(combined_output),
                    'num_sequences': combined_summary['num_sequences'],
                    'total_tokens': combined_summary['total_tokens'],
                    'timestamp': datetime.now().isoformat(),
                }
                return True

        merged_stats = self._merge_sequence_files(sources, combined_output)
        self.sequence_files['combined'] = combined_output
        self.sequence_counts['combined'] = merged_stats['num_sequences']
        self.sequence_metadata['combined'] = {
            'path': str(combined_output),
            'num_sequences': merged_stats['num_sequences'],
            'sequence_length': merged_stats.get('sequence_length'),
            'total_tokens': merged_stats.get('total_tokens'),
        }

        combined_params = temp_analyzer.calculate_training_params(
            merged_stats['num_sequences'],
            config_path=str(self.config_path),
            overrides=self.epoch_overrides,
        )
        self._apply_epoch_override(combined_params)
        self.stats['training_params'] = combined_params

        if self.max_sequences_limit and merged_stats['num_sequences'] > self.max_sequences_limit:
            trimmed_path = self.base_dir / "data" / "token_sequences_full_trimmed.json"
            trimmed_bundle = self._trim_sequences(combined_output, trimmed_path, self.max_sequences_limit)
            if trimmed_bundle:
                self.sequence_files['training'] = trimmed_path
                self.sequence_counts['training'] = trimmed_bundle['num_sequences']
                self.sequence_metadata['training'] = {
                    'path': str(trimmed_path),
                    'num_sequences': trimmed_bundle['num_sequences'],
                    'sequence_length': trimmed_bundle.get('sequence_length'),
                    'total_tokens': trimmed_bundle.get('total_tokens'),
                    'trimmed': True,
                    'max_sequences_limit': self.max_sequences_limit,
                    'source': 'combined',
                }
                trimmed_params = temp_analyzer.calculate_training_params(
                    trimmed_bundle['num_sequences'],
                    config_path=str(self.config_path),
                    overrides=self.epoch_overrides,
                )
                self._apply_epoch_override(trimmed_params)
                self.stats['training_params'] = trimmed_params
        else:
            self.sequence_files['training'] = combined_output
            self.sequence_counts['training'] = merged_stats['num_sequences']
            self.sequence_metadata['training'] = {
                'path': str(combined_output),
                'num_sequences': merged_stats['num_sequences'],
                'sequence_length': merged_stats.get('sequence_length'),
                'total_tokens': merged_stats.get('total_tokens'),
                'trimmed': False,
                'max_sequences_limit': self.max_sequences_limit,
                'source': 'combined',
            }

        logger.info("\n✓ PHASE 2C COMPLETE")
        logger.info(f"  Combined sequences: {merged_stats['num_sequences']}")
        self.results['phase_2c_merge'] = {
            'status': 'complete',
            'output_file': str(combined_output),
            'num_sequences': merged_stats['num_sequences'],
            'total_tokens': merged_stats['total_tokens'],
            'timestamp': datetime.now().isoformat(),
        }
        return True
    
    def phase_3_embeddings(self) -> bool:
        """
        Phase 3: Embedding generation
        """
        logger.info("\n" + "#" * 70)
        logger.info("# PHASE 3: EMBEDDING GENERATION")
        logger.info("#" * 70)
        
        input_file = self.base_dir / "data" / "git_history_rich.jsonl"
        output_file = self.base_dir / "embeddings" / "qdrant_points.json"

        # NEW: compute script path and short-circuit if missing
        script_path = self.project_root / "embeddings" / "embedding_generator.py"
        if not script_path.exists():
            logger.warning(
                "Embeddings script not found at %s; skipping Phase 3 and continuing.",
                script_path,
            )
            self.results['phase_3_embeddings'] = {
                'status': 'skipped',
                'reason': 'embedding_generator.py not found',
                'output_file': None,
                'duration_seconds': 0.0,
                'timestamp': datetime.now().isoformat(),
            }
            return True  # do NOT fail the pipeline
        
        logger.info(f"\nWhat's happening:")
        logger.info(f"  • Creating 768-dimensional vectors")
        logger.info(f"  • Using all-mpnet-base-v2 model")
        logger.info(f"  • Processing in batches of 128")
        logger.info(f"  • Formatting for Qdrant vector DB\n")
        
        cmd = [
            "python3",
            str(script_path),  # was self.base_dir / "embeddings" / "embedding_generator.py"
            "--input", str(input_file),
            "--qdrant-output", str(output_file),
            "--stats",
            "--mode", "commit",
        ]
        
        if self.verbose:
            cmd.append("--verbose")
        
        if not self.force and output_file.exists():
            size_mb = output_file.stat().st_size / 1e6
            logger.info("\n➤ PHASE 3 SKIPPED (existing embeddings)")
            logger.info(f"  Output: {output_file.name}")
            logger.info(f"    Size: {size_mb:.1f} MB")
            self.results['phase_3_embeddings'] = {
                'status': 'skipped',
                'output_file': str(output_file),
                'size_mb': size_mb,
                'duration_seconds': 0.0,
                'timestamp': datetime.now().isoformat(),
            }
            return True
        
        start = time.time()
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            elapsed = time.time() - start
            
            if output_file.exists():
                size_mb = output_file.stat().st_size / 1e6
                dimension = self._infer_embedding_dimension(output_file)
                dimension_label = dimension if dimension else "unknown"
                
                logger.info(f"\n✓ PHASE 3 COMPLETE ({self.format_time(elapsed)})")
                logger.info(f"  Output: {output_file.name}")
                logger.info(f"    Size: {size_mb:.1f} MB")
                logger.info(f"    Embedding dimension: {dimension_label}")
                logger.info(f"    Qdrant compatibility: ✓\n")
                
                self.results['phase_3_embeddings'] = {
                    'status': 'complete',
                    'output_file': str(output_file),
                    'size_mb': size_mb,
                    'embedding_dimension': 768,
                    'duration_seconds': elapsed,
                    'timestamp': datetime.now().isoformat(),
                }
                return True
        except subprocess.CalledProcessError as e:
            logger.error(f"\n✗ PHASE 3 FAILED")
            logger.error(f"  {e.stderr}")
            self.results['phase_3_embeddings'] = {'status': 'failed', 'error': str(e)}
            return False

    def phase_3b_snapshot_embeddings(self) -> bool:
        """
        Phase 3B: Embeddings for file snapshot chunks
        """
        input_file = self.base_dir / "data" / "file_snapshots.jsonl"
        output_file = self.base_dir / "embeddings" / "qdrant_points_files.json"
        script_path = self.project_root / "embeddings" / "embedding_generator.py"

        if not input_file.exists():
            logger.warning("Snapshot JSONL missing; skipping file embeddings")
            self.results['phase_3b_embeddings'] = {
                'status': 'skipped',
                'reason': 'snapshot jsonl missing',
                'output_file': None,
            }
            return True

        cmd = [
            "python3",
            str(script_path),
            "--input", str(input_file),
            "--qdrant-output", str(output_file),
            "--mode", "snapshot",
            "--stats",
        ]
        if self.verbose:
            cmd.append("--verbose")

        if not self.force and output_file.exists():
            size_mb = output_file.stat().st_size / 1e6
            logger.info("\n➤ PHASE 3B SKIPPED (existing snapshot embeddings)")
            logger.info(f"  Snapshot embeddings file: {output_file.name} ({size_mb:.1f} MB)")
            self.results['phase_3b_embeddings'] = {
                'status': 'skipped',
                'output_file': str(output_file),
                'size_mb': size_mb,
                'duration_seconds': 0.0,
                'timestamp': datetime.now().isoformat(),
            }
            return True

        start = time.time()
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            elapsed = time.time() - start
            size_mb = output_file.stat().st_size / 1e6 if output_file.exists() else 0
            logger.info(f"\n✓ PHASE 3B COMPLETE ({self.format_time(elapsed)})")
            logger.info(f"  Snapshot embeddings file: {output_file.name} ({size_mb:.1f} MB)")
            self.results['phase_3b_embeddings'] = {
                'status': 'complete',
                'output_file': str(output_file),
                'size_mb': size_mb,
                'duration_seconds': elapsed,
                'timestamp': datetime.now().isoformat(),
            }
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"\n✗ PHASE 3B FAILED")
            logger.error(f"  {e.stderr}")
            self.results['phase_3b_embeddings'] = {'status': 'failed', 'error': str(e)}
            return False

    def _infer_embedding_dimension(self, path: Path) -> Optional[int]:
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return None

        if isinstance(data, list) and data:
            first = data[0]
            if isinstance(first, dict):
                vector = first.get("vector")
                if isinstance(vector, list):
                    return len(vector)
        return None

    def _merge_sequence_files(self, labeled_files: List[Tuple[str, Path]], output_path: Path) -> Dict[str, Any]:
        """Merge multiple token sequence files into one."""
        combined_sequences: List[List[int]] = []
        combined_metadata: Dict[str, Dict] = {}
        vocab_size = None
        seq_length = None
        total_tokens = 0
        offset = 0

        for label, file_path in labeled_files:
            with open(file_path, 'r') as f:
                data = json.load(f)
            sequences = data.get("token_sequences", [])
            combined_sequences.extend(sequences)
            total_tokens += sum(len(seq) for seq in sequences if isinstance(seq, list))

            src_meta = data.get("metadata", {})
            for idx in range(len(sequences)):
                meta = src_meta.get(str(idx)) or src_meta.get(idx) or {}
                meta = dict(meta)
                meta['source'] = label
                combined_metadata[str(offset + idx)] = meta

            if vocab_size is None:
                vocab_size = data.get("vocab_size")
            if seq_length is None and sequences:
                seq_length = data.get("sequence_length", len(sequences[0]))
            offset += len(sequences)

        bundle = {
            "token_sequences": combined_sequences,
            "num_sequences": len(combined_sequences),
            "sequence_length": seq_length or 0,
            "total_tokens": total_tokens,
            "vocab_size": vocab_size,
            "metadata": combined_metadata,
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(bundle, f, indent=2)
        return bundle

    def _trim_sequences(self, source: Path, dest: Path, limit: int) -> Optional[Dict[str, Any]]:
        """Create a trimmed sequence file limited to the first `limit` sequences."""
        if not source.exists():
            return None
        with open(source, 'r') as f:
            data = json.load(f)
        sequences = data.get("token_sequences", [])
        if len(sequences) <= limit:
            return None

        metadata = data.get("metadata", {})
        selected_indices = self._select_trimmed_indices(metadata, len(sequences), limit)
        trimmed_sequences = [sequences[i] for i in selected_indices]
        trimmed_metadata: Dict[str, Dict] = {}
        for new_idx, orig_idx in enumerate(selected_indices):
            trimmed_metadata[str(new_idx)] = metadata.get(str(orig_idx), metadata.get(orig_idx, {}))

        total_tokens = sum(len(seq) for seq in trimmed_sequences if isinstance(seq, list))

        trimmed_bundle = {
            "token_sequences": trimmed_sequences,
            "num_sequences": len(trimmed_sequences),
            "sequence_length": data.get("sequence_length", len(trimmed_sequences[0]) if trimmed_sequences else 0),
            "total_tokens": total_tokens,
            "vocab_size": data.get("vocab_size"),
            "metadata": trimmed_metadata,
        }

        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, 'w') as f:
            json.dump(trimmed_bundle, f, indent=2)

        logger.warning(
            "Trimmed sequences (priority order) from %d to %d (limit %d); writing to %s",
            len(sequences),
            len(trimmed_sequences),
            limit,
            dest,
        )

        return trimmed_bundle
    
    def _select_trimmed_indices(
        self,
        metadata: Dict[str, Dict],
        num_sequences: int,
        limit: int,
    ) -> List[int]:
        """Select indices to keep based on trim priority configuration."""
        priority_list = self.training_cfg.get("training", {}).get("trim_priority", [])
        priority_map = {source: idx for idx, source in enumerate(priority_list)}
        default_priority = len(priority_map)

        indices = list(range(num_sequences))

        def seq_priority(idx: int) -> Tuple[int, int]:
            meta = metadata.get(str(idx), metadata.get(idx, {}))
            source = meta.get("source")
            prio = priority_map.get(source, default_priority)
            return (prio, idx)

        indices.sort(key=seq_priority)
        return indices[:limit]

    def phase_4_training(self, training_params: Dict) -> bool:
        """
        Phase 4: Dynamic model training (using actual sequence count)
        """
        num_epochs = training_params['epochs']
        
        logger.info("\n" + "#" * 70)
        logger.info(f"# PHASE 4: MODEL TRAINING ({num_epochs} EPOCHS CALCULATED)")
        logger.info("#" * 70)
        
        sequence_path = (
            self.sequence_files.get('training')
            or self.sequence_files.get('combined')
            or self.sequence_files.get('commits')
            or (self.base_dir / "data" / "token_sequences_rich.json")
        )
        input_file = Path(sequence_path)
        training_source_label = "unknown"
        if self.sequence_files.get('training') and input_file == self.sequence_files.get('training'):
            training_source_label = "merged_trimmed" if (
                self.max_sequences_limit
                and self.sequence_counts.get('training')
                and self.sequence_counts.get('combined')
                and self.sequence_counts['training'] < self.sequence_counts['combined']
            ) else "merged_full"
        elif self.sequence_files.get('combined') and input_file == self.sequence_files.get('combined'):
            training_source_label = "merged_full"
        elif self.sequence_files.get('commits') and input_file == self.sequence_files.get('commits'):
            training_source_label = "commits_only"
        training_meta = self.sequence_metadata.setdefault('training', {})
        training_meta.setdefault('path', str(input_file))
        training_meta['source_label'] = training_source_label
        training_meta['num_sequences'] = training_params.get('num_sequences')
        training_meta['epochs'] = num_epochs
        self.training_source_label = training_source_label
        output_dir = self.base_dir / "models" / "the-block-git-model-final"
        config_file = self.project_root / "training_config.yaml"
        
        logger.info(f"\nWhat's happening:")
        logger.info(f"  • Training GPT-2-medium on your code patterns")
        logger.info(f"  • Training source: {training_source_label} -> {input_file}")
        logger.info(f"  • {num_epochs} epochs (formula-determined from {training_params['num_sequences']} sequences)")
        logger.info(f"  • {training_params['steps_per_epoch']} steps per epoch")
        logger.info(f"  • Total {training_params['total_steps']} training steps")
        logger.info(f"  • Hardware-optimized batch size")
        logger.info(f"  • Early stopping + validation enabled\n")
        
        cmd = [
            "python3",
            str(self.project_root / "training" / "model_trainer_unified.py"),
            "--config", str(config_file),
            "--sequences", str(input_file),
            "--epochs", str(num_epochs),
            "--output", str(output_dir),
        ]
        
        if self.verbose:
            cmd.append("--verbose")
        
        start = time.time()
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            elapsed = time.time() - start
            
            if (output_dir / "pytorch_model.bin").exists():
                model_size_mb = (output_dir / "pytorch_model.bin").stat().st_size / 1e6
                
                # Load training report if it exists
                training_report = None
                report_path = output_dir / "training_report.json"
                if report_path.exists():
                    with open(report_path) as f:
                        training_report = json.load(f)
                
                logger.info(f"\n✓ PHASE 4 COMPLETE ({self.format_time(elapsed)})")
                logger.info(f"  Model: {output_dir.name}")
                logger.info(f"    Size: {model_size_mb:.0f} MB")
                logger.info(f"    Parameters: 345M (GPT-2-medium)")
                logger.info(f"    Training duration: {self.format_time(elapsed)}")
                logger.info(f"    Epochs completed: {num_epochs}")
                
                if training_report:
                    logger.info(f"\n  Training metrics:")
                    logger.info(f"    Final train loss: {training_report['training']['final_train_loss']:.4f}")
                    logger.info(f"    Final val loss: {training_report['training']['final_val_loss']:.4f}")
                    logger.info(f"    Final perplexity: {training_report['training']['final_perplexity']:.2f}")
                    logger.info(f"    Peak GPU memory: {training_report['hardware']['peak_gpu_memory_mb']:.0f}MB\n")
                
                self.results['phase_4_training'] = {
                    'status': 'complete',
                    'model_dir': str(output_dir),
                    'model_size_mb': model_size_mb,
                    'epochs': num_epochs,
                    'duration_seconds': elapsed,
                    'training_source': training_source_label,
                    'sequence_file': str(input_file),
                    'num_sequences': training_params.get('num_sequences'),
                    'training_report': training_report,
                    'timestamp': datetime.now().isoformat(),
                }
                return True
        except subprocess.CalledProcessError as e:
            logger.error(f"\n✗ PHASE 4 FAILED")
            logger.error("  Command: %s", " ".join(cmd))
            logger.error("  Exit code: %s", e.returncode)
            logger.error("  STDOUT:\n%s", e.stdout)
            logger.error("  STDERR:\n%s", e.stderr)
            self.results['phase_4_training'] = {'status': 'failed', 'error': str(e)}
            return False
    
    def format_time(self, seconds: float) -> str:
        """Format seconds to human readable"""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            return f"{seconds/3600:.2f}h"
    
    def generate_final_report(self):
        """
        Generate comprehensive final report with training metrics integrated
        """
        total_time = time.time() - self.start_time
        
        logger.info("\n" + "#" * 70)
        logger.info("# FINAL REPORT")
        logger.info("#" * 70)
        
        # Phase completion status
        logger.info("\nPhase Status:")
        for phase, result in self.results.items():
            status = result.get('status', 'unknown')
            symbol = "✓" if status == 'complete' else "✗"
            logger.info(f"  {symbol} {phase}: {status}")
        
        # Repository stats
        logger.info(f"\nRepository Statistics:")
        repo_stats = self.stats.get('repository', {})
        logger.info(f"  Commits analyzed: {repo_stats.get('unique_commits', 'N/A')}")
        logger.info(f"  Branches: {repo_stats.get('branches', 'N/A')}")
        logger.info(f"  Authors: {repo_stats.get('unique_authors', 'N/A')}")
        
        # Training stats
        train_params = self.stats.get('training_params', {})
        logger.info(f"\nTraining Parameters (Formula-Based):")
        logger.info(f"  Sequences: {train_params.get('num_sequences', 'N/A')}")
        logger.info(f"  Epochs: {train_params.get('epochs', 'N/A')}")
        logger.info(f"  Total steps: {train_params.get('total_steps', 'N/A')}")
        logger.info(f"  Warmup steps: {train_params.get('warmup_steps', 'N/A')}")
        logger.info(f"  Target tokens: {train_params.get('target_tokens', 'N/A')}")
        
        logger.info(f"\nSequence Summary:")
        if not self.sequence_metadata:
            logger.info("  No sequence metadata recorded")
        else:
            for label in ['commits', 'snapshots', 'combined', 'training']:
                meta = self.sequence_metadata.get(label)
                if not meta:
                    continue
                extra = []
                if meta.get('trimmed'):
                    extra.append(f"trimmed<= {meta.get('max_sequences_limit')}")
                if meta.get('source_label'):
                    extra.append(meta.get('source_label'))
                extra_str = f" ({', '.join(extra)})" if extra else ""
                logger.info(
                    f"  {label}: {meta.get('num_sequences', 'N/A')} sequences -> {meta.get('path', 'N/A')}{extra_str}"
                )
                if meta.get('total_tokens') is not None:
                    logger.info(f"     Total tokens: {meta.get('total_tokens')}")
        
        # Training metrics (if training completed)
        training_report = self.results.get('phase_4_training', {}).get('training_report')
        if training_report:
            logger.info(f"\nTraining Results:")
            train_metrics = training_report.get('training', {})
            logger.info(f"  Final train loss: {train_metrics.get('final_train_loss', 'N/A'):.4f}")
            logger.info(f"  Final val loss: {train_metrics.get('final_val_loss', 'N/A'):.4f}")
            logger.info(f"  Final perplexity: {train_metrics.get('final_perplexity', 'N/A'):.2f}")
            logger.info(f"  Best val loss: {train_metrics.get('best_val_loss', 'N/A'):.4f}")
            
            grad_stats = training_report.get('gradients', {})
            logger.info(f"\nGradient Statistics:")
            logger.info(f"  Min norm: {grad_stats.get('min_norm', 'N/A'):.4f}")
            logger.info(f"  Max norm: {grad_stats.get('max_norm', 'N/A'):.4f}")
            
            lr_stats = training_report.get('learning_rate', {})
            logger.info(f"\nLearning Rate:")
            logger.info(f"  Min: {lr_stats.get('min', 'N/A'):.2e}")
            logger.info(f"  Max: {lr_stats.get('max', 'N/A'):.2e}")
            
            hw_stats = training_report.get('hardware', {})
            logger.info(f"\nHardware Peaks:")
            logger.info(f"  Peak GPU memory: {hw_stats.get('peak_gpu_memory_mb', 'N/A'):.0f}MB")
            logger.info(f"  Peak RAM: {hw_stats.get('peak_ram_percent', 'N/A'):.1f}%")
        
        # Timing summary
        logger.info(f"\nExecution Summary:")
        logger.info(f"  Total time: {self.format_time(total_time)}")
        logger.info(f"  Model location: {self.base_dir / 'models' / 'the-block-git-model-final'}")
        
        # Save manifest
        manifest = {
            'execution_timestamp': datetime.now().isoformat(),
            'total_execution_time_seconds': total_time,
            'repository_stats': self.stats.get('repository', {}),
            'training_parameters': self.stats.get('training_params', {}),
            'training_source_label': self.training_source_label,
            'sequence_metadata': self.sequence_metadata,
            'phase_results': self.results,
            'training_report': training_report,
        }
        
        manifest_file = self.base_dir / "MANIFEST_DYNAMIC.json"
        with open(manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        logger.info(f"\nManifest saved to: {manifest_file}")
        logger.info("#" * 70 + "\n")
        
        return manifest
    
    def run(self):
        """
        Run complete pipeline
        """
        try:
            logger.info("\n" + "="*70)
            logger.info("DYNAMIC PIPELINE START")
            logger.info("="*70)
            
            # Phase 0: Analysis
            stats = self.phase_0_analyze_repository()
            
            # Phase 1: Scraping
            if not self.phase_1_scrape(stats):
                return False
            if not self.phase_1b_snapshot_files():
                return False
            
            # Phase 2: Tokenization
            if not self.phase_2_tokenize():
                return False
            if not self.phase_2b_tokenize_snapshots():
                return False
            if not self.phase_2c_merge_sequences():
                return False
            
            # Phase 3: Embeddings
            if not self.phase_3_embeddings():
                return False
            if not self.phase_3b_snapshot_embeddings():
                return False
            
            # Phase 4: Training (using UPDATED training params from Phase 2)
            if not self.phase_4_training(self.stats['training_params']):
                return False
            
            # Final report
            self.generate_final_report()
            
            return True
        
        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Dynamic pipeline orchestrator")
    parser.add_argument("--repo", type=str, required=True, help="Repository path")
    parser.add_argument("--base-dir", type=str, help="Base directory for output")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    parser.add_argument("--force", action="store_true", help="Rerun every phase even if output files already exist")
    parser.add_argument("--sequence-length", type=int, help="Override tokenizer sequence length")
    parser.add_argument("--overlap", type=int, help="Override tokenizer overlap between sequences")
    parser.add_argument("--target-tokens", type=int, help="Override target tokens used for epoch calculation")
    parser.add_argument("--min-epochs", type=int, help="Lower bound on auto-computed epochs")
    parser.add_argument("--max-epochs", type=int, help="Upper bound on auto-computed epochs")
    parser.add_argument("--force-epochs", type=int, help="Force a fixed number of epochs (overrides auto calculation)")
    
    args = parser.parse_args()
    
    epoch_overrides: Dict[str, Any] = {}
    if args.target_tokens:
        epoch_overrides["target_tokens"] = args.target_tokens
    if args.min_epochs:
        epoch_overrides["min_epochs"] = args.min_epochs
    if args.max_epochs:
        epoch_overrides["max_epochs"] = args.max_epochs

    orchestrator = DynamicPipelineOrchestrator(
        repo_path=args.repo,
        base_dir=args.base_dir,
        verbose=args.verbose,
        force=args.force,
        sequence_length=args.sequence_length or 512,
        overlap=args.overlap or 128,
        epoch_overrides=epoch_overrides,
        force_epochs=args.force_epochs,
    )
    
    success = orchestrator.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
