#!/usr/bin/env python3
"""
Dynamic Git Scraper for The Block

Automatically detects:
- All commits across all branches
- True commit count (no assumptions)
- All branch information
- Complete metadata extraction

Optimized for Ryzen 5 3800X + RTX 2060 Super
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, asdict, field
from collections import defaultdict
import logging
import math
import yaml

try:
    from git import Repo, GitCommandError, Actor
except ImportError:
    Repo = None
    GitCommandError = None
    Actor = None

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

try:
    import pygit2
except ImportError:
    pygit2 = None

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GitAnalyzer:
    """Analyzes Git repository to get accurate commit counts and metadata"""
    
    def __init__(self, repo_path: str, verbose: bool = False):
        if Repo is None:
            raise ImportError("GitPython is required: pip install GitPython")
        if tqdm is None:
            raise ImportError("tqdm is required: pip install tqdm")

        self.repo_path = Path(repo_path)
        self.verbose = verbose

        try:
            self.repo = Repo(str(self.repo_path))
        except Exception as e:
            raise RuntimeError(f"Failed to initialize GitPython repo: {e}")

        # pygit2 is optional; only initialize when available
        self.repo_pg = None
        if pygit2 is not None:
            try:
                self.repo_pg = pygit2.Repository(str(self.repo_path))
            except Exception as e:
                logger.warning(f"pygit2 repository init failed; continuing without pygit2: {e}")
    
    def _log(self, msg: str):
        if self.verbose:
            logger.info(msg)
    
    def get_all_commits(self) -> Tuple[Dict, int, int, List[str], Dict[str, int]]:
        """
        Get all unique commits across all branches.
        Returns: (commits_dict, total_count, unique_count, branches, stats)
        """
        logger.info("\n" + "="*70)
        logger.info("Analyzing Git repository to get ACCURATE commit counts")
        logger.info("="*70)
        
        all_commits = {}
        branch_info = {}
        commits_per_branch = defaultdict(int)
        
        # Get all branches
        logger.info("\nScanning all branches...")
        branches = []
        try:
            for branch in self.repo.branches:
                branches.append(branch.name)
                logger.info(f"  Found branch: {branch.name}")
        except Exception as e:
            logger.warning(f"Error reading branches: {e}")
            branches = ["main", "master"]  # Fallback
        
        # Get commits from each branch
        logger.info(f"\nExtracting commits from {len(branches)} branches...")
        for branch_name in tqdm(branches, desc="Analyzing branches"):
            try:
                branch_commits = 0
                for commit in self.repo.iter_commits(branch_name):
                    all_commits[commit.hexsha] = commit
                    commits_per_branch[branch_name] += 1
                    branch_commits += 1
                
                logger.info(f"  {branch_name}: {branch_commits} commits")
            except Exception as e:
                logger.warning(f"  Error on {branch_name}: {e}")
        
        total_commits_across_branches = sum(commits_per_branch.values())
        unique_commits = len(all_commits)
        
        logger.info(f"\n" + "-"*70)
        logger.info("COMMIT ANALYSIS RESULTS")
        logger.info("-"*70)
        logger.info(f"Total commits across all branches: {total_commits_across_branches}")
        logger.info(f"Unique commits: {unique_commits}")
        logger.info(f"Branches analyzed: {len(branches)}")
        logger.info(f"\nBranch breakdown:")
        for branch, count in sorted(commits_per_branch.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {branch}: {count} commits")
        logger.info("-"*70 + "\n")
        
        return all_commits, total_commits_across_branches, unique_commits, branches, dict(commits_per_branch)
    
    def get_repository_stats(self) -> Dict:
        """Get comprehensive repository statistics"""
        logger.info("Calculating repository statistics...")
        
        all_commits, total_across, unique, branches, per_branch = self.get_all_commits()
        
        # Get authors
        authors = set()
        for commit in all_commits.values():
            authors.add(commit.author.email)
        
        # Get date range
        timestamps = [c.committed_date for c in all_commits.values()]
        if timestamps:
            first_commit_time = min(timestamps)
            last_commit_time = max(timestamps)
            time_span_days = (last_commit_time - first_commit_time) / 86400
        else:
            first_commit_time = 0
            last_commit_time = 0
            time_span_days = 0
        
        stats = {
            'total_commits_across_branches': total_across,
            'unique_commits': unique,
            'unique_authors': len(authors),
            'branches': len(branches),
            'branch_names': branches,
            'commits_per_branch': per_branch,
            'first_commit_timestamp': first_commit_time,
            'last_commit_timestamp': last_commit_time,
            'time_span_days': time_span_days,
            'commits_per_day': unique / time_span_days if time_span_days > 0 else 0,
        }
        
        return stats, all_commits
    
    def calculate_training_params(
        self,
        num_sequences: int,
        config_path: Optional[str] = None,
        overrides: Optional[Dict[str, Any]] = None,
) -> Dict:
        """Calculate optimal training parameters using actual sequence count and config.
        
        If a config_path is provided, use the epoch_calculation targets from training_config.yaml;
        otherwise, fall back to the legacy heuristic.
        """
        import math
        
        overrides = overrides or {}
        
        # If no config provided, use the legacy heuristic (backwards-compatible)
        if config_path is None:
            if num_sequences < 20:
                base_epochs = 10
            elif num_sequences < 50:
                base_epochs = 8
            elif num_sequences < 100:
                base_epochs = 6
            elif num_sequences < 200:
                base_epochs = 5
            else:
                base_epochs = 4
            
            batch_size = 8
            steps_per_epoch = num_sequences // batch_size
            if num_sequences % batch_size != 0:
                steps_per_epoch += 1
            
            total_steps = steps_per_epoch * base_epochs
            # Warmup steps: 10% of total, but clamp to reasonable bounds
            # For small datasets (66 steps), min bound is too high - use 10% instead
            warmup_ratio = 0.1
            warmup_steps = max(10, int(total_steps * warmup_ratio))  # 10% with min of 10
            warmup_steps = min(warmup_steps, 1000)  # Cap at 1000
            estimated_time_seconds = total_steps * 1.5
            estimated_time_minutes = estimated_time_seconds / 60
            
            params = {
                'num_sequences': num_sequences,
                'epochs': base_epochs,
                'steps_per_epoch': steps_per_epoch,
                'total_steps': total_steps,
                'warmup_steps': warmup_steps,
                'batch_size': batch_size,
                'estimated_time_minutes': estimated_time_minutes,
                'estimated_time_hours': estimated_time_minutes / 60,
            }
        else:
            # Config-driven formula (no heavy ML imports required here)
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            epoch_cfg = cfg.get("epoch_calculation", {})
            train_cfg = cfg.get("training", {})

            target_tokens = overrides.get(
                "target_tokens", epoch_cfg.get("target_tokens", 20000000)
            )
            min_epochs = overrides.get(
                "min_epochs", epoch_cfg.get("min_epochs", 3)
            )
            max_epochs = overrides.get(
                "max_epochs", epoch_cfg.get("max_epochs", 10)
            )
            total_tokens = num_sequences * 2048

            if total_tokens > 0:
                ideal_epochs = target_tokens / total_tokens
            else:
                ideal_epochs = max_epochs

            epochs = int(max(min_epochs, min(max_epochs, math.floor(ideal_epochs))))
            batch_size = train_cfg.get("batch_size_reference", 4)
            steps_per_epoch = max(1, num_sequences // batch_size)
            if num_sequences % batch_size != 0:
                steps_per_epoch += 1

            total_steps = steps_per_epoch * epochs
            warmup_min = train_cfg.get("warmup_steps_min", 10)
            warmup_max = train_cfg.get("warmup_steps_max", 1000)
            warmup_steps = min(max(warmup_min, int(0.1 * total_steps)), warmup_max)
            estimated_time_minutes = (total_steps * 1.5) / 60

            params = {
                "num_sequences": num_sequences,
                "epochs": epochs,
                "steps_per_epoch": steps_per_epoch,
                "total_steps": total_steps,
                "warmup_steps": warmup_steps,
                "batch_size": batch_size,
                "estimated_time_minutes": estimated_time_minutes,
                "estimated_time_hours": estimated_time_minutes / 60,
                "target_tokens": target_tokens,
            }
        
        logger.info("\nTraining Parameters Calculated:")
        logger.info(f"  Token sequences: {num_sequences}")
        logger.info(f"  Determined epochs: {params['epochs']}")
        logger.info(f"  Steps per epoch: {params['steps_per_epoch']}")
        logger.info(f"  Total training steps: {params['total_steps']}")
        logger.info(f"  Warmup steps: {params['warmup_steps']}")
        logger.info(f"  Estimated training time: {params['estimated_time_minutes']:.1f} minutes ({params['estimated_time_hours']:.2f} hours)")
        logger.info("")
        
        return params


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Dynamic Git analyzer")
    parser.add_argument("--repo", type=str, required=True, help="Repository path")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    analyzer = GitAnalyzer(args.repo, verbose=args.verbose)
    stats, _ = analyzer.get_repository_stats()
    
    # Estimate token sequences (roughly 6 commits per 2048-token sequence)
    estimated_sequences = max(1, stats['unique_commits'] // 6)
    
    # Use config-driven formula if available
    config_path = Path(__file__).resolve().parent.parent / "training_config.yaml"
    if config_path.exists():
        training_params = analyzer.calculate_training_params(estimated_sequences, config_path=str(config_path))
    else:
        training_params = analyzer.calculate_training_params(estimated_sequences)
    


if __name__ == "__main__":
    main()
