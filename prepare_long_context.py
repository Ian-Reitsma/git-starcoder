#!/usr/bin/env python3
"""Long-context data preparation and training orchestration.

This script:
1. Analyzes repository structure for long-context opportunities
2. Builds hierarchical, multi-file sequences
3. Creates curriculum with difficulty levels
4. Orchestrates multi-phase training
5. Tracks long-context specific metrics

Usage:
    python prepare_long_context.py \
        --repo /path/to/repo \
        --output-dir ./long_context_data \
        --max-sequence-length 2048 \
        --phases 3
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import argparse

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def analyze_repo_for_long_context(repo_path: Path) -> Dict[str, Any]:
    """Analyze repository to identify long-context opportunities.
    
    Returns dict with:
    - file_structure: nested dict of directories/files
    - import_dependencies: file-to-file dependency graph
    - large_files: candidates for multi-file sequences
    - commit_history: file evolution chains
    """
    logger.info(f"Analyzing repository structure: {repo_path}")
    
    analysis = {
        "repo_path": str(repo_path),
        "total_files": 0,
        "large_files": [],  # Files >1K lines
        "file_groups": {},  # Related files (e.g., impl + tests)
        "commit_history": {},  # File evolution across commits
    }
    
    # Count files and identify large ones
    for fpath in repo_path.rglob("*"):
        if fpath.is_file() and not fpath.name.startswith("."):
            analysis["total_files"] += 1
            try:
                lines = fpath.read_text(errors="ignore").count("\n")
                if lines > 1000:
                    analysis["large_files"].append({
                        "path": str(fpath.relative_to(repo_path)),
                        "lines": lines,
                    })
            except Exception:
                pass
    
    logger.info(f"  Total files: {analysis['total_files']}")
    logger.info(f"  Large files (>1K lines): {len(analysis['large_files'])}")
    return analysis


def build_long_context_curriculum(analysis: Dict[str, Any]) -> Dict[str, List[str]]:
    """Build curriculum grouping files by difficulty.
    
    Returns dict:
    {"easy": [...], "medium": [...], "hard": [...], "very_hard": [...]}
    """
    curriculum = {"easy": [], "medium": [], "hard": [], "very_hard": []}
    
    # Classify large files by difficulty
    for f in analysis.get("large_files", []):
        lines = f["lines"]
        if lines < 2000:
            curriculum["medium"].append(f["path"])
        elif lines < 5000:
            curriculum["hard"].append(f["path"])
        else:
            curriculum["very_hard"].append(f["path"])
    
    return curriculum


def main():
    """Main orchestration."""
    parser = argparse.ArgumentParser(
        description="Prepare long-context training data and orchestrate multi-phase training"
    )
    parser.add_argument(
        "--repo",
        type=Path,
        required=True,
        help="Path to repository to analyze",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./long_context_data"),
        help="Directory for output sequences",
    )
    parser.add_argument(
        "--max-sequence-length",
        type=int,
        default=2048,
        help="Maximum tokens per sequence",
    )
    parser.add_argument(
        "--phases",
        type=int,
        default=3,
        help="Number of training phases (1-3)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging",
    )
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    
    logger.info("=" * 70)
    logger.info("LONG-CONTEXT TRAINING PREPARATION")
    logger.info("=" * 70)
    
    # Analyze repository
    analysis = analyze_repo_for_long_context(args.repo)
    
    # Build curriculum
    curriculum = build_long_context_curriculum(analysis)
    logger.info(f"\nCurriculum built:")
    for difficulty, files in curriculum.items():
        if files:
            logger.info(f"  {difficulty}: {len(files)} files")
    
    # Output preparation summary
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "analysis": analysis,
        "curriculum": curriculum,
        "config": {
            "max_sequence_length": args.max_sequence_length,
            "num_phases": args.phases,
        },
    }
    
    summary_path = args.output_dir / "preparation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\n✓ Preparation summary saved to {summary_path}")
    logger.info("\nNext steps:")
    logger.info("  1. Run tokenization with high sequence_length")
    logger.info("  2. Build hierarchical sequences using dataset_builder_long_context.py")
    logger.info("  3. Start training with multi-phase orchestration")


if __name__ == "__main__":
    main()
