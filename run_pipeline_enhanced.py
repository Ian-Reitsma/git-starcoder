#!/usr/bin/env python3
"""
Enhanced Unified Training Pipeline

Runs the enhanced training pipeline:
- Phase 0: Validate repository
- Phase 1: Git scraping (now required; invokes git_scraper.py)
- Phase 2: Enhanced semantic chunking
- Phase 3: Enhanced tokenization
- Phase 4: Enhanced dataset building

This file is updated to remove placeholder behavior in Phase 1.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from semantic_chunker_enhanced import EnhancedSemanticChunker
from tokenizer_enhanced import EnhancedCodeTokenizer, VocabularyBuilder
from dataset_builder_enhanced import EnhancedDatasetBuilder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnhancedPipelineOrchestrator:
    """Orchestrates the enhanced training pipeline."""

    def __init__(
        self,
        repo_path: str,
        base_dir: Optional[str] = None,
        verbose: bool = False,
        config_path: Optional[str] = None,
    ):
        self.repo_path = Path(repo_path).expanduser().resolve()
        self.base_dir = Path(base_dir).expanduser().resolve() if base_dir else Path.cwd()
        self.verbose = verbose
        self.config_path = Path(config_path).expanduser().resolve() if config_path else (self.base_dir / "training_config.yaml")

        self.data_dir = self.base_dir / "data_enhanced"
        self.data_dir.mkdir(exist_ok=True, parents=True)

        self.models_dir = self.base_dir / "models_enhanced"
        self.models_dir.mkdir(exist_ok=True, parents=True)

        self.manifest_path = self.base_dir / "MANIFEST_ENHANCED.json"

        self.manifest: Dict[str, object] = {
            "pipeline_name": "Enhanced Training Pipeline",
            "created_at": datetime.now().isoformat(),
            "repo_path": str(self.repo_path),
            "base_dir": str(self.base_dir),
            "phases": {},
        }

    def _save_manifest(self) -> None:
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(self.manifest, f, indent=2)

    def _log_phase(self, phase_num: int, phase_name: str, status: str, details: Optional[Dict] = None) -> None:
        details = details or {}
        self.manifest["phases"][f"phase_{phase_num}"] = {
            "name": phase_name,
            "status": status,
            "timestamp": datetime.now().isoformat(),
            **details,
        }
        logger.info("\n" + "=" * 70)
        logger.info(f"Phase {phase_num}: {phase_name} - {status}")
        logger.info("=" * 70)
        self._save_manifest()

    # -------------------------
    # Phase 0
    # -------------------------
    def phase_0_validate(self) -> bool:
        logger.info("\nPhase 0: Repository Validation")
        logger.info("=" * 70)

        try:
            if not self.repo_path.exists():
                self._log_phase(0, "Repository Validation", "FAILED", {"error": f"Repo not found: {self.repo_path}"})
                return False
            if not (self.repo_path / ".git").exists():
                self._log_phase(0, "Repository Validation", "FAILED", {"error": f"Not a git repo: {self.repo_path}"})
                return False

            if not self.config_path.exists():
                logger.warning(f"Config not found: {self.config_path}")

            self._log_phase(0, "Repository Validation", "SUCCESS", {"repo_path": str(self.repo_path)})
            return True
        except Exception as e:
            self._log_phase(0, "Repository Validation", "FAILED", {"error": str(e)})
            return False

    # -------------------------
    # Phase 1
    # -------------------------
    def phase_1_scrape_git(self) -> bool:
        """Scrape git history into commits_rich.json by invoking git_scraper.py."""
        logger.info("\nPhase 1: Git Scraping")
        logger.info("=" * 70)

        commits_file = self.data_dir / "commits_rich.json"

        try:
            if commits_file.exists() and commits_file.stat().st_size > 0:
                self._log_phase(1, "Git Scraping", "SKIPPED", {"commits_file": str(commits_file)})
                return True

            scraper_script = (self.base_dir / "git_scraper.py")
            if not scraper_script.exists():
                scraper_script = Path(__file__).parent / "git_scraper.py"

            if not scraper_script.exists():
                msg = f"git_scraper.py not found: {scraper_script}"
                self._log_phase(1, "Git Scraping", "FAILED", {"error": msg})
                return False

            cmd = [sys.executable, str(scraper_script), "--repo", str(self.repo_path), "--output", str(commits_file)]
            logger.info(f"Running: {' '.join(cmd)}")

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                msg = f"git_scraper.py failed (exit={result.returncode})\nSTDERR:\n{result.stderr.strip()}"
                self._log_phase(1, "Git Scraping", "FAILED", {"error": msg})
                return False

            if not commits_file.exists() or commits_file.stat().st_size == 0:
                msg = f"Git scraping completed but output missing/empty: {commits_file}"
                self._log_phase(1, "Git Scraping", "FAILED", {"error": msg})
                return False

            self._log_phase(1, "Git Scraping", "SUCCESS", {"commits_file": str(commits_file), "stdout": result.stdout.strip()})
            return True

        except Exception as e:
            self._log_phase(1, "Git Scraping", "FAILED", {"error": str(e)})
            return False

    # -------------------------
    # Phase 2
    # -------------------------
    def phase_2_enhanced_chunking(self) -> bool:
        logger.info("\nPhase 2: Enhanced Semantic Chunking")
        logger.info("=" * 70)

        commits_file = self.data_dir / "commits_rich.json"
        chunks_file = self.data_dir / "chunks_enhanced.jsonl"

        try:
            if not commits_file.exists():
                self._log_phase(2, "Enhanced Semantic Chunking", "FAILED", {"error": f"Missing {commits_file}"})
                return False

            chunker = EnhancedSemanticChunker(str(commits_file), str(chunks_file), include_cross_file=True)
            chunker.process_all()
            stats = chunker.save_statistics()
            chunker.save_jsonl()

            self._log_phase(2, "Enhanced Semantic Chunking", "SUCCESS", {"chunks_file": str(chunks_file), **stats})
            return True
        except Exception as e:
            self._log_phase(2, "Enhanced Semantic Chunking", "FAILED", {"error": str(e)})
            return False

    # -------------------------
    # Phase 3
    # -------------------------
    def phase_3_enhanced_tokenization(self) -> bool:
        logger.info("\nPhase 3: Enhanced Tokenization")
        logger.info("=" * 70)

        chunks_file = self.data_dir / "chunks_enhanced.jsonl"
        vocab_file = self.data_dir / "vocab_enhanced.json"
        tokens_file = self.data_dir / "tokens_enhanced.pt"

        try:
            if not chunks_file.exists():
                self._log_phase(3, "Enhanced Tokenization", "FAILED", {"error": f"Missing {chunks_file}"})
                return False

            vocab = VocabularyBuilder(vocab_size=50257)
            vocab.build_from_chunks(str(chunks_file))
            vocab.save(str(vocab_file))

            tokenizer = EnhancedCodeTokenizer(vocab)
            all_tokens, metadata = tokenizer.tokenize_file(str(chunks_file))

            try:
                import torch  # type: ignore

                torch.save({"tokens": all_tokens, "metadata": metadata}, str(tokens_file))
            except Exception as e:
                # Fallback to json
                with open(tokens_file.with_suffix(".json"), "w", encoding="utf-8") as f:
                    json.dump({"tokens": all_tokens, "metadata": metadata}, f)

            with_context = sum(1 for m in metadata if m.get("has_cross_file_context", False))
            self._log_phase(
                3,
                "Enhanced Tokenization",
                "SUCCESS",
                {
                    "vocab_file": str(vocab_file),
                    "tokens_file": str(tokens_file),
                    "vocab_size": int(vocab.next_id),
                    "chunks_with_context": int(with_context),
                },
            )
            return True

        except Exception as e:
            self._log_phase(3, "Enhanced Tokenization", "FAILED", {"error": str(e)})
            return False

    # -------------------------
    # Phase 4
    # -------------------------
    def phase_4_enhanced_dataset_building(self) -> bool:
        logger.info("\nPhase 4: Enhanced Dataset Building")
        logger.info("=" * 70)

        tokens_file = self.data_dir / "tokens_enhanced.pt"
        # metadata is embedded in tokens file; keep parameter for compatibility
        output_dir = self.data_dir / "dataset_enhanced"
        output_dir.mkdir(exist_ok=True, parents=True)

        try:
            if not tokens_file.exists():
                # allow JSON fallback
                json_fallback = tokens_file.with_suffix(".json")
                if json_fallback.exists():
                    tokens_file = json_fallback
                else:
                    self._log_phase(4, "Enhanced Dataset Building", "FAILED", {"error": f"Missing {tokens_file}"})
                    return False

            builder = EnhancedDatasetBuilder(
                tokens_file=str(tokens_file),
                metadata_file=str(tokens_file),
                context_window=2048,
                target_window=256,
                output_dir=str(output_dir),
                commit_based=True,
            )

            result = builder.process(prefix="training_data_enhanced")
            self._log_phase(4, "Enhanced Dataset Building", "SUCCESS", {"output_dir": str(output_dir), **result.get("stats", {})})
            return True

        except Exception as e:
            self._log_phase(4, "Enhanced Dataset Building", "FAILED", {"error": str(e)})
            return False

    def run_all(self) -> bool:
        phases = [
            (self.phase_0_validate, "Repository Validation"),
            (self.phase_1_scrape_git, "Git Scraping"),
            (self.phase_2_enhanced_chunking, "Enhanced Semantic Chunking"),
            (self.phase_3_enhanced_tokenization, "Enhanced Tokenization"),
            (self.phase_4_enhanced_dataset_building, "Enhanced Dataset Building"),
        ]

        for fn, name in phases:
            ok = fn()
            if not ok:
                logger.error(f"Pipeline failed at: {name}")
                return False

        logger.info("\nEnhanced pipeline completed successfully")
        logger.info(f"Manifest: {self.manifest_path}")
        return True


def main() -> None:
    p = argparse.ArgumentParser(description="Enhanced pipeline (fixed Phase 1)")
    p.add_argument("--repo", required=True, help="Path to git repository")
    p.add_argument("--base-dir", default=None, help="Base dir for outputs")
    p.add_argument("--config", default=None, help="Optional training config (not required for preprocessing)")
    p.add_argument("--verbose", action="store_true", help="Verbose output")

    args = p.parse_args()

    orch = EnhancedPipelineOrchestrator(
        repo_path=args.repo,
        base_dir=args.base_dir,
        verbose=args.verbose,
        config_path=args.config,
    )

    raise SystemExit(0 if orch.run_all() else 1)


if __name__ == "__main__":
    main()
