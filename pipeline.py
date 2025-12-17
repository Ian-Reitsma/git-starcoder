#!/usr/bin/env python3
"""
End-to-End Pipeline Orchestrator

Coordinates the entire flow:
1. Scrape Git repository
2. Tokenize commits
3. Generate embeddings
4. Fine-tune model
5. Prepare for RAG system

Usage:
    python pipeline.py --repo /Users/ianreitsma/projects/the-block --run all
    python pipeline.py --repo /Users/ianreitsma/projects/the-block --run scrape,tokenize
"""

import sys
import os
from pathlib import Path
import subprocess
import argparse
import json
from datetime import datetime


class Pipeline:
    """End-to-end orchestration"""
    
    def __init__(self, repo_path: str, base_dir: str):
        self.repo_path = Path(repo_path)
        self.base_dir = Path(base_dir)
        self.data_dir = self.base_dir / "data"
        self.model_dir = self.base_dir / "models"
        self.embedding_dir = self.base_dir / "embeddings"
        
        # Create directories
        for d in [self.data_dir, self.model_dir, self.embedding_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        self.script_dir = self.base_dir
        self.log_file = self.base_dir / "pipeline.log"
    
    def _log(self, msg: str):
        """Log message with timestamp"""
        timestamp = datetime.now().isoformat()
        log_msg = f"[{timestamp}] {msg}"
        print(log_msg)
        with open(self.log_file, 'a') as f:
            f.write(log_msg + '\n')
    
    def _run_command(self, cmd: list, description: str) -> bool:
        """Run shell command and handle errors"""
        self._log(f"Starting: {description}")
        self._log(f"Command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.script_dir),
                capture_output=False,
                text=True,
                check=True
            )
            self._log(f"✓ Completed: {description}")
            return True
        except subprocess.CalledProcessError as e:
            self._log(f"✗ Failed: {description}")
            self._log(f"Error: {e}")
            return False
        except Exception as e:
            self._log(f"✗ Exception: {description}")
            self._log(f"Error: {e}")
            return False
    
    def scrape(self) -> bool:
        """Step 1: Scrape Git repository"""
        self._log("\n" + "="*60)
        self._log("STEP 1: Scraping Git Repository")
        self._log("="*60)
        
        output_file = self.data_dir / "git_history.jsonl"
        
        cmd = [
            "python",
            str(self.script_dir / "scrapers" / "git_scraper.py"),
            "--repo", str(self.repo_path),
            "--output", str(output_file),
            "--stats",
            "--verbose",
        ]
        
        success = self._run_command(cmd, "Git Scraping")
        
        if success and output_file.exists():
            with open(output_file, 'r') as f:
                num_lines = sum(1 for _ in f)
            self._log(f"Scraped {num_lines} commits")
        
        return success
    
    def tokenize(self) -> bool:
        """Step 2: Tokenize commits"""
        self._log("\n" + "="*60)
        self._log("STEP 2: Tokenizing Commits")
        self._log("="*60)
        
        input_file = self.data_dir / "git_history.jsonl"
        output_file = self.data_dir / "tokenized_commits.jsonl"
        sequences_file = self.data_dir / "token_sequences.json"
        
        cmd = [
            "python",
            str(self.script_dir / "tokenizers" / "git_tokenizer.py"),
            "--input", str(input_file),
            "--output", str(output_file),
            "--sequences", str(sequences_file),
            "--strategy", "semantic",
            "--model", "gpt2",
            "--stats",
        ]
        
        return self._run_command(cmd, "Tokenization")
    
    def embed(self) -> bool:
        """Step 3: Generate embeddings"""
        self._log("\n" + "="*60)
        self._log("STEP 3: Generating Embeddings")
        self._log("="*60)
        
        input_file = self.data_dir / "git_history.jsonl"
        output_file = self.embedding_dir / "commits.jsonl"
        qdrant_file = self.embedding_dir / "qdrant_points.json"
        
        cmd = [
            "python",
            str(self.script_dir / "embeddings" / "embedding_generator.py"),
            "--input", str(input_file),
            "--output", str(output_file),
            "--qdrant-output", str(qdrant_file),
            "--model", "all-MiniLM-L6-v2",
            "--stats",
        ]
        
        return self._run_command(cmd, "Embedding Generation")
    
    def train(self) -> bool:
        """Step 4: Train model"""
        self._log("\n" + "="*60)
        self._log("STEP 4: Training Model")
        self._log("="*60)
        
        input_file = self.data_dir / "token_sequences.json"
        
        cmd = [
            "python",
            str(self.script_dir / "training" / "model_trainer.py"),
            "--input", str(input_file),
            "--model-name", "gpt2",
            "--output-dir", str(self.model_dir),
            "--batch-size", "4",
            "--epochs", "3",
            "--evaluate",
        ]
        
        return self._run_command(cmd, "Model Training")
    
    def run_all(self) -> bool:
        """Run complete pipeline"""
        self._log("\n" + "#"*60)
        self._log("# COMPLETE PIPELINE EXECUTION")
        self._log("#"*60)
        
        steps = [
            ("scrape", self.scrape),
            ("tokenize", self.tokenize),
            ("embed", self.embed),
            ("train", self.train),
        ]
        
        results = {}
        for step_name, step_func in steps:
            results[step_name] = step_func()
            if not results[step_name]:
                self._log(f"\n✗ Pipeline stopped at {step_name}")
                return False
        
        # Print summary
        self._log("\n" + "="*60)
        self._log("PIPELINE SUMMARY")
        self._log("="*60)
        for step_name, success in results.items():
            status = "✓" if success else "✗"
            self._log(f"{status} {step_name}")
        
        # Create manifest
        manifest = {
            "timestamp": datetime.now().isoformat(),
            "repo_path": str(self.repo_path),
            "base_dir": str(self.base_dir),
            "steps_completed": [k for k, v in results.items() if v],
            "data_dir": str(self.data_dir),
            "model_dir": str(self.model_dir),
            "embedding_dir": str(self.embedding_dir),
        }
        
        manifest_file = self.base_dir / "manifest.json"
        with open(manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        self._log(f"\nManifest saved to {manifest_file}")
        self._log(f"Log file: {self.log_file}")
        
        return all(results.values())
    
    def run_steps(self, steps: list) -> bool:
        """Run specific steps"""
        step_map = {
            "scrape": self.scrape,
            "tokenize": self.tokenize,
            "embed": self.embed,
            "train": self.train,
        }
        
        results = {}
        for step in steps:
            if step not in step_map:
                self._log(f"Unknown step: {step}")
                return False
            
            results[step] = step_map[step]()
            if not results[step]:
                self._log(f"Step {step} failed, stopping")
                return False
        
        return True


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end Git scraping and model training pipeline"
    )
    parser.add_argument(
        "--repo",
        type=str,
        default="/Users/ianreitsma/projects/the-block",
        help="Path to Git repository"
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        help="Base directory for pipeline (default: ~/.perplexity/git-scrape-scripting)"
    )
    parser.add_argument(
        "--run",
        type=str,
        default="all",
        help="Comma-separated steps: scrape,tokenize,embed,train or 'all'"
    )
    
    args = parser.parse_args()
    
    # Determine base directory
    if args.base_dir:
        base_dir = Path(args.base_dir)
    else:
        base_dir = Path.home() / "projects" / "the-block" / ".perplexity" / "git-scrape-scripting"
    
    print(f"Pipeline Configuration")
    print(f"  Repository: {args.repo}")
    print(f"  Base Dir: {base_dir}")
    print()
    
    pipeline = Pipeline(args.repo, str(base_dir))
    
    if args.run.lower() == "all":
        success = pipeline.run_all()
    else:
        steps = [s.strip() for s in args.run.split(',')]
        success = pipeline.run_steps(steps)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
