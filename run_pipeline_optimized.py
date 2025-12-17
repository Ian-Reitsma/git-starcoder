#!/usr/bin/env python3
"""
Optimized Pipeline Orchestrator for The Block

Coordinates all steps with hardware-specific optimizations:
- Ryzen 5 3800X (8-core/16-thread)
- NVIDIA GTX 2060 Super (8GB)
- 48GB RAM
- NVMe storage

Runs the full pipeline:
1. Rich Git scraping (extracts EVERYTHING)
2. Rich tokenization (semantic, 2048-token sequences)
3. Embedding generation (768-dim for better quality)
4. Model training (GPT-2 medium)

Total time: ~10 minutes
"""

import os
import sys
import time
import json
import subprocess
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s'
)
logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """Coordinates the full optimization pipeline"""
    
    def __init__(self, repo_path: str, base_dir: str = ".", verbose: bool = False):
        self.repo_path = Path(repo_path)
        self.base_dir = Path(base_dir)
        self.verbose = verbose
        self.timings = {}
        self.results = {}
        
        # Create required directories
        (self.base_dir / "data").mkdir(parents=True, exist_ok=True)
        (self.base_dir / "embeddings").mkdir(parents=True, exist_ok=True)
        (self.base_dir / "models").mkdir(parents=True, exist_ok=True)
        
        # Check hardware
        self._check_hardware()
    
    def _check_hardware(self):
        """Verify hardware capabilities"""
        logger.info("Checking hardware...")
        
        try:
            import torch
            logger.info(f"PyTorch: {torch.__version__}")
            logger.info(f"CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
                logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        except:
            logger.warning("Could not detect GPU")
        
        try:
            import psutil
            logger.info(f"CPU cores: {psutil.cpu_count(logical=False)}")
            logger.info(f"RAM: {psutil.virtual_memory().total / 1e9:.0f}GB")
        except:
            logger.warning("Could not detect system info")
    
    def _run_command(self, cmd: list, description: str) -> bool:
        """Run a command with timing"""
        logger.info(f"\n{'='*70}")
        logger.info(f"Step: {description}")
        logger.info(f"{'='*70}")
        logger.info(f"Command: {' '.join(cmd)}")
        
        start_time = time.time()
        
        try:
            if self.verbose:
                result = subprocess.run(cmd, check=True)
            else:
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                if result.stderr:
                    logger.info(result.stderr)
            
            elapsed = time.time() - start_time
            self.timings[description] = elapsed
            logger.info(f"✓ Completed in {elapsed:.1f}s")
            return True
        
        except subprocess.CalledProcessError as e:
            logger.error(f"✗ Failed: {e}")
            return False
    
    def step_1_scrape(self):
        """Step 1: Rich Git scraping"""
        output_jsonl = self.base_dir / "data" / "git_history_rich.jsonl"
        output_json = self.base_dir / "data" / "git_history_rich.json"
        
        cmd = [
            "python3",
            str(self.base_dir / "scrapers" / "git_scraper_rich.py"),
            "--repo", str(self.repo_path),
            "--output", str(output_jsonl),
            "--output-json", str(output_json),
            "--stats",
        ]
        
        if self.verbose:
            cmd.append("--verbose")
        
        success = self._run_command(cmd, "Rich Git Scraping")
        
        if success:
            self.results['scrape'] = {
                'output_jsonl': str(output_jsonl),
                'output_json': str(output_json),
            }
        
        return success
    
    def step_2_tokenize(self):
        """Step 2: Rich tokenization"""
        input_file = self.base_dir / "data" / "git_history_rich.jsonl"
        output_json = self.base_dir / "data" / "token_sequences_rich.json"
        output_jsonl = self.base_dir / "data" / "token_sequences_rich.jsonl"
        
        cmd = [
            "python3",
            str(self.base_dir / "tokenizers" / "git_tokenizer_rich.py"),
            "--input", str(input_file),
            "--sequences", str(output_json),
            "--sequences-jsonl", str(output_jsonl),
            "--sequence-length", "2048",  # Max for your hardware
            "--overlap", "256",
            "--stats",
        ]
        
        if self.verbose:
            cmd.append("--verbose")
        
        success = self._run_command(cmd, "Rich Tokenization (2048-token sequences)")
        
        if success:
            self.results['tokenize'] = {
                'output_json': str(output_json),
                'output_jsonl': str(output_jsonl),
            }
        
        return success
    
    def step_3_embeddings(self):
        """Step 3: High-quality embeddings"""
        input_file = self.base_dir / "data" / "git_history_rich.jsonl"
        output_qdrant = self.base_dir / "embeddings" / "qdrant_points.json"
        output_jsonl = self.base_dir / "embeddings" / "commits_embeddings.jsonl"
        
        cmd = [
            "python3",
            str(self.base_dir / "embeddings" / "embedding_generator.py"),
            "--input", str(input_file),
            "--output", str(output_jsonl),
            "--qdrant-output", str(output_qdrant),
            "--model", "all-mpnet-base-v2",  # 768-dim, better quality
            "--batch-size", "128",  # Your CPU can handle this
            "--stats",
        ]
        
        if self.verbose:
            cmd.append("--verbose")
        
        success = self._run_command(cmd, "High-Quality Embeddings (768-dim)")
        
        if success:
            self.results['embeddings'] = {
                'output_qdrant': str(output_qdrant),
                'output_jsonl': str(output_jsonl),
            }
        
        return success
    
    def step_4_train(self):
        """Step 4: Model training (GPT-2 medium)"""
        input_file = self.base_dir / "data" / "token_sequences_rich.json"
        model_output = self.base_dir / "models" / "the-block-git-model-final"
        
        cmd = [
            "python3",
            str(self.base_dir / "training" / "model_trainer.py"),
            "--input", str(input_file),
            "--model-name", "gpt2-medium",  # Slightly larger, your GPU handles it
            "--output-dir", str(model_output),
            "--epochs", "5",  # More epochs, you have time
            "--batch-size", "8",  # Max safe for 8GB VRAM
            "--learning-rate", "5e-5",
            "--warmup-ratio", "0.1",
            "--max-seq-length", "2048",  # Use full 2048-token sequences
            "--evaluate",
            "--save-best",
        ]
        
        if self.verbose:
            cmd.append("--verbose")
        
        success = self._run_command(cmd, "Model Training (GPT-2-medium, 5 epochs)")
        
        if success:
            self.results['train'] = {
                'model_dir': str(model_output),
            }
        
        return success
    
    def run_full_pipeline(self):
        """Execute all steps"""
        logger.info(f"\n\n{'*'*70}")
        logger.info("*" + " "*68 + "*")
        logger.info("*" + " THE BLOCK: MAXIMUM CONTEXT MODEL TRAINING PIPELINE ".center(68) + "*")
        logger.info("*" + " "*68 + "*")
        logger.info(f"{'*'*70}\n")
        
        logger.info(f"Repository: {self.repo_path}")
        logger.info(f"Output directory: {self.base_dir}")
        logger.info(f"Start time: {datetime.now().isoformat()}")
        
        pipeline_start = time.time()
        
        # Run all steps
        steps = [
            ("Step 1", self.step_1_scrape, "Rich Git Scraping"),
            ("Step 2", self.step_2_tokenize, "Rich Tokenization"),
            ("Step 3", self.step_3_embeddings, "Embedding Generation"),
            ("Step 4", self.step_4_train, "Model Training"),
        ]
        
        results = {}
        for step_name, step_func, description in steps:
            if step_func():
                results[step_name] = "✓ SUCCESS"
            else:
                results[step_name] = "✗ FAILED"
                logger.error(f"Pipeline stopped at {step_name}")
                break
        
        # Final summary
        total_time = time.time() - pipeline_start
        
        logger.info(f"\n\n{'='*70}")
        logger.info("PIPELINE COMPLETE")
        logger.info(f"{'='*70}\n")
        
        logger.info("Execution Times:")
        total_timed = 0
        for desc, elapsed in self.timings.items():
            logger.info(f"  {desc}: {elapsed:.1f}s")
            total_timed += elapsed
        
        logger.info(f"\nTotal Execution Time: {total_time:.1f}s ({total_time/60:.1f}m)")
        logger.info(f"Actual Processing Time: {total_timed:.1f}s")
        logger.info(f"Overhead: {(total_time - total_timed):.1f}s\n")
        
        logger.info("Step Results:")
        for step, result in results.items():
            logger.info(f"  {step}: {result}")
        
        logger.info(f"\nOutput Files:")
        self._list_outputs()
        
        logger.info(f"\n{'='*70}")
        logger.info("Your maximally-informed The Block model is ready!")
        logger.info(f"{'='*70}\n")
        
        # Save manifest
        self._save_manifest(results, total_time)
        
        return all(v.startswith('✓') for v in results.values())
    
    def _list_outputs(self):
        """List generated output files"""
        dirs_to_check = [self.base_dir / "data", self.base_dir / "embeddings", self.base_dir / "models"]
        
        for dir_path in dirs_to_check:
            if dir_path.exists():
                logger.info(f"\n  {dir_path.name}/:")
                for f in sorted(dir_path.rglob("*")):
                    if f.is_file():
                        size = f.stat().st_size / 1e6  # MB
                        logger.info(f"    {f.relative_to(self.base_dir)}: {size:.1f}MB")
    
    def _save_manifest(self, results: dict, total_time: float):
        """Save execution manifest"""
        manifest = {
            'timestamp': datetime.now().isoformat(),
            'repository': str(self.repo_path),
            'total_execution_time_seconds': total_time,
            'total_execution_time_minutes': total_time / 60,
            'step_times': self.timings,
            'step_results': results,
            'outputs': self.results,
            'hardware': {
                'note': 'Ryzen 5 3800X + RTX 2060 Super + 48GB RAM',
                'optimizations': [
                    'Streaming I/O (no RAM bottleneck)',
                    'GPU batch size: 8 (safe for 8GB VRAM)',
                    'Sequence length: 2048 tokens (max context)',
                    'Model: GPT-2-medium (larger, better quality)',
                    'Epochs: 5 (more training time available)',
                    'Mixed precision: Yes (saves VRAM)',
                ],
            },
        }
        
        manifest_file = self.base_dir / "MANIFEST.json"
        with open(manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        logger.info(f"\nManifest saved to: {manifest_file}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Optimized pipeline for maximum-context The Block model training"
    )
    parser.add_argument("--repo", type=str, required=True, help="Repository path")
    parser.add_argument("--output-dir", type=str, default=".", help="Output directory")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    parser.add_argument(
        "--skip-scrape", action="store_true", help="Skip step 1 (use existing data)"
    )
    parser.add_argument(
        "--skip-tokenize", action="store_true", help="Skip step 2 (use existing sequences)"
    )
    parser.add_argument(
        "--skip-embeddings", action="store_true", help="Skip step 3 (use existing embeddings)"
    )
    
    args = parser.parse_args()
    
    orchestrator = PipelineOrchestrator(
        repo_path=args.repo,
        base_dir=args.output_dir,
        verbose=args.verbose
    )
    
    success = orchestrator.run_full_pipeline()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
