#!/usr/bin/env python3
"""
Unified Training Pipeline - Supports Multiple Model Architectures

Run the complete training pipeline with:
- Phase 0: Repository analysis
- Phase 1: Git scraping
- Phase 2: Tokenization
- Phase 3: Embeddings (optional, for future RAG)
- Phase 4: Model training (now with model flexibility)

Uses OptimizedModelTrainer for flexible model support.
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import subprocess

try:
    from scrapers.git_scraper_dynamic import GitAnalyzer
    from training.model_trainer_unified import OptimizedModelTrainer
    from tqdm import tqdm
except ImportError as e:
    print(f"Import error: {e}")
    print("Install: pip install -r requirements.txt")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class UnifiedPipelineOrchestrator:
    """Orchestrates the complete training pipeline with model flexibility"""
    
    def __init__(self, repo_path: str, base_dir: str = None, verbose: bool = False, config_path: str = None):
        self.repo_path = Path(repo_path)
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self.verbose = verbose
        self.config_path = config_path or (self.base_dir / "training_config.yaml")
        
        # Create directories
        for dir_name in ['data', 'embeddings', 'models', 'training']:
            (self.base_dir / dir_name).mkdir(parents=True, exist_ok=True)
        
        self.results = {}
        self.stats = {}
        self.start_time = time.time()
    
    def _log(self, msg: str, level: str = 'info'):
        if self.verbose or level in ['error', 'warning']:
            getattr(logger, level)(msg)
    
    def phase_0_analyze_repository(self) -> Dict:
        """Phase 0: Analyze repository"""
        logger.info("\n" + "#"*70)
        logger.info("# PHASE 0: REPOSITORY ANALYSIS")
        logger.info("#"*70)
        
        analyzer = GitAnalyzer(str(self.repo_path), verbose=self.verbose)
        stats, all_commits = analyzer.get_repository_stats()
        
        estimated_sequences = max(1, stats['unique_commits'] // 6)
        training_params = analyzer.calculate_training_params(estimated_sequences)
        
        self.stats['repository'] = stats
        self.stats['training_params'] = training_params
        self.stats['estimated_sequences'] = estimated_sequences
        
        logger.info(f"\nRepository Analysis:")
        logger.info(f"  ✓ Total unique commits: {stats['unique_commits']}")
        logger.info(f"  ✓ Branches: {stats['branches']}")
        logger.info(f"  ✓ Unique authors: {stats['unique_authors']}")
        logger.info(f"\nEstimated Processing:")
        logger.info(f"  ✓ Token sequences: {estimated_sequences}")
        logger.info(f"  ✓ Training epochs: {training_params['epochs']}")
        logger.info(f"  ✓ Total steps: {training_params['total_steps']}")
        logger.info("\n")
        
        self.results['phase_0_analyze'] = {
            'status': 'complete',
            'repository_stats': stats,
            'training_parameters': training_params,
            'timestamp': datetime.now().isoformat(),
        }
        
        return stats
    
    def phase_1_scrape(self, stats: Dict) -> bool:
        """Phase 1: Git scraping"""
        logger.info("\n" + "#"*70)
        logger.info(f"# PHASE 1: GIT SCRAPING ({stats['unique_commits']} COMMITS)")
        logger.info("#"*70 + "\n")
        
        output_jsonl = self.base_dir / "data" / "git_history_rich.jsonl"
        
        cmd = [
            "python3",
            str(self.base_dir / "scrapers" / "git_scraper_rich.py"),
            "--repo", str(self.repo_path),
            "--output", str(output_jsonl),
            "--stats",
        ]
        
        if self.verbose:
            cmd.append("--verbose")
        
        start = time.time()
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            elapsed = time.time() - start
            
            if output_jsonl.exists():
                size_mb = output_jsonl.stat().st_size / 1e6
                lines = sum(1 for _ in open(output_jsonl))
                
                logger.info(f"✓ PHASE 1 COMPLETE ({elapsed:.1f}s)")
                logger.info(f"  Commits processed: {lines}")
                logger.info(f"  Size: {size_mb:.1f} MB\n")
                
                self.results['phase_1_scrape'] = {
                    'status': 'complete',
                    'commits_processed': lines,
                    'size_mb': size_mb,
                    'duration_seconds': elapsed,
                }
                return True
        except subprocess.CalledProcessError as e:
            logger.error(f"✗ PHASE 1 FAILED: {e.stderr}")
            self.results['phase_1_scrape'] = {'status': 'failed'}
            return False
    
    def phase_2_tokenize(self) -> bool:
        """Phase 2: Tokenization"""
        logger.info("\n" + "#"*70)
        logger.info("# PHASE 2: TOKENIZATION")
        logger.info("#"*70 + "\n")
        
        input_file = self.base_dir / "data" / "git_history_rich.jsonl"
        output_file = self.base_dir / "data" / "token_sequences_rich.json"
        
        cmd = [
            "python3",
            str(self.base_dir / "tokenizers" / "git_tokenizer_rich.py"),
            "--input", str(input_file),
            "--sequences", str(output_file),
            "--stats",
        ]
        
        if self.verbose:
            cmd.append("--verbose")
        
        start = time.time()
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            elapsed = time.time() - start
            
            if output_file.exists():
                with open(output_file) as f:
                    sequences = json.load(f)
                
                num_sequences = len(sequences)
                
                logger.info(f"✓ PHASE 2 COMPLETE ({elapsed:.1f}s)")
                logger.info(f"  Sequences: {num_sequences}")
                logger.info(f"  Total tokens: {num_sequences * 2048:,}")
                logger.info(f"\n  Re-computing training parameters...")
                
                # Update epochs based on actual sequence count
                analyzer = GitAnalyzer(str(self.repo_path), verbose=False)
                updated_params = analyzer.calculate_training_params(num_sequences)
                self.stats['training_params'] = updated_params
                logger.info(f"  Updated epochs: {updated_params['epochs']}\n")
                
                self.results['phase_2_tokenize'] = {
                    'status': 'complete',
                    'num_sequences': num_sequences,
                    'total_tokens': num_sequences * 2048,
                    'duration_seconds': elapsed,
                }
                return True
        except subprocess.CalledProcessError as e:
            logger.error(f"✗ PHASE 2 FAILED: {e.stderr}")
            self.results['phase_2_tokenize'] = {'status': 'failed'}
            return False
    
    def phase_3_embeddings(self) -> bool:
        """Phase 3: Embedding generation (optional, for future RAG)"""
        logger.info("\n" + "#"*70)
        logger.info("# PHASE 3: EMBEDDING GENERATION (OPTIONAL)")
        logger.info("#"*70 + "\n")
        
        input_file = self.base_dir / "data" / "git_history_rich.jsonl"
        output_file = self.base_dir / "embeddings" / "qdrant_points.json"
        
        logger.info("Skipping Phase 3 (embeddings not needed for training)")
        logger.info("Embeddings can be generated later if needed for RAG\n")
        
        self.results['phase_3_embeddings'] = {
            'status': 'skipped',
            'reason': 'not_needed_for_training',
        }
        return True
    
    def phase_4_training(self, training_params: Dict) -> bool:
        """Phase 4: Model training (with flexible architecture)"""
        num_epochs = training_params['epochs']
        
        logger.info("\n" + "#"*70)
        logger.info(f"# PHASE 4: MODEL TRAINING ({num_epochs} EPOCHS)")
        logger.info("#"*70 + "\n")
        
        input_file = self.base_dir / "data" / "token_sequences_rich.json"
        output_dir = self.base_dir / "models" / "the-block-git-model-final"
        
        if not input_file.exists():
            logger.error(f"Sequences file not found: {input_file}")
            return False
        
        try:
            trainer = OptimizedModelTrainer(str(self.config_path))
            stats = trainer.train(
                sequences_file=str(input_file),
                num_epochs=num_epochs,
                output_dir=str(output_dir),
            )
            
            self.results['phase_4_training'] = {
                'status': 'complete',
                'training_stats': stats,
                'model_dir': str(output_dir),
            }
            
            return True
        except Exception as e:
            logger.error(f"✗ PHASE 4 FAILED: {e}")
            self.results['phase_4_training'] = {'status': 'failed', 'error': str(e)}
            return False
    
    def run_complete_pipeline(self) -> Dict:
        """Run complete unified pipeline"""
        logger.info("\n" + "="*70)
        logger.info("UNIFIED TRAINING PIPELINE (Multi-Architecture)")
        logger.info("="*70)
        logger.info(f"Repository: {self.repo_path}")
        logger.info(f"Config: {self.config_path}")
        logger.info("="*70 + "\n")
        
        # Phase 0
        stats = self.phase_0_analyze_repository()
        if not stats:
            return {'status': 'failed', 'phase': 0}
        
        # Phase 1
        if not self.phase_1_scrape(stats):
            return {'status': 'failed', 'phase': 1}
        
        # Phase 2
        if not self.phase_2_tokenize():
            return {'status': 'failed', 'phase': 2}
        
        # Phase 3 (optional)
        self.phase_3_embeddings()
        
        # Phase 4
        training_params = self.stats['training_params']
        if not self.phase_4_training(training_params):
            return {'status': 'failed', 'phase': 4}
        
        # Success
        total_time = time.time() - self.start_time
        
        logger.info("\n" + "="*70)
        logger.info("PIPELINE COMPLETE")
        logger.info("="*70)
        logger.info(f"Total time: {total_time/60:.1f} minutes")
        logger.info(f"Status: SUCCESS")
        logger.info("="*70 + "\n")
        
        # Save manifest
        manifest = {
            'status': 'complete',
            'total_execution_time_seconds': total_time,
            'repository_stats': self.stats['repository'],
            'training_parameters': self.stats['training_params'],
            'phase_results': self.results,
            'timestamp': datetime.now().isoformat(),
        }
        
        manifest_path = self.base_dir / "MANIFEST_UNIFIED.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        logger.info(f"Manifest saved: {manifest_path}")
        
        return manifest


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Unified Training Pipeline with Model Flexibility"
    )
    parser.add_argument(
        "--repo",
        required=True,
        help="Path to Git repository"
    )
    parser.add_argument(
        "--config",
        default="training_config.yaml",
        help="Path to training config (default: training_config.yaml)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    orchestrator = UnifiedPipelineOrchestrator(
        repo_path=args.repo,
        config_path=args.config,
        verbose=args.verbose,
    )
    
    result = orchestrator.run_complete_pipeline()
    
    if result['status'] == 'complete':
        print("\n✓ Training complete! Model saved to models/the-block-git-model-final/")
        print("\nTo use the model:")
        print("  from transformers import AutoModelForCausalLM, AutoTokenizer")
        print("  model = AutoModelForCausalLM.from_pretrained('models/the-block-git-model-final')")
        print("  tokenizer = AutoTokenizer.from_pretrained('models/the-block-git-model-final')")
        sys.exit(0)
    else:
        print(f"\n✗ Pipeline failed at phase {result['phase']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
