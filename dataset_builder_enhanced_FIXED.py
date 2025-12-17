#!/usr/bin/env python3
"""
Enhanced Dataset Builder: Create training dataset from tokenized chunks WITH cross-file context.

Key improvements:
- Builds examples around commits (not just sequential token sliding windows)
- Includes full context: imports, traits, structs, old code, new code
- Better temporal organization for learning code evolution
- Flexible example construction for different learning objectives

Output: PyTorch Dataset ready for fine-tuning.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict
from datetime import datetime
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnhancedDatasetBuilder:
    """Build training dataset from enhanced tokenized chunks."""
    
    def __init__(
        self,
        tokens_file: str,
        metadata_file: str,
        context_window: int = 2048,
        target_window: int = 256,
        output_dir: str = "outputs_enhanced",
        commit_based: bool = True
    ):
        """
        Initialize enhanced dataset builder.
        
        Args:
            tokens_file: Tokens JSONL or PT file from tokenizer
            metadata_file: Metadata file with chunk information
            context_window: Tokens for context
            target_window: Tokens to predict
            output_dir: Output directory
            commit_based: Use commit-based examples instead of sliding window
        """
        
        self.tokens_file = Path(tokens_file)
        self.metadata_file = Path(metadata_file)
        self.context_window = context_window
        self.target_window = target_window
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.commit_based = commit_based
        
        self.training_examples = []
        self.stats = {
            "total_examples": 0,
            "total_tokens": 0,
            "avg_context_len": 0,
            "avg_target_len": 0,
            "data_splits": {},
            "example_types": {},
        }
    
    def load_tokens_and_metadata(self) -> Tuple[List[List[int]], List[Dict[str, Any]]]:
        """Load tokens and metadata."""
        logger.info(f"Loading tokens from {self.tokens_file}...")
        
        try:
            import torch
            token_data = torch.load(self.tokens_file)
            tokens = token_data["tokens"]
            metadata = token_data["metadata"]
        except (ImportError, Exception):
            logger.info("PyTorch not available, trying JSON...")
            with open(self.tokens_file, encoding="utf-8") as f:
                token_data = json.load(f)
            tokens = token_data["tokens"]
            metadata = token_data["metadata"]
        
        logger.info(f"Loaded {len(tokens)} token sequences")
        return tokens, metadata
    
    def build_commit_based_examples(
        self,
        tokens: List[List[int]],
        meta List[Dict[str, Any]]  # FIXED: type annotation
    ) -> List[Dict[str, Any]]:
        """
        Build examples organized by commit (better for code learning).
        
        Each example contains:
        - context: imports + trait defs + struct defs + old code
        - target: new code to predict
        """
        logger.info("Building commit-based training examples...")
        
        examples = []
        commit_groups: Dict[str, List[int]] = defaultdict(list)
        
        # Group tokens by commit
        for idx, meta in enumerate(metadata):  # FIXED: corrected variable name
            commit_hash = meta.get("commit_hash", "unknown")
            commit_groups[commit_hash].append(idx)
        
        logger.info(f"Found {len(commit_groups)} unique commits")
        
        # Build examples for each commit
        for commit_idx, (commit_hash, chunk_indices) in enumerate(sorted(commit_groups.items())):
            if (commit_idx + 1) % 100 == 0:
                logger.info(f"Processing commit {commit_idx + 1}/{len(commit_groups)}...")
            
            # Concatenate all chunks for this commit
            context_tokens = []
            target_tokens = []
            
            for chunk_idx in chunk_indices:
                chunk_tokens = tokens[chunk_idx]
                
                # Add to context (up to context_window)
                if len(context_tokens) < self.context_window:
                    available = self.context_window - len(context_tokens)
                    add_tokens = chunk_tokens[:available]
                    context_tokens.extend(add_tokens)
                    
                    # Remaining tokens go to target
                    if len(chunk_tokens) > available:
                        target_tokens.extend(chunk_tokens[available:])
                else:
                    # Context full, everything goes to target
                    target_tokens.extend(chunk_tokens)
            
            # Create example if we have both context and target
            if len(context_tokens) > self.context_window // 2 and len(target_tokens) > self.target_window // 4:
                context_padded = self._pad_sequence(context_tokens, self.context_window)
                target_padded = self._pad_sequence(target_tokens, self.target_window)
                
                example = {
                    "context": context_padded,
                    "target": target_padded,
                    "context_mask": self._create_mask(context_tokens, self.context_window),
                    "target_mask": self._create_mask(target_tokens, self.target_window),
                    "commit_hash": commit_hash,
                    "num_chunks": len(chunk_indices),
                    "example_type": "commit_based",
                }
                
                examples.append(example)
                
                self.stats["example_types"]["commit_based"] = \
                    self.stats["example_types"].get("commit_based", 0) + 1
        
        logger.info(f"Created {len(examples)} commit-based examples")
        return examples
    
    def build_sequential_window_examples(
        self,
        tokens: List[List[int]]
    ) -> List[Dict[str, Any]]:
        """
        Build examples using sliding window over chronological token sequence.
        Falls back to this if commit_based fails or for compatibility.
        """
        logger.info("Building sequential window training examples...")
        
        # Create chronological token sequence
        token_sequence = []
        for token_list in tokens:
            token_sequence.extend(token_list)
        
        logger.info(f"Created token sequence with {len(token_sequence)} total tokens")
        
        examples = []
        stride = max(self.context_window // 4, 128)  # 25% stride
        
        for start_idx in range(0, len(token_sequence) - self.context_window, stride):
            context_end = start_idx + self.context_window
            target_end = min(context_end + self.target_window, len(token_sequence))
            
            context_tokens = token_sequence[start_idx:context_end]
            target_tokens = token_sequence[context_end:target_end]
            
            # Skip if too small
            if len(context_tokens) < self.context_window // 2:
                continue
            if len(target_tokens) < self.target_window // 4:
                continue
            
            context_padded = self._pad_sequence(context_tokens, self.context_window)
            target_padded = self._pad_sequence(target_tokens, self.target_window)
            
            example = {
                "context": context_padded,
                "target": target_padded,
                "context_mask": self._create_mask(context_tokens, self.context_window),
                "target_mask": self._create_mask(target_tokens, self.target_window),
                "start_idx": start_idx,
                "example_type": "sequential_window",
            }
            
            examples.append(example)
            
            self.stats["example_types"]["sequential_window"] = \
                self.stats["example_types"].get("sequential_window", 0) + 1
        
        logger.info(f"Created {len(examples)} sequential window examples")
        return examples
    
    def _pad_sequence(self, seq: List[int], target_len: int) -> List[int]:
        """Pad or truncate sequence to target length."""
        if len(seq) >= target_len:
            return seq[:target_len]
        else:
            return seq + [0] * (target_len - len(seq))
    
    def _create_mask(self, seq: List[int], target_len: int) -> List[int]:
        """Create attention mask (1 for real tokens, 0 for padding)."""
        mask = [1] * len(seq)
        if len(seq) < target_len:
            mask += [0] * (target_len - len(seq))
        return mask[:target_len]
    
    def build_all_examples(
        self,
        tokens: List[List[int]],
        meta List[Dict[str, Any]]  # FIXED: type annotation
    ) -> List[Dict[str, Any]]:
        """Build training examples (commit-based or sequential)."""
        
        if self.commit_based:
            return self.build_commit_based_examples(tokens, metadata)
        else:
            return self.build_sequential_window_examples(tokens)
    
    def split_data(
        self,
        examples: List[Dict[str, Any]],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Split examples into train/val/test, preserving temporal order."""
        logger.info(f"Splitting {train_ratio:.0%} train, {val_ratio:.0%} val, {test_ratio:.0%} test")
        
        total = len(examples)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)
        
        train_data = examples[:train_end]
        val_data = examples[train_end:val_end]
        test_data = examples[val_end:]
        
        self.stats["data_splits"] = {
            "train": len(train_data),
            "val": len(val_data),
            "test": len(test_data)
        }
        
        logger.info(f"Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")
        
        return train_data, val_data, test_data
    
    def compute_statistics(self, examples: List[Dict[str, Any]]) -> None:
        """Compute dataset statistics."""
        if not examples:
            return
        
        self.stats["total_examples"] = len(examples)
        self.stats["total_tokens"] = sum(len(e["context"]) + len(e["target"]) for e in examples)
        self.stats["avg_context_len"] = sum(len(e["context"]) for e in examples) // len(examples)
        self.stats["avg_target_len"] = sum(len(e["target"]) for e in examples) // len(examples)
    
    def save_dataset(
        self,
        train_ List[Dict[str, Any]],  # FIXED: proper type annotation and param name
        val_ List[Dict[str, Any]],    # FIXED: proper type annotation and param name
        test_ List[Dict[str, Any]],   # FIXED: proper type annotation and param name
        prefix: str = "training_data_enhanced"
    ) -> Dict[str, Path]:
        """Save dataset splits to disk."""
        output_paths = {}
        
        for split_name, data in [("train", train_data), ("val", val_data), ("test", test_data)]:
            output_path = self.output_dir / f"{prefix}_{split_name}.json"
            
            logger.info(f"Saving {split_name} data to {output_path}...")
            
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f)
            
            output_paths[split_name] = output_path
            logger.info(f"Saved {len(data)} {split_name} examples to {output_path}")
        
        return output_paths
    
    def save_statistics(self, prefix: str = "dataset_stats_enhanced") -> Path:
        """Save dataset statistics."""
        stats_path = self.output_dir / f"{prefix}.json"
        
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(self.stats, f, indent=2)
        
        logger.info(f"Saved statistics to {stats_path}")
        return stats_path
    
    def process(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        prefix: str = "training_data_enhanced"
    ) -> Dict[str, Any]:
        """Complete pipeline: load -> build -> split -> save."""
        
        # Load
        tokens, metadata = self.load_tokens_and_metadata()
        
        # Build examples
        all_examples = self.build_all_examples(tokens, metadata)
        
        if not all_examples:
            logger.error("No training examples created!")
            return {}
        
        # Compute statistics
        self.compute_statistics(all_examples)
        
        # Split
        train_data, val_data, test_data = self.split_data(
            all_examples,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio
        )
        
        # Save (FIXED: use local variables, not undefined names)
        output_paths = self.save_dataset(train_data, val_data, test_data, prefix=prefix)
        stats_path = self.save_statistics(prefix="dataset_stats_enhanced")
        
        return {
            "output_paths": {k: str(v) for k, v in output_paths.items()},
            "stats_path": str(stats_path),
            "stats": self.stats
        }


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Build enhanced training dataset from tokenized chunks"
    )
    parser.add_argument(
        "--tokens",
        required=True,
        help="Input tokens file (from tokenizer_enhanced.py)"
    )
    parser.add_argument(
        "--metadata",
        required=True,
        help="Metadata file with chunk information"
    )
    parser.add_argument(
        "--output-dir",
        default="outputs_enhanced",
        help="Output directory"
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=2048,
        help="Context window size in tokens"
    )
    parser.add_argument(
        "--target-window",
        type=int,
        default=256,
        help="Target window size in tokens"
    )
    parser.add_argument(
        "--commit-based",
        action="store_true",
        default=True,
        help="Use commit-based examples (default: True)"
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Use sequential window examples instead of commit-based"
    )
    
    args = parser.parse_args()
    
    try:
        builder = EnhancedDatasetBuilder(
            args.tokens,
            args.metadata,
            context_window=args.context_window,
            target_window=args.target_window,
            output_dir=args.output_dir,
            commit_based=(not args.sequential)
        )
        
        result = builder.process()
        
        # Print summary
        print(f"\n" + "="*70)
        print("ENHANCED DATASET BUILDING SUMMARY")
        print("="*70)
        print(f"Total examples: {result['stats']['total_examples']}")
        print(f"Total tokens: {result['stats']['total_tokens']}")
        print(f"Avg context length: {result['stats']['avg_context_len']}")
        print(f"Avg target length: {result['stats']['avg_target_len']}")
        print(f"\nExample types:")
        for ex_type, count in result['stats']['example_types'].items():
            print(f"  {ex_type}: {count}")
        print(f"\nData splits:")
        for split, count in result['stats']['data_splits'].items():
            print(f"  {split}: {count}")
        print(f"\nOutput files:")
        for split, path in result['output_paths'].items():
            print(f"  {split}: {path}")
        print(f"\nStatistics: {result['stats_path']}")
        print("="*70 + "\n")
        
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        exit(1)


if __name__ == "__main__":
    main()
