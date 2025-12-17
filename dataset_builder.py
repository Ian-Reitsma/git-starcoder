#!/usr/bin/env python3
"""
Dataset Builder: Create training dataset from tokenized chunks.

Key principles:
- Preserve chronological order (model learns progression)
- Create context windows (e.g., last 10 commits as context, predict next)
- Branch-aware contextualization
- Handle variable-length sequences

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


class DatasetBuilder:
    """Build training dataset from tokenized chunks."""
    
    def __init__(
        self,
        vocab_file: str,
        chunks_file: str,
        commits_file: str,
        context_window: int = 2048,
        target_window: int = 256,
        output_dir: str = "outputs"
    ):
        """
        Initialize dataset builder.
        
        Args:
            vocab_file: Vocabulary JSON from tokenizer
            chunks_file: Chunks JSONL file
            commits_file: Original commits JSON (for ordering)
            context_window: Number of tokens to use as context
            target_window: Number of tokens to predict
            output_dir: Directory to save output
        """
        
        self.vocab_file = Path(vocab_file)
        self.chunks_file = Path(chunks_file)
        self.commits_file = Path(commits_file)
        self.context_window = context_window
        self.target_window = target_window
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load vocabulary
        with open(vocab_file) as f:
            vocab_data = json.load(f)
        self.vocab = vocab_data["token_to_id"]
        self.pad_token_id = self.vocab.get("<PAD>", 0)
        
        # Load commit order
        with open(commits_file) as f:
            data = json.load(f)
        self.commits_by_hash = {c["metadata"]["hash"]: c for c in data.get("commits", [])}
        
        self.training_examples = []
        self.stats = {
            "total_examples": 0,
            "total_tokens": 0,
            "avg_context_len": 0,
            "avg_target_len": 0,
            "data_splits": {}
        }
    
    def build_chronological_sequence(self) -> Tuple[List[Dict], List[int]]:
        """
        Build chronological sequence of all chunks.
        
        Returns:
            chunks in order, token sequence
        """
        logger.info("Building chronological sequence...")
        
        # Load chunks in order
        chunks = []
        with open(self.chunks_file) as f:
            for line in f:
                chunk = json.loads(line)
                chunks.append(chunk)
        
        # Order by commit timestamp (should already be ordered but let's be sure)
        chunks_by_commit = defaultdict(list)
        for chunk in chunks:
            commit_hash = chunk["commit_hash"]
            chunks_by_commit[commit_hash].append(chunk)
        
        # Flatten in chronological order
        ordered_chunks = []
        for commit_hash in sorted(
            chunks_by_commit.keys(),
            key=lambda h: self.commits_by_hash.get(h, {}).get("metadata", {}).get("timestamp_unix", 0)
        ):
            ordered_chunks.extend(chunks_by_commit[commit_hash])
        
        logger.info(f"Ordered {len(ordered_chunks)} chunks chronologically")
        
        # Convert chunks to token sequence
        token_sequence = []
        chunk_positions = []  # Track which chunks are in sequence
        
        for i, chunk in enumerate(ordered_chunks):
            # Load pre-tokenized tokens if available, otherwise tokenize
            tokens = self._chunk_to_tokens(chunk)
            
            # Mark chunk boundaries
            chunk_positions.append(len(token_sequence))
            token_sequence.extend(tokens)
            
            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i+1}/{len(ordered_chunks)} chunks")
        
        logger.info(f"Created token sequence with {len(token_sequence)} total tokens")
        
        return ordered_chunks, token_sequence
    
    def _chunk_to_tokens(self, chunk: Dict[str, Any]) -> List[int]:
        """
        Convert chunk dict to token IDs.
        
        This mirrors the tokenization logic from tokenizer.py
        """
        tokens = []
        
        # Structural tokens
        tokens.append(self.vocab.get("<COMMIT_START>", 0))
        tokens.append(self.vocab.get(f"<FILE:{chunk.get('file_path', '')}>", self.vocab.get("<UNK>", 1)))
        tokens.append(self.vocab.get(f"<CHANGE:{chunk.get('change_type', '')}>", self.vocab.get("<UNK>", 1)))
        
        # Code content (simplified - in production, use pre-tokenized)
        code = chunk.get("new_code", "")
        for word in code.split()[:256]:  # Limit to 256 words
            word_id = self.vocab.get(word, self.vocab.get("<UNK>", 1))
            tokens.append(word_id)
        
        return tokens[:self.context_window + self.target_window]
    
    def create_context_target_pairs(
        self,
        token_sequence: List[int],
        stride: int = 128
    ) -> List[Dict[str, Any]]:
        """
        Create (context, target) pairs from token sequence.
        
        Args:
            token_sequence: Full token sequence from chronological chunks
            stride: How many tokens to advance for next example
        
        Returns:
            List of training examples
        """
        logger.info(f"Creating context-target pairs (stride={stride})...")
        
        examples = []
        
        for start_idx in range(0, len(token_sequence) - self.context_window, stride):
            context_end = start_idx + self.context_window
            target_end = min(context_end + self.target_window, len(token_sequence))
            
            context_tokens = token_sequence[start_idx:context_end]
            target_tokens = token_sequence[context_end:target_end]
            
            # Skip if either is too small
            if len(context_tokens) < self.context_window // 2:
                continue
            if len(target_tokens) < self.target_window // 4:
                continue
            
            # Pad sequences
            context_padded = self._pad_sequence(context_tokens, self.context_window)
            target_padded = self._pad_sequence(target_tokens, self.target_window)
            
            examples.append({
                "context": context_padded,
                "target": target_padded,
                "context_mask": self._create_mask(context_tokens, self.context_window),
                "target_mask": self._create_mask(target_tokens, self.target_window),
                "start_idx": start_idx
            })
            
            if len(examples) % 100 == 0:
                logger.info(f"Created {len(examples)} examples...")
        
        logger.info(f"Created {len(examples)} context-target pairs")
        return examples
    
    def _pad_sequence(self, seq: List[int], target_len: int) -> List[int]:
        """Pad or truncate sequence to target length."""
        if len(seq) >= target_len:
            return seq[:target_len]
        else:
            return seq + [self.pad_token_id] * (target_len - len(seq))
    
    def _create_mask(self, seq: List[int], target_len: int) -> List[int]:
        """Create attention mask (1 for real tokens, 0 for padding)."""
        mask = [1] * len(seq)
        if len(seq) < target_len:
            mask += [0] * (target_len - len(seq))
        return mask[:target_len]
    
    def split_data(
        self,
        examples: List[Dict],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Split examples into train/val/test preserving chronological order.
        
        This ensures we don't have temporal leakage.
        """
        logger.info(f"Splitting  {train_ratio:.0%} train, {val_ratio:.0%} val, {test_ratio:.0%} test")
        
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
    
    def save_dataset(
        self,
        train_ List[Dict],
        val_ List[Dict],
        test_ List[Dict],
        prefix: str = "training_data"
    ) -> Dict[str, Path]:
        """
        Save dataset splits to disk.
        
        Supports both PyTorch and pickle formats.
        """
        logger.info(f"Saving dataset to {self.output_dir}...")
        
        output_files = {}
        
        try:
            import torch
            
            # Convert to PyTorch format
            train_tensors = self._to_pytorch_format(train_data)
            val_tensors = self._to_pytorch_format(val_data)
            test_tensors = self._to_pytorch_format(test_data)
            
            train_path = self.output_dir / f"{prefix}_train.pt"
            val_path = self.output_dir / f"{prefix}_val.pt"
            test_path = self.output_dir / f"{prefix}_test.pt"
            
            torch.save(train_tensors, train_path)
            torch.save(val_tensors, val_path)
            torch.save(test_tensors, test_path)
            
            output_files = {
                "train": train_path,
                "val": val_path,
                "test": test_path
            }
            
            logger.info(f"Saved PyTorch datasets")
            
        except ImportError:
            logger.warning("PyTorch not available, using pickle")
            
            import pickle
            
            train_path = self.output_dir / f"{prefix}_train.pkl"
            val_path = self.output_dir / f"{prefix}_val.pkl"
            test_path = self.output_dir / f"{prefix}_test.pkl"
            
            with open(train_path, 'wb') as f:
                pickle.dump(train_data, f)
            with open(val_path, 'wb') as f:
                pickle.dump(val_data, f)
            with open(test_path, 'wb') as f:
                pickle.dump(test_data, f)
            
            output_files = {
                "train": train_path,
                "val": val_path,
                "test": test_path
            }
        
        # Save metadata
        metadata_path = self.output_dir / f"{prefix}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump({
                "context_window": self.context_window,
                "target_window": self.target_window,
                "vocab_size": len(self.vocab),
                "splits": self.stats["data_splits"],
                "created": datetime.utcnow().isoformat()
            }, f, indent=2)
        
        output_files["metadata"] = metadata_path
        
        return output_files
    
    def _to_pytorch_format(self, examples: List[Dict]) -> Dict[str, Any]:
        """Convert examples to PyTorch tensors."""
        import torch
        
        contexts = torch.tensor([ex["context"] for ex in examples], dtype=torch.long)
        targets = torch.tensor([ex["target"] for ex in examples], dtype=torch.long)
        context_masks = torch.tensor([ex["context_mask"] for ex in examples], dtype=torch.bool)
        target_masks = torch.tensor([ex["target_mask"] for ex in examples], dtype=torch.bool)
        
        return {
            "contexts": contexts,
            "targets": targets,
            "context_masks": context_masks,
            "target_masks": target_masks
        }
    
    def compute_statistics(self, examples: List[Dict]) -> Dict[str, Any]:
        """Compute dataset statistics."""
        context_lens = [sum(ex["context_mask"]) for ex in examples]
        target_lens = [sum(ex["target_mask"]) for ex in examples]
        
        return {
            "total_examples": len(examples),
            "total_tokens": sum(context_lens) + sum(target_lens),
            "avg_context_len": sum(context_lens) / len(context_lens) if context_lens else 0,
            "avg_target_len": sum(target_lens) / len(target_lens) if target_lens else 0,
            "min_context_len": min(context_lens) if context_lens else 0,
            "max_context_len": max(context_lens) if context_lens else 0,
        }
    
    def build(self) -> Dict[str, Path]:
        """Build complete dataset."""
        logger.info("Building complete dataset...")
        
        # Step 1: Create chronological sequence
        chunks, token_sequence = self.build_chronological_sequence()
        
        # Step 2: Create context-target pairs
        examples = self.create_context_target_pairs(token_sequence)
        
        # Step 3: Split data
        train_data, val_data, test_data = self.split_data(examples)
        
        # Step 4: Save datasets
        output_files = self.save_dataset(train_data, val_data, test_data)
        
        # Step 5: Compute and save statistics
        self.stats["train_stats"] = self.compute_statistics(train_data)
        self.stats["val_stats"] = self.compute_statistics(val_data)
        self.stats["test_stats"] = self.compute_statistics(test_data)
        
        stats_path = self.output_dir / "dataset_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(self.stats, f, indent=2, default=str)
        
        output_files["stats"] = stats_path
        
        return output_files


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Build training dataset from tokenized chunks"
    )
    parser.add_argument(
        "--vocab",
        required=True,
        help="Vocabulary JSON file (from tokenizer.py)"
    )
    parser.add_argument(
        "--chunks",
        required=True,
        help="Chunks JSONL file (from semantic_chunker.py)"
    )
    parser.add_argument(
        "--commits",
        required=True,
        help="Commits JSON file (from git_scraper.py)"
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=2048,
        help="Context window size"
    )
    parser.add_argument(
        "--target-window",
        type=int,
        default=256,
        help="Target window size"
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    try:
        builder = DatasetBuilder(
            vocab_file=args.vocab,
            chunks_file=args.chunks,
            commits_file=args.commits,
            context_window=args.context_window,
            target_window=args.target_window,
            output_dir=args.output_dir
        )
        
        output_files = builder.build()
        
        # Print summary
        print(f"\n" + "="*60)
        print("DATASET BUILDING SUMMARY")
        print("="*60)
        print(f"Context window: {args.context_window}")
        print(f"Target window: {args.target_window}")
        print(f"\nData splits:")
        for split, count in builder.stats["data_splits"].items():
            print(f"  {split.upper()}: {count}")
        print(f"\nTrain statistics:")
        for key, val in builder.stats["train_stats"].items():
            if isinstance(val, float):
                print(f"  {key}: {val:.2f}")
            else:
                print(f"  {key}: {val}")
        print(f"\nOutput files:")
        for name, path in output_files.items():
            print(f"  {name}: {path}")
        print("="*60 + "\n")
        
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        exit(1)


if __name__ == "__main__":
    main()
