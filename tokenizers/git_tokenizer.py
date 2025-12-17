#!/usr/bin/env python3
"""
Git Data Tokenizer

Converts Git commit history into optimized token sequences for LLM training.
Creates multiple tokenization strategies:
1. Flat - Simple token-by-token representation
2. Semantic - Groups related commits and metadata
3. Hierarchical - Maintains branching and temporal relationships
4. Diff-aware - Focuses on code changes with context

Usage:
    python git_tokenizer.py --input data/git_history.jsonl --strategy semantic
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass
from enum import Enum
import argparse
from tqdm import tqdm

try:
    from transformers import AutoTokenizer
    import numpy as np
except ImportError:
    print("Please install transformers: pip install transformers")
    sys.exit(1)


class TokenizationStrategy(Enum):
    """Different ways to tokenize git data"""
    FLAT = "flat"
    SEMANTIC = "semantic"
    HIERARCHICAL = "hierarchical"
    DIFF_AWARE = "diff_aware"


@dataclass
class TokenizedCommit:
    """A commit after tokenization"""
    commit_hash: str
    tokens: List[int]
    token_count: int
    token_string: str  # Human-readable token representation
    meta Dict  # Original commit metadata
    context_window: str  # Surrounding commits for context


class GitTokenizer:
    """Tokenizes Git commit history for model training"""
    
    def __init__(self, model_name: str = "gpt2", strategy: TokenizationStrategy = TokenizationStrategy.SEMANTIC):
        """Initialize tokenizer with HF model"""
        self.model_name = model_name
        self.strategy = strategy
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Add special tokens for git operations
        self.special_tokens = {
            "<COMMIT>": "<COMMIT>",
            "<MERGE>": "<MERGE>",
            "<FILE_ADD>": "<FILE_ADD>",
            "<FILE_MOD>": "<FILE_MOD>",
            "<FILE_DEL>": "<FILE_DEL>",
            "<AUTHOR>": "<AUTHOR>",
            "<BRANCH>": "<BRANCH>",
            "<MESSAGE>": "<MESSAGE>",
            "<DIFF_STAT>": "<DIFF_STAT>",
            "<INSERTIONS>": "<INSERTIONS>",
            "<DELETIONS>": "<DELETIONS>",
            "<PARENT>": "<PARENT>",
        }
        
        self.tokenizer.add_tokens(list(self.special_tokens.values()))
        self.tokenized_commits: List[TokenizedCommit] = []
    
    def _load_jsonl(self, path: str) -> List[Dict]:
        """Load JSONL commit data"""
        commits = []
        with open(path, 'r') as f:
            for line in f:
                if line.strip():
                    commits.append(json.loads(line))
        return commits
    
    def _format_flat(self, commit: Dict) -> str:
        """
        Flat tokenization: Simple sequential representation
        
        Format:
        <COMMIT> hash author subject [insertions/deletions]
        """
        parts = [
            self.special_tokens["<COMMIT>"],
            commit["commit_hash"][:8],
            self.special_tokens["<AUTHOR>"],
            commit["author_name"],
            self.special_tokens["<MESSAGE>"],
            commit["message_subject"],
        ]
        
        if commit["is_merge"]:
            parts.insert(1, self.special_tokens["<MERGE>"])
        
        parts.extend([
            self.special_tokens["<INSERTIONS>"],
            str(commit["insertions"]),
            self.special_tokens["<DELETIONS>"],
            str(commit["deletions"]),
        ])
        
        return " ".join(parts)
    
    def _format_semantic(self, commit: Dict, context_commits: List[Dict] = None) -> str:
        """
        Semantic tokenization: Structured with metadata
        
        Format:
        <COMMIT> {hash}
        <AUTHOR> {author} <TIMESTAMP> {ts}
        <BRANCH> {branch}
        <MESSAGE> {subject}
        <FILES> <ADD> {count} <MOD> {count} <DEL> {count}
        <CHANGES> <INSERTIONS> {count} <DELETIONS> {count}
        <MODIFIED_FILES> {files separated by semicolon}
        """
        parts = [
            self.special_tokens["<COMMIT>"],
            commit["commit_hash"][:8],
        ]
        
        # Author context
        parts.extend([
            self.special_tokens["<AUTHOR>"],
            commit["author_name"].replace(" ", "_"),
            commit["author_email"],
        ])
        
        # Branch context
        parts.extend([
            self.special_tokens["<BRANCH>"],
            commit["branch"].replace("/", "_"),
        ])
        
        # Message
        parts.extend([
            self.special_tokens["<MESSAGE>"],
            commit["message_subject"][:100],  # Limit length
        ])
        
        # File changes with semantic markers
        if commit["files_added"]:
            parts.extend([
                self.special_tokens["<FILE_ADD>"],
                str(len(commit["files_added"]))
            ])
        
        if commit["files_modified"]:
            parts.extend([
                self.special_tokens["<FILE_MOD>"],
                str(len(commit["files_modified"]))
            ])
        
        if commit["files_deleted"]:
            parts.extend([
                self.special_tokens["<FILE_DEL>"],
                str(len(commit["files_deleted"]))
            ])
        
        # Change statistics
        parts.extend([
            self.special_tokens["<INSERTIONS>"],
            str(commit["insertions"]),
            self.special_tokens["<DELETIONS>"],
            str(commit["deletions"]),
        ])
        
        # Modified files (important for context)
        if commit["files_modified"]:
            files_str = ";".join(commit["files_modified"][:5])  # Top 5 files
            parts.extend(["FILES", files_str])
        
        # Parent context (for understanding commit relationships)
        if commit["parent_hashes"]:
            parts.extend([
                self.special_tokens["<PARENT>"],
                commit["parent_hashes"][0][:8]
            ])
        
        return " ".join(parts)
    
    def _format_hierarchical(self, commits: List[Dict], commit_index: int) -> str:
        """
        Hierarchical tokenization: Maintains branch relationships and temporal order
        
        Includes commit chains and branching information
        """
        commit = commits[commit_index]
        parts = []
        
        # Add predecessor context (up to 2 previous commits on same branch)
        predecessors = []
        for i in range(max(0, commit_index - 2), commit_index):
            if commits[i]["branch"] == commit["branch"]:
                predecessors.append(commits[i]["commit_hash"][:8])
        
        if predecessors:
            parts.extend(["<PRED>", "|".join(predecessors)])
        
        # Main commit
        parts.extend([
            self.special_tokens["<COMMIT>"],
            commit["commit_hash"][:8],
            self.special_tokens["<BRANCH>"],
            commit["branch"],
        ])
        
        # Merge information
        if commit["is_merge"]:
            parts.extend([
                self.special_tokens["<MERGE>"],
                "|" .join(p[:8] for p in commit["parent_hashes"])
            ])
        
        # Author and message
        parts.extend([
            self.special_tokens["<AUTHOR>"],
            commit["author_name"],
            self.special_tokens["<MESSAGE>"],
            commit["message_subject"][:80],
        ])
        
        # Changes
        parts.extend([
            self.special_tokens["<DIFF_STAT>"],
            commit["diff_summary"],
        ])
        
        # Modified files
        all_files = (
            commit["files_added"] +
            commit["files_modified"] +
            commit["files_deleted"]
        )
        if all_files:
            parts.extend(["FILES", "|".join(f.split("/")[-1] for f in all_files[:10])])
        
        return " ".join(parts)
    
    def _format_diff_aware(self, commit: Dict) -> str:
        """
        Diff-aware tokenization: Emphasizes code changes
        
        Format includes file paths, change types, and statistics per file
        """
        parts = [
            self.special_tokens["<COMMIT>"],
            commit["commit_hash"][:8],
            self.special_tokens["<AUTHOR>"],
            commit["author_name"],
        ]
        
        # Detailed file changes with per-file stats
        for filepath, stats in list(commit["stats"].items())[:10]:  # Top 10 files
            # Determine file type
            if filepath.endswith('.rs'):
                file_marker = "<RS_FILE>"
            elif filepath.endswith('.toml'):
                file_marker = "<TOML_FILE>"
            elif filepath.endswith('.proto'):
                file_marker = "<PROTO_FILE>"
            else:
                file_marker = "<FILE>"
            
            # Add file change info
            parts.extend([
                file_marker,
                filepath.split("/")[-1],  # Just filename
                self.special_tokens["<INSERTIONS>"],
                str(stats.get("insertions", 0)),
                self.special_tokens["<DELETIONS>"],
                str(stats.get("deletions", 0)),
            ])
        
        # Summary stats
        parts.extend([
            self.special_tokens["<DIFF_STAT>"],
            commit["diff_summary"],
        ])
        
        # Message for context
        parts.extend([
            self.special_tokens["<MESSAGE>"],
            commit["message_subject"][:100],
        ])
        
        return " ".join(parts)
    
    def tokenize(self, input_path: str) -> List[TokenizedCommit]:
        """Tokenize all commits from input file"""
        
        print(f"Loading commits from {input_path}...")
        commits = self._load_jsonl(input_path)
        print(f"Loaded {len(commits)} commits")
        
        print(f"Tokenizing using {self.strategy.value} strategy...")
        
        for i, commit in enumerate(tqdm(commits, desc="Tokenizing")):
            # Select formatting strategy
            if self.strategy == TokenizationStrategy.FLAT:
                token_string = self._format_flat(commit)
            elif self.strategy == TokenizationStrategy.SEMANTIC:
                token_string = self._format_semantic(commit)
            elif self.strategy == TokenizationStrategy.HIERARCHICAL:
                token_string = self._format_hierarchical(commits, i)
            elif self.strategy == TokenizationStrategy.DIFF_AWARE:
                token_string = self._format_diff_aware(commit)
            else:
                raise ValueError(f"Unknown strategy: {self.strategy}")
            
            # Tokenize the string
            tokens = self.tokenizer.encode(token_string)
            
            # Create context window (surrounding commits)
            context_commits = [
                commits[j] for j in range(max(0, i-2), min(len(commits), i+3))
                if j != i
            ]
            context_string = " ".join([self._format_semantic(c) for c in context_commits])
            
            tokenized = TokenizedCommit(
                commit_hash=commit["commit_hash"],
                tokens=tokens,
                token_count=len(tokens),
                token_string=token_string,
                metadata=commit,
                context_window=context_string
            )
            
            self.tokenized_commits.append(tokenized)
        
        return self.tokenized_commits
    
    def save_tokens(self, output_path: str):
        """Save tokenized data as JSONL"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"Saving {len(self.tokenized_commits)} tokenized commits to {output_path}...")
        
        with open(output_file, 'w') as f:
            for tc in self.tokenized_commits:
                data = {
                    "commit_hash": tc.commit_hash,
                    "tokens": tc.tokens,
                    "token_count": tc.token_count,
                    "token_string": tc.token_string,
                    "context_window": tc.context_window,
                    "metadata": tc.metadata,
                }
                f.write(json.dumps(data) + '\n')
    
    def save_token_sequences(self, output_path: str, max_length: int = 2048):
        """
        Save as contiguous token sequences for training
        
        This format is optimal for language model training.
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Concatenate all tokens with separators
        all_tokens = []
        metadata_map = {}
        
        print("Building token sequences...")
        for tc in tqdm(self.tokenized_commits):
            all_tokens.extend(tc.tokens)
            all_tokens.append(self.tokenizer.eos_token_id)  # Separator
        
        print(f"Total tokens: {len(all_tokens)}")
        
        # Split into training sequences
        sequences = []
        for i in range(0, len(all_tokens), max_length):
            seq = all_tokens[i:i+max_length]
            if len(seq) >= 512:  # Minimum sequence length
                sequences.append(seq)
        
        print(f"Created {len(sequences)} training sequences")
        print(f"Saving to {output_path}...")
        
        with open(output_file, 'w') as f:
            json.dump({
                "token_sequences": sequences,
                "vocab_size": self.tokenizer.vocab_size,
                "num_sequences": len(sequences),
                "total_tokens": len(all_tokens),
            }, f)
    
    def get_statistics(self) -> Dict:
        """Get tokenization statistics"""
        if not self.tokenized_commits:
            return {}
        
        token_counts = [tc.token_count for tc in self.tokenized_commits]
        
        return {
            "total_commits": len(self.tokenized_commits),
            "total_tokens": sum(token_counts),
            "avg_tokens_per_commit": sum(token_counts) / len(token_counts),
            "min_tokens": min(token_counts),
            "max_tokens": max(token_counts),
            "median_tokens": sorted(token_counts)[len(token_counts)//2],
            "vocab_size": self.tokenizer.vocab_size,
        }


def main():
    parser = argparse.ArgumentParser(description="Tokenize Git commit history")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file from git scraper")
    parser.add_argument(
        "--output",
        type=str,
        default="data/tokenized_commits.jsonl",
        help="Output file for tokenized commits"
    )
    parser.add_argument(
        "--sequences",
        type=str,
        default="data/token_sequences.json",
        help="Output file for token sequences (training format)"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=[s.value for s in TokenizationStrategy],
        default="semantic",
        help="Tokenization strategy"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        help="HuggingFace tokenizer model"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print statistics"
    )
    
    args = parser.parse_args()
    
    strategy = TokenizationStrategy(args.strategy)
    tokenizer = GitTokenizer(model_name=args.model, strategy=strategy)
    
    tokenizer.tokenize(args.input)
    tokenizer.save_tokens(args.output)
    tokenizer.save_token_sequences(args.sequences)
    
    if args.stats:
        stats = tokenizer.get_statistics()
        print("\n" + "="*60)
        print(f"Tokenization Statistics ({args.strategy})")
        print("="*60)
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"{key}: {value:.2f}")
            else:
                print(f"{key}: {value}")


if __name__ == "__main__":
    main()
