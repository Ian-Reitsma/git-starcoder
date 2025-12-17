#!/usr/bin/env python3
"""
Maximally Rich Git Tokenizer for The Block

Creates semantic, context-aware token sequences that encode:
- Commit relationships (parent-child, merge patterns)
- File ownership and change frequency
- Complexity patterns
- Author collaboration patterns
- Branch evolution
- Time-series patterns (when things happened)
- Architectural changes

Optimized for 512-token sequences on your hardware.

Output: Token sequences optimized for GPT-2-medium and beyond.
"""

import json
import sys
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, Counter
import math

try:
    from transformers import AutoTokenizer, PreTrainedTokenizer
    from tqdm import tqdm
except ImportError:
    print("Install: pip install transformers tqdm")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Special tokens for semantic structure
SPECIAL_TOKENS = {
    '<COMMIT>': 50258,
    '</COMMIT>': 50259,
    '<SUBJECT>': 50260,
    '<BODY>': 50261,
    '<MERGE>': 50262,
    '<FILES>': 50263,
    '<AUTHOR>': 50264,
    '<TIMESTAMP>': 50265,
    '<BRANCH>': 50266,
    '<COMPLEXITY>': 50267,
    '<RELATED>': 50268,
}


class RichGitTokenizer:
    """Semantically-aware Git tokenizer"""
    
    def __init__(
        self,
        tokenizer_name: str = "gpt2",
        tokenizer: Optional[PreTrainedTokenizer] = None,
        trust_remote_code: bool = False,
        verbose: bool = False,
        include_diff_text: bool = False,
    ):
        self.tokenizer_name = tokenizer_name
        self.verbose = verbose
        self.include_diff_text = include_diff_text
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=trust_remote_code,
        )
        self.vocab_size = self.tokenizer.vocab_size
        self.token_sequences: List[List[int]] = []
        self.metadata_map: Dict[int, Dict] = {}  # seq_idx -> metadata
    
    def _log(self, msg: str):
        if self.verbose:
            logger.info(msg)
    
    def _format_commit_semantic(self, commit: Dict) -> str:
        """
        Format commit with semantic markers for rich understanding.
        Each commit becomes a structured text block.
        """
        parts = []
        
        # Commit header with metadata
        parts.append(f"<COMMIT> {commit['abbrev_hash']}")
        
        # Timestamp and author
        parts.append(f"<AUTHOR> {commit['author_name']}")
        parts.append(f"<TIMESTAMP> {commit['commit_timestamp']}")
        
        # Branch information
        if commit.get('branches'):
            branches_str = ' '.join(commit['branches'])
            parts.append(f"<BRANCH> {branches_str}")
        
        # Complexity signal
        complexity = commit.get('complexity_score', 0)
        parts.append(f"<COMPLEXITY> {complexity:.2f}")
        
        # Merge signal
        if commit.get('is_merge'):
            parts.append(f"<MERGE> parents:{len(commit.get('parents', []))} files:{commit.get('files_changed', 0)}")
        
        # Subject
        parts.append(f"<SUBJECT> {commit['subject']}")
        
        # Body if exists
        if commit.get('body'):
            parts.append(f"<BODY> {commit['body'][:200]}")
        
        # File changes with crate info
        if commit.get('files_added') or commit.get('files_modified') or commit.get('files_deleted'):
            parts.append("<FILES>")
            
            # Group by crate
            files_by_crate = commit.get('files_by_crate', {})
            for crate, files in files_by_crate.items():
                parts.append(f"  {crate}: {' '.join(files[:3])}")
        
        # Statistics
        if commit.get('insertions') > 0 or commit.get('deletions') > 0:
            parts.append(f"  +{commit.get('insertions', 0)} -{commit.get('deletions', 0)}")
        
        # Related issues and commits
        if commit.get('related_issues'):
            parts.append(f"<RELATED> issues: {' '.join(commit['related_issues'])}")
        
        # Time since parent (pattern signal)
        if commit.get('time_since_parent', 0) > 0:
            hours_since = commit['time_since_parent'] / 3600
            parts.append(f"  hours_since_parent: {hours_since:.1f}")

        if self.include_diff_text:
            for diff in commit.get('diff_stats', []):
                diff_text = diff.get('diff_text') if isinstance(diff, dict) else None
                if diff_text:
                    path = diff.get('path') if isinstance(diff, dict) else 'unknown'
                    parts.append(f"<DIFF> {path}")
                    parts.append(diff_text.strip())
                    parts.append("</DIFF>")
        
        parts.append("</COMMIT>")
        
        return '\n'.join(parts)
    
    def _extract_primary_directory(self, commit: Dict) -> Optional[str]:
        """Extract a representative directory name from the commit"""
        for key in ('files_modified', 'files_added', 'files_deleted'):
            for path in commit.get(key, []):
                if not path:
                    continue
                if '/' in path:
                    dir_name = path.split('/', 1)[0]
                else:
                    dir_name = path
                if dir_name:
                    return dir_name
        return None
    
    def _create_hierarchical_context(self, commits: List[Dict]) -> Dict[str, List[str]]:
        """
        Build hierarchical relationships:
        - Branch evolution
        - Merge patterns
        - Author collaboration
        """
        context = {
            'branch_evolution': [],
            'merge_chains': [],
            'author_patterns': defaultdict(list),
            'file_hotspots': [],
        }
        
        # Track branch tip commits
        branch_tips = {}
        for commit in commits:
            for branch in commit.get('branches', []):
                branch_tips[branch] = commit['hash']
        
        # Identify merge patterns
        for i, commit in enumerate(commits):
            if commit.get('is_merge'):
                # Merge pattern: parent commits -> this commit
                parents = commit.get('parents', [])
                context['merge_chains'].append({
                    'commit': commit['abbrev_hash'],
                    'parent_count': len(parents),
                    'complexity': commit.get('complexity_score', 0)
                })
            
            # Author pattern
            author = commit['author_name']
            context['author_patterns'][author].append({
                'commit': commit['abbrev_hash'],
                'insertions': commit.get('insertions', 0)
            })
        
        # Find file hotspots (frequently changed)
        file_change_freq = Counter()
        for commit in commits:
            for f in commit.get('files_modified', []):
                file_change_freq[f] += 1
        
        context['file_hotspots'] = file_change_freq.most_common(20)
        
        return context
    
    def tokenize_commits(
        self,
        commits: List[Dict],
        sequence_length: int = 512,
        overlap: int = 128,
    ) -> List[List[int]]:
        """
        Tokenize commits into sequences with context windows.
        
        Creates overlapping sequences to maintain continuity.
        Each sequence has: previous context + current commit + next context
        """
        
        logger.info(f"Tokenizing {len(commits)} commits into {sequence_length}-token sequences")
        
        sequences = []
        token_buffer = []
        sequence_metadata = []
        
        # Sort commits by timestamp to maintain chronological order
        commits_sorted = sorted(commits, key=lambda c: c.get('commit_timestamp', 0))
        
        for idx, commit in enumerate(tqdm(commits_sorted, desc="Tokenizing")):
            # Format commit with rich semantics
            commit_text = self._format_commit_semantic(commit)
            
            # Tokenize
            tokens = self.tokenizer.encode(commit_text, add_special_tokens=False)
            
            # Add to buffer
            token_buffer.extend(tokens)
            
            # When buffer gets large, create a sequence
            while len(token_buffer) >= sequence_length:
                # Take sequence_length tokens
                sequence = token_buffer[:sequence_length]
                sequences.append(sequence)
                
                # Keep overlap for continuity
                token_buffer = token_buffer[sequence_length - overlap:]
                
                # Track metadata
                sequence_metadata.append({
                    'start_commit_idx': max(0, idx - len(token_buffer) // 100),
                    'end_commit_idx': idx,
                    'num_commits': idx + 1,
                    'total_tokens': len(sequence),
                    'sample_commit': commit['abbrev_hash'],
                    'author_name': commit.get('author_name'),
                    'commit_timestamp': commit.get('commit_timestamp'),
                    'primary_directory': self._extract_primary_directory(commit),
                })
        
        # Handle remaining tokens
        if token_buffer:
            if len(token_buffer) < sequence_length:
                padding = [self.tokenizer.eos_token_id] * (sequence_length - len(token_buffer))
                token_buffer.extend(padding)
            
            sequences.append(token_buffer)
            last_commit = commits_sorted[-1] if commits_sorted else {}
            sequence_metadata.append({
                'final_sequence': True,
                'total_tokens': len(token_buffer),
                'sample_commit': last_commit.get('abbrev_hash'),
                'author_name': last_commit.get('author_name'),
                'commit_timestamp': last_commit.get('commit_timestamp'),
                'primary_directory': self._extract_primary_directory(last_commit) if last_commit else None,
            })
        
        self.token_sequences = sequences
        self.metadata_map = {i: m for i, m in enumerate(sequence_metadata)}
        
        logger.info(f"Created {len(sequences)} token sequences")
        logger.info(f"Total tokens: {sum(len(s) for s in sequences)}")
        
        if len(sequences) > 0:
            logger.info(
                f"Avg tokens per sequence: "
                f"{sum(len(s) for s in sequences) / len(sequences):.0f}"
            )
        else:
            logger.warning("No sequences generated!")

        return sequences
    
    def save_sequences_json(self, output_path: str):
        """Save token sequences as JSON"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'token_sequences': self.token_sequences,
            'vocab_size': self.vocab_size,
            'num_sequences': len(self.token_sequences),
            'total_tokens': sum(len(s) for s in self.token_sequences),
            'sequence_length': len(self.token_sequences[0]) if self.token_sequences else 0,
            'metadata': self.metadata_map
        }
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved sequences to {output_path}")
    
    def save_sequences_jsonl(self, output_path: str):
        """Save token sequences as JSONL (one per line)"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            for idx, sequence in enumerate(self.token_sequences):
                data = {
                    'sequence_id': idx,
                    'tokens': sequence,
                    'length': len(sequence),
                    'metadata': self.metadata_map.get(idx, {})
                }
                f.write(json.dumps(data) + '\n')
        
        logger.info(f"Saved {len(self.token_sequences)} sequences to {output_path}")
    
    def get_statistics(self) -> Dict:
        """Get tokenization statistics"""
        if not self.token_sequences:
            return {}
        
        lengths = [len(s) for s in self.token_sequences]
        
        return {
            'num_sequences': len(self.token_sequences),
            'total_tokens': sum(lengths),
            'avg_sequence_length': sum(lengths) / len(lengths),
            'min_sequence_length': min(lengths),
            'max_sequence_length': max(lengths),
            'vocab_size': self.vocab_size,
        }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Rich Git tokenizer")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL from git_scraper_rich")
    parser.add_argument("--sequences", type=str, default="data/token_sequences_rich.json", 
                       help="Output token sequences JSON")
    parser.add_argument("--sequences-jsonl", type=str, help="Also save as JSONL")
    parser.add_argument("--sequence-length", type=int, default=512, help="Token sequence length")
    parser.add_argument("--overlap", type=int, default=128, help="Sequence overlap")
    parser.add_argument("--model", type=str, default="gpt2", help="Tokenizer/model name to use (e.g. bigcode/starcoder2-3b)")
    parser.add_argument("--trust-remote-code", action="store_true", help="Allow remote code when loading tokenizer")
    parser.add_argument("--stats", action="store_true", help="Print statistics")
    parser.add_argument("--include-diff-text", action="store_true", help="Include raw diff text in sequences")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Load commits
    logger.info(f"Loading commits from {args.input}")
    commits = []
    with open(args.input, 'r') as f:
        for line in f:
            commits.append(json.loads(line))
    
    logger.info(f"Loaded {len(commits)} commits")
    
    # Tokenize
    tokenizer = RichGitTokenizer(
        tokenizer_name=args.model,
        trust_remote_code=args.trust_remote_code,
        verbose=args.verbose,
        include_diff_text=args.include_diff_text,
    )
    tokenizer.tokenize_commits(
        commits,
        sequence_length=args.sequence_length,
        overlap=args.overlap
    )
    
    # Save
    tokenizer.save_sequences_json(args.sequences)
    if args.sequences_jsonl:
        tokenizer.save_sequences_jsonl(args.sequences_jsonl)
    
    # Stats
    if args.stats:
        stats = tokenizer.get_statistics()
        print("\n" + "="*60)
        print("Tokenization Statistics")
        print("="*60)
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"{key}: {value:.2f}")
            else:
                print(f"{key}: {value}")


if __name__ == "__main__":
    main()
