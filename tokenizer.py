#!/usr/bin/env python3
"""
Custom Tokenizer: Convert chunks to tokens preserving semantic structure.

Key insight: Traditional BPE tokenizers lose code structure.
This tokenizer uses:
- Special tokens for structural boundaries
- Rust-specific vocabulary (macros, keywords, attributes)
- Position encoding for temporal context

Output: PyTorch tensors ready for training.
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from collections import Counter, defaultdict
import logging
import numpy as np
from dataclasses import dataclass

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class Token:
    """Represents a token."""
    text: str
    token_id: int
    token_type: str  # 'special', 'keyword', 'identifier', 'literal', 'operator', etc
    frequency: int = 0


class VocabularyBuilder:
    """Build vocabulary from chunks."""
    
    # Special tokens
    SPECIAL_TOKENS = [
        "<PAD>",
        "<UNK>",
        "<COMMIT_START>",
        "<COMMIT_END>",
        "<FILE_START>",
        "<FILE_END>",
        "<CHANGE_START>",
        "<CHANGE_END>",
        "<CODE_START>",
        "<CODE_END>",
        "<TEST_START>",
        "<TEST_END>",
        "<NEWLINE>",
        "<INDENT>",
        "<DEDENT>",
    ]
    
    # Branch/file context tokens (template)
    # Will be filled dynamically: <BRANCH:main>, <FILE:src/lib.rs>, etc
    DYNAMIC_SPECIAL_PATTERNS = [
        "<BRANCH:{branch}>",
        "<FILE:{file}>",
        "<AUTHOR:{author}>",
        "<MODULE:{module}>",
        "<CHANGE:{type}>",
        "<TIMESTAMP:{date}>",
    ]
    
    # Rust keywords and common patterns
    RUST_KEYWORDS = [
        "fn", "struct", "impl", "trait", "enum", "mod", "pub", "async",
        "await", "use", "let", "const", "static", "mut", "unsafe", "loop",
        "while", "for", "if", "else", "match", "return", "break", "continue",
        "type", "where", "as", "move", "dyn", "ref", "extern", "crate",
        "self", "super", "in", "Box", "Vec", "String", "Result", "Option",
    ]
    
    # Rust macros
    RUST_MACROS = [
        "println!", "eprintln!", "format!", "vec!", "assert!",
        "panic!", "unwrap!", "expect!", "try!", "macro_rules!",
        "derive", "cfg", "test", "tokio", "serde", "async_trait",
    ]
    
    def __init__(self, vocab_size: int = 50257):
        self.vocab_size = vocab_size
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.token_frequency: Counter = Counter()
        self.reserved_space = len(self.SPECIAL_TOKENS) + 5000  # Space for dynamic tokens
        self._initialize_special_tokens()
    
    def _initialize_special_tokens(self) -> None:
        """Initialize special tokens."""
        idx = 0
        
        # Add special tokens
        for token in self.SPECIAL_TOKENS:
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
            idx += 1
        
        # Reserve space for dynamic special tokens
        self.dynamic_start_idx = idx
        idx += 5000
        
        # Add Rust keywords
        for keyword in self.RUST_KEYWORDS:
            token = f"<KW:{keyword}>"
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
            idx += 1
        
        # Add Rust macros
        for macro in self.RUST_MACROS:
            token = f"<MACRO:{macro}>"
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
            idx += 1
        
        self.next_id = idx
    
    def add_token(self, token: str) -> int:
        """Add token to vocabulary, return its ID."""
        if token in self.token_to_id:
            return self.token_to_id[token]
        
        if self.next_id >= self.vocab_size:
            # Vocabulary full, return UNK
            return self.token_to_id["<UNK>"]
        
        token_id = self.next_id
        self.token_to_id[token] = token_id
        self.id_to_token[token_id] = token
        self.next_id += 1
        
        return token_id
    
    def get_id(self, token: str, add_if_missing: bool = True) -> int:
        """Get token ID, optionally adding if missing."""
        if token in self.token_to_id:
            return self.token_to_id[token]
        
        if add_if_missing:
            return self.add_token(token)
        else:
            return self.token_to_id["<UNK>"]
    
    def build_from_chunks(self, chunks_file: str) -> None:
        """Build vocabulary from chunks file."""
        logger.info("Building vocabulary from chunks...")
        
        # Scan chunks and count token frequencies
        with open(chunks_file, "r") as f:
            for line_num, line in enumerate(f):
                if (line_num + 1) % 1000 == 0:
                    logger.info(f"Processing chunk {line_num + 1}...")
                
                chunk = json.loads(line)
                
                # Extract tokens from code
                code = chunk.get("new_code", "")
                message = chunk.get("commit_metadata", {}).get("message", "")
                
                # Tokenize code
                tokens = self._tokenize_code(code)
                for token in tokens:
                    self.token_frequency[token] += 1
                
                # Tokenize message
                tokens = self._tokenize_text(message)
                for token in tokens:
                    self.token_frequency[token] += 1
        
        # Sort by frequency and add to vocabulary
        sorted_tokens = sorted(
            self.token_frequency.items(),
            key=lambda x: -x[1]
        )
        
        logger.info(f"Found {len(sorted_tokens)} unique tokens")
        
        for token, freq in sorted_tokens:
            if self.next_id >= self.vocab_size:
                logger.info(f"Vocabulary full at {self.next_id} tokens")
                break
            
            if token not in self.token_to_id:
                self.add_token(token)
        
        logger.info(f"Built vocabulary with {self.next_id} tokens")
    
    def _tokenize_code(self, code: str) -> List[str]:
        """Tokenize code into meaningful units."""
        tokens = []
        
        # Simple tokenization: split on whitespace and punctuation
        # Keep operators and brackets together
        import re
        pattern = r'[a-zA-Z_][a-zA-Z0-9_]*|[0-9]+|[+\-*/%=<>!&|^~?:|;,.(\[\]{}]'
        
        for match in re.finditer(pattern, code):
            token = match.group()
            tokens.append(token)
        
        return tokens
    
    def _tokenize_text(self, text: str) -> List[str]:
        """Tokenize natural language text."""
        import re
        tokens = []
        
        # Split on whitespace, keep punctuation
        for word in text.split():
            # Keep words together but mark punctuation
            if word.endswith('.') or word.endswith(','):
                tokens.append(word[:-1])
                tokens.append(word[-1])
            else:
                tokens.append(word)
        
        return tokens
    
    def save(self, output_file: str) -> None:
        """Save vocabulary to file."""
        logger.info(f"Saving vocabulary to {output_file}...")
        
        vocab_data = {
            "token_to_id": self.token_to_id,
            "id_to_token": {int(k): v for k, v in self.id_to_token.items()},
            "vocab_size": self.next_id,
            "special_tokens": self.SPECIAL_TOKENS,
            "rust_keywords": self.RUST_KEYWORDS,
        }
        
        with open(output_file, "w") as f:
            json.dump(vocab_data, f, indent=2)
        
        logger.info(f"Saved vocabulary with {self.next_id} tokens")


class CodeTokenizer:
    """Tokenize code chunks with structural awareness."""
    
    def __init__(self, vocab: VocabularyBuilder):
        self.vocab = vocab
        self.token_sequences = []
    
    def tokenize_chunk(self, chunk: Dict[str, Any], chunk_index: int) -> List[int]:
        """Tokenize a single chunk, preserving structure."""
        tokens = []
        
        # Commit boundary
        tokens.append(self.vocab.token_to_id["<COMMIT_START>"])
        
        # File context
        file_path = chunk.get("file_path", "unknown")
        tokens.append(self.vocab.get_id(f"<FILE:{file_path}>"))
        
        # Change type
        change_type = chunk.get("change_type", "unknown")
        tokens.append(self.vocab.get_id(f"<CHANGE:{change_type}>"))
        
        # Author context
        author = chunk.get("commit_metadata", {}).get("author_email", "unknown")
        tokens.append(self.vocab.get_id(f"<AUTHOR:{author}>"))
        
        # Timestamp context (year-month for bucketing)
        timestamp = chunk.get("commit_metadata", {}).get("timestamp", "")
        if timestamp:
            date_bucket = timestamp[:7]  # YYYY-MM
            tokens.append(self.vocab.get_id(f"<TIMESTAMP:{date_bucket}>"))
        
        # Module path if available
        module = chunk.get("module_path")
        if module:
            tokens.append(self.vocab.get_id(f"<MODULE:{module}>"))
        
        # Commit message
        message = chunk.get("commit_metadata", {}).get("message", "")
        tokens.append(self.vocab.token_to_id["<CODE_START>"])
        message_tokens = self._tokenize_text(message)
        for token in message_tokens[:32]:  # Limit message length
            tokens.append(self.vocab.get_id(token, add_if_missing=False))
        tokens.append(self.vocab.token_to_id["<CODE_END>"])
        
        # Actual code diff
        old_code = chunk.get("old_code", "")
        new_code = chunk.get("new_code", "")
        
        tokens.append(self.vocab.token_to_id["<CODE_START>"])
        
        # Tokenize new code
        code_tokens = self._tokenize_code(new_code)
        for token in code_tokens[:1024]:  # Limit code length
            token_id = self.vocab.get_id(token, add_if_missing=False)
            tokens.append(token_id)
        
        tokens.append(self.vocab.token_to_id["<CODE_END>"])
        tokens.append(self.vocab.token_to_id["<COMMIT_END>"])
        
        return tokens
    
    def _tokenize_code(self, code: str) -> List[str]:
        """Tokenize code."""
        import re
        pattern = r'[a-zA-Z_][a-zA-Z0-9_]*|[0-9]+|[+\-*/%=<>!&|^~?:|;,.(\[\]{}]'
        tokens = []
        for match in re.finditer(pattern, code):
            tokens.append(match.group())
        return tokens
    
    def _tokenize_text(self, text: str) -> List[str]:
        """Tokenize text."""
        tokens = []
        for word in text.split():
            if word.endswith('.') or word.endswith(','):
                tokens.append(word[:-1])
                tokens.append(word[-1])
            else:
                tokens.append(word)
        return tokens
    
    def tokenize_file(self, chunks_file: str) -> Tuple[List[List[int]], List[Dict]]:
        """Tokenize all chunks in file."""
        logger.info(f"Tokenizing {chunks_file}...")
        
        all_tokens = []
        metadata = []
        
        with open(chunks_file, "r") as f:
            for idx, line in enumerate(f):
                if (idx + 1) % 1000 == 0:
                    logger.info(f"Tokenizing chunk {idx + 1}...")
                
                chunk = json.loads(line)
                tokens = self.tokenize_chunk(chunk, idx)
                all_tokens.append(tokens)
                
                # Store metadata for later reference
                metadata.append({
                    "chunk_id": chunk.get("chunk_id"),
                    "commit_hash": chunk.get("commit_hash"),
                    "file_path": chunk.get("file_path"),
                    "token_count": len(tokens)
                })
        
        logger.info(f"Tokenized {len(all_tokens)} chunks")
        return all_tokens, metadata


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Tokenize semantic chunks"
    )
    parser.add_argument(
        "--chunks",
        required=True,
        help="Input chunks JSONL file (from semantic_chunker.py)"
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=50257,
        help="Vocabulary size"
    )
    parser.add_argument(
        "--output-vocab",
        default="vocab.json",
        help="Output vocabulary file"
    )
    parser.add_argument(
        "--output-tokens",
        default="tokens.pt",
        help="Output tokens file (PyTorch format)"
    )
    
    args = parser.parse_args()
    
    try:
        # Build vocabulary
        vocab = VocabularyBuilder(vocab_size=args.vocab_size)
        vocab.build_from_chunks(args.chunks)
        vocab.save(args.output_vocab)
        
        # Tokenize all chunks
        tokenizer = CodeTokenizer(vocab)
        all_tokens, metadata = tokenizer.tokenize_file(args.chunks)
        
        # Save tokens as PyTorch tensor
        try:
            import torch
            
            # Convert to PyTorch tensors
            token_data = {
                "tokens": all_tokens,
                "metadata": metadata
            }
            
            torch.save(token_data, args.output_tokens)
            logger.info(f"Saved tokens to {args.output_tokens}")
        except ImportError:
            logger.warning("PyTorch not available, saving as pickle instead")
            with open(args.output_tokens.replace('.pt', '.pkl'), 'wb') as f:
                pickle.dump(token_data, f)
        
        # Print summary
        print(f"\n" + "="*60)
        print("TOKENIZATION SUMMARY")
        print("="*60)
        print(f"Vocabulary size: {vocab.next_id}")
        print(f"Chunks tokenized: {len(all_tokens)}")
        print(f"Avg tokens per chunk: {sum(len(t) for t in all_tokens) / len(all_tokens):.0f}")
        print(f"\nOutput files:")
        print(f"  Vocabulary: {args.output_vocab}")
        print(f"  Tokens: {args.output_tokens}")
        print("="*60 + "\n")
        
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        exit(1)


if __name__ == "__main__":
    main()
