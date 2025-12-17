#!/usr/bin/env python3
"""
File Snapshot Tokenizer

Converts the JSONL output of file_snapshot_scraper into fixed-length token sequences
that preserve file metadata and literal code chunks.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

try:
    from transformers import AutoTokenizer, PreTrainedTokenizer
    from tqdm import tqdm
except ImportError:
    print("Install: pip install transformers tqdm")
    raise

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class FileSnapshotTokenizer:
    def __init__(
        self,
        tokenizer_name: str,
        trust_remote_code: bool,
        sequence_length: int,
        overlap: int,
        verbose: bool = False,
    ):
        self.tokenizer_name = tokenizer_name
        self.verbose = verbose
        self.sequence_length = sequence_length
        self.overlap = overlap
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.sequences: List[List[int]] = []
        self.metadata: Dict[int, Dict] = {}

    def _format_chunk(self, chunk: Dict) -> str:
        parts = [
            f"<FILE> {chunk.get('file_path')}",
            f"<BRANCH> {chunk.get('branch')}",
            f"<COMMIT> {chunk.get('commit_hash')}",
            f"<RANGE> {chunk.get('start_line')}:{chunk.get('end_line')}",
        ]
        text = chunk.get("text", "")
        parts.append("<CODE>")
        parts.append(text)
        parts.append("</CODE>")
        return "\n".join(parts)

    def tokenize(self, chunks: List[Dict]) -> None:
        buffer: List[int] = []
        seq_idx = 0
        for chunk in tqdm(chunks, desc="Tokenizing snapshots"):
            text = self._format_chunk(chunk)
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            buffer.extend(tokens)
            while len(buffer) >= self.sequence_length:
                seq = buffer[: self.sequence_length]
                self.sequences.append(seq)
                self.metadata[len(self.sequences) - 1] = {
                    "source": "snapshot",
                    "file_path": chunk.get("file_path"),
                    "branch": chunk.get("branch"),
                    "start_line": chunk.get("start_line"),
                    "end_line": chunk.get("end_line"),
                }
                buffer = buffer[self.sequence_length - self.overlap :]
                seq_idx += 1
        if buffer:
            padding = [self.tokenizer.eos_token_id] * (self.sequence_length - len(buffer))
            buffer.extend(padding)
            self.sequences.append(buffer)
            self.metadata[len(self.sequences) - 1] = {
                "source": "snapshot",
                "file_path": chunks[-1].get("file_path") if chunks else None,
                "branch": chunks[-1].get("branch") if chunks else None,
                "start_line": chunks[-1].get("start_line") if chunks else None,
                "end_line": chunks[-1].get("end_line") if chunks else None,
                "final_sequence": True,
            }

    def save(self, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "token_sequences": self.sequences,
            "vocab_size": len(self.tokenizer),
            "num_sequences": len(self.sequences),
            "sequence_length": self.sequence_length,
            "metadata": self.metadata,
        }
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info("Saved %d snapshot sequences -> %s", len(self.sequences), output_path)


def load_chunks(path: Path) -> List[Dict]:
    chunks = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunks.append(json.loads(line))
    logger.info("Loaded %d snapshot chunks from %s", len(chunks), path)
    return chunks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tokenize file snapshot chunks.")
    parser.add_argument("--input", required=True, help="Snapshot JSONL from file_snapshot_scraper.")
    parser.add_argument("--output", default="data/token_sequences_snapshot.json", help="Output JSON file.")
    parser.add_argument("--model", default="gpt2", help="Tokenizer/model name.")
    parser.add_argument("--sequence-length", type=int, default=512, help="Sequence length.")
    parser.add_argument("--overlap", type=int, default=128, help="Token overlap between sequences.")
    parser.add_argument("--trust-remote-code", action="store_true", help="Allow remote code when loading tokenizer.")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging.")
    parser.add_argument("--stats", action="store_true", help="Print summary statistics.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = FileSnapshotTokenizer(
        tokenizer_name=args.model,
        trust_remote_code=args.trust_remote_code,
        sequence_length=args.sequence_length,
        overlap=args.overlap,
        verbose=args.verbose,
    )
    chunks = load_chunks(Path(args.input))
    tokenizer.tokenize(chunks)
    tokenizer.save(Path(args.output))

    if args.stats:
        total_tokens = sum(len(seq) for seq in tokenizer.sequences)
        logger.info(
            "Snapshot tokenization stats: %d sequences | %d tokens (sequence length %d, overlap %d)",
            len(tokenizer.sequences),
            total_tokens,
            tokenizer.sequence_length,
            tokenizer.overlap,
        )


if __name__ == "__main__":
    main()
