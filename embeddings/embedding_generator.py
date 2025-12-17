#!/usr/bin/env python3
"""
Embedding generator for rich Git commits.

Reads the JSONL output from the scraper, builds contextual text per commit,
and produces Qdrant-compatible vectors using a SentenceTransformer model.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Any

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None  # Guard so we can skip embedding generation gracefully


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate embeddings for commit JSONL output."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input JSONL file (e.g., data/git_history_rich.jsonl).",
    )
    parser.add_argument(
        "--qdrant-output",
        required=True,
        help="Path to output Qdrant points JSON file.",
    )
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-mpnet-base-v2",
        help="SentenceTransformer model name.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for embedding generation.",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="If set, prints statistics.",
    )
    parser.add_argument(
        "--mode",
        choices=["commit", "snapshot"],
        default="commit",
        help="Input format: commit history or file snapshots.",
    )
    return parser.parse_args()


def load_commits(path: Path) -> List[Dict[str, Any]]:
    commits: List[Dict[str, Any]] = []
    if not path.exists():
        logger.warning("Input file %s does not exist; returning empty list.", path)
        return commits

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            commits.append(json.loads(line))
    return commits


def format_commit_text(commit: Dict[str, Any]) -> str:
    """Build a descriptive text block for embedding."""
    subject = commit.get("subject") or ""
    body = commit.get("body") or ""
    author = commit.get("author_name") or commit.get("committer_name") or "unknown"
    files = commit.get("files_modified", []) or commit.get("files_added", [])
    file_summary = " ".join(files[:5])
    complexity = commit.get("complexity_score", 0.0)
    related = " ".join(commit.get("related_issues", [])[:5])

    parts = [
        f"Commit {commit.get('abbrev_hash', commit.get('hash', 'unknown'))}",
        f"Author: {author}",
        f"Subject: {subject}",
        body,
    ]
    if file_summary:
        parts.append(f"Files: {file_summary}")
    parts.append(f"Complexity: {complexity:.2f}")
    if related:
        parts.append(f"Issues: {related}")
    return "\n".join(part for part in parts if part)


def format_snapshot_text(chunk: Dict[str, Any]) -> str:
    path = chunk.get("file_path")
    branch = chunk.get("branch")
    rng = f"{chunk.get('start_line')}:{chunk.get('end_line')}"
    body = chunk.get("text", "")
    return "\n".join(
        [
            f"File {path}",
            f"Branch: {branch}",
            f"Lines: {rng}",
            body,
        ]
    )


def build_payload(record: Dict[str, Any], mode: str) -> Dict[str, Any]:
    if mode == "snapshot":
        return {
            "type": "file_chunk",
            "file_path": record.get("file_path"),
            "branch": record.get("branch"),
            "commit_hash": record.get("commit_hash"),
            "start_line": record.get("start_line"),
            "end_line": record.get("end_line"),
        }
    return {
        "type": "commit",
        "hash": record.get("hash"),
        "abbrev_hash": record.get("abbrev_hash"),
        "author": record.get("author_name"),
        "timestamp": record.get("commit_timestamp"),
        "subject": record.get("subject"),
        "complexity": record.get("complexity_score"),
        "files_changed": record.get("files_changed"),
    }


def generate_embeddings(
    texts: List[str],
    model_name: str,
    batch_size: int,
) -> List[List[float]]:
    if SentenceTransformer is None:
        raise RuntimeError(
            "sentence-transformers is required for embedding generation. "
            "Install with `pip install sentence-transformers`."
        )
    model = SentenceTransformer(model_name)
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return [vec.tolist() for vec in vectors]


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    in_path = Path(args.input)
    out_path = Path(args.qdrant_output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    commits = load_commits(in_path)
    if not commits:
        logger.warning("No commits found; writing empty embedding list.")
        with out_path.open("w", encoding="utf-8") as f:
            json.dump([], f)
        return

    if SentenceTransformer is None:
        logger.warning(
            "sentence-transformers not installed; skipping embeddings and writing empty list."
        )
        with out_path.open("w", encoding="utf-8") as f:
            json.dump([], f)
        return

    if args.mode == "snapshot":
        texts = [format_snapshot_text(chunk) for chunk in commits]
    else:
        texts = [format_commit_text(commit) for commit in commits]
    payloads = [build_payload(record, args.mode) for record in commits]
    ids = [
        record.get("hash")
        or record.get("abbrev_hash")
        or record.get("chunk_id")
        or idx
        for idx, record in enumerate(commits)
    ]

    vectors = generate_embeddings(texts, args.model, args.batch_size)
    dimension = len(vectors[0]) if vectors else 0

    points = [
        {
            "id": commit_id,
            "vector": vector,
            "payload": payload,
        }
        for commit_id, vector, payload in zip(ids, vectors, payloads)
    ]

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(points, f)

    if args.stats:
        logger.info("Embedded %d commits; dimension=%d; output=%s", len(points), dimension, out_path)


if __name__ == "__main__":
    main()
