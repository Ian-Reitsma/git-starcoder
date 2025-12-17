#!/usr/bin/env python3
"""
File Snapshot Scraper

Walks a working tree (e.g., the-block) and emits overlapping text chunks for every
source, config, and documentation file. This gives the pipeline literal access to the
current workspace contents, independent of commit history.
"""

import argparse
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

logger = logging.getLogger(__name__)

DEFAULT_INCLUDE_EXTENSIONS = {
    ".rs", ".toml", ".md", ".txt", ".yaml", ".yml", ".json",
    ".py", ".sh", ".bash", ".ps1", ".sql", ".proto", ".ts",
    ".tsx", ".js", ".jsx", ".html", ".css",
}

DEFAULT_EXCLUDE_DIRS = {
    ".git", "target", "node_modules", "venv", ".venv", "__pycache__",
    "dist", "build", ".idea", ".vscode", ".pytest_cache",
}

DEFAULT_MAX_FILE_BYTES = 2_000_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Snapshot the current repo files into chunks.")
    parser.add_argument("--repo", required=True, help="Path to working tree (e.g., the-block).")
    parser.add_argument(
        "--output",
        default="data/file_snapshots.jsonl",
        help="Output JSONL path (default: data/file_snapshots.jsonl).",
    )
    parser.add_argument(
        "--lines-per-chunk",
        type=int,
        default=200,
        help="Number of lines per chunk (default: 200).",
    )
    parser.add_argument(
        "--overlap-lines",
        type=int,
        default=40,
        help="Line overlap between chunks (default: 40).",
    )
    parser.add_argument(
        "--max-file-bytes",
        type=int,
        default=DEFAULT_MAX_FILE_BYTES,
        help="Skip files larger than this many bytes (default: 2MB).",
    )
    parser.add_argument(
        "--include-ext",
        nargs="*",
        default=sorted(DEFAULT_INCLUDE_EXTENSIONS),
        help="File extensions to include (default: common source/config/docs).",
    )
    parser.add_argument(
        "--exclude-dirs",
        nargs="*",
        default=sorted(DEFAULT_EXCLUDE_DIRS),
        help="Directory names to skip entirely.",
    )
    parser.add_argument(
        "--detect-branch",
        action="store_true",
        help="Detect the current branch via git instead of relying on HEAD only.",
    )
    return parser.parse_args()


def is_binary(path: Path) -> bool:
    """Simple binary file heuristic."""
    try:
        chunk = path.read_bytes()[:1024]
    except OSError:
        return True
    return b"\0" in chunk


def chunk_lines(text: str, lines_per_chunk: int, overlap: int) -> Iterator[Tuple[int, int, str]]:
    """Split text into overlapping line ranges."""
    lines = text.splitlines()
    total = len(lines)
    if total == 0:
        return
    start = 0
    stride = max(1, lines_per_chunk - overlap)
    while start < total:
        end = min(total, start + lines_per_chunk)
        chunk_text = "\n".join(lines[start:end])
        yield start + 1, end, chunk_text
        if end == total:
            break
        start += stride


def get_git_metadata(repo: Path, detect_branch: bool) -> Dict[str, str]:
    """Return HEAD commit and branch information for the repo."""
    def run_git(args: Sequence[str]) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=str(repo),
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()

    head = run_git(["rev-parse", "HEAD"])
    branch = "detached"
    if detect_branch:
        try:
            branch = run_git(["rev-parse", "--abbrev-ref", "HEAD"])
        except subprocess.CalledProcessError:
            branch = "detached"
    return {"commit_hash": head, "branch": branch}


def iter_files(base: Path, include_ext: Iterable[str], exclude_dirs: Iterable[str], max_bytes: int) -> Iterator[Path]:
    include = {ext.lower() for ext in include_ext}
    exclude = set(exclude_dirs)
    for root, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if d not in exclude]
        for name in files:
            path = Path(root) / name
            try:
                if path.stat().st_size > max_bytes:
                    continue
            except OSError:
                continue
            if path.suffix.lower() not in include:
                continue
            if is_binary(path):
                continue
            yield path


def snapshot_repo(args: argparse.Namespace) -> int:
    repo = Path(args.repo).resolve()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    meta = get_git_metadata(repo, args.detect_branch)
    total_chunks = 0

    with output.open("w", encoding="utf-8") as f:
        for file_path in iter_files(repo, args.include_ext, args.exclude_dirs, args.max_file_bytes):
            rel_path = file_path.relative_to(repo).as_posix()
            try:
                text = file_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                # Fallback: try latin-1; if still fails, skip
                try:
                    text = file_path.read_text(encoding="latin-1")
                except UnicodeDecodeError:
                    logger.warning("Skipping non-text file: %s", rel_path)
                    continue

            for idx, (start_line, end_line, chunk_text) in enumerate(
                chunk_lines(text, args.lines_per_chunk, args.overlap_lines)
            ):
                record = {
                    "chunk_id": f"{rel_path}::chunk-{idx}",
                    "file_path": rel_path,
                    "language": file_path.suffix.lstrip("."),
                    "branch": meta["branch"],
                    "commit_hash": meta["commit_hash"],
                    "start_line": start_line,
                    "end_line": end_line,
                    "total_lines": text.count("\n") + 1,
                    "text": chunk_text,
                }
                f.write(json.dumps(record) + "\n")
                total_chunks += 1

    logger.info("Snapshot complete: %d chunks -> %s", total_chunks, output)
    return total_chunks


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    args = parse_args()
    snapshot_repo(args)


if __name__ == "__main__":
    main()
