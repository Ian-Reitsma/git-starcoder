#!/usr/bin/env python3
"""
Git Scraper: Extract all commits, branches, merges, and diffs from repo.

This is designed to preserve:
- Branch lineage (understand parallel development)
- Merge decisions (learn architectural coordination)
- Author/timestamp metadata (learn pace + team patterns)
- Complete diffs (learn exact changes)

Output: JSON-structured commit graph ready for semantic chunking.
"""

import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging
from dataclasses import dataclass, asdict
import re
from collections import defaultdict

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass  # FIXED: added missing colon
class CommitMeta:
    """Core commit information."""
    hash: str
    abbrev_hash: str
    author: str
    author_email: str
    timestamp: str  # ISO 8601
    timestamp_unix: int
    message: str
    message_body: str
    parents: List[str]  # parent commit hashes
    is_merge: bool
    files_changed: int
    insertions: int
    deletions: int


@dataclass
class FileDiff:
    """Per-file diff information."""
    filename: str
    change_type: str  # 'A' (add), 'M' (modify), 'D' (delete), 'R' (rename), 'T' (type change)
    old_file: Optional[str]  # for renames
    new_file: Optional[str]  # for renames
    additions: int
    deletions: int
    patch: str  # actual diff content
    is_binary: bool
    is_merge_conflict_resolution: bool = False


@dataclass
class BranchInfo:
    """Branch information at point of commit."""
    branch_name: str
    is_current: bool
    is_remote: bool
    upstream: Optional[str] = None


@dataclass  
class CommitRecord:
    """Complete commit record with all context."""
    meta: CommitMeta  # FIXED: added type annotation colon, changed CommitMetadata to CommitMeta
    branches: List[BranchInfo]
    diffs: List[FileDiff]
    tags: List[str]
    commit_number: int  # chronological order
    context: Dict[str, Any]  # additional context


class GitScraper:
    """Scrape git repository comprehensively."""
    
    def __init__(self, repo_path: str, output_dir: str = "outputs"):
        self.repo_path = Path(repo_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        if not (self.repo_path / ".git").exists():
            raise ValueError(f"Not a git repository: {repo_path}")
        
        logger.info(f"Initialized GitScraper for {repo_path}")
    
    def _run_git(self, *args: str) -> str:
        """Run git command, return stdout."""
        try:
            result = subprocess.run(
                ["git", "-C", str(self.repo_path)] + list(args),
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode != 0:
                logger.warning(f"Git command failed: {' '.join(args)}\nError: {result.stderr}")
                return ""
            return result.stdout
        except subprocess.TimeoutExpired:
            logger.error(f"Git command timed out: {' '.join(args)}")
            return ""
    
    def get_all_commits_chronological(self) -> List[str]:
        """Get all commits in chronological order (oldest first)."""
        logger.info("Fetching all commits chronologically...")
        output = self._run_git("rev-list", "--all", "--date-order")
        commits = [line.strip() for line in output.strip().split("\n") if line.strip()]
        logger.info(f"Found {len(commits)} commits")
        return commits
    
    def get_commit_metadata(self, commit_hash: str) -> CommitMeta:  # FIXED: added missing colon after return type
        """Extract metadata for a single commit."""
        # Use %x00 as delimiter for reliable parsing
        fmt = (
            "%H%x00"  # full hash
            "%h%x00"  # abbrev hash
            "%an%x00"  # author name
            "%ae%x00"  # author email
            "%aI%x00"  # author date ISO 8601
            "%at%x00"  # author date unix timestamp
            "%s%x00"  # subject (message first line)
            "%b%x00"  # body
            "%P%x00"  # parent hashes (space-separated)
            "%f"      # raw subject
        )
        
        output = self._run_git("show", "-s", f"--format={fmt}", commit_hash)
        parts = output.strip().split("\x00")
        
        if len(parts) < 10:
            logger.warning(f"Could not parse commit {commit_hash}")
            raise ValueError(f"Failed to parse commit {commit_hash}")
        
        parents = parts[8].split() if parts[8].strip() else []
        is_merge = len(parents) > 1
        
        # Get file stats
        stats_output = self._run_git("show", "--stat=100", "--format=", commit_hash)
        files_changed, insertions, deletions = self._parse_stat(stats_output)
        
        return CommitMeta(
            hash=parts[0],
            abbrev_hash=parts[1],
            author=parts[2],
            author_email=parts[3],
            timestamp=parts[4],
            timestamp_unix=int(parts[5]) if parts[5] else 0,
            message=parts[6],
            message_body=parts[7],
            parents=parents,
            is_merge=is_merge,
            files_changed=files_changed,
            insertions=insertions,
            deletions=deletions
        )
    
    def _parse_stat(self, stat_output: str) -> tuple:
        """Parse git stat output to extract file count and insertions/deletions."""
        lines = stat_output.strip().split("\n")
        files_changed = 0
        insertions = 0
        deletions = 0
        
        for line in lines:
            if " changed" in line:
                # Last line like: "5 files changed, 250 insertions(+), 100 deletions(-)"
                parts = line.split()
                try:
                    files_changed = int(parts[0])
                    # Find insertions
                    for i, part in enumerate(parts):
                        if part.endswith("insertion") or part.endswith("insertions(+)"):
                            insertions = int(parts[i-1]) if i > 0 else 0
                        elif part.endswith("deletion") or part.endswith("deletions(-)"):
                            deletions = int(parts[i-1]) if i > 0 else 0
                except ValueError:
                    pass
                break
        
        return files_changed, insertions, deletions
    
    def get_commit_diffs(self, commit_hash: str) -> List[FileDiff]:
        """Extract all file diffs for a commit."""
        diffs = []
        
        # Get list of changed files with their change types
        output = self._run_git("diff-tree", "--no-commit-id", "-r", "-M", "-C", "--name-status", commit_hash)
        
        file_changes = {}
        for line in output.strip().split("\n"):
            if not line.strip():
                continue
            parts = line.split("\t")
            change_type = parts[0][0]  # A, M, D, R, T
            
            if change_type in ['R', 'C']:  # Rename or Copy
                old_file = parts[1]
                new_file = parts[2]
                file_changes[new_file] = (change_type, old_file, new_file)
            else:
                filename = parts[1] if len(parts) > 1 else parts[0]
                file_changes[filename] = (change_type, None, None)
        
        # Get actual patches for each file
        for filename, (change_type, old_file, new_file) in file_changes.items():
            try:
                patch_output = self._run_git(
                    "show",
                    f"{commit_hash}:{filename}",
                )
                
                # Get file stats
                stats = self._run_git(
                    "show",
                    f"--stat=100",
                    "--format=",
                    f"{commit_hash} -- {filename}",
                )
                
                stats_parts = stats.split(",")
                additions = 0
                deletions = 0
                
                for part in stats_parts:
                    if "insertion" in part:
                        try:
                            additions = int(part.split()[0])
                        except (ValueError, IndexError):
                            pass
                    elif "deletion" in part:
                        try:
                            deletions = int(part.split()[0])
                        except (ValueError, IndexError):
                            pass
                
                is_binary = "Bin" in stats
                
                diff = FileDiff(
                    filename=filename,
                    change_type=change_type,
                    old_file=old_file,
                    new_file=new_file,
                    additions=additions,
                    deletions=deletions,
                    patch=patch_output,
                    is_binary=is_binary,
                )
                diffs.append(diff)
            except Exception as e:
                logger.warning(f"Could not get diff for {filename}: {e}")
        
        return diffs
    
    def get_commit_branches(self, commit_hash: str) -> List[BranchInfo]:
        """Get branches associated with this commit."""
        branches = []
        
        # Get branches pointing to this commit
        output = self._run_git("branch", "-a", "--contains", commit_hash, "--format=%(refname:short)")
        
        for branch in output.strip().split("\n"):
            if branch.strip():
                is_remote = "/" in branch
                is_current = branch.startswith("*")
                if is_current:
                    branch = branch[2:]
                
                branches.append(BranchInfo(
                    branch_name=branch,
                    is_current=is_current,
                    is_remote=is_remote,
                ))
        
        return branches
    
    def get_commit_tags(self, commit_hash: str) -> List[str]:
        """Get tags for this commit."""
        output = self._run_git("tag", "--contains", commit_hash)
        tags = [tag.strip() for tag in output.strip().split("\n") if tag.strip()]
        return tags
    
    def scrape_repository(self) -> Dict[str, Any]:
        """Scrape entire repository and return structured data."""
        logger.info("Starting repository scrape...")
        
        commits_list = self.get_all_commits_chronological()
        commit_records = []
        
        for idx, commit_hash in enumerate(commits_list):
            if (idx + 1) % 100 == 0:
                logger.info(f"Processing commit {idx + 1}/{len(commits_list)}...")
            
            try:
                meta = self.get_commit_metadata(commit_hash)
                diffs = self.get_commit_diffs(commit_hash)
                branches = self.get_commit_branches(commit_hash)
                tags = self.get_commit_tags(commit_hash)
                
                record = CommitRecord(
                    meta=meta,
                    branches=branches,
                    diffs=diffs,
                    tags=tags,
                    commit_number=idx,
                    context={},
                )
                
                commit_records.append(record)
            except Exception as e:
                logger.error(f"Error processing commit {commit_hash}: {e}")
        
        return {
            "repo_path": str(self.repo_path),
            "scraped_at": datetime.now().isoformat(),
            "commits": [asdict(c) for c in commit_records],
            "total_commits": len(commit_records),
            "branches_found": len(set(b.branch_name for c in commit_records for b in c.branches)),
        }
    
    def save_results(self,  Dict[str, Any], output_file: Optional[str] = None) -> Path:
        """Save scrape results to JSON."""
        if output_file is None:
            output_file = self.output_dir / "commits_rich.json"
        else:
            output_file = Path(output_file)
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved results to {output_file}")
        return output_file


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Scrape git repository")
    parser.add_argument("--repo", required=True, help="Path to git repository")
    parser.add_argument("--output", default=None, help="Output JSON file")
    parser.add_argument("--output-dir", default="outputs", help="Output directory")
    
    args = parser.parse_args()
    
    try:
        scraper = GitScraper(args.repo, output_dir=args.output_dir)
        data = scraper.scrape_repository()
        output_path = scraper.save_results(data, args.output)
        
        print(f"\n" + "="*70)
        print("GIT SCRAPER SUMMARY")
        print("="*70)
        print(f"Total commits: {data['total_commits']}")
        print(f"Branches found: {data['branches_found']}")
        print(f"Output file: {output_path}")
        print("="*70 + "\n")
        
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        exit(1)


if __name__ == "__main__":
    main()
