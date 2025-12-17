#!/usr/bin/env python3
"""
Comprehensive Git Repository Scraper

Extracts all commits, merges, and metadata from a Git repository.
Designed to capture the complete evolutionary history of the codebase
for model training and analysis.

Usage:
    python git_scraper.py --repo /path/to/repo --output data/git_history.jsonl
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import argparse

try:
    from git import Repo, GitCommandError, Actor
    import pygit2
except ImportError:
    print("Please install GitPython: pip install GitPython pygit2")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable


@dataclass
class CommitMetadata:
    """Complete metadata for a single commit"""
    commit_hash: str
    parent_hashes: List[str]
    author_name: str
    author_email: str
    author_timestamp: int  # Unix timestamp
    committer_name: str
    committer_email: str
    commit_timestamp: int  # Unix timestamp
    message: str
    message_subject: str
    message_body: str
    branch: str
    is_merge: bool
    files_changed: int
    insertions: int
    deletions: int
    files_modified: List[str]
    files_added: List[str]
    files_deleted: List[str]
    stats: Dict[str, int]  # Per-file stats {filename: {insertions, deletions}}
    tree_hash: str
    diff_summary: str  # Summary of changes


class GitScraper:
    """Scrapes comprehensive Git history from a repository"""
    
    def __init__(self, repo_path: str, verbose: bool = False):
        self.repo_path = Path(repo_path)
        self.verbose = verbose
        
        # Initialize both GitPython and pygit2 for best compatibility
        try:
            self.repo_gp = Repo(str(self.repo_path))
            self.repo_pg = pygit2.Repository(str(self.repo_path))
        except Exception as e:
            print(f"Error initializing repository: {e}")
            sys.exit(1)
        
        self.commits_data: List[CommitMetadata] = []
        self.commit_hashes_seen = set()
    
    def _log(self, msg: str):
        """Conditional logging"""
        if self.verbose:
            print(f"[*] {msg}")
    
    def _get_all_branches(self) -> List[str]:
        """Get all local and remote branches"""
        try:
            result = subprocess.run(
                ["git", "-C", str(self.repo_path), "for-each-ref", "--format=%(refname)", "refs/heads", "refs/remotes"],
                capture_output=True,
                text=True,
                check=True
            )
            branches: List[str] = []
            for ref in result.stdout.splitlines():
                ref = ref.strip()
                if not ref:
                    continue
                if ref.endswith("/HEAD"):
                    continue

                if ref.startswith("refs/heads/"):
                    branches.append(ref[len("refs/heads/"):])
                elif ref.startswith("refs/remotes/"):
                    branches.append(ref[len("refs/remotes/"):])
                else:
                    branches.append(ref)

            return sorted(set(branches))
        except subprocess.CalledProcessError as e:
            self._log(f"Error getting branches: {e.stderr.strip()}")
        except Exception as e:
            self._log(f"Error getting branches: {e}")

        return []
    
    def _get_commit_diff_stats(self, commit) -> Tuple[int, int, Dict[str, Dict[str, int]]]:
        """Get detailed diff statistics for a commit"""
        file_stats = {}

        try:
            stats = commit.stats
            total_insertions = stats.total.get("insertions", 0)
            total_deletions = stats.total.get("deletions", 0)

            for filepath, values in stats.files.items():
                file_stats[filepath] = {
                    "insertions": values.get("insertions", 0),
                    "deletions": values.get("deletions", 0)
                }
        except Exception as e:
            self._log(f"Error getting diff stats: {e}")
            total_insertions = 0
            total_deletions = 0

        return total_insertions, total_deletions, file_stats
    
    def _get_file_changes(self, commit) -> Tuple[List[str], List[str], List[str]]:
        """Determine which files were added, modified, deleted"""
        added = []
        modified = []
        deleted = []
        
        try:
            if commit.parents:
                parent = commit.parents[0]
                diffs = parent.diff(commit)
            else:
                diffs = commit.diff(None)
            
            for diff_item in diffs:
                filepath = diff_item.b_path or diff_item.a_path
                
                if diff_item.new_file or not diff_item.a_path:
                    added.append(filepath)
                elif diff_item.deleted_file or not diff_item.b_path:
                    deleted.append(filepath)
                else:
                    modified.append(filepath)
        
        except Exception as e:
            self._log(f"Error getting file changes: {e}")
        
        return added, modified, deleted
    
    def _extract_message_parts(self, message: str) -> Tuple[str, str]:
        """Split commit message into subject and body"""
        lines = message.strip().split('\n', 1)
        subject = lines[0] if lines else ""
        body = lines[1].strip() if len(lines) > 1 else ""
        return subject, body
    
    def _get_commit_for_branch(self, branch: str) -> Optional[object]:
        """Get commit object for a branch"""
        try:
            # Handle remote branches
            ref_name = f"refs/remotes/origin/{branch}" if f"origin/{branch}" in self.repo_gp.remotes.origin.refs else f"refs/heads/{branch}"
            return self.repo_gp.commit(branch)
        except Exception as e:
            self._log(f"Error getting commit for branch {branch}: {e}")
            return None
    
    def scrape_all_commits(self) -> List[CommitMetadata]:
        """Scrape all commits from all branches"""
        self._log("Starting comprehensive Git scraping...")
        
        # Get all branches
        branches = self._get_all_branches()
        self._log(f"Found {len(branches)} branches")
        
        # Collect all unique commits
        all_commits = {}
        
        for branch in tqdm(branches, desc="Scanning branches"):
            try:
                # Get commits for this branch
                for commit in self.repo_gp.iter_commits(branch):
                    if commit.hexsha not in all_commits:
                        all_commits[commit.hexsha] = (commit, branch)
            except GitCommandError as e:
                self._log(f"Error iterating branch {branch}: {e}")
                continue
        
        self._log(f"Found {len(all_commits)} unique commits")
        
        # Process each commit
        for commit_hash, (commit, discovered_branch) in tqdm(
            all_commits.items(),
            desc="Processing commits",
            total=len(all_commits)
        ):
            if commit_hash in self.commit_hashes_seen:
                continue
            
            self.commit_hashes_seen.add(commit_hash)
            
            try:
                # Extract basic info
                parent_hashes = [p.hexsha for p in commit.parents]
                is_merge = len(parent_hashes) > 1
                
                # Extract message
                subject, body = self._extract_message_parts(commit.message)
                
                # Get file changes
                added, modified, deleted = self._get_file_changes(commit)
                files_changed = len(added) + len(modified) + len(deleted)
                
                # Get diff stats
                insertions, deletions, stats = self._get_commit_diff_stats(commit)
                
                # Create diff summary
                diff_summary = f"Files changed: {files_changed}, Insertions: +{insertions}, Deletions: -{deletions}"
                
                # Create metadata object
                metadata = CommitMetadata(
                    commit_hash=commit_hash,
                    parent_hashes=parent_hashes,
                    author_name=commit.author.name,
                    author_email=commit.author.email,
                    author_timestamp=int(commit.authored_datetime.timestamp()),
                    committer_name=commit.committer.name,
                    committer_email=commit.committer.email,
                    commit_timestamp=int(commit.committed_datetime.timestamp()),
                    message=commit.message,
                    message_subject=subject,
                    message_body=body,
                    branch=discovered_branch,
                    is_merge=is_merge,
                    files_changed=files_changed,
                    insertions=insertions,
                    deletions=deletions,
                    files_modified=modified,
                    files_added=added,
                    files_deleted=deleted,
                    stats=stats,
                    tree_hash=commit.tree.hexsha,
                    diff_summary=diff_summary
                )
                
                self.commits_data.append(metadata)
            
            except Exception as e:
                self._log(f"Error processing commit {commit_hash}: {e}")
                continue
        
        # Sort by timestamp
        self.commits_data.sort(key=lambda x: x.commit_timestamp)
        
        self._log(f"Successfully scraped {len(self.commits_data)} commits")
        return self.commits_data
    
    def save_jsonl(self, output_path: str):
        """Save scraped data as JSONL"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        self._log(f"Writing {len(self.commits_data)} commits to {output_path}")
        
        with open(output_file, 'w') as f:
            for commit in self.commits_data:
                f.write(json.dumps(asdict(commit)) + '\n')
        
        self._log(f"Saved to {output_path}")
    
    def save_json(self, output_path: str):
        """Save scraped data as JSON array"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        self._log(f"Writing {len(self.commits_data)} commits to {output_path}")
        
        with open(output_file, 'w') as f:
            json.dump([asdict(commit) for commit in self.commits_data], f, indent=2)
        
        self._log(f"Saved to {output_path}")
    
    def get_statistics(self) -> Dict:
        """Generate statistics about the scraped repository"""
        if not self.commits_data:
            return {}
        
        total_commits = len(self.commits_data)
        total_merges = sum(1 for c in self.commits_data if c.is_merge)
        total_insertions = sum(c.insertions for c in self.commits_data)
        total_deletions = sum(c.deletions for c in self.commits_data)
        
        # Get time span
        first_commit = self.commits_data[0]
        last_commit = self.commits_data[-1]
        time_span_days = (last_commit.commit_timestamp - first_commit.commit_timestamp) / 86400
        
        # Get unique authors
        authors = set()
        for commit in self.commits_data:
            authors.add(commit.author_email)
        
        # Get branches
        branches = set(c.branch for c in self.commits_data)
        
        return {
            "total_commits": total_commits,
            "total_merges": total_merges,
            "merge_percentage": (total_merges / total_commits * 100) if total_commits > 0 else 0,
            "total_insertions": total_insertions,
            "total_deletions": total_deletions,
            "average_insertions_per_commit": total_insertions / total_commits if total_commits > 0 else 0,
            "average_deletions_per_commit": total_deletions / total_commits if total_commits > 0 else 0,
            "unique_authors": len(authors),
            "unique_branches": len(branches),
            "time_span_days": time_span_days,
            "commits_per_day": total_commits / time_span_days if time_span_days > 0 else 0,
            "first_commit_timestamp": first_commit.commit_timestamp,
            "last_commit_timestamp": last_commit.commit_timestamp,
        }


def main():
    parser = argparse.ArgumentParser(
        description="Scrape comprehensive Git history from a repository"
    )
    parser.add_argument(
        "--repo",
        type=str,
        required=True,
        help="Path to Git repository"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/git_history.jsonl",
        help="Output file path (JSONL format)"
    )
    parser.add_argument(
        "--output-json",
        type=str,
        help="Also save as JSON (prettified)"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print statistics"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    scraper = GitScraper(args.repo, verbose=args.verbose)
    scraper.scrape_all_commits()
    scraper.save_jsonl(args.output)
    
    if args.output_json:
        scraper.save_json(args.output_json)
    
    if args.stats:
        stats = scraper.get_statistics()
        print("\n" + "="*60)
        print("Repository Statistics")
        print("="*60)
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"{key}: {value:.2f}")
            else:
                print(f"{key}: {value}")


if __name__ == "__main__":
    main()
