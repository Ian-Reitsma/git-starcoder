#!/usr/bin/env python3
"""
Maximally Rich Git Scraper for The Block

Extracts EVERYTHING from Git:
- All commits with complete metadata
- Merge commit details (parents, resolution strategy)
- Full diffs with context
- Branch lineage and relationships
- Commit timing (to understand work patterns)
- File ownership and change frequency
- Author collaboration patterns
- Commit dependencies and relationships

Optimized for:
- Ryzen 5 3800X (8-core, 16-thread)
- RTX 2060 Super (8GB VRAM)
- 48GB RAM
- NVMe storage

Output: Richest possible commit graph for model training
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, asdict, field
from collections import defaultdict
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib

try:
    from git import Repo, GitCommandError, Actor
    import pygit2
    from tqdm import tqdm
except ImportError:
    print("Install: pip install GitPython pygit2 tqdm")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class DiffStats:
    """Per-file diff statistics"""
    path: str
    insertions: int
    deletions: int
    lines_of_context: int
    change_type: str  # 'add', 'modify', 'delete', 'rename'
    original_path: Optional[str] = None
    hunks: int = 0  # Number of diff hunks
    complexity: float = 0.0  # (insertions + deletions) / (insertions + deletions + unchanged)
    diff_text: str = ""  # Raw diff text for downstream tokenization


@dataclass
class AuthorStats:
    """Author contribution data"""
    name: str
    email: str
    num_commits: int = 0
    total_insertions: int = 0
    total_deletions: int = 0
    num_merges: int = 0
    first_commit_timestamp: int = 0
    last_commit_timestamp: int = 0
    files_touched: Set[str] = field(default_factory=set)


@dataclass
class BranchInfo:
    """Branch information"""
    name: str
    tip_hash: str
    creation_timestamp: int
    num_commits: int
    parent_branch: Optional[str] = None
    merge_base_hash: Optional[str] = None


@dataclass
class CommitMetadata:
    """Complete, rich commit metadata"""
    # Core identification
    hash: str
    abbrev_hash: str
    parents: List[str]
    tree_hash: str
    
    # Author info
    author_name: str
    author_email: str
    author_timestamp: int
    author_timezone: str
    
    # Committer info
    committer_name: str
    committer_email: str
    commit_timestamp: int
    committer_timezone: str
    
    # Message
    subject: str
    body: str
    full_message: str
    
    # Classification
    is_merge: bool
    is_squash: bool = False
    is_fixup: bool = False
    is_revert: bool = False
    
    # Branch info
    branches: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    
    # Change statistics
    files_added: List[str] = field(default_factory=list)
    files_modified: List[str] = field(default_factory=list)
    files_deleted: List[str] = field(default_factory=list)
    files_renamed: Dict[str, str] = field(default_factory=dict)  # old -> new
    
    # Detailed diff
    insertions: int = 0
    deletions: int = 0
    files_changed: int = 0
    diff_stats: List[DiffStats] = field(default_factory=list)
    
    # Merge-specific
    merge_parents: List[str] = field(default_factory=list)
    merge_base: Optional[str] = None
    merge_conflict_resolution: Optional[str] = None
    
    # Relationships
    related_issues: List[str] = field(default_factory=list)  # Refs to #123, etc
    related_commits: List[str] = field(default_factory=list)  # Commits mentioned
    
    # Work patterns
    time_since_parent: int = 0  # Seconds since parent commit
    work_hour: int = 0  # Hour of day (0-23)
    day_of_week: int = 0  # 0=Monday, 6=Sunday
    
    # Complexity metrics
    complexity_score: float = 0.0
    is_high_risk: bool = False  # Many files, big changes
    
    # File ownership patterns
    files_by_crate: Dict[str, List[str]] = field(default_factory=dict)  # Crate -> files


class RichGitScraper:
    """Maximally rich Git history extractor"""
    
    def __init__(self, repo_path: str, max_workers: int = 8, verbose: bool = False):
        self.repo_path = Path(repo_path)
        self.verbose = verbose
        self.max_workers = max_workers
        
        logger.info(f"Initializing repository: {repo_path}")
        try:
            self.repo = Repo(str(self.repo_path))
            self.repo_pg = pygit2.Repository(str(self.repo_path))
        except Exception as e:
            logger.error(f"Failed to initialize: {e}")
            sys.exit(1)
        
        self.commits: List[CommitMetadata] = []
        self.author_stats: Dict[str, AuthorStats] = {}
        self.branch_info: Dict[str, BranchInfo] = {}
        self.file_frequency: Dict[str, int] = defaultdict(int)
        self.commit_branch_map: Dict[str, List[str]] = self._build_commit_branch_map()
        self.commit_tag_map: Dict[str, List[str]] = self._build_commit_tag_map()
    
    def _log(self, msg: str):
        if self.verbose:
            logger.info(msg)
    
    def _extract_issue_refs(self, text: str) -> List[str]:
        """Extract GitHub issue references (#123)"""
        import re
        return re.findall(r'#(\d+)', text)
    
    def _extract_commit_refs(self, text: str) -> List[str]:
        """Extract commit hash references"""
        import re
        return re.findall(r'\b([0-9a-f]{7,40})\b', text, re.IGNORECASE)
    
    def _classify_commit(self, message: str, parents: List) -> Tuple[bool, bool, bool]:
        """Classify commit type"""
        is_merge = len(parents) > 1
        is_squash = 'squash' in message.lower()
        is_fixup = message.startswith('fixup!')
        is_revert = message.startswith('Revert')
        
        return is_squash, is_fixup, is_revert
    
    def _extract_crate_from_path(self, path: str) -> str:
        """Extract crate name from file path"""
        parts = path.split('/')
        if parts[0] == 'crates' and len(parts) > 1:
            return parts[1]
        return 'root'
    
    def _build_commit_branch_map(self) -> Dict[str, List[str]]:
        """Precompute branch membership for each commit"""
        branch_map: Dict[str, Set[str]] = defaultdict(set)
        try:
            for branch in self.repo.branches:
                for commit in self.repo.iter_commits(branch):
                    branch_map[commit.hexsha].add(branch.name)
        except Exception as e:
            self._log(f"Branch mapping error: {e}")
        return {commit_hash: sorted(list(branches)) for commit_hash, branches in branch_map.items()}

    def _build_commit_tag_map(self) -> Dict[str, List[str]]:
        """Precompute tag membership for each commit"""
        tag_map: Dict[str, List[str]] = defaultdict(list)
        try:
            for tag in self.repo.tags:
                if tag.commit:
                    tag_map[tag.commit.hexsha].append(tag.name)
        except Exception as e:
            self._log(f"Tag mapping error: {e}")
        return tag_map

    def _collect_commit_diffs(self, commit) -> List:
        """Compute commit diffs once and reuse downstream"""
        try:
            if commit.parents:
                parent = commit.parents[0]
                diffs = parent.diff(commit)
            else:
                diffs = commit.diff(None)
            return list(diffs)
        except Exception as e:
            self._log(f"Diff collection error: {e}")
            return []

    def _get_diff_context(self, diffs: List) -> Tuple[int, int, List[DiffStats]]:
        """Extract rich diff information"""
        total_insertions = 0
        total_deletions = 0
        diff_stats_list = []
        
        for diff_item in diffs:
            diff_text = ''
            if hasattr(diff_item, 'diff') and diff_item.diff is not None:
                raw_diff = diff_item.diff
                if isinstance(raw_diff, bytes):
                    diff_text = raw_diff.decode('utf-8', errors='ignore')
                elif isinstance(raw_diff, str):
                    diff_text = raw_diff
                else:
                    diff_text = str(raw_diff)
                insertions = diff_text.count('\n+')
                deletions = diff_text.count('\n-')
                hunks = diff_text.count('\n@@')
            else:
                insertions = 0
                deletions = 0
                hunks = 0

            filepath = diff_item.b_path or diff_item.a_path

            if diff_item.new_file or not diff_item.a_path:
                change_type = 'add'
            elif diff_item.deleted_file or not diff_item.b_path:
                change_type = 'delete'
            elif diff_item.renamed_file:
                change_type = 'rename'
            else:
                change_type = 'modify'

            total_changes = insertions + deletions
            complexity = total_changes / (total_changes + 1.0) if total_changes > 0 else 0.0

            stats = DiffStats(
                path=filepath,
                insertions=insertions,
                deletions=deletions,
                lines_of_context=0,
                change_type=change_type,
                hunks=hunks,
                complexity=complexity,
                diff_text=diff_text,
            )

            diff_stats_list.append(stats)
            total_insertions += insertions
            total_deletions += deletions
        
        return total_insertions, total_deletions, diff_stats_list
    
    def _get_file_changes(self, diffs: List) -> Tuple[List[str], List[str], List[str], Dict[str, str]]:
        """Categorize file changes"""
        added = []
        modified = []
        deleted = []
        renamed = {}
        
        for diff_item in diffs:
            filepath = diff_item.b_path or diff_item.a_path

            if diff_item.renamed_file:
                renamed[diff_item.a_path] = diff_item.b_path
            elif diff_item.new_file or not diff_item.a_path:
                added.append(filepath)
            elif diff_item.deleted_file or not diff_item.b_path:
                deleted.append(filepath)
            else:
                modified.append(filepath)
        
        return added, modified, deleted, renamed
    
    def _get_branch_info(self) -> Dict[str, BranchInfo]:
        """Get information about all branches"""
        branches = {}
        
        try:
            for branch in self.repo.branches:
                commit = branch.commit
                branches[branch.name] = BranchInfo(
                    name=branch.name,
                    tip_hash=commit.hexsha,
                    creation_timestamp=int(commit.committed_date),
                    num_commits=len(list(self.repo.iter_commits(branch)))
                )
        except Exception as e:
            self._log(f"Branch info error: {e}")
        
        return branches
    
    def _get_all_commits(self) -> List:
        """Get all unique commits across all branches"""
        all_commits = {}
        
        try:
            # Get commits from all references (branches, tags)
            for ref in self.repo.refs:
                try:
                    for commit in self.repo.iter_commits(ref):
                        all_commits[commit.hexsha] = commit
                except:
                    pass
        
        except Exception as e:
            self._log(f"Commit collection error: {e}")
        
        return list(all_commits.values())
    
    def _get_commit_branches(self, commit_hash: str) -> List[str]:
        """Find all branches containing a commit"""
        return self.commit_branch_map.get(commit_hash, [])
    
    def _get_commit_tags(self, commit_hash: str) -> List[str]:
        """Find all tags on a commit"""
        return self.commit_tag_map.get(commit_hash, [])
    
    def _calculate_complexity(self, commit_data: Dict) -> float:
        """Calculate commit complexity score (0.0-1.0)"""
        insertions = commit_data.get('insertions', 0)
        deletions = commit_data.get('deletions', 0)
        files_changed = commit_data.get('files_changed', 1)
        is_merge = commit_data.get('is_merge', False)
        
        # Base complexity from changes
        change_score = min((insertions + deletions) / 500, 1.0)  # Normalize at 500 lines
        
        # File diversity score
        file_score = min(files_changed / 50, 1.0)  # Normalize at 50 files
        
        # Merge complexity multiplier
        merge_multiplier = 1.5 if is_merge else 1.0
        
        complexity = ((change_score * 0.6 + file_score * 0.4) * merge_multiplier)
        return min(complexity, 1.0)
    
    def scrape_all_commits(self, progress: bool = True) -> List[CommitMetadata]:
        """Extract all commits with maximum richness"""
        
        logger.info("Starting rich Git scrape...")
        
        # Get branch info first
        self.branch_info = self._get_branch_info()
        logger.info(f"Found {len(self.branch_info)} branches")
        
        # Get all unique commits
        all_commits = self._get_all_commits()
        logger.info(f"Found {len(all_commits)} unique commits")
        
        # Sort by date
        all_commits.sort(key=lambda c: c.committed_date)
        
        # Process each commit
        iterator = tqdm(all_commits, desc="Extracting commits", disable=not progress)
        
        for commit in iterator:
            try:
                # Basic info
                commit_hash = commit.hexsha
                parent_hashes = [p.hexsha for p in commit.parents]
                
                # Messages
                subject, body = self._split_message(commit.message)
                
                # Reuse diffs across downstream computations
                diffs = self._collect_commit_diffs(commit)
                
                # File changes
                added, modified, deleted, renamed = self._get_file_changes(diffs)
                
                # Diff stats
                insertions, deletions, diff_stats = self._get_diff_context(diffs)
                
                # Classification
                is_squash, is_fixup, is_revert = self._classify_commit(commit.message, commit.parents)
                
                # Branch and tag info
                branches = self._get_commit_branches(commit_hash)
                tags = self._get_commit_tags(commit_hash)
                
                # Extract references
                issue_refs = self._extract_issue_refs(commit.message)
                commit_refs = self._extract_commit_refs(commit.message)
                
                # Work patterns
                author_date = datetime.fromtimestamp(commit.authored_date)
                work_hour = author_date.hour
                day_of_week = author_date.weekday()

                # Time since parent
                time_since_parent = 0
                if commit.parents:
                    time_since_parent = int(commit.authored_date - commit.parents[0].authored_date)
                
                # Files by crate
                files_by_crate = defaultdict(list)
                for f in added + modified:
                    crate = self._extract_crate_from_path(f)
                    files_by_crate[crate].append(f)
                
                # Complexity
                complexity_score = self._calculate_complexity({
                    'insertions': insertions,
                    'deletions': deletions,
                    'files_changed': len(added) + len(modified) + len(deleted),
                    'is_merge': len(parent_hashes) > 1
                })
                
                is_high_risk = (insertions + deletions > 500) or len(added) + len(modified) + len(deleted) > 20
                
                # Build metadata
                metadata = CommitMetadata(
                    hash=commit_hash,
                    abbrev_hash=commit_hash[:8],
                    parents=parent_hashes,
                    tree_hash=commit.tree.hexsha,
                    
                    author_name=commit.author.name,
                    author_email=commit.author.email,
                    author_timestamp=int(commit.authored_date),
                    author_timezone=str(commit.author.name),  # Placeholder
                    
                    committer_name=commit.committer.name,
                    committer_email=commit.committer.email,
                    commit_timestamp=int(commit.committed_date),
                    committer_timezone=str(commit.committer.name),  # Placeholder
                    
                    subject=subject,
                    body=body,
                    full_message=commit.message,
                    
                    is_merge=len(parent_hashes) > 1,
                    is_squash=is_squash,
                    is_fixup=is_fixup,
                    is_revert=is_revert,
                    
                    branches=branches,
                    tags=tags,
                    
                    files_added=added,
                    files_modified=modified,
                    files_deleted=deleted,
                    files_renamed=renamed,
                    
                    insertions=insertions,
                    deletions=deletions,
                    files_changed=len(added) + len(modified) + len(deleted),
                    diff_stats=diff_stats,
                    
                    merge_parents=parent_hashes[1:] if len(parent_hashes) > 1 else [],
                    
                    related_issues=[f"#{i}" for i in issue_refs],
                    related_commits=commit_refs,
                    
                    time_since_parent=time_since_parent,
                    work_hour=work_hour,
                    day_of_week=day_of_week,
                    
                    complexity_score=complexity_score,
                    is_high_risk=is_high_risk,
                    
                    files_by_crate=dict(files_by_crate)
                )
                
                self.commits.append(metadata)
                
                # Update author stats
                author_key = f"{commit.author.name}:{commit.author.email}"
                if author_key not in self.author_stats:
                    self.author_stats[author_key] = AuthorStats(
                        name=commit.author.name,
                        email=commit.author.email,
                        first_commit_timestamp=int(commit.authored_date)
                    )
                
                stats = self.author_stats[author_key]
                stats.num_commits += 1
                stats.total_insertions += insertions
                stats.total_deletions += deletions
                if len(parent_hashes) > 1:
                    stats.num_merges += 1
                stats.last_commit_timestamp = int(commit.authored_date)
                stats.files_touched.update(added + modified)
                
                # Track file frequency
                for f in added + modified:
                    self.file_frequency[f] += 1
            
            except Exception as e:
                logger.warning(f"Error processing commit {commit.hexsha[:8]}: {e}")
                continue
        
        logger.info(f"Extracted {len(self.commits)} commits successfully")
        return self.commits
    
    def _split_message(self, message: str) -> Tuple[str, str]:
        """Split commit message into subject and body"""
        lines = message.strip().split('\n', 1)
        subject = lines[0] if lines else ""
        body = lines[1].strip() if len(lines) > 1 else ""
        return subject, body
    
    def save_jsonl(self, output_path: str):
        """Save to JSONL (one commit per line)"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Writing {len(self.commits)} commits to {output_path}")
        
        with open(output_file, 'w') as f:
            for commit in self.commits:
                # Convert dataclasses to dict, handling nested structures
                data = asdict(commit)
                # Convert sets to lists
                if 'files_by_crate' in data:
                    data['files_by_crate'] = {k: list(v) if isinstance(v, set) else v for k, v in data['files_by_crate'].items()}
                # Convert diff_stats
                data['diff_stats'] = [asdict(ds) for ds in commit.diff_stats]
                f.write(json.dumps(data) + '\n')
        
        logger.info(f"Successfully saved to {output_path}")
    
    def save_json(self, output_path: str):
        """Save to JSON (pretty-printed)"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        data = []
        for commit in self.commits:
            d = asdict(commit)
            d['diff_stats'] = [asdict(ds) for ds in commit.diff_stats]
            d['files_by_crate'] = {k: list(v) if isinstance(v, set) else v 
                                  for k, v in d['files_by_crate'].items()}
            data.append(d)
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved pretty JSON to {output_path}")
    
    def get_statistics(self) -> Dict:
        """Generate comprehensive statistics"""
        if not self.commits:
            return {}
        
        # Time span
        first_timestamp = min(c.commit_timestamp for c in self.commits)
        last_timestamp = max(c.commit_timestamp for c in self.commits)
        time_span_days = (last_timestamp - first_timestamp) / 86400
        
        # Changes
        total_insertions = sum(c.insertions for c in self.commits)
        total_deletions = sum(c.deletions for c in self.commits)
        total_merges = sum(1 for c in self.commits if c.is_merge)
        
        # Files
        all_files = set()
        for c in self.commits:
            all_files.update(c.files_added + c.files_modified + c.files_deleted)
        
        return {
            "total_commits": len(self.commits),
            "total_merges": total_merges,
            "merge_percentage": (total_merges / len(self.commits) * 100) if self.commits else 0,
            "total_insertions": total_insertions,
            "total_deletions": total_deletions,
            "avg_insertions_per_commit": total_insertions / len(self.commits) if self.commits else 0,
            "avg_deletions_per_commit": total_deletions / len(self.commits) if self.commits else 0,
            "total_files_touched": len(all_files),
            "unique_authors": len(self.author_stats),
            "unique_branches": len(self.branch_info),
            "time_span_days": time_span_days,
            "commits_per_day": len(self.commits) / time_span_days if time_span_days > 0 else 0,
            "high_risk_commits": sum(1 for c in self.commits if c.is_high_risk),
        }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Rich Git history scraper for The Block")
    parser.add_argument("--repo", type=str, required=True, help="Repository path")
    parser.add_argument("--output", type=str, default="data/git_history_rich.jsonl", help="Output JSONL")
    parser.add_argument("--output-json", type=str, help="Also save as JSON")
    parser.add_argument("--workers", type=int, default=8, help="Thread workers")
    parser.add_argument("--stats", action="store_true", help="Print statistics")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    scraper = RichGitScraper(args.repo, max_workers=args.workers, verbose=args.verbose)
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
