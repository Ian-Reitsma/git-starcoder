#!/usr/bin/env python3
"""
Semantic Chunker: Split diffs into meaningful units for tokenization.

This preserves code structure and semantic meaning:
- Function-level changes
- Module boundaries
- Test additions
- Configuration changes
- Refactoring patterns

Output: JSONL (one chunk per line) ready for tokenization.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import logging
from enum import Enum

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ChangeType(Enum):
    """Type of change in a chunk."""
    NEW_FUNCTION = "new_function"
    NEW_STRUCT = "new_struct"
    NEW_TEST = "new_test"
    MODIFIED_FUNCTION = "modified_function"
    MODIFIED_LOGIC = "modified_logic"
    REFACTORING = "refactoring"
    BUG_FIX = "bug_fix"
    OPTIMIZATION = "optimization"
    DOCUMENTATION = "documentation"
    CONFIGURATION = "configuration"
    IMPORTS = "imports"
    DELETION = "deletion"
    UNKNOWN = "unknown"


@dataclass
class CodeChunk:
    """A semantically meaningful chunk of code changes."""
    
    chunk_id: str  # commit_hash + chunk_index
    commit_hash: str
    file_path: str
    file_type: str  # rs, toml, md, etc
    
    change_type: ChangeType
    change_category: str  # file operation: A/M/D/R
    
    # Code content
    old_code: str  # code before change
    new_code: str  # code after change
    patch: str  # actual diff
    
    # Context
    function_name: Optional[str]  # if applicable
    module_path: Optional[str]  # module hierarchy
    line_range: Tuple[int, int]  # (start, end) in original file
    
    # Metadata
    additions: int
    deletions: int
    context_lines: str  # surrounding code for understanding
    
    # Relationships
    commit_meta: Dict[str, Any]  # author, timestamp, message, etc
    related_files: List[str]  # other files changed in same commit
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dict."""
        d = asdict(self)
        d['change_type'] = self.change_type.value
        return d


class RustCodeAnalyzer:
    """Analyze Rust code structure."""
    
    # Rust patterns
    FN_PATTERN = re.compile(r'^\s*(pub\s+)?(async\s+)?fn\s+([a-z_][a-z0-9_]*)\s*[(<]')
    STRUCT_PATTERN = re.compile(r'^\s*(pub\s+)?struct\s+([A-Z][a-zA-Z0-9_]*)\s*[{(<]')
    IMPL_PATTERN = re.compile(r'^\s*impl(?:\s*<[^>]+>)?\s+([A-Z][a-zA-Z0-9_:]*)')
    TEST_PATTERN = re.compile(r'^\s*#\[(?:tokio::)?test\]|^\s*#\[test\]|fn\s+test_')
    MOD_PATTERN = re.compile(r'^\s*mod\s+([a-z_][a-z0-9_]*)')
    USE_PATTERN = re.compile(r'^\s*use\s+')
    TRAIT_PATTERN = re.compile(r'^\s*(?:pub\s+)?trait\s+([A-Z][a-zA-Z0-9_]*)')
    ENUM_PATTERN = re.compile(r'^\s*(?:pub\s+)?enum\s+([A-Z][a-zA-Z0-9_]*)')
    CONST_PATTERN = re.compile(r'^\s*(?:pub\s+)?const\s+([A-Z_][A-Z0-9_]*)')
    
    @classmethod
    def extract_function_name(cls, lines: List[str]) -> Optional[str]:
        """Extract function name from code block."""
        for line in lines:
            match = cls.FN_PATTERN.search(line)
            if match:
                return match.group(3)
        return None
    
    @classmethod
    def extract_struct_name(cls, lines: List[str]) -> Optional[str]:
        """Extract struct name from code block."""
        for line in lines:
            match = cls.STRUCT_PATTERN.search(line)
            if match:
                return match.group(2)
        return None
    
    @classmethod
    def is_test_code(cls, code: str) -> bool:
        """Check if code block is a test."""
        return bool(cls.TEST_PATTERN.search(code))
    
    @classmethod
    def is_import(cls, code: str) -> bool:
        """Check if line is import statement."""
        return bool(cls.USE_PATTERN.search(code))
    
    @classmethod
    def get_module_path(cls, file_path: str) -> str:
        """Convert file path to module path."""
        # Convert src/energy_market/lib.rs -> energy_market
        parts = Path(file_path).parts
        
        if 'src' in parts:
            idx = parts.index('src')
            module_parts = parts[idx+1:]
        else:
            module_parts = parts
        
        # Remove file extension and lib.rs/mod.rs
        if module_parts:
            if module_parts[-1] in ['lib.rs', 'mod.rs']:
                module_parts = module_parts[:-1]
            elif module_parts[-1].endswith('.rs'):
                module_parts = (module_parts[:-1]) + (module_parts[-1][:-3],)
        
        return '::'.join(module_parts)
    
    @classmethod
    def infer_change_type(cls, code: str, added_lines: List[str], 
                         removed_lines: List[str]) -> ChangeType:
        """Infer type of change from code."""
        
        # Check for new functions
        for line in added_lines:
            if cls.FN_PATTERN.search(line):
                return ChangeType.NEW_FUNCTION
            if cls.STRUCT_PATTERN.search(line):
                return ChangeType.NEW_STRUCT
            if cls.TEST_PATTERN.search(code):
                return ChangeType.NEW_TEST
        
        # Check for tests
        if cls.is_test_code(code):
            if added_lines:  # Adding test
                return ChangeType.NEW_TEST
            else:
                return ChangeType.MODIFIED_FUNCTION
        
        # Check for refactoring (same logic, different code)
        if len(added_lines) > 0 and len(removed_lines) > 0:
            # Simple heuristic: significant rewrite
            if abs(len(added_lines) - len(removed_lines)) < 5:
                return ChangeType.REFACTORING
            else:
                return ChangeType.MODIFIED_LOGIC
        
        # Check for bug fixes (look for common patterns)
        fix_keywords = ['fix', 'bug', 'error', 'panic', 'unwrap', 'expect']
        if any(kw in code.lower() for kw in fix_keywords):
            return ChangeType.BUG_FIX
        
        # Check for optimizations
        opt_keywords = ['cache', 'lazy', 'inline', 'optimize', 'perf', 'batch']
        if any(kw in code.lower() for kw in opt_keywords):
            return ChangeType.OPTIMIZATION
        
        return ChangeType.UNKNOWN


class YAMLAnalyzer:
    """Analyze YAML/TOML configuration files."""
    
    @staticmethod
    def infer_change_type(file_path: str, code: str) -> ChangeType:
        """Infer change type for config files."""
        if file_path.endswith(('.toml', '.yaml', '.yml')):
            return ChangeType.CONFIGURATION
        return ChangeType.CONFIGURATION


class SemanticChunker:
    """Chunk code changes semantically."""
    
    def __init__(self, input_file: str, output_file: str = "chunks.jsonl"):
        self.input_file = Path(input_file)
        self.output_file = Path(output_file)
        self.chunks: List[CodeChunk] = []
        self.stats = {
            "total_commits": 0,
            "total_files": 0,
            "total_chunks": 0,
            "chunk_types": {}
        }
    
    def load_commits(self) -> List[Dict[str, Any]]:
        """Load commits from git scraper output."""
        logger.info(f"Loading commits from {self.input_file}...")
        
        with open(self.input_file, "r") as f:
            data = json.load(f)
        
        commits = data.get("commits", [])
        logger.info(f"Loaded {len(commits)} commits")
        
        return commits
    
    def chunk_commit(self, commit: Dict[str, Any]) -> List[CodeChunk]:
        """Chunk a single commit into semantic units."""
        chunks = []
        commit_hash = commit["metadata"]["hash"]
        commit_metadata = commit["metadata"]
        commit_message = commit["metadata"]["message"]
        related_files = [d["filename"] for d in commit["diffs"]]
        
        for diff_idx, diff in enumerate(commit["diffs"]):
            file_path = diff["filename"]
            file_type = Path(file_path).suffix.lstrip(".")
            
            # Skip binary files
            if diff.get("is_binary", False):
                logger.debug(f"Skipping binary file: {file_path}")
                continue
            
            patch = diff.get("patch", "")
            change_cat = diff.get("change_type", "M")
            
            # Parse patch into additions/deletions
            added_lines, removed_lines = self._parse_patch(patch)
            old_code = "\n".join(removed_lines)
            new_code = "\n".join(added_lines)
            
            # Analyze code
            if file_type == "rs":
                change_type = RustCodeAnalyzer.infer_change_type(
                    new_code, added_lines, removed_lines
                )
                function_name = RustCodeAnalyzer.extract_function_name(added_lines)
                module_path = RustCodeAnalyzer.get_module_path(file_path)
            elif file_type in ["toml", "yaml", "yml"]:
                change_type = ChangeType.CONFIGURATION
                function_name = None
                module_path = Path(file_path).stem
            else:
                change_type = ChangeType.DOCUMENTATION
                function_name = None
                module_path = None
            
            # Create chunk
            chunk_id = f"{commit_hash[:8]}_{diff_idx}"
            
            chunk = CodeChunk(
                chunk_id=chunk_id,
                commit_hash=commit_hash,
                file_path=file_path,
                file_type=file_type,
                change_type=change_type,
                change_category=change_cat,
                old_code=old_code[:2000],  # Limit size
                new_code=new_code[:2000],
                patch=patch[:5000],  # Limit patch size
                function_name=function_name,
                module_path=module_path,
                line_range=(0, len(added_lines)),  # Approximate
                additions=len(added_lines),
                deletions=len(removed_lines),
                context_lines="",  # Could be enriched
                commit_metadata=commit_metadata,
                related_files=related_files
            )
            
            chunks.append(chunk)
            
            # Track stats
            change_type_key = change_type.value
            self.stats["chunk_types"][change_type_key] = \
                self.stats["chunk_types"].get(change_type_key, 0) + 1
        
        return chunks
    
    def _parse_patch(self, patch: str) -> Tuple[List[str], List[str]]:
        """Parse unified diff format into added/removed lines."""
        added_lines = []
        removed_lines = []
        
        in_hunk = False
        for line in patch.split("\n"):
            # Skip file headers
            if line.startswith("---") or line.startswith("+++"):
                continue
            
            # Parse hunk headers
            if line.startswith("@@"):
                in_hunk = True
                continue
            
            if not in_hunk:
                continue
            
            if line.startswith("+") and not line.startswith("+++"):
                added_lines.append(line[1:])
            elif line.startswith("-") and not line.startswith("---"):
                removed_lines.append(line[1:])
        
        return added_lines, removed_lines
    
    def process_all(self) -> List[CodeChunk]:
        """Process all commits and generate chunks."""
        commits = self.load_commits()
        
        all_chunks = []
        for i, commit in enumerate(commits):
            if (i + 1) % 100 == 0:
                logger.info(f"Processing commit {i+1}/{len(commits)}...")
            
            try:
                chunks = self.chunk_commit(commit)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.warning(f"Error chunking commit {commit['metadata']['hash']}: {e}")
                continue
        
        self.chunks = all_chunks
        self.stats["total_commits"] = len(commits)
        self.stats["total_files"] = len(set(c.file_path for c in all_chunks))
        self.stats["total_chunks"] = len(all_chunks)
        
        return all_chunks
    
    def save_jsonl(self) -> Path:
        """Save chunks to JSONL (one per line)."""
        logger.info(f"Saving {len(self.chunks)} chunks to {self.output_file}...")
        
        with open(self.output_file, "w") as f:
            for chunk in self.chunks:
                f.write(json.dumps(chunk.to_dict()) + "\n")
        
        logger.info(f"Saved to {self.output_file}")
        return self.output_file
    
    def save_statistics(self) -> Dict[str, Any]:
        """Save chunking statistics."""
        stats_file = self.output_file.parent / "chunking_stats.json"
        
        with open(stats_file, "w") as f:
            json.dump(self.stats, f, indent=2)
        
        logger.info(f"Saved statistics to {stats_file}")
        return self.stats


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Semantically chunk git diffs"
    )
    parser.add_argument(
        "--commits",
        required=True,
        help="Input commits JSON file (from git_scraper.py)"
    )
    parser.add_argument(
        "--output",
        default="chunks.jsonl",
        help="Output JSONL file"
    )
    
    args = parser.parse_args()
    
    try:
        chunker = SemanticChunker(args.commits, args.output)
        chunks = chunker.process_all()
        chunker.save_jsonl()
        stats = chunker.save_statistics()
        
        # Print summary
        print(f"\n" + "="*60)
        print("SEMANTIC CHUNKING SUMMARY")
        print("="*60)
        print(f"Total chunks created: {stats['total_chunks']}")
        print(f"From {stats['total_commits']} commits")
        print(f"Across {stats['total_files']} files")
        print(f"\nChunk distribution:")
        for chunk_type, count in sorted(stats['chunk_types'].items(), key=lambda x: -x[1]):
            print(f"  {chunk_type}: {count}")
        print(f"\nOutput saved to: {args.output}")
        print("="*60 + "\n")
        
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        exit(1)


if __name__ == "__main__":
    main()
