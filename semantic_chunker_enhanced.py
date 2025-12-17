#!/usr/bin/env python3
"""
Enhanced Semantic Chunker: Split diffs into meaningful units WITH cross-file context.

Improvements over original:
- Include related files from same commit
- Expand truncation limits (2000 → 4000 chars for code)
- Include old_code for before/after patterns
- Preserve full import statements and type definitions
- Add module-level context from related files

Output: JSONL (one chunk per line) ready for tokenization.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field
import logging
from enum import Enum
from collections import defaultdict

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
class CrossFileContext:
    """Context from related files in the same commit."""
    file_path: str
    snippet: str  # Relevant code snippet from related file
    snippet_type: str  # "imports", "trait_def", "struct_def", "impl_block", "function_stub"
    line_range: Tuple[int, int]  # Line numbers in original file


@dataclass
class CodeChunk:
    """A semantically meaningful chunk of code changes with cross-file context."""
    
    chunk_id: str  # commit_hash + chunk_index
    commit_hash: str
    file_path: str
    file_type: str  # rs, toml, md, etc
    
    change_type: ChangeType
    change_category: str  # file operation: A/M/D/R
    
    # Code content (expanded limits)
    old_code: str  # code before change (up to 4000 chars)
    new_code: str  # code after change (up to 4000 chars)
    patch: str  # actual diff (up to 10000 chars)
    
    # Context
    function_name: Optional[str]  # if applicable
    module_path: Optional[str]  # module hierarchy
    line_range: Tuple[int, int]  # (start, end) in original file
    
    # Metadata
    additions: int
    deletions: int
    context_lines: str  # surrounding code for understanding
    
    # Cross-file context (NEW)
    cross_file_context: List[CrossFileContext] = field(default_factory=list)
    related_files: List[str] = field(default_factory=list)
    
    # Relationships (FIXED: added missing colon)
    commit_meta: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dict."""
        d = asdict(self)
        d['change_type'] = self.change_type.value
        d['cross_file_context'] = [
            {
                'file_path': ctx.file_path,
                'snippet': ctx.snippet,
                'snippet_type': ctx.snippet_type,
                'line_range': ctx.line_range,
            }
            for ctx in self.cross_file_context
        ]
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
    def extract_trait_name(cls, lines: List[str]) -> Optional[str]:
        """Extract trait name from code block."""
        for line in lines:
            match = cls.TRAIT_PATTERN.search(line)
            if match:
                return match.group(1)
        return None
    
    @classmethod
    def extract_imports(cls, code: str) -> List[str]:
        """Extract all use statements from code."""
        lines = code.split('\n')
        imports = []
        for line in lines:
            if cls.USE_PATTERN.search(line):
                imports.append(line.strip())
        return imports
    
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
        parts = Path(file_path).parts
        
        if 'src' in parts:
            idx = parts.index('src')
            module_parts = parts[idx+1:]
        else:
            module_parts = parts
        
        if module_parts:
            if module_parts[-1] in ['lib.rs', 'mod.rs']:
                module_parts = module_parts[:-1]
            elif module_parts[-1].endswith('.rs'):
                module_parts = tuple(list(module_parts[:-1]) + [module_parts[-1][:-3]])
        
        return '::'.join(module_parts)
    
    @classmethod
    def infer_change_type(cls, code: str, added_lines: List[str], 
                         removed_lines: List[str]) -> ChangeType:
        """Infer type of change from code."""
        
        for line in added_lines:
            if cls.FN_PATTERN.search(line):
                return ChangeType.NEW_FUNCTION
            if cls.STRUCT_PATTERN.search(line):
                return ChangeType.NEW_STRUCT
            if cls.TEST_PATTERN.search(code):
                return ChangeType.NEW_TEST
        
        if cls.is_test_code(code):
            if added_lines:
                return ChangeType.NEW_TEST
            else:
                return ChangeType.MODIFIED_FUNCTION
        
        if len(added_lines) > 0 and len(removed_lines) > 0:
            if abs(len(added_lines) - len(removed_lines)) < 5:
                return ChangeType.REFACTORING
            else:
                return ChangeType.MODIFIED_LOGIC
        
        fix_keywords = ['fix', 'bug', 'error', 'panic', 'unwrap', 'expect']
        if any(kw in code.lower() for kw in fix_keywords):
            return ChangeType.BUG_FIX
        
        opt_keywords = ['cache', 'lazy', 'inline', 'optimize', 'perf', 'batch']
        if any(kw in code.lower() for kw in opt_keywords):
            return ChangeType.OPTIMIZATION
        
        return ChangeType.UNKNOWN


class EnhancedSemanticChunker:
    """Chunk code changes semantically WITH cross-file context."""
    
    def __init__(self, input_file: str, output_file: str = "chunks_enhanced.jsonl", 
                 include_cross_file: bool = True):
        self.input_file = Path(input_file)
        self.output_file = Path(output_file)
        self.include_cross_file = include_cross_file
        self.chunks: List[CodeChunk] = []
        self.stats = {
            "total_commits": 0,
            "total_files": 0,
            "total_chunks": 0,
            "chunks_with_cross_file_context": 0,
            "chunk_types": {},
            "avg_chunk_size_chars": 0,
            "max_chunk_size_chars": 0,
        }
    
    def load_commits(self) -> List[Dict[str, Any]]:
        """Load commits from git scraper output."""
        logger.info(f"Loading commits from {self.input_file}...")
        
        with open(self.input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        commits = data.get("commits", [])
        logger.info(f"Loaded {len(commits)} commits")
        
        return commits
    
    def _extract_relevant_imports(self, commit: Dict, target_file: str) -> List[str]:
        """Extract import statements relevant to the target file."""
        imports = []
        
        for diff in commit.get("diffs", []):
            file_path = diff.get("filename", "")
            if file_path.endswith(".rs") and file_path != target_file:
                code = diff.get("new_code", "") or diff.get("patch", "")
                file_imports = RustCodeAnalyzer.extract_imports(code)
                imports.extend(file_imports[:5])  # Limit to 5 per file
        
        return imports[:10]  # Max 10 imports
    
    def _extract_trait_and_struct_defs(self, commit: Dict, target_file: str) -> Dict[str, List[str]]:
        """Extract trait and struct definitions from related files."""
        defs = {"traits": [], "structs": [], "impl_blocks": []}
        
        for diff in commit.get("diffs", []):
            file_path = diff.get("filename", "")
            if file_path.endswith(".rs") and file_path != target_file:
                code = diff.get("new_code", "") or diff.get("patch", "")
                lines = code.split('\n')
                
                for i, line in enumerate(lines):
                    if RustCodeAnalyzer.TRAIT_PATTERN.search(line):
                        trait_block = '\n'.join(lines[i:min(i+5, len(lines))])
                        defs["traits"].append(trait_block)
                    elif RustCodeAnalyzer.STRUCT_PATTERN.search(line):
                        struct_block = '\n'.join(lines[i:min(i+5, len(lines))])
                        defs["structs"].append(struct_block)
                    elif RustCodeAnalyzer.IMPL_PATTERN.search(line):
                        impl_block = '\n'.join(lines[i:min(i+3, len(lines))])
                        defs["impl_blocks"].append(impl_block)
        
        return defs
    
    def _build_cross_file_context(self, commit: Dict, target_file: str) -> List[CrossFileContext]:
        """Build cross-file context from related files in same commit."""
        if not self.include_cross_file:
            return []
        
        contexts = []
        
        imports = self._extract_relevant_imports(commit, target_file)
        if imports:
            contexts.append(CrossFileContext(
                file_path="<imports>",
                snippet='\n'.join(imports),
                snippet_type="imports",
                line_range=(0, len(imports))
            ))
        
        defs = self._extract_trait_and_struct_defs(commit, target_file)
        
        for trait_def in defs["traits"]:
            contexts.append(CrossFileContext(
                file_path="<trait_defs>",
                snippet=trait_def,
                snippet_type="trait_def",
                line_range=(0, 0)
            ))
        
        for struct_def in defs["structs"]:
            contexts.append(CrossFileContext(
                file_path="<struct_defs>",
                snippet=struct_def,
                snippet_type="struct_def",
                line_range=(0, 0)
            ))
        
        for impl_block in defs["impl_blocks"][:3]:
            contexts.append(CrossFileContext(
                file_path="<impl_blocks>",
                snippet=impl_block,
                snippet_type="impl_block",
                line_range=(0, 0)
            ))
        
        return contexts
    
    def _parse_patch(self, patch: str) -> Tuple[List[str], List[str]]:
        """Parse unified diff into added and removed lines."""
        added = []
        removed = []
        
        for line in patch.split('\n'):
            if line.startswith('+') and not line.startswith('+++ '):
                added.append(line[1:])
            elif line.startswith('-') and not line.startswith('--- '):
                removed.append(line[1:])
        
        return added, removed
    
    def chunk_commit(self, commit: Dict[str, Any]) -> List[CodeChunk]:
        """Chunk a single commit into semantic units with cross-file context."""
        chunks = []
        commit_hash = commit["meta"]["hash"]
        commit_metadata = commit["meta"]
        related_files = [d["filename"] for d in commit.get("diffs", [])]
        
        for diff_idx, diff in enumerate(commit.get("diffs", [])):
            file_path = diff["filename"]
            file_type = Path(file_path).suffix.lstrip(".")
            
            if diff.get("is_binary", False):
                logger.debug(f"Skipping binary file: {file_path}")
                continue
            
            patch = diff.get("patch", "")
            change_cat = diff.get("change_type", "M")
            
            added_lines, removed_lines = self._parse_patch(patch)
            old_code = "\n".join(removed_lines)
            new_code = "\n".join(added_lines)
            
            old_code = old_code[:4000]
            new_code = new_code[:4000]
            patch = patch[:10000]
            
            if file_type == "rs":
                change_type = RustCodeAnalyzer.infer_change_type(
                    new_code, added_lines, removed_lines
                )
                function_name = RustCodeAnalyzer.extract_function_name(added_lines)
                module_path = RustCodeAnalyzer.get_module_path(file_path)
            else:
                change_type = ChangeType.DOCUMENTATION
                function_name = None
                module_path = None
            
            cross_file_ctx = self._build_cross_file_context(commit, file_path)
            
            chunk_id = f"{commit_hash[:8]}_{diff_idx}"
            
            chunk = CodeChunk(
                chunk_id=chunk_id,
                commit_hash=commit_hash,
                file_path=file_path,
                file_type=file_type,
                change_type=change_type,
                change_category=change_cat,
                old_code=old_code,
                new_code=new_code,
                patch=patch,
                function_name=function_name,
                module_path=module_path,
                line_range=(0, 0),
                additions=len(added_lines),
                deletions=len(removed_lines),
                context_lines="",
                cross_file_context=cross_file_ctx,
                related_files=related_files,
                commit_meta=commit_metadata,
            )
            
            chunks.append(chunk)
            self.stats["chunk_types"][change_type.value] = self.stats["chunk_types"].get(change_type.value, 0) + 1
        
        return chunks
    
    def process_all(self) -> List[CodeChunk]:
        """Process all commits into chunks."""
        logger.info("Processing all commits into chunks...")
        
        commits = self.load_commits()
        self.stats["total_commits"] = len(commits)
        
        all_chunks = []
        for commit in commits:
            chunks = self.chunk_commit(commit)
            all_chunks.extend(chunks)
        
        self.chunks = all_chunks
        self.stats["total_chunks"] = len(all_chunks)
        
        if all_chunks:
            chunk_sizes = [len(c.new_code) for c in all_chunks]
            self.stats["avg_chunk_size_chars"] = sum(chunk_sizes) // len(chunk_sizes)
            self.stats["max_chunk_size_chars"] = max(chunk_sizes)
            self.stats["chunks_with_cross_file_context"] = sum(
                1 for c in all_chunks if c.cross_file_context
            )
        
        logger.info(f"Created {len(all_chunks)} chunks")
        return all_chunks
    
    def save_jsonl(self) -> None:
        """Save chunks as JSONL."""
        with open(self.output_file, "w", encoding="utf-8") as f:
            for chunk in self.chunks:
                f.write(json.dumps(chunk.to_dict()) + "\n")
        logger.info(f"Saved {len(self.chunks)} chunks to {self.output_file}")
    
    def save_statistics(self) -> Dict[str, Any]:
        """Return statistics."""
        return self.stats


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced semantic chunker")
    parser.add_argument("--commits", required=True, help="Input commits file")
    parser.add_argument("--output", default="chunks_enhanced.jsonl", help="Output chunks file")
    parser.add_argument("--no-cross-file", action="store_true", help="Disable cross-file context")
    
    args = parser.parse_args()
    
    chunker = EnhancedSemanticChunker(
        input_file=args.commits,
        output_file=args.output,
        include_cross_file=not args.no_cross_file
    )
    
    chunker.process_all()
    chunker.save_jsonl()
    stats = chunker.save_statistics()
    
    logger.info(f"Statistics: {stats}")


if __name__ == "__main__":
    main()
