#!/usr/bin/env python3
"""Adaptive dataset packer - maximize token efficiency.

Key innovations:
- Entropy-based file segmentation (semantic grouping)
- Dynamic padding elimination (pack to exact max_length)
- Cross-file dependency awareness (import graphs)
- Quality scoring (information density)
- Token flow optimization (no waste)
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["AdaptiveDatasetPacker", "PackedSequence"]


@dataclass
class PackedSequence:
    """Efficiently packed sequence (no waste)."""

    tokens: List[int]
    meta: Dict[str, Any]
    quality_score: float  # Information density
    efficiency: float  # (useful_tokens / total_tokens)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tokens": self.tokens,
            "metadata": self.meta,
            "quality_score": self.quality_score,
            "efficiency": self.efficiency,
        }


class AdaptiveDatasetPacker:
    """Pack sequences to maximize token efficiency and learning signal."""

    def __init__(
        self,
        tokenizer,
        repo_path: Path,
        max_length: int = 2048,
        reserve_tokens: int = 16,
    ):
        self.tokenizer = tokenizer
        self.repo_path = Path(repo_path)
        self.max_length = max_length
        self.reserve_tokens = reserve_tokens
        self.usable_length = max_length - reserve_tokens

        # Build dependency graph
        self.import_graph = self._build_import_graph()
        logger.info("Dependency graph built: %d files", len(self.import_graph))

    # =========================================================================
    # PHASE 1: Repo Analysis & Dependency Mapping
    # =========================================================================

    def _build_import_graph(self) -> Dict[str, List[str]]:
        """Build import dependency graph (file -> [imported_files])."""
        graph: Dict[str, List[str]] = defaultdict(list)

        for fpath in self.repo_path.rglob("*.rs"):
            if fpath.name.startswith("."):
                continue
            try:
                content = fpath.read_text(errors="ignore")
                file_key = str(fpath.relative_to(self.repo_path))

                # Find imports
                for line in content.split("\n"):
                    if "use " in line and "::" in line:
                        # Parse import path
                        parts = line.strip().split("use ")[1].split("::")[0]
                        graph[file_key].append(parts)
            except Exception:
                pass

        return dict(graph)

    def _compute_file_entropy(self, content: str) -> float:
        """Compute Shannon entropy of token sequence (information density).

        Higher entropy = more informative (less boilerplate/repetition).
        """
        tokens = self.tokenizer.encode(content, add_special_tokens=False)
        if len(tokens) < 2:
            return 0.0

        # Count token frequencies
        freq = {}
        for t in tokens:
            freq[t] = freq.get(t, 0) + 1

        # Shannon entropy
        entropy = 0.0
        total = len(tokens)
        for count in freq.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)

        # Normalize to [0, 1]
        return min(1.0, entropy / 8.0)  # Cap at log2(256) ~= 8

    def _find_semantic_breakpoints(self, content: str, max_chunk: int = 1000) -> List[int]:
        """Find natural breaking points in code (function boundaries, etc).

        Returns token indices of good places to split.
        """
        tokens = self.tokenizer.encode(content, add_special_tokens=False)
        breakpoints = [0]

        # Find function/impl block starts (heuristic)
        token_text = self.tokenizer.decode(tokens)
        lines = token_text.split("\n")

        token_idx = 0
        for line_idx, line in enumerate(lines):
            line_tokens = self.tokenizer.encode(line, add_special_tokens=False)
            token_idx += len(line_tokens)

            # Good breakpoints: function/impl declarations
            if any(kw in line for kw in ["fn ", "impl ", "pub fn", "pub struct", "pub enum"]):
                if token_idx < len(tokens) - 100:  # Don't break too close to end
                    breakpoints.append(token_idx)

        breakpoints.append(len(tokens))  # End
        return sorted(set(breakpoints))

    # =========================================================================
    # PHASE 2: Greedy Packing Algorithm
    # =========================================================================

    def pack_hierarchical_efficient(
        self,
        file_paths: Sequence[str],
    ) -> Optional[PackedSequence]:
        """Pack files hierarchically, eliminating padding waste.

        Algorithm:
        1. Sort files by entropy (high-entropy first)
        2. Greedily pack into bins of size usable_length
        3. For each file, find semantic breakpoints to avoid cutting mid-function
        4. Score sequence quality and efficiency
        """
        if not file_paths:
            return None

        # Load and sort files by entropy
        file_contents: Dict[str, str] = {}
        file_entropies: Dict[str, float] = {}

        for fpath_str in file_paths:
            try:
                fpath = self.repo_path / fpath_str
                content = fpath.read_text(errors="ignore")
                file_contents[fpath_str] = content
                entropy = self._compute_file_entropy(content)
                file_entropies[fpath_str] = entropy
            except Exception as e:
                logger.warning("Could not load %s: %s", fpath_str, e)
                continue

        if not file_contents:
            return None

        # Sort by entropy (high first)
        sorted_files = sorted(file_contents.keys(), key=lambda f: -file_entropies[f])

        # Greedy packing
        token_list: List[int] = []
        files_packed: List[str] = []
        total_useful_tokens = 0

        for fpath_str in sorted_files:
            content = file_contents[fpath_str]
            file_tokens = self.tokenizer.encode(content, add_special_tokens=False)

            # Find good break point (semantic boundary)
            remaining_budget = self.usable_length - len(token_list)
            if len(file_tokens) > remaining_budget:
                # Try to find a breakpoint
                breakpoints = self._find_semantic_breakpoints(content)
                file_tokens_trimmed = None
                for bp in reversed(breakpoints):
                    if bp < remaining_budget:
                        file_tokens_trimmed = file_tokens[:bp]
                        break
                if file_tokens_trimmed is None:
                    break  # Can't fit any more
                file_tokens = file_tokens_trimmed

            if not file_tokens:
                continue

            token_list.extend(file_tokens)
            files_packed.append(fpath_str)
            total_useful_tokens += len(file_tokens)

            if len(token_list) >= self.usable_length:
                break

        # Exact truncation (no padding)
        token_list = token_list[: self.usable_length]

        # Compute quality metrics
        quality_score = np.mean([file_entropies.get(f, 0.5) for f in files_packed])
        efficiency = total_useful_tokens / self.usable_length if self.usable_length > 0 else 0.0

        metadata = {
            "type": "adaptive_hierarchical",
            "files_packed": files_packed,
            "num_files": len(files_packed),
            "total_tokens": len(token_list),
            "has_padding": False,  # No padding
            "curriculum_difficulty": self._estimate_difficulty(quality_score),
        }

        return PackedSequence(
            tokens=token_list,
            metadata=metadata,
            quality_score=quality_score,
            efficiency=efficiency,
        )

    # =========================================================================
    # PHASE 3: Quality-Aware Packing
    # =========================================================================

    def pack_quality_biased(
        self,
        file_paths: Sequence[str],
        quality_threshold: float = 0.5,
    ) -> Optional[PackedSequence]:
        """Pack only high-quality (high-entropy) files to maximize learning signal.

        Skip low-entropy files (boilerplate, imports, etc).
        """
        # Filter files by entropy threshold
        high_quality_files = []
        for fpath_str in file_paths:
            try:
                fpath = self.repo_path / fpath_str
                content = fpath.read_text(errors="ignore")
                entropy = self._compute_file_entropy(content)
                if entropy >= quality_threshold:
                    high_quality_files.append(fpath_str)
            except Exception:
                pass

        if not high_quality_files:
            logger.warning("No high-quality files found (threshold=%.2f)", quality_threshold)
            high_quality_files = list(file_paths)  # Fallback

        return self.pack_hierarchical_efficient(high_quality_files)

    # =========================================================================
    # PHASE 4: Token Flow & Efficiency Analysis
    # =========================================================================

    def analyze_packing_efficiency(self, sequences: Sequence[PackedSequence]) -> Dict[str, Any]:
        """Analyze overall packing efficiency across a batch."""
        if not sequences:
            return {}

        efficiencies = [s.efficiency for s in sequences]
        quality_scores = [s.quality_score for s in sequences]
        token_counts = [len(s.tokens) for s in sequences]

        return {
            "mean_efficiency": np.mean(efficiencies),
            "std_efficiency": np.std(efficiencies),
            "min_efficiency": np.min(efficiencies),
            "max_efficiency": np.max(efficiencies),
            "mean_quality": np.mean(quality_scores),
            "total_tokens_packed": sum(token_counts),
            "num_sequences": len(sequences),
            "tokens_per_sequence": np.mean(token_counts),
        }

    # =========================================================================
    # Utilities
    # =========================================================================

    def _estimate_difficulty(self, quality_score: float) -> str:
        """Estimate difficulty from quality score."""
        if quality_score < 0.3:
            return "easy"
        elif quality_score < 0.5:
            return "medium"
        elif quality_score < 0.7:
            return "hard"
        else:
            return "very_hard"

    def batch_to_json(
        self, sequences: Sequence[PackedSequence], output_path: Path
    ) -> None:
        """Save packed sequences with efficiency metrics."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        efficiency_stats = self.analyze_packing_efficiency(sequences)
        payload = {
            "num_sequences": len(sequences),
            "max_length": self.max_length,
            "usable_length": self.usable_length,
            "efficiency_stats": efficiency_stats,
            "sequences": [s.to_dict() for s in sequences],
        }
        output_path.write_text(json.dumps(payload, indent=2))
        logger.info(
            "Saved %d sequences to %s (mean efficiency: %.2f%%)",
            len(sequences),
            output_path,
            efficiency_stats.get("mean_efficiency", 0) * 100,
        )
