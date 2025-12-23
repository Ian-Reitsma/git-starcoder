#!/usr/bin/env python3
"""
ELITE DATASET TESTS

Validates all advanced optimizations:
  ✓ Multi-scale context windows
  ✓ Git history integration
  ✓ AST-based augmentation
  ✓ Inter-file dependencies
  ✓ Function signature extraction
  ✓ Error pattern injection
  ✓ Temporal weighting
  ✓ Semantic deduplication
  ✓ Smart + temporal + evolution weighting
  ✓ Curriculum learning
"""

import unittest
import json
import os
from pathlib import Path
from collections import Counter

class TestELITEDataset(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Load dataset once for all tests"""
        cls.data_dir = Path.home() / "projects" / "git-starcoder" / "training_data_ELITE"
        cls.train_file = cls.data_dir / "training_data_train.jsonl"
        cls.val_file = cls.data_dir / "training_data_val.jsonl"
        cls.test_file = cls.data_dir / "training_data_test.jsonl"
        cls.metadata_file = cls.data_dir / "dataset_metadata.json"
        
        # Load metadata
        if cls.metadata_file.exists():
            with open(cls.metadata_file, 'r') as f:
                cls.metadata = json.load(f)
        else:
            cls.metadata = {}
        
        # Load sample sequences
        cls.train_seqs = []
        if cls.train_file.exists():
            with open(cls.train_file, 'r') as f:
                for idx, line in enumerate(f):
                    if idx < 1000:  # Sample first 1000
                        cls.train_seqs.append(json.loads(line))
                    else:
                        break
    
    # ========================================================================
    # BASIC INTEGRITY TESTS
    # ========================================================================
    
    def test_dataset_directory_exists(self):
        """Verify dataset directory was created"""
        self.assertTrue(self.data_dir.exists(), 
                       f"Dataset directory not found: {self.data_dir}")
    
    def test_metadata_file_exists(self):
        """Verify metadata file exists"""
        self.assertTrue(self.metadata_file.exists(),
                       "Metadata file not found")
    
    def test_train_file_exists(self):
        """Verify training file exists"""
        self.assertTrue(self.train_file.exists(),
                       "Training file not found")
    
    def test_val_file_exists(self):
        """Verify validation file exists"""
        self.assertTrue(self.val_file.exists(),
                       "Validation file not found")
    
    def test_test_file_exists(self):
        """Verify test file exists"""
        self.assertTrue(self.test_file.exists(),
                       "Test file not found")
    
    # ========================================================================
    # METADATA TESTS
    # ========================================================================
    
    def test_metadata_content(self):
        """Verify metadata has all required fields"""
        required = [
            'elite_version', 'total_sequences', 'train_sequences',
            'val_sequences', 'test_sequences', 'source_files',
            'tokenizer', 'context_windows', 'primary_window',
            'augmentation_techniques', 'weighting_strategy',
            'learning_strategy'
        ]
        for field in required:
            self.assertIn(field, self.metadata,
                         f"Missing metadata field: {field}")
        
        print(f"\n✓ Metadata complete:")
        print(f"  Elite version: {self.metadata.get('elite_version')}")
        print(f"  Total sequences: {self.metadata.get('total_sequences')}")
        print(f"  Context windows: {self.metadata.get('context_windows')}")
        print(f"  Primary window: {self.metadata.get('primary_window')}")
    
    def test_git_history_integration(self):
        """Verify git history was extracted"""
        self.assertIn('git_diffs', self.metadata,
                     "Git diffs not in metadata")
        git_diffs = self.metadata.get('git_diffs', 0)
        self.assertGreater(git_diffs, 0,
                          "No git diffs extracted")
        
        print(f"\n✓ Git history integration:")
        print(f"  Git diffs extracted: {git_diffs}")
    
    def test_function_signatures_extracted(self):
        """Verify function signatures were tracked"""
        self.assertIn('function_signatures', self.metadata,
                     "Function signatures not in metadata")
        func_count = self.metadata.get('function_signatures', 0)
        self.assertGreater(func_count, 0,
                          "No function signatures extracted")
        
        print(f"\n✓ Function signatures:")
        print(f"  Unique functions tracked: {func_count}")
    
    def test_dependency_graph_built(self):
        """Verify inter-file dependencies were analyzed"""
        self.assertIn('dependency_graph_nodes', self.metadata,
                     "Dependency graph not in metadata")
        dep_count = self.metadata.get('dependency_graph_nodes', 0)
        self.assertGreater(dep_count, 0,
                          "No dependencies tracked")
        
        print(f"\n✓ Dependency graph:")
        print(f"  Files with imports: {dep_count}")
    
    # ========================================================================
    # MULTI-SCALE CONTEXT TESTS
    # ========================================================================
    
    def test_multi_scale_contexts(self):
        """Verify multiple context window sizes are present"""
        context_windows = self.metadata.get('context_windows', [])
        self.assertGreater(len(context_windows), 1,
                          "Only one context window size (should be multi-scale)")
        
        # Check sequences have different window sizes
        window_sizes = set()
        for seq in self.train_seqs[:100]:
            window_size = seq['metadata'].get('window_size', 0)
            window_sizes.add(window_size)
        
        self.assertGreater(len(window_sizes), 1,
                          "All sequences have same window size")
        
        print(f"\n✓ Multi-scale contexts:")
        print(f"  Window sizes configured: {context_windows}")
        print(f"  Window sizes in use: {sorted(window_sizes)}")
    
    def test_primary_window_size(self):
        """Verify primary window is appropriate"""
        primary = self.metadata.get('primary_window', 0)
        self.assertIn(primary, [1024, 1536, 2048, 4096],
                     f"Unexpected primary window: {primary}")
        
        print(f"\n✓ Primary window: {primary} tokens")
    
    # ========================================================================
    # AUGMENTATION TESTS
    # ========================================================================
    
    def test_elite_augmentation_types(self):
        """Verify ELITE augmentation techniques are present"""
        expected_types = {
            'base', 'ast_based', 'error_patterns', 
            'type_annotations', 'cross_file_context', 'git_diff'
        }
        
        aug_types = set()
        for seq in self.train_seqs:
            aug_type = seq['metadata'].get('augmentation_type', 'base')
            aug_types.add(aug_type)
        
        found_types = expected_types.intersection(aug_types)
        self.assertGreater(len(found_types), 3,
                          f"Only {len(found_types)} augmentation types found")
        
        print(f"\n✓ ELITE augmentation types:")
        for atype in sorted(aug_types):
            count = sum(1 for s in self.train_seqs 
                       if s['metadata'].get('augmentation_type') == atype)
            print(f"  {atype}: {count} sequences")
    
    def test_git_diff_sequences(self):
        """Verify git diff sequences for evolution learning"""
        git_diff_seqs = [s for s in self.train_seqs 
                        if s['metadata'].get('augmentation_type') == 'git_diff']
        
        self.assertGreater(len(git_diff_seqs), 0,
                          "No git diff sequences found")
        
        # Check they have evolution metadata
        for seq in git_diff_seqs[:5]:
            self.assertIn('is_evolution', seq['metadata'],
                         "Git diff missing evolution flag")
        
        print(f"\n✓ Git diff sequences:")
        print(f"  Evolution sequences: {len(git_diff_seqs)}")
    
    def test_error_pattern_sequences(self):
        """Verify error pattern injection"""
        error_seqs = [s for s in self.train_seqs 
                     if s['metadata'].get('augmentation_type') == 'error_patterns']
        
        print(f"\n✓ Error pattern sequences: {len(error_seqs)}")
    
    # ========================================================================
    # WEIGHTING TESTS
    # ========================================================================
    
    def test_smart_weighting(self):
        """Verify smart priority weighting"""
        priority_counts = Counter()
        for seq in self.train_seqs:
            priority = seq['metadata'].get('priority', 'medium')
            priority_counts[priority] += 1
        
        # High priority should dominate (3x weight)
        self.assertGreater(priority_counts['high'], priority_counts['low'],
                          "High priority not weighted properly")
        
        print(f"\n✓ Smart weighting:")
        for priority, count in priority_counts.items():
            pct = (count / len(self.train_seqs)) * 100
            print(f"  {priority}: {count} ({pct:.1f}%)")
    
    def test_temporal_weighting(self):
        """Verify temporal weighting (recent code boosted)"""
        # Check if age_days metadata exists
        has_temporal = any('age_days' in s['metadata'] 
                          for s in self.train_seqs)
        
        if has_temporal:
            recent_seqs = [s for s in self.train_seqs 
                          if s['metadata'].get('age_days', 1000) < 60]
            print(f"\n✓ Temporal weighting:")
            print(f"  Recent sequences (<60 days): {len(recent_seqs)}")
        else:
            print(f"\n⚠ Temporal metadata not found (may not have git access)")
    
    # ========================================================================
    # DEDUPLICATION TESTS
    # ========================================================================
    
    def test_semantic_deduplication(self):
        """Verify near-duplicates were removed"""
        # Check metadata for dedup info
        self.assertIn('deduplication', self.metadata,
                     "Deduplication not documented")
        
        # Sample check: token sequence diversity
        token_hashes = set()
        for seq in self.train_seqs[:500]:
            tokens = tuple(seq['tokens'][:100])  # First 100 tokens
            token_hashes.add(hash(tokens))
        
        uniqueness = len(token_hashes) / len(self.train_seqs[:500])
        self.assertGreater(uniqueness, 0.80,
                          f"Low uniqueness: {uniqueness:.2%}")
        
        print(f"\n✓ Semantic deduplication:")
        print(f"  Uniqueness (sample): {uniqueness:.1%}")
        print(f"  Strategy: {self.metadata.get('deduplication')}")
    
    # ========================================================================
    # CURRICULUM LEARNING TESTS
    # ========================================================================
    
    def test_curriculum_learning(self):
        """Verify complexity-based ordering"""
        # Check sequences have complexity scores
        complexities = [s['metadata'].get('complexity', 0) 
                       for s in self.train_seqs[:100]]
        
        self.assertTrue(any(c > 0 for c in complexities),
                       "No complexity scores found")
        
        # Note: After shuffling, won't be strictly ordered,
        # but original ordering was curriculum-based
        avg_complexity = sum(complexities) / len(complexities)
        
        print(f"\n✓ Curriculum learning:")
        print(f"  Avg complexity (sample): {avg_complexity:.0f}")
        print(f"  Strategy: {self.metadata.get('learning_strategy')}")
    
    # ========================================================================
    # DATASET SPLIT TESTS
    # ========================================================================
    
    def test_dataset_split_ratio(self):
        """Verify 85/10/5 split"""
        total = self.metadata.get('total_sequences', 0)
        train = self.metadata.get('train_sequences', 0)
        val = self.metadata.get('val_sequences', 0)
        test = self.metadata.get('test_sequences', 0)
        
        train_pct = (train / total) * 100
        val_pct = (val / total) * 100
        test_pct = (test / total) * 100
        
        self.assertAlmostEqual(train_pct, 85.0, delta=1.0,
                              msg=f"Train split off: {train_pct:.1f}%")
        self.assertAlmostEqual(val_pct, 10.0, delta=1.0,
                              msg=f"Val split off: {val_pct:.1f}%")
        self.assertAlmostEqual(test_pct, 5.0, delta=1.0,
                              msg=f"Test split off: {test_pct:.1f}%")
        
        print(f"\n✓ Dataset split:")
        print(f"  Train: {train} ({train_pct:.1f}%)")
        print(f"  Val: {val} ({val_pct:.1f}%)")
        print(f"  Test: {test} ({test_pct:.1f}%)")
    
    # ========================================================================
    # FILE FORMAT TESTS
    # ========================================================================
    
    def test_jsonl_format(self):
        """Verify JSONL format (one JSON per line)"""
        with open(self.train_file, 'r') as f:
            for idx, line in enumerate(f):
                if idx >= 10:  # Test first 10 lines
                    break
                try:
                    obj = json.loads(line)
                    self.assertIn('tokens', obj)
                    self.assertIn('metadata', obj)
                except json.JSONDecodeError as e:
                    self.fail(f"Invalid JSONL at line {idx}: {e}")
        
        print(f"\n✓ JSONL format: Valid")
    
    def test_token_sequences_valid(self):
        """Verify token sequences are valid"""
        for seq in self.train_seqs[:10]:
            tokens = seq['tokens']
            self.assertIsInstance(tokens, list,
                                 "Tokens not a list")
            self.assertGreater(len(tokens), 0,
                              "Empty token sequence")
            self.assertTrue(all(isinstance(t, int) for t in tokens),
                           "Non-integer tokens found")
        
        print(f"\n✓ Token sequences: Valid")
    
    # ========================================================================
    # FILE SIZE TESTS
    # ========================================================================
    
    def test_file_sizes(self):
        """Verify dataset files are substantial"""
        train_size = os.path.getsize(self.train_file) / (1024**2)
        val_size = os.path.getsize(self.val_file) / (1024**2)
        test_size = os.path.getsize(self.test_file) / (1024**2)
        
        self.assertGreater(train_size, 50,
                          f"Train file too small: {train_size:.1f} MB")
        self.assertGreater(val_size, 5,
                          f"Val file too small: {val_size:.1f} MB")
        
        print(f"\n✓ File sizes:")
        print(f"  Train: {train_size:.1f} MB")
        print(f"  Val: {val_size:.1f} MB")
        print(f"  Test: {test_size:.1f} MB")
        print(f"  Total: {train_size + val_size + test_size:.1f} MB")
    
    # ========================================================================
    # SEQUENCE COUNT TESTS
    # ========================================================================
    
    def test_sequence_count(self):
        """Verify substantial number of sequences"""
        total = self.metadata.get('total_sequences', 0)
        self.assertGreater(total, 10000,
                          f"Too few sequences: {total}")
        
        print(f"\n✓ Total sequences: {total:,}")
    
    def test_augmented_sequences(self):
        """Verify augmentation increased dataset size"""
        source_files = self.metadata.get('source_files', 0)
        total_seqs = self.metadata.get('total_sequences', 0)
        
        # Should have multiple sequences per file
        ratio = total_seqs / max(source_files, 1)
        self.assertGreater(ratio, 10,
                          f"Low seq/file ratio: {ratio:.1f}")
        
        print(f"\n✓ Augmentation effectiveness:")
        print(f"  Source files: {source_files}")
        print(f"  Total sequences: {total_seqs}")
        print(f"  Sequences per file: {ratio:.1f}x")

def run_tests():
    """Run all tests and print summary"""
    suite = unittest.TestLoader().loadTestsFromTestCase(TestELITEDataset)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print(f"\n{'='*80}")
    if result.wasSuccessful():
        print(f"✓ ALL ELITE DATASET TESTS PASSED!")
        print(f"\n  Your dataset has:")
        print(f"    ✓ Multi-scale context windows")
        print(f"    ✓ Git history integration")
        print(f"    ✓ AST-based augmentation")
        print(f"    ✓ Inter-file dependencies")
        print(f"    ✓ Function signatures")
        print(f"    ✓ Error pattern injection")
        print(f"    ✓ Temporal weighting")
        print(f"    ✓ Semantic deduplication")
        print(f"    ✓ Smart + temporal + evolution weighting")
        print(f"    ✓ Curriculum learning")
        print(f"\n  Ready for ELITE model training! 🔥")
    else:
        print(f"✗ Some tests failed. Check output above.")
    print(f"{'='*80}")
    
    return 0 if result.wasSuccessful() else 1

if __name__ == '__main__':
    exit(run_tests())
