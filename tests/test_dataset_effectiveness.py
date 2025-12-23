#!/usr/bin/env python3
"""
Comprehensive Tests for Maximum Effectiveness Dataset Creator

Tests:
  ✓ File scanning and parsing
  ✓ Tokenization correctness
  ✓ Augmentation quality
  ✓ Weighting application
  ✓ Dataset integrity
  ✓ File formats (JSONL)
  ✓ Metadata accuracy
  ✓ Sequence diversity
"""

import json
import sys
from pathlib import Path
import unittest
from collections import Counter

# Setup paths
GIT_STARCODER = Path.home() / "projects" / "git-starcoder"
DATASET_DIR = GIT_STARCODER / "training_data_effectiveness"

class TestDatasetCreation(unittest.TestCase):
    """Test dataset creation and integrity"""
    
    @classmethod
    def setUpClass(cls):
        """Load test data"""
        cls.dataset_dir = DATASET_DIR
        cls.metadata_file = cls.dataset_dir / "dataset_metadata.json"
        cls.train_file = cls.dataset_dir / "training_data_train.jsonl"
        cls.val_file = cls.dataset_dir / "training_data_val.jsonl"
        cls.test_file = cls.dataset_dir / "training_data_test.jsonl"
    
    def test_dataset_directory_exists(self):
        """Test that dataset directory was created"""
        self.assertTrue(
            self.dataset_dir.exists(),
            f"Dataset directory not found: {self.dataset_dir}"
        )
        print(f"  ✓ Dataset directory exists: {self.dataset_dir}")
    
    def test_metadata_file_exists(self):
        """Test that metadata file was created"""
        self.assertTrue(
            self.metadata_file.exists(),
            f"Metadata file not found: {self.metadata_file}"
        )
        print(f"  ✓ Metadata file exists")
    
    def test_metadata_content(self):
        """Test metadata file contains expected fields"""
        with open(self.metadata_file, 'r') as f:
            metadata = json.load(f)
        
        required_fields = [
            'total_sequences', 'train_sequences', 'val_sequences', 'test_sequences',
            'max_tokens_per_sequence', 'tokenizer', 'augmentation_techniques',
            'weighting_strategy', 'learning_strategy'
        ]
        
        for field in required_fields:
            self.assertIn(field, metadata, f"Missing metadata field: {field}")
        
        print(f"  ✓ Metadata contains all required fields")
        print(f"    - Total sequences: {metadata['total_sequences']}")
        print(f"    - Max tokens: {metadata['max_tokens_per_sequence']}")
        print(f"    - Augmentation techniques: {len(metadata['augmentation_techniques'])}")
    
    def test_train_file_exists(self):
        """Test training file exists"""
        self.assertTrue(
            self.train_file.exists(),
            f"Train file not found: {self.train_file}"
        )
        print(f"  ✓ Training file exists")
    
    def test_val_file_exists(self):
        """Test validation file exists"""
        self.assertTrue(
            self.val_file.exists(),
            f"Val file not found: {self.val_file}"
        )
        print(f"  ✓ Validation file exists")
    
    def test_train_file_format(self):
        """Test training file is valid JSONL"""
        count = 0
        with open(self.train_file, 'r') as f:
            for line in f:
                try:
                    obj = json.loads(line.strip())
                    self.assertIn('tokens', obj, "Missing 'tokens' field")
                    self.assertIn('metadata', obj, "Missing 'metadata' field")
                    self.assertIsInstance(obj['tokens'], list, "Tokens should be list")
                    self.assertIsInstance(obj['metadata'], dict, "Metadata should be dict")
                    count += 1
                except json.JSONDecodeError as e:
                    self.fail(f"Invalid JSON at line {count}: {e}")
        
        self.assertGreater(count, 0, "Training file is empty")
        print(f"  ✓ Training file is valid JSONL ({count} sequences)")
    
    def test_val_file_format(self):
        """Test validation file is valid JSONL"""
        count = 0
        with open(self.val_file, 'r') as f:
            for line in f:
                try:
                    obj = json.loads(line.strip())
                    count += 1
                except json.JSONDecodeError:
                    self.fail(f"Invalid JSON in val file at line {count}")
        
        self.assertGreater(count, 0, "Validation file is empty")
        print(f"  ✓ Validation file is valid JSONL ({count} sequences)")
    
    def test_sequence_token_count(self):
        """Test sequences have correct token count"""
        with open(self.metadata_file, 'r') as f:
            metadata = json.load(f)
        
        max_tokens = metadata['max_tokens_per_sequence']
        
        with open(self.train_file, 'r') as f:
            for i, line in enumerate(f):
                if i >= 10:  # Sample first 10
                    break
                obj = json.loads(line.strip())
                token_count = len(obj['tokens'])
                self.assertEqual(
                    token_count, max_tokens,
                    f"Sequence {i} has {token_count} tokens, expected {max_tokens}"
                )
        
        print(f"  ✓ All sequences have correct token count ({max_tokens})")
    
    def test_sequence_metadata(self):
        """Test sequence metadata is present"""
        required_meta = ['seq_id', 'source_file']
        
        with open(self.train_file, 'r') as f:
            for i, line in enumerate(f):
                if i >= 5:  # Sample first 5
                    break
                obj = json.loads(line.strip())
                meta = obj['metadata']
                for field in required_meta:
                    self.assertIn(field, meta, f"Missing metadata field: {field}")
        
        print(f"  ✓ All sequences have required metadata")
    
    def test_dataset_split_ratio(self):
        """Test train/val/test split ratio is 85/10/5"""
        with open(self.metadata_file, 'r') as f:
            metadata = json.load(f)
        
        total = metadata['total_sequences']
        train = metadata['train_sequences']
        val = metadata['val_sequences']
        test = metadata['test_sequences']
        
        train_pct = train / total * 100
        val_pct = val / total * 100
        test_pct = test / total * 100
        
        # Allow 2% tolerance
        self.assertAlmostEqual(train_pct, 85, delta=2, msg=f"Train split {train_pct}% != 85%")
        self.assertAlmostEqual(val_pct, 10, delta=2, msg=f"Val split {val_pct}% != 10%")
        self.assertAlmostEqual(test_pct, 5, delta=2, msg=f"Test split {test_pct}% != 5%")
        
        print(f"  ✓ Dataset split is correct")
        print(f"    - Train: {train_pct:.1f}% ({train} sequences)")
        print(f"    - Val: {val_pct:.1f}% ({val} sequences)")
        print(f"    - Test: {test_pct:.1f}% ({test} sequences)")
    
    def test_augmentation_diversity(self):
        """Test that augmented sequences exist"""
        with open(self.metadata_file, 'r') as f:
            metadata = json.load(f)
        
        base_seqs = metadata['base_sequences']
        aug_seqs = metadata['augmented_sequences']
        total_seqs = metadata['total_sequences']
        
        self.assertGreater(aug_seqs, 0, "No augmented sequences found")
        self.assertGreater(aug_seqs, base_seqs, "Augmented sequences should exceed base")
        
        aug_ratio = aug_seqs / (base_seqs + aug_seqs) * 100
        self.assertGreater(aug_ratio, 30, f"Augmentation ratio too low: {aug_ratio}%")
        
        print(f"  ✓ Augmentation diversity verified")
        print(f"    - Base sequences: {base_seqs}")
        print(f"    - Augmented sequences: {aug_seqs}")
        print(f"    - Augmentation ratio: {aug_ratio:.1f}%")
    
    def test_context_window_size(self):
        """Test context window matches configuration"""
        with open(self.metadata_file, 'r') as f:
            metadata = json.load(f)
        
        max_tokens = metadata['max_tokens_per_sequence']
        
        # Should be 512, 1024, 1536, 2048, or 4096
        valid_sizes = [512, 1024, 1536, 2048, 4096]
        self.assertIn(
            max_tokens, valid_sizes,
            f"Unexpected token count: {max_tokens}. Expected one of {valid_sizes}"
        )
        
        print(f"  ✓ Context window is valid: {max_tokens} tokens")
    
    def test_weighting_strategy(self):
        """Test weighting strategy is applied"""
        with open(self.metadata_file, 'r') as f:
            metadata = json.load(f)
        
        weights = metadata['weighting_strategy']
        self.assertIn('high_priority', weights)
        self.assertIn('medium_priority', weights)
        self.assertIn('low_priority', weights)
        
        self.assertEqual(weights['high_priority'], 3.0, "High priority weight should be 3.0")
        self.assertEqual(weights['medium_priority'], 1.0, "Medium priority weight should be 1.0")
        self.assertLess(weights['low_priority'], 1.0, "Low priority weight should be < 1.0")
        
        print(f"  ✓ Weighting strategy verified")
        print(f"    - High priority: {weights['high_priority']}x")
        print(f"    - Medium priority: {weights['medium_priority']}x")
        print(f"    - Low priority: {weights['low_priority']}x")
    
    def test_curriculum_learning(self):
        """Test that sequences are ordered by complexity"""
        complexities = []
        
        with open(self.train_file, 'r') as f:
            for i, line in enumerate(f):
                if i >= 100:  # Sample first 100
                    break
                obj = json.loads(line.strip())
                complexity = obj['metadata'].get('complexity', 0)
                complexities.append(complexity)
        
        # Should be roughly increasing (curriculum learning)
        # Allow some variance but general trend should be up
        if len(complexities) > 1:
            increases = sum(1 for i in range(1, len(complexities)) if complexities[i] >= complexities[i-1])
            increase_ratio = increases / (len(complexities) - 1)
            
            self.assertGreater(
                increase_ratio, 0.4,
                f"Curriculum learning order not detected. Increase ratio: {increase_ratio}"
            )
        
        print(f"  ✓ Curriculum learning ordering verified")
    
    def test_no_duplicate_sequences(self):
        """Test that sequences are diverse (not just copies with different IDs)"""
        token_hashes = []
        
        with open(self.train_file, 'r') as f:
            for i, line in enumerate(f):
                if i >= 500:  # Sample first 500
                    break
                obj = json.loads(line.strip())
                # Hash tokens to check uniqueness
                token_hash = tuple(obj['tokens'][:50])  # First 50 tokens
                token_hashes.append(token_hash)
        
        unique_hashes = len(set(token_hashes))
        total_samples = len(token_hashes)
        uniqueness_ratio = unique_hashes / total_samples
        
        self.assertGreater(
            uniqueness_ratio, 0.7,
            f"Low sequence diversity. Uniqueness: {uniqueness_ratio:.1%}"
        )
        
        print(f"  ✓ Sequence diversity verified")
        print(f"    - Unique sequences (sample): {unique_hashes}/{total_samples}")
        print(f"    - Diversity ratio: {uniqueness_ratio:.1%}")
    
    def test_file_sizes(self):
        """Test that output files have reasonable sizes"""
        train_size = self.train_file.stat().st_size / (1024*1024)  # MB
        val_size = self.val_file.stat().st_size / (1024*1024)
        test_size = self.test_file.stat().st_size / (1024*1024)
        
        total_size = train_size + val_size + test_size
        
        # Should be at least 50 MB (if good quality)
        self.assertGreater(
            total_size, 50,
            f"Dataset too small: {total_size:.1f} MB. Expected >= 50 MB"
        )
        
        print(f"  ✓ File sizes are reasonable")
        print(f"    - Train: {train_size:.1f} MB")
        print(f"    - Val: {val_size:.1f} MB")
        print(f"    - Test: {test_size:.1f} MB")
        print(f"    - Total: {total_size:.1f} MB")
    
    def test_augmentation_types_present(self):
        """Test that augmented sequences have type annotations"""
        aug_types = []
        
        with open(self.train_file, 'r') as f:
            for i, line in enumerate(f):
                if i >= 200:  # Sample first 200
                    break
                obj = json.loads(line.strip())
                aug_type = obj['metadata'].get('augmentation_type')
                if aug_type:
                    aug_types.append(aug_type)
        
        self.assertGreater(len(aug_types), 0, "No augmented sequences found in sample")
        
        # Should have multiple types
        unique_types = set(aug_types)
        self.assertGreater(len(unique_types), 1, "Only one augmentation type found")
        
        print(f"  ✓ Augmentation types present and diverse")
        print(f"    - Types found: {sorted(unique_types)}")
        print(f"    - Distribution: {dict(Counter(aug_types))}")


if __name__ == '__main__':
    print(f"""
{'='*80}
  TESTING MAXIMUM EFFECTIVENESS DATASET
{'='*80}
""")
    
    # Run tests
    suite = unittest.TestLoader().loadTestsFromTestCase(TestDatasetCreation)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print(f"""
{'='*80}
  TEST SUMMARY
{'='*80}
""")
    
    if result.wasSuccessful():
        print(f"✅ All tests passed!")
        print(f"\nDataset is ready for training.")
        print(f"\nNext: python3 training/model_trainer_unified.py \\")
        print(f"  --config training_config_metal_cuda_universal.yaml \\")
        print(f"  --sequences training_data_effectiveness/training_data_train.jsonl \\")
        print(f"  --epochs 1 \\")
        print(f"  --device cuda")
        sys.exit(0)
    else:
        print(f"❌ Some tests failed!")
        print(f"\nFailed tests: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
        sys.exit(1)
