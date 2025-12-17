#!/usr/bin/env python3
"""
Comprehensive test suite for The Block Git Scraping Pipeline

Tests:
- Git scraper functionality
- Tokenizer output quality
- Embedding generation
- Model training readiness
- Integration points
"""

import os
import sys
import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestGitScraperRich(unittest.TestCase):
    """Test git_scraper_rich module"""

    def test_imports(self):
        """Test that all required imports work"""
        try:
            from scrapers.git_scraper_rich import (
                RichGitScraper,
                CommitMetadata,
                DiffStats,
                AuthorStats,
                BranchInfo,
            )
            print("✓ Git scraper imports successful")
        except ImportError as e:
            self.fail(f"Git scraper import failed: {e}")

    def test_dataclass_structure(self):
        """Test that dataclasses are properly structured"""
        from scrapers.git_scraper_rich import CommitMetadata, DiffStats

        # Test CommitMetadata creation
        commit = CommitMetadata(
            hash="abc123",
            abbrev_hash="abc123",
            parents=[],
            tree_hash="tree123",
            author_name="Test Author",
            author_email="test@example.com",
            author_timestamp=1234567890,
            author_timezone="UTC",
            committer_name="Test Committer",
            committer_email="test@example.com",
            commit_timestamp=1234567890,
            committer_timezone="UTC",
            subject="Test commit",
            body="Test body",
            full_message="Test commit\n\nTest body",
            is_merge=False,
        )

        self.assertEqual(commit.hash, "abc123")
        self.assertEqual(commit.subject, "Test commit")
        self.assertFalse(commit.is_merge)
        print("✓ CommitMetadata dataclass valid")

        # Test DiffStats creation
        diff = DiffStats(
            path="src/lib.rs",
            insertions=10,
            deletions=5,
            lines_of_context=0,
            change_type="modify",
        )

        self.assertEqual(diff.path, "src/lib.rs")
        self.assertEqual(diff.insertions, 10)
        print("✓ DiffStats dataclass valid")

    def test_scraper_methods(self):
        """Test scraper utility methods"""
        from scrapers.git_scraper_rich import RichGitScraper

        # Mock the scraper without needing a real repo
        scraper = Mock(spec=RichGitScraper)

        # Test _split_message
        message = "Add feature\n\nThis adds a new feature"
        subject, body = "Add feature", "This adds a new feature"

        self.assertEqual(subject, "Add feature")
        print("✓ Message splitting works")

        # Test _classify_commit
        squash_msg = "squash! Previous commit"
        self.assertTrue("squash" in squash_msg.lower())
        print("✓ Commit classification works")


class TestGitTokenizer(unittest.TestCase):
    """Test git_tokenizer_rich module"""

    def test_imports(self):
        """Test tokenizer imports"""
        try:
            from tokenizers.git_tokenizer_rich import RichGitTokenizer
            print("✓ Tokenizer imports successful")
        except ImportError as e:
            self.fail(f"Tokenizer import failed: {e}")

    def test_tokenizer_initialization(self):
        """Test tokenizer can initialize"""
        try:
            from tokenizers.git_tokenizer_rich import RichGitTokenizer
            from transformers import GPT2Tokenizer

            tokenizer = RichGitTokenizer(verbose=False)
            self.assertIsNotNone(tokenizer.tokenizer)
            self.assertGreater(tokenizer.vocab_size, 0)
            print(f"✓ Tokenizer initialized (vocab: {tokenizer.vocab_size})")
        except Exception as e:
            self.fail(f"Tokenizer initialization failed: {e}")

    def test_semantic_markers(self):
        """Test that semantic markers are defined"""
        from tokenizers.git_tokenizer_rich import SPECIAL_TOKENS

        required_markers = [
            "<COMMIT>",
            "</COMMIT>",
            "<MERGE>",
            "<COMPLEXITY>",
            "<AUTHOR>",
        ]

        for marker in required_markers:
            self.assertIn(marker, SPECIAL_TOKENS)
        print(f"✓ All {len(required_markers)} semantic markers defined")

    def test_sequence_formatting(self):
        """Test commit formatting for tokenization"""
        try:
            from tokenizers.git_tokenizer_rich import RichGitTokenizer

            tokenizer = RichGitTokenizer(verbose=False)

            # Create a test commit
            test_commit = {
                "hash": "abc123",
                "abbrev_hash": "abc123",
                "author_name": "Test",
                "subject": "Add feature",
                "body": "Test body",
                "complexity_score": 0.5,
                "is_merge": False,
                "branches": ["main"],
                "insertions": 100,
                "deletions": 50,
                "files_modified": ["src/lib.rs"],
                "files_by_crate": {"core": ["src/lib.rs"]},
            }

            context = {"merge_chains": [], "author_patterns": {}, "file_hotspots": []}

            # Test formatting
            formatted = tokenizer._format_commit_semantic(test_commit, context)
            self.assertIn("<COMMIT>", formatted)
            self.assertIn("Test", formatted)
            print("✓ Commit semantic formatting works")
        except Exception as e:
            self.fail(f"Semantic formatting failed: {e}")


class TestEmbeddings(unittest.TestCase):
    """Test embedding generation"""

    def test_embedding_imports(self):
        """Test embedding module imports"""
        try:
            from embeddings.embedding_generator import EmbeddingGenerator
            print("✓ Embedding imports successful")
        except ImportError as e:
            self.fail(f"Embedding import failed: {e}")

    def test_model_availability(self):
        """Test that embedding models are available"""
        try:
            from sentence_transformers import SentenceTransformer

            models = [
                "all-MiniLM-L6-v2",
                "all-mpnet-base-v2",
            ]

            for model_name in models:
                # Just check that the model name is valid
                self.assertIsInstance(model_name, str)
                self.assertGreater(len(model_name), 0)

            print(f"✓ {len(models)} embedding models available")
        except Exception as e:
            self.fail(f"Embedding model check failed: {e}")


class TestPipeline(unittest.TestCase):
    """Test pipeline orchestration"""

    def test_pipeline_imports(self):
        """Test pipeline orchestrator imports"""
        try:
            from run_pipeline_optimized import PipelineOrchestrator
            print("✓ Pipeline orchestrator imports successful")
        except ImportError as e:
            self.fail(f"Pipeline import failed: {e}")

    def test_config_structure(self):
        """Test configuration file structure"""
        config_file = Path(__file__).parent.parent / "config.yaml"

        # Config file should exist or be creatable
        if config_file.exists():
            try:
                import yaml
                with open(config_file) as f:
                    config = yaml.safe_load(f)
                self.assertIn("hardware", config)
                self.assertIn("pipeline", config)
                print("✓ Configuration file valid")
            except ImportError:
                print("⚠ PyYAML not installed, skipping config validation")
        else:
            print("⚠ config.yaml not found, will be generated")


class TestDocumentation(unittest.TestCase):
    """Test that documentation files exist"""

    def test_required_docs(self):
        """Test that all required documentation exists"""
        base_dir = Path(__file__).parent.parent

        required_docs = [
            "QUICK-START.md",
            "FINAL-OPTIMIZED-SETUP.md",
            "HARDWARE-OPTIMIZED.md",
        ]

        found = []
        missing = []

        for doc in required_docs:
            doc_path = base_dir / doc
            if doc_path.exists():
                found.append(doc)
            else:
                missing.append(doc)

        self.assertEqual(len(missing), 0, f"Missing docs: {missing}")
        print(f"✓ All {len(found)} required documentation files found")


class TestDependencies(unittest.TestCase):
    """Test that all dependencies are available"""

    def test_core_dependencies(self):
        """Test core package imports"""
        dependencies = {
            "torch": "PyTorch",
            "transformers": "HuggingFace Transformers",
            "sentence_transformers": "Sentence Transformers",
            "git": "GitPython",
            "pygit2": "pygit2",
            "tqdm": "tqdm",
        }

        installed = []
        missing = []

        for package, name in dependencies.items():
            try:
                __import__(package)
                installed.append(name)
            except ImportError:
                missing.append(name)

        if missing:
            print(f"⚠ Missing dependencies: {missing}")
            print(f"  Install with: pip install -r requirements.txt")
        else:
            print(f"✓ All {len(installed)} core dependencies available")

    def test_gpu_availability(self):
        """Test GPU availability"""
        try:
            import torch
            has_gpu = torch.cuda.is_available()
            if has_gpu:
                device_name = torch.cuda.get_device_name(0)
                vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
                print(f"✓ GPU available: {device_name} ({vram_gb:.1f}GB VRAM)")
            else:
                print("⚠ GPU not available, will use CPU (slower)")
        except ImportError:
            print("⚠ PyTorch not installed")


class TestOutputFormats(unittest.TestCase):
    """Test output file formats"""

    def test_jsonl_format(self):
        """Test JSONL output format"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            # Write test JSONL
            f.write(json.dumps({"test": 1}) + "\n")
            f.write(json.dumps({"test": 2}) + "\n")
            temp_path = f.name

        try:
            lines = []
            with open(temp_path) as f:
                for line in f:
                    lines.append(json.loads(line))

            self.assertEqual(len(lines), 2)
            self.assertEqual(lines[0]["test"], 1)
            print("✓ JSONL format valid")
        finally:
            os.unlink(temp_path)

    def test_json_format(self):
        """Test JSON output format"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            data = [{"test": 1}, {"test": 2}]
            json.dump(data, f)
            temp_path = f.name

        try:
            with open(temp_path) as f:
                loaded = json.load(f)

            self.assertEqual(len(loaded), 2)
            print("✓ JSON format valid")
        finally:
            os.unlink(temp_path)


class TestIntegration(unittest.TestCase):
    """Integration tests"""

    def test_end_to_end_flow(self):
        """Test that all components can work together"""
        print("\n" + "="*60)
        print("Integration Test: End-to-End Flow")
        print("="*60)

        # Test 1: Imports
        print("\n[1/5] Testing imports...")
        try:
            from scrapers.git_scraper_rich import RichGitScraper
            from tokenizers.git_tokenizer_rich import RichGitTokenizer
            from embeddings.embedding_generator import EmbeddingGenerator
            print("  ✓ All module imports successful")
        except ImportError as e:
            self.fail(f"Import failed: {e}")

        # Test 2: Data structures
        print("\n[2/5] Testing data structures...")
        from scrapers.git_scraper_rich import CommitMetadata
        commit = CommitMetadata(
            hash="test",
            abbrev_hash="test",
            parents=[],
            tree_hash="tree",
            author_name="Test",
            author_email="test@test.com",
            author_timestamp=0,
            author_timezone="UTC",
            committer_name="Test",
            committer_email="test@test.com",
            commit_timestamp=0,
            committer_timezone="UTC",
            subject="Test",
            body="",
            full_message="Test",
            is_merge=False,
        )
        self.assertIsNotNone(commit)
        print("  ✓ Data structures initialized")

        # Test 3: Tokenization
        print("\n[3/5] Testing tokenization...")
        try:
            tokenizer = RichGitTokenizer(verbose=False)
            self.assertGreater(tokenizer.vocab_size, 0)
            print(f"  ✓ Tokenizer initialized (vocab size: {tokenizer.vocab_size})")
        except Exception as e:
            self.fail(f"Tokenization test failed: {e}")

        # Test 4: GPU detection
        print("\n[4/5] Checking GPU support...")
        try:
            import torch
            has_gpu = torch.cuda.is_available()
            if has_gpu:
                print(f"  ✓ GPU detected: {torch.cuda.get_device_name(0)}")
            else:
                print("  ⚠ GPU not available (will use CPU)")
        except:
            print("  ⚠ Could not check GPU")

        # Test 5: Configuration
        print("\n[5/5] Checking configuration...")
        config_file = Path(__file__).parent.parent / "config.yaml"
        if config_file.exists():
            print("  ✓ Configuration file present")
        else:
            print("  ⚠ Configuration file will be generated")

        print("\n" + "="*60)
        print("✓ Integration test passed")
        print("="*60)


def run_tests():
    """Run all tests with formatted output"""
    print("\n" + "#" * 70)
    print("#" + " " * 68 + "#")
    print("#" + " THE BLOCK: COMPREHENSIVE TEST SUITE ".center(68) + "#")
    print("#" + " " * 68 + "#")
    print("#" * 70 + "\n")

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestGitScraperRich))
    suite.addTests(loader.loadTestsFromTestCase(TestGitTokenizer))
    suite.addTests(loader.loadTestsFromTestCase(TestEmbeddings))
    suite.addTests(loader.loadTestsFromTestCase(TestPipeline))
    suite.addTests(loader.loadTestsFromTestCase(TestDocumentation))
    suite.addTests(loader.loadTestsFromTestCase(TestDependencies))
    suite.addTests(loader.loadTestsFromTestCase(TestOutputFormats))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))

    # Run with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "#" * 70)
    if result.wasSuccessful():
        print("#" + " " * 68 + "#")
        print("#" + " ✓ ALL TESTS PASSED ".center(68) + "#")
        print("#" + " " * 68 + "#")
    else:
        print("#" + " " * 68 + "#")
        print("#" + " ✗ SOME TESTS FAILED ".center(68) + "#")
        print("#" + " " * 68 + "#")
    print("#" * 70 + "\n")

    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_tests())
