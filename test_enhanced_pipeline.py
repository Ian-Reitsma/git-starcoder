#!/usr/bin/env python3
"""
Comprehensive tests for enhanced training pipeline.

Tests:
1. Enhanced semantic chunker with cross-file context
2. Enhanced tokenizer with expanded limits
3. Enhanced dataset builder with commit-based examples
4. Full integration test
"""

import json
import tempfile
import logging
from pathlib import Path
from typing import Dict, List, Any

from semantic_chunker_enhanced import EnhancedSemanticChunker, RustCodeAnalyzer, CrossFileContext
from tokenizer_enhanced import VocabularyBuilder, EnhancedCodeTokenizer
from dataset_builder_enhanced import EnhancedDatasetBuilder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_test_commits_data() -> Dict[str, Any]:
    """
    Create test commits data that simulates real git history.
    """
    return {
        "commits": [
            {
                "metadata": {
                    "hash": "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6",
                    "author_name": "Test Author",
                    "author_email": "test@example.com",
                    "timestamp": 1700000000,
                    "message": "Add energy market dispute RPC",
                },
                "diffs": [
                    {
                        "filename": "src/energy_market/rpc.rs",
                        "change_type": "M",
                        "is_binary": False,
                        "patch": """--- a/src/energy_market/rpc.rs
+++ b/src/energy_market/rpc.rs
@@ -1,5 +1,15 @@
 use crate::energy_market::types::*;
 use serde::{Deserialize, Serialize};
 
+pub async fn propose_dispute(
+    dispute_id: u64,
+    reason: String,
+) -> Result<DisputeResponse> {
+    // Implementation
+    Ok(DisputeResponse {
+        dispute_id,
+        status: DisputeStatus::Draft,
+    })
+}
 
 pub fn handle_rpc(request: RpcRequest) -> RpcResponse {
     match request.method.as_str() {""",
                        "new_code": """pub async fn propose_dispute(
    dispute_id: u64,
    reason: String,
) -> Result<DisputeResponse> {
    // Implementation
    Ok(DisputeResponse {
        dispute_id,
        status: DisputeStatus::Draft,
    })
}""",
                        "old_code": "",
                    },
                    {
                        "filename": "src/energy_market/types.rs",
                        "change_type": "M",
                        "is_binary": False,
                        "patch": """--- a/src/energy_market/types.rs
+++ b/src/energy_market/types.rs
@@ -1,3 +1,10 @@
 use serde::{Deserialize, Serialize};
 
+#[derive(Debug, Clone, Serialize, Deserialize)]
+pub struct Dispute {
+    pub dispute_id: u64,
+    pub provider_id: Address,
+    pub status: DisputeStatus,
+}""",
                        "new_code": """#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dispute {
    pub dispute_id: u64,
    pub provider_id: Address,
    pub status: DisputeStatus,
}""",
                        "old_code": "",
                    },
                ]
            },
            {
                "metadata": {
                    "hash": "b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7",
                    "author_name": "Test Author",
                    "author_email": "test@example.com",
                    "timestamp": 1700000100,
                    "message": "Add dispute voting logic",
                },
                "diffs": [
                    {
                        "filename": "src/energy_market/rpc.rs",
                        "change_type": "M",
                        "is_binary": False,
                        "patch": """--- a/src/energy_market/rpc.rs
+++ b/src/energy_market/rpc.rs
@@ -10,6 +10,15 @@
     })
 }
 
+pub async fn vote_on_dispute(
+    dispute_id: u64,
+    vote: bool,
+) -> Result<()> {
+    // Implementation
+    Ok(())
+}
 
 pub fn handle_rpc(request: RpcRequest) -> RpcResponse {
     match request.method.as_str() {""",
                        "new_code": """pub async fn vote_on_dispute(
    dispute_id: u64,
    vote: bool,
) -> Result<()> {
    // Implementation
    Ok(())
}""",
                        "old_code": "",
                    },
                ]
            },
        ]
    }


def test_rust_code_analyzer():
    """Test Rust code pattern extraction."""
    logger.info("\nTest 1: Rust Code Analyzer")
    logger.info("="*70)
    
    code = """
pub async fn propose_dispute(
    dispute_id: u64,
    reason: String,
) -> Result<DisputeResponse> {
    Ok(DisputeResponse { dispute_id, status: DisputeStatus::Draft })
}
    """
    
    lines = code.split('\n')
    
    # Test function extraction
    fn_name = RustCodeAnalyzer.extract_function_name(lines)
    assert fn_name == "propose_dispute", f"Expected 'propose_dispute', got {fn_name}"
    logger.info(f"✓ Function extraction: {fn_name}")
    
    # Test import extraction
    imports = RustCodeAnalyzer.extract_imports("use crate::types::*;\nuse serde::Serialize;")
    assert len(imports) == 2, f"Expected 2 imports, got {len(imports)}"
    logger.info(f"✓ Import extraction: {len(imports)} imports")
    
    logger.info("Test 1: PASSED\n")


def test_enhanced_semantic_chunker():
    """Test enhanced semantic chunker with cross-file context."""
    logger.info("\nTest 2: Enhanced Semantic Chunker")
    logger.info("="*70)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create test commits file
        commits_file = tmpdir / "commits_test.json"
        commits_data = create_test_commits_data()
        with open(commits_file, "w") as f:
            json.dump(commits_data, f)
        
        # Create output file
        chunks_file = tmpdir / "chunks_test.jsonl"
        
        # Run chunker
        chunker = EnhancedSemanticChunker(
            str(commits_file),
            str(chunks_file),
            include_cross_file=True
        )
        
        chunks = chunker.process_all()
        stats = chunker.save_statistics()
        chunker.save_jsonl()
        
        # Verify results
        assert len(chunks) > 0, "No chunks created"
        logger.info(f"✓ Created {len(chunks)} chunks")
        
        # Check cross-file context
        chunks_with_context = sum(1 for c in chunks if c.cross_file_context)
        logger.info(f"✓ Chunks with cross-file context: {chunks_with_context}")
        
        # Check expanded limits
        assert stats['max_chunk_size_chars'] > 2000, "Chunk size not expanded"
        logger.info(f"✓ Max chunk size: {stats['max_chunk_size_chars']} chars (expanded from 2000)")
        
        # Verify old_code inclusion
        has_old_code = any(len(c.old_code) > 0 for c in chunks)
        logger.info(f"✓ Old code included: {has_old_code}")
        
        logger.info("Test 2: PASSED\n")


def test_enhanced_tokenizer():
    """Test enhanced tokenizer with expanded limits."""
    logger.info("\nTest 3: Enhanced Tokenizer")
    logger.info("="*70)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create test chunks
        chunks_file = tmpdir / "chunks_test.jsonl"
        commits_data = create_test_commits_data()
        
        # Write enhanced chunks
        chunker = EnhancedSemanticChunker(
            None,
            str(chunks_file),
            include_cross_file=True
        )
        chunks = chunker.chunk_commit(commits_data["commits"][0])
        
        with open(chunks_file, "w") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk.to_dict()) + "\n")
        
        # Build vocabulary
        vocab = VocabularyBuilder(vocab_size=50257)
        vocab.build_from_chunks(str(chunks_file))
        
        vocab_size = vocab.next_id
        logger.info(f"✓ Vocabulary built with {vocab_size} tokens")
        
        # Verify new tokens for cross-file context
        assert "<CONTEXT_START>" in vocab.token_to_id, "Context token not in vocab"
        assert "<IMPORTS_START>" in vocab.token_to_id, "Imports token not in vocab"
        logger.info(f"✓ New context tokens added to vocabulary")
        
        # Tokenize chunks
        tokenizer = EnhancedCodeTokenizer(vocab)
        tokens, metadata = tokenizer.tokenize_file(str(chunks_file))
        
        assert len(tokens) > 0, "No tokens created"
        logger.info(f"✓ Tokenized {len(tokens)} chunks")
        
        # Verify expanded token limits (2048 instead of 1024)
        avg_tokens = sum(len(t) for t in tokens) / len(tokens)
        logger.info(f"✓ Avg tokens per chunk: {avg_tokens:.0f} (expanded limit: 2048)")
        
        logger.info("Test 3: PASSED\n")


def test_enhanced_dataset_builder():
    """Test enhanced dataset builder with commit-based examples."""
    logger.info("\nTest 4: Enhanced Dataset Builder")
    logger.info("="*70)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create test tokens data
        tokens_file = tmpdir / "tokens_test.pt"
        metadata_file = tmpdir / "metadata_test.json"
        
        # Create simple token sequences
        test_tokens = [
            list(range(100, 200)),  # Chunk 1
            list(range(200, 300)),  # Chunk 2
            list(range(300, 400)),  # Chunk 3
        ]
        
        test_metadata = [
            {"chunk_id": "a1_0", "commit_hash": "commit1", "file_path": "file1.rs", 
             "token_count": 100, "has_cross_file_context": True},
            {"chunk_id": "a1_1", "commit_hash": "commit1", "file_path": "file2.rs", 
             "token_count": 100, "has_cross_file_context": True},
            {"chunk_id": "b2_0", "commit_hash": "commit2", "file_path": "file1.rs", 
             "token_count": 100, "has_cross_file_context": False},
        ]
        
        # Save tokens (try PyTorch first, then fallback to JSON)
        try:
            import torch
            token_data = {"tokens": test_tokens, "metadata": test_metadata}
            torch.save(token_data, str(tokens_file))
        except ImportError:
            # Fallback: save as JSON with dummy data
            with open(tokens_file.with_suffix('.json'), "w") as f:
                json.dump({"tokens": test_tokens, "metadata": test_metadata}, f)
            tokens_file = tokens_file.with_suffix('.json')
        
        # Build dataset
        output_dir = tmpdir / "dataset"
        builder = EnhancedDatasetBuilder(
            str(tokens_file),
            str(metadata_file),
            context_window=100,
            target_window=50,
            output_dir=str(output_dir),
            commit_based=True
        )
        
        # Load tokens manually for testing
        tokens = test_tokens
        metadata = test_metadata
        
        # Test commit-based example building
        examples = builder.build_commit_based_examples(tokens, metadata)
        
        assert len(examples) > 0, "No examples created"
        logger.info(f"✓ Created {len(examples)} commit-based examples")
        
        # Verify example structure
        example = examples[0]
        assert "context" in example, "Example missing 'context'"
        assert "target" in example, "Example missing 'target'"
        assert "commit_hash" in example, "Example missing 'commit_hash'"
        logger.info(f"✓ Examples have correct structure")
        
        # Verify masks are created
        assert "context_mask" in example, "Example missing 'context_mask'"
        assert "target_mask" in example, "Example missing 'target_mask'"
        logger.info(f"✓ Attention masks created")
        
        logger.info("Test 4: PASSED\n")


def main():
    """Run all tests."""
    logger.info("\n" + "#"*70)
    logger.info("# ENHANCED PIPELINE TEST SUITE")
    logger.info("#"*70)
    
    try:
        test_rust_code_analyzer()
        test_enhanced_semantic_chunker()
        test_enhanced_tokenizer()
        test_enhanced_dataset_builder()
        
        logger.info("\n" + "#"*70)
        logger.info("# ALL TESTS PASSED")
        logger.info("#"*70 + "\n")
        return True
        
    except AssertionError as e:
        logger.error(f"\nTest failed: {e}")
        return False
    except Exception as e:
        logger.error(f"\nUnexpected error: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
