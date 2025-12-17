# Implementation Notes: Git Scrape + Tokenization + Training

## Overview

This document provides deep technical details for each component of the system.

---

## 1. Git Scraper (`git_scraper.py`)

### Design Principles

**Why not just `git log`?**

Standard `git log` loses critical information:

```
# Standard output (loses context)
git log --oneline
# Output: abc1234 Add energy market RPC

# We extract:
# - Full diffs for every file
# - Branch lineage (which branches contained this commit)
# - Merge decisions (why was this merged)
# - Author patterns (who writes what)
# - Timestamp progression (when was it developed)
```

### Key Methods

#### `get_all_commits_chronological()`

```python
git rev-list --all --date-order
```

This command:
- `--all`: All branches, tags, refs (not just current branch)
- `--date-order`: Chronological order (oldest first), helping model learn progression

#### `get_commit_metadata()`

Extracts:

```python
# git show -s --format=...
{
    "hash": "abc123def...",  # Full hash for unique identification
    "author": "Ian Reitsma",  # Learn author patterns
    "timestamp": "2024-01-15T10:30:00Z",  # Temporal context
    "message": "Add energy market RPC",  # Intent
    "parents": ["parent1", "parent2"],  # parent2 = merge commit
    "is_merge": true,  # Important for understanding coordination
    "insertions": 250,
    "deletions": 50
}
```

#### `get_commit_diffs()`

For each file in commit:

```python
# git show abc123:file.rs (full file content)
# git diff abc123~1 abc123 -- file.rs (patch)
{
    "filename": "src/energy_market/lib.rs",
    "change_type": "M",  # Added, Modified, Deleted, Renamed
    "additions": 145,
    "deletions": 32,
    "patch": "@@ -10,5 +10,8 @@...",  # Unified diff format
    "is_binary": false
}
```

The patch is in standard unified diff format:
```
@@ -10,5 +10,8 @@
 old line
 old line
-removed line
+added line
+added line
 old line
```

### Data Structure

Output JSON structure (commits.json):

```json
{
  "metadata": {
    "scrape_timestamp": "2024-12-09T...",
    "total_commits": 2847,
    "total_authors": 3,
    "merge_commits": 127,
    "total_files_touched": 412,
    "unique_branches": 34,
    "branches": ["main", "develop", "feature/energy-market", ...],
    "total_insertions": 450000,
    "total_deletions": 120000,
    "date_range": {
      "first": "2023-01-15T...",
      "last": "2024-12-09T..."
    }
  },
  "commits": [
    {
      "metadata": {
        "hash": "abc123...",
        "abbrev_hash": "abc123",
        "author": "Ian Reitsma",
        "author_email": "ian@example.com",
        "timestamp": "2024-01-15T10:30:00Z",
        "timestamp_unix": 1705320600,
        "message": "Add energy market RPC endpoints",
        "message_body": "Longer description...",
        "parents": ["parent1hash"],
        "is_merge": false,
        "files_changed": 3,
        "insertions": 250,
        "deletions": 50
      },
      "branches": [
        {"branch_name": "main", "is_current": false, "is_remote": false}
      ],
      "diffs": [
        {
          "filename": "src/energy_market/lib.rs",
          "change_type": "M",
          "additions": 145,
          "deletions": 32,
          "patch": "@@ -10,5 +10,8 @@...",
          "is_binary": false
        }
      ],
      "tags": ["v0.1.0"],
      "commit_number": 0,
      "context": {
        "is_merge": false,
        "has_tags": true,
        "file_count": 3,
        "total_lines_changed": 300
      }
    }
  ]
}
```

### Performance Optimizations

1. **Batch Processing**: Commits extracted in order, no sorting overhead
2. **Diff Caching**: Git diffs buffered to minimize subprocess calls
3. **Lazy Loading**: Branch info only fetched when needed
4. **Error Resilience**: Malformed commits logged but don't crash pipeline

### Known Limitations

1. **Large Repos**: Takes 2-5 min for 2500+ commits. Timeout at 30 sec per commit.
2. **Binary Files**: Diffs skipped (detected by `git show` output)
3. **Merge Commits**: Diffs only show changes relative to first parent
4. **Large Files**: Diffs truncated at 10KB

---

## 2. Semantic Chunker (`semantic_chunker.py`)

### Problem Statement

**Raw diffs are too granular for LLM learning:**

```python
# Without chunking:
Tokens: [<FILE>, <src/energy_market/lib.rs>, <ADD>, <fn>, <energy_dispute_handler>, ...]
# Model learns: "files have names, diffs have functions"

# With semantic chunking:
Chunk1: [<COMMIT_START>, <FILE>, <MODULE:energy_market>, <CHANGE:new_function>, 
         <fn>, <energy_dispute_handler>, ..., <COMMIT_END>]
Chunk2: [<COMMIT_START>, <FILE>, <MODULE:energy_market>, <CHANGE:new_test>,
         <#[test]>, <fn>, <test_energy_dispute>, ..., <COMMIT_END>]
# Model learns: "functions come with tests", "author structure matters"
```

### Analysis Strategy

For **Rust files** (RustCodeAnalyzer):

```
Pattern                         Signal
---------                       ------
^\s*fn\s+([a-z_]+)             New/modified function
^\s*struct\s+([A-Z]+)          New/modified struct
^\s*#\[test\]                  Test code
^\s*impl\s+([A-Z]+)            Implementation block
^\s*trait\s+([A-Z]+)           Trait definition
fix|bug|error|panic            Bug fix (by keyword)
cache|lazy|inline|perf         Optimization (by keyword)
```

For **Config files** (YAML/TOML):
- Mark as `CONFIGURATION` type
- Track which sections changed

For **Documentation** (Markdown):
- Mark as `DOCUMENTATION` type
- Extract section headers as context

### Chunk Structure

```python
CodeChunk {
    chunk_id: "abc123_0",  # commit_hash + index
    commit_hash: "abc123...",
    file_path: "src/energy_market/lib.rs",
    file_type: "rs",
    
    change_type: ChangeType.NEW_FUNCTION,  # Inferred from code analysis
    change_category: "M",  # Git category (A/M/D/R)
    
    old_code: "fn old_handler() { ... }",  # Code before
    new_code: "fn energy_dispute_handler() { ... }",  # Code after
    patch: "@@ -10,5 +10,8 @@...",  # Raw diff
    
    function_name: "energy_dispute_handler",  # Extracted from new_code
    module_path: "energy_market::disputes",  # From file path
    line_range: (10, 45),  # Approximate
    
    additions: 35,
    deletions: 5,
    
    commit_meta {...},  # Author, timestamp, message
    related_files: ["src/energy_market/mod.rs", "src/cli/..."]
}
```

### Output Format (JSONL)

One chunk per line:

```json
{"chunk_id": "abc123_0", "commit_hash": "...", ...}
{"chunk_id": "abc123_1", "commit_hash": "...", ...}
{"chunk_id": "def456_0", "commit_hash": "...", ...}
```

Benefit: Can stream process without loading all in memory.

### Key Insights

1. **Function boundaries** preserve semantic units
2. **Test code separation** teaches model "functions come with tests"
3. **Module context** helps model understand architecture
4. **Change type inference** captures intent (refactor vs bug fix)
5. **Related files** show cross-file coordination

---

## 3. Tokenizer (`tokenizer.py`)

### Tokenization Strategy

**Key insight:** Standard BPE loses code structure.

```python
# Standard BPE:
Tokens: [3, 142, 5, 200, 15, ...]  # Raw numbers, no structure

# Our hybrid tokenizer:
Tokens: [<COMMIT_START>, <FILE:src/lib.rs>, <CHANGE:new_function>,
         <AUTHOR:ian@example.com>, <TIMESTAMP:2024-01-15>,
         <MODULE:energy_market>, <CODE_START>,
         fn, energy_dispute_handler, (, &, mut, self, ...,
         <CODE_END>, <COMMIT_END>]
```

Model learns:
1. Structural patterns (commits, files, change types)
2. Authorship patterns (who makes what changes)
3. Temporal patterns (when changes happen)
4. Architectural patterns (module structure)

### Special Token Categories

1. **Structural Boundaries**
   - `<COMMIT_START>/<COMMIT_END>` - Commit boundaries
   - `<FILE_START>/<FILE_END>` - File boundaries
   - `<CODE_START>/<CODE_END>` - Code section boundaries

2. **Context Tokens** (dynamic)
   - `<BRANCH:main>` - Branch information
   - `<FILE:src/energy_market/lib.rs>` - File path
   - `<AUTHOR:ian@example.com>` - Author attribution
   - `<MODULE:energy_market>` - Module hierarchy
   - `<CHANGE:new_function>` - Change type
   - `<TIMESTAMP:2024-01>` - Time bucket (year-month)

3. **Language Tokens** (reserved)
   - `<KW:fn>`, `<KW:struct>`, etc. - Rust keywords
   - `<MACRO:println!>` - Rust macros

4. **Padding/Special**
   - `<PAD>` - Padding for batch processing
   - `<UNK>` - Unknown token
   - `<NEWLINE>` - Line breaks (in expanded version)

### Vocabulary Building

```python
vocab = VocabularyBuilder(vocab_size=50257)
vocab.build_from_chunks("chunks.jsonl")
```

Process:
1. Scan all chunks, extract tokens
2. Count frequency of each token
3. Sort by frequency
4. Reserve first ~5000 for special tokens
5. Add remaining up to vocab_size

Result: Vocabulary JSON

```json
{
  "token_to_id": {
    "<PAD>": 0,
    "<UNK>": 1,
    "<COMMIT_START>": 2,
    "<COMMIT_END>": 3,
    ...,
    "fn": 5015,
    "struct": 5016,
    "pub": 5017,
    "energy": 5200,
    "dispute": 5201,
    ...
  },
  "id_to_token": {reverse mapping},
  "vocab_size": 50257
}
```

### Per-Chunk Tokenization

For each chunk:

```python
tokens = [
    TOKEN_ID["<COMMIT_START>"],
    TOKEN_ID[f"<FILE:{chunk['file_path']}>"],
    TOKEN_ID[f"<CHANGE:{chunk['change_type']}>"],
    TOKEN_ID[f"<AUTHOR:{chunk['commit_metadata']['author_email']}>"],
    TOKEN_ID[f"<TIMESTAMP:{chunk['commit_metadata']['timestamp'][:7]}>"],
    TOKEN_ID[f"<MODULE:{chunk['module_path']}>"],
    TOKEN_ID["<CODE_START>"],
    # Tokenize actual code
    *tokenize_code(chunk['new_code']),
    TOKEN_ID["<CODE_END>"],
    TOKEN_ID["<COMMIT_END>"]
]
```

### Code Tokenization

```python
pattern = r'[a-zA-Z_][a-zA-Z0-9_]*|[0-9]+|[+\-*/%=<>!&|^~?:|;,.(\[\]{}]'
```

Tokenizes:
- `identifier` → stays together
- `123` → number
- `+` → operator
- `(` → bracket

Result: Semantically meaningful tokens preserving structure.

### Output Format

PyTorch tensor:

```python
token_data = {
    "tokens": [  # List of token sequences
        [2, 5010, 5011, 5012, ..., 3],  # Chunk 1 tokenized
        [2, 5020, 5021, ..., 3],  # Chunk 2 tokenized
        ...
    ],
    "metadata": [  # Parallel metadata
        {"chunk_id": "abc123_0", "commit_hash": "...", ...},
        {"chunk_id": "abc123_1", ...},
        ...
    ]
}
torch.save(token_data, "tokens.pt")
```

---

## 4. Dataset Builder (`dataset_builder.py`)

### Problem: Creating Meaningful Training Pairs

**Goal:** Teach model to predict next change given repository state.

```python
# Naive approach:
Context: [token1, token2, ..., token2048]
Target: [token2049, token2050, ..., token2304]
# Problem: Model doesn't learn structure, just predicts next tokens

# Better approach:
Context: [last 10 commits, last 5 branches, current file]
Target: [next 3 commits, likely files to change]
# Problem: Too specific, hard to generalize

# Our approach:
Context: [chronological sequence of commits up to point N]
Target: [next commit(s)]
# Benefit: Model learns architectural patterns from full progression
```

### Chronological Ordering (Critical!)

```python
def build_chronological_sequence(self):
    # Sort chunks by commit timestamp (NOT by hash)
    ordered_chunks = sorted(
        all_chunks,
        key=lambda c: commits_by_hash[c['commit_hash']]['timestamp_unix']
    )
    # This preserves temporal structure
    # Model learns: "first we added module X, then tests, then optimized"
```

### Context-Target Pairs

```python
def create_context_target_pairs(self, token_sequence, stride=128):
    # Sliding window approach
    for start_idx in range(0, len(token_sequence) - context_window, stride):
        context = token_sequence[start_idx : start_idx + context_window]
        target = token_sequence[start_idx + context_window : start_idx + context_window + target_window]
        
        yield {
            "context": context,
            "target": target,
            "context_mask": [1 if token != PAD else 0 for token in context],
            "target_mask": [1 if token != PAD else 0 for token in target]
        }
```

Example with small windows:

```
token_sequence: [2, 100, 101, ..., 3, 2, 200, 201, ..., 3, ...]
                 \________________commit 1__________________/ \__commit 2__/

Context window = 50, Target window = 10, Stride = 10

Example 1:
  context: [2, 100, 101, ..., (50 tokens total)]
  target: [(next 10 tokens)]

Example 2 (stride=10):
  context: [100, 101, 102, ..., (50 tokens starting from idx 10)]
  target: [(next 10 tokens)]
```

### Attention Masks

Critical for variable-length sequences:

```python
def _create_mask(self, seq, target_len):
    mask = [1] * len(seq)  # Real tokens
    if len(seq) < target_len:
        mask += [0] * (target_len - len(seq))  # Padding
    return mask
```

During training:
- Model only computes loss on positions with mask=1
- Padding doesn't affect gradient updates
- Allows variable-length sequences in same batch

### Data Splits (Temporal!)

**CRITICAL:** Preserve temporal order to avoid information leakage.

```python
def split_data(self, examples, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    # NO SHUFFLING!
    # Chronologically split
    total = len(examples)
    train_end = int(total * 0.7)
    val_end = train_end + int(total * 0.15)
    
    return (
        examples[:train_end],      # First 70% (oldest commits)
        examples[train_end:val_end],  # Middle 15% (recent commits)
        examples[val_end:]          # Last 15% (newest commits)
    )
```

Benefit: Model trained on old commits, validated on recent ones = true generalization test.

### PyTorch Dataset Format

```python
token_data = {
    "contexts": torch.tensor(  # Shape: (N, context_window)
        [[2, 100, 101, ..., 0],
         [2, 200, 201, ..., 0],
         ...],
        dtype=torch.long
    ),
    "targets": torch.tensor(  # Shape: (N, target_window)
        [[102, 103, 104, ..., 0],
         [201, 202, 203, ..., 0],
         ...],
        dtype=torch.long
    ),
    "context_masks": torch.tensor(  # Shape: (N, context_window)
        [[1, 1, 1, ..., 0],
         [1, 1, 1, ..., 0],
         ...],
        dtype=torch.bool
    ),
    "target_masks": torch.tensor(  # Shape: (N, target_window)
        [[1, 1, 1, ..., 0],
         [1, 1, 1, ..., 0],
         ...],
        dtype=torch.bool
    )
}
torch.save(token_data, "training_data_train.pt")
```

---

## 5. Model Training (`train_model.py`)

### Training Objectives

**Standard objective:** Minimize next-token prediction loss.

```python
loss = cross_entropy_loss(logits, target_tokens, weight=target_mask)
```

**Our objective:** Model learns your codebase patterns.

```
Given: [commit1, commit2, ..., commitN]
Predict: [commitN+1 structure, likely files, change types]
```

### Hardware Constraints

RTX 2060 Super (8GB VRAM):

```
llama2-7b (13GB unquantized) → Q4_K_M (4-bit) → 3.5GB
Activations + batch: 4-5GB
Total: ~7-8GB ✓ Fits

llama2-70b (130GB unquantized) → Q4_K_M → 20GB
Activations + batch: 10GB
Total: ~30GB ✗ Doesn't fit
```

Optimizations for 8GB VRAM:
1. **4-bit quantization**: Reduces model size 4x
2. **Gradient checkpointing**: Trade compute for memory
3. **Small batch size** (2-4)
4. **Flash attention**: Reduce attention VRAM
5. **LoRA fine-tuning**: Only train small adapter (vs full model)

### Training Loop Pseudocode

```python
def train_epoch(model, train_loader, optimizer, scheduler):
    total_loss = 0
    
    for batch_idx, batch in enumerate(train_loader):
        contexts = batch["contexts"].to(device)  # (batch, 2048)
        targets = batch["targets"].to(device)  # (batch, 256)
        masks = batch["target_masks"].to(device)  # (batch, 256)
        
        # Forward pass
        logits = model(contexts)  # (batch, 256, vocab_size)
        
        # Compute loss (only on non-padding tokens)
        loss = cross_entropy_loss(
            logits.view(-1, vocab_size),  # Flatten: (batch*256, vocab_size)
            targets.view(-1),  # Flatten: (batch*256)
            reduction='none'
        ).view(batch_idx, target_window)
        
        # Apply mask
        loss = (loss * masks).sum() / masks.sum()
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)
```

### Evaluation Metrics

1. **Perplexity**: Standard LLM metric
   ```
   perplexity = exp(loss)
   ```

2. **Token Accuracy**: % of correctly predicted tokens
   ```
   accuracy = (predictions == targets).sum() / masks.sum()
   ```

3. **BLEU/ROUGE**: Sequence-level metrics (more relevant for code)
   ```
   BLEU-4: Compare generated tokens against reference
   ```

### Hyperparameter Tuning

Recommended starting values:

```yaml
learning_rate: 5.0e-5      # Standard for fine-tuning
warmup_steps: 500          # 5% of training
batch_size: 4              # Max for 8GB VRAM
max_grad_norm: 1.0         # Prevent exploding gradients
weight_decay: 0.01         # L2 regularization
scheduler: cosine          # Annealing (better than constant)
epochs: 3                  # Usually sufficient for code patterns
```

### Expected Training Time

```
Dataset: ~10k training examples
Batch size: 4
Epochs: 3
Iterations: 10000 / 4 * 3 = 7,500

LLaMA2-7b on RTX 2060:
  ~1.5 sec per iteration
  7,500 * 1.5s = 11,250s ≈ 3 hours per epoch
  3 epochs = 9 hours
  + validation: +2 hours
  Total: 10-12 hours
```

---

## 6. Pipeline Integration

### Execution Flow

```
python pipeline.py --config config.yaml
  ↓
[1] git_scraper.py
     Inputs: repo_path
     Output: commits.json (500MB)
     Logs: INFO progress every 100 commits
  ↓
[2] semantic_chunker.py
     Inputs: commits.json
     Output: chunks.jsonl (800MB), chunking_stats.json
     Logs: Chunk type distribution
  ↓
[3] tokenizer.py
     Inputs: chunks.jsonl
     Output: vocab.json (2MB), tokens.pt (1.2GB)
     Logs: Vocabulary statistics
  ↓
[4] dataset_builder.py
     Inputs: vocab.json, chunks.jsonl, commits.json
     Output: training_data_*.pt (1.5GB total), dataset_stats.json
     Logs: Data split breakdown
  ↓
[5] train_model.py
     Inputs: vocab.json, training_data_*.pt
     Output: model_weights.pt (13GB), training_logs/
     Logs: Loss curves, validation metrics
```

### Error Handling

Each stage can fail independently. Pipeline is resilient:

```python
if continue_on_error:
    # Next stage tries with available inputs
    # E.g., if git_scraper fails, chunker skips
else:
    # Stop at first failure
    sys.exit(1)
```

### Checkpoint/Resume

Not yet implemented but critical for long training:

```python
# Resume from checkpoint
python train_model.py --resume outputs/checkpoints/epoch2.pt

# Saves every N steps:
torch.save({
    'epoch': epoch,
    'model_state': model.state_dict(),
    'optimizer_state': optimizer.state_dict(),
    'loss': loss
}, 'checkpoints/epoch2.pt')
```

---

## 7. Performance Tuning

### For Your Hardware

**RTX 2060 Super (8GB VRAM):**

```yaml
Batch size: 2-4 (test with 2 first)
Context window: 2048 (standard)
Target window: 256 (prediction length)
Gradient accumulation: 2 (effective batch = 4-8)
Mixed precision: false (not critical for 7B model)
```

**To see VRAM usage:**
```python
import torch
print(torch.cuda.memory_allocated())  # Currently used
print(torch.cuda.max_memory_allocated())  # Peak usage
```

### CPU Optimization

Ryzen 5 3800X (8 cores):

```yaml
num_workers: 4  # DataLoader workers
scraper_batch_size: 50  # Process commits in batches
```

---

## 8. Debugging

### Enable Verbose Logging

```yaml
logging:
  level: DEBUG  # Instead of INFO
```

Or CLI:
```bash
python git_scraper.py --repo . --verbose
```

### Sample Data

```python
# After git_scraper
with open("outputs/commits.json") as f:
    data = json.load(f)
    print(data['commits'][0])  # See first commit

# After semantic_chunker
with open("outputs/chunks.jsonl") as f:
    for i, line in enumerate(f):
        if i >= 3:
            break
        chunk = json.loads(line)
        print(f"Chunk {i}: {chunk['function_name']} in {chunk['file_path']}")

# After tokenizer
import torch
data = torch.load("outputs/tokens.pt")
print(f"Token sequences: {len(data['tokens'])}")
print(f"First sequence length: {len(data['tokens'][0])}")
```

### Unit Tests

Each script has embedded error handling. For custom testing:

```bash
# Test just git scraper
python git_scraper.py --repo . --output test_commits.json

# Test semantic chunker on first 100 commits
head -n 100 test_commits.json > test_subset.json
python semantic_chunker.py --commits test_subset.json --output test_chunks.jsonl
```

---

## 9. Integration with n8n

After training completes, load model in n8n:

```python
# n8n Custom Code Node
import torch
import json

# Load model
weights = torch.load('~/.perplexity/model_weights.pt')
vocab = json.load(open('~/.perplexity/vocab.json'))

# Use in workflow
def predict_next_commit(context_tokens):
    # context_tokens: last 10 commits as token sequence
    # Returns: predicted next files, change types, etc
    ...
```

---

## 10. Future Enhancements

1. **Incremental Updates**: Re-run only on new commits
2. **Multi-model Ensemble**: Combine predictions from multiple models
3. **Retrieval Augmentation**: Use RAG to fetch relevant code chunks
4. **Fine-grained Change Prediction**: Predict line-by-line changes
5. **Determinism Validation**: Integrate with The Block's replay tests
6. **Real-time Monitoring**: Stream model predictions as you code

---

## References

- [Git Documentation](https://git-scm.com/docs)
- [LLaMA2 Paper](https://arxiv.org/abs/2307.09288)
- [Code-as-Sequence](https://arxiv.org/abs/2004.13018)
- [Unified Diff Format](https://en.wikipedia.org/wiki/Diff#Unified_format)
