# 🚀 START HERE: Dynamic System Quick Start

## What's New

Your system now:
- ✅ **Auto-detects** all commits across all branches (not hardcoded 287)
- ✅ **Calculates optimal** epochs based on actual data (not hardcoded 5)
- ✅ **Shows comprehensive** training statistics (losses, perplexity, hardware)
- ✅ **Verifies everything** is covered before starting

---

## Run It (3 Steps)

### Step 1: Install & Test
```bash
cd ~/.perplexity/git-scrape-scripting
bash INSTALL.sh
bash run_tests.sh
```

### Step 2: Activate and Run
```bash
source venv/bin/activate
python3 run_pipeline_dynamic.py \
  --repo /Users/ianreitsma/projects/the-block \
  --verbose
```

### Step 3: Wait & Watch

You'll see:

**Phase 0 (10-30s)**:
```
Scanning all branches...
  Found: main (156 commits)
  Found: develop (342 commits)
  Found: feature/energy-markets (89 commits)
  Found: fix/edge-cases (34 commits)
  Found: experimental/governance (12 commits)

Total unique: 467
Calculated epochs: 6
Estimated time: 1.5m
```

**Phase 1-3 (5-10 min)**: Processing data

**Phase 4 (1-2 min)**: Training with detailed per-epoch output
```
Epoch 1/6: Loss: 4.52 | Val Loss: 3.89 | Perplexity: 49.23
Epoch 2/6: Loss: 3.78 | Val Loss: 3.12 | Perplexity: 22.65
Epoch 3/6: Loss: 2.95 | Val Loss: 2.45 | Perplexity: 11.62 ✓ better
...
```

**Done**: Check `MANIFEST_DYNAMIC.json` for complete statistics

---

## What You Get

### Files
```
data/
  git_history_rich.jsonl       ← All commits + metadata
  token_sequences_rich.json    ← 2048-token sequences

embeddings/
  qdrant_points.json           ← 768-dim vectors for RAG

models/
  the-block-git-model-final/   ← Trained model

MANIFEST_DYNAMIC.json          ← Complete statistics
```

### Manifest Contents
```json
{
  "repository_stats": {
    "unique_commits": 467,
    "branches": 5,
    "branch_names": ["main", "develop", ...],
    "commits_per_branch": { ... },
    "unique_authors": 12,
    "commits_per_day": 1.90
  },
  "training_parameters": {
    "epochs": 6,
    "total_steps": 60,
    "estimated_time_minutes": 1.5
  },
  "phase_results": {
    "phase_0_analyze": { "status": "complete", ... },
    "phase_1_scrape": { "status": "complete", "commits_processed": 467, ... },
    "phase_2_tokenize": { "status": "complete", "sequences": 78, ... },
    "phase_3_embeddings": { "status": "complete", "size_mb": 38.2, ... },
    "phase_4_training": { "status": "complete", "epochs": 6, ... }
  }
}
```

---

## Key Features

### Auto-Detection

No assumptions anymore:
```
Before: Assumed 287 commits, hardcoded 5 epochs
After:  Measured from Git, dynamic epochs calculated
```

### Smart Epoch Calculation

Based on data size:
```python
< 20 sequences   →  10 epochs (thorough training)
< 50 sequences   →  8 epochs
< 100 sequences  →  6 epochs
< 200 sequences  →  5 epochs
> 200 sequences  →  4 epochs (sufficient)
```

### Comprehensive Training Stats

Every epoch shows:
- Training loss
- Validation loss  
- Perplexity
- Gradient norms
- Learning rate
- Hardware utilization
- Time remaining

---

## Documentation

| Document | What It Is |
|----------|----------|
| **START_HERE_DYNAMIC.md** | This file - quick start |
| **DYNAMIC_SYSTEM_GUIDE.md** | Detailed explanation |
| **README.md** | Updated overview |
| **DYNAMIC-CHANGES.md** | Summary of changes |

---

## Key Differences from Old System

| Aspect | Old | New |
|--------|-----|-----|
| Commits | 287 (hardcoded) | Auto-detected |
| Branches | Not verified | All scanned |
| Epochs | 5 (fixed) | 3-10 (calculated) |
| Training stats | Basic | Comprehensive |
| Hardware monitor | None | Full monitoring |
| Verification | No | Complete |

---

## The Command

```bash
python3 run_pipeline_dynamic.py --repo /Users/ianreitsma/projects/the-block --verbose
```

That's it. The system handles everything else.

---

## What Happens Step-by-Step

### Phase 0: Repository Analysis (30s)
1. Connect to Git repository
2. Find all branches
3. Count commits on each
4. Get statistics
5. Calculate training parameters
6. Show estimates

### Phase 1: Scraping (2-5 min)
1. Process all detected commits
2. Extract 30+ metadata fields
3. Save to git_history_rich.jsonl
4. Report file size and statistics

### Phase 2: Tokenization (1-2 min)
1. Load all commits
2. Create 2048-token sequences
3. Add semantic markers
4. Save to token_sequences_rich.json
5. Report sequence count

### Phase 3: Embeddings (3-5 min)
1. Load commit data
2. Generate 768-dimensional vectors
3. Format for Qdrant
4. Save to qdrant_points.json
5. Report file size

### Phase 4: Training (1-3 min)
1. Load token sequences
2. Initialize model (GPT-2-medium)
3. Run calculated epochs
4. Track losses and metrics
5. Early stopping if validation plateaus
6. Save model checkpoint
7. Report final statistics

---

## Examples

### Small Repository (50 commits)
```
Phase 0: Detect 50 commits → ~8-10 sequences
Calculate: 8 epochs
Time: ~5-7 minutes total
```

### Medium Repository (300 commits)
```
Phase 0: Detect 300 commits → ~50 sequences
Calculate: 6 epochs
Time: ~10-15 minutes total
```

### Large Repository (1000+ commits)
```
Phase 0: Detect 1000+ commits → ~150+ sequences
Calculate: 4 epochs
Time: ~15-20 minutes total
```

---

## Verify It Works

After running:

```bash
# Check manifest exists
ls -lh MANIFEST_DYNAMIC.json

# See repository stats
jq '.repository_stats' MANIFEST_DYNAMIC.json

# See training parameters
jq '.training_parameters' MANIFEST_DYNAMIC.json

# See phase results
jq '.phase_results' MANIFEST_DYNAMIC.json

# Check model exists
ls -lh models/the-block-git-model-final/pytorch_model.bin

# Test loading model
python3 << 'EOF'
from transformers import GPT2LMHeadModel
model = GPT2LMHeadModel.from_pretrained('models/the-block-git-model-final')
print("✓ Model loaded successfully")
EOF
```

---

## Troubleshooting

### Phase 0 hangs
```bash
# Make sure repo is valid
git -C /Users/ianreitsma/projects/the-block status
git -C /Users/ianreitsma/projects/the-block branch -a
```

### Wrong commit count
```bash
# Verify with git
git -C /Users/ianreitsma/projects/the-block rev-list --all --count

# Check manifest
jq '.repository_stats.unique_commits' MANIFEST_DYNAMIC.json
```

### Training fails
```bash
# Check GPU
nvidia-smi

# Check space
df -h .

# Check logs
jq '.phase_results.phase_4_training' MANIFEST_DYNAMIC.json
```

---

## What's Different

### Detection
```python
# OLD: Hardcoded
commits_count = 287
epochs = 5

# NEW: Detected
commits_count = analyze_git_repo()  # Actually scans Git
epochs = calculate_epochs(num_sequences)  # Formula-based
```

### Reporting
```python
# OLD: Limited
print("Training complete")

# NEW: Comprehensive
print(f"Epoch 1/6: Loss {loss:.4f} | Val {val:.4f} | PPL {ppl:.2f}")
print(f"GPU: {gpu}% | CPU: {cpu}% | RAM: {ram}%")
print(f"Grad norm: {grad_norm:.2f} | LR: {lr:.2e}")
print(f"Time: {elapsed}s | Remaining: {remaining}s")
```

---

## Summary

**Before**: Hardcoded assumptions  
**After**: Dynamic detection and comprehensive reporting

**Before**: Limited visibility  
**After**: Complete transparency at every step

**Before**: Fixed training  
**After**: Intelligent, adaptive training

---

## Ready?

Run this now:

```bash
cd ~/.perplexity/git-scrape-scripting && \
source venv/bin/activate && \
python3 run_pipeline_dynamic.py \
  --repo /Users/ianreitsma/projects/the-block \
  --verbose
```

Then come back in ~15 minutes and check `MANIFEST_DYNAMIC.json`.

Your maximally-informed Block model awaits! 🚀
