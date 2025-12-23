# Comprehensive Pipeline: Full Codebase to 6,464+ Training Sequences

## The Problem

Your old setup generated **6,464 sequences with 3.0B params** from full codebase analysis.

The new `fresh_start_extraction.py` only generates **136 sequences** by tokenizing commit metadata (not code).

```
Old: Scrape commits → Extract code/diffs → Chunk semantically → 
     Tokenize all chunks → Build big dataset → 6,464 sequences

New: Scrape commits → Tokenize metadata only → 136 sequences ❌
```

**This is a 47x reduction in training data.**

## The Solution

**Use `comprehensive_pipeline.py`** to run the FULL pipeline in one command:

```bash
cd /home/projects/git-starcoder
python3 comprehensive_pipeline.py
```

## What It Does

### Step 1: Scrape Rich Git History
```bash
python3 scrapers/git_scraper_rich.py \
  --repo /home/projects/the-block \
  --output data/the-block/git_history_rich.jsonl \
  --output-json data/the-block/git_history_rich.json \
  --stats
```

**Output:** All 513 commits with:
- Complete metadata (author, timestamp, branches, tags)
- Full diffs for every file change
- Merge details and conflict resolution
- File ownership and change frequency
- Complexity scores
- Author collaboration patterns

**Time: 2-5 minutes**

### Step 2: Chunk Code/Diffs into Semantic Pieces
```bash
python3 semantic_chunker_enhanced_FIXED.py \
  --repo /home/projects/the-block \
  --commits data/the-block/git_history_rich.json \
  --output data/the-block/chunks_semantic.jsonl \
  --max-chunk-tokens 512 \
  --min-chunk-tokens 64
```

**Output:** 1,000-3,000 semantic chunks, where each chunk is:
- One file change (added/modified/deleted)
- Associated diff context
- Commit metadata
- File path and change type

**This is where the 47x expansion happens**: 513 commits → 1,500-3,000 chunks

**Time: 5-15 minutes**

### Step 3: Tokenize with CodeBERT
```bash
python3 tokenizers/file_snapshot_tokenizer.py \
  --input data/the-block/chunks_semantic.jsonl \
  --sequences data/the-block/chunks_tokenized.json \
  --model microsoft/codebert-base \
  --sequence-length 512 \
  --overlap 128 \
  --stats
```

**Output:** Token sequences (512 tokens each) from CodeBERT:
- Vocabulary size: ~50K tokens
- Semantic understanding of code patterns
- Overlapping windows create MORE sequences
- Total: 3,000-6,464+ token sequences

**Note:** First run downloads CodeBERT (~400 MB), takes extra 5-10 minutes

**Time: 10-30 minutes**

### Step 4: Build Large Chronological Dataset
```bash
python3 dataset_builder_enhanced_v2_optimized.py \
  --vocab data/the-block/chunks_tokenized.json \
  --chunks data/the-block/chunks_tokenized.json \
  --commits data/the-block/git_history_rich.json \
  --context-window 2048 \
  --target-window 256 \
  --output-dir data/the-block/dataset
```

**Output:** Training/validation/test splits with:
- 2048-token context windows (what the model "sees")
- 256-token target windows (what it predicts)
- Chronological ordering (preserves git history patterns)
- Overlapping windows for more training examples

**Expected output:**
- `training_data_train.json`: ~85% of sequences
- `training_data_val.json`: ~10% of sequences
- `training_data_test.json`: ~5% of sequences
- **Total: 3,000-6,464+ sequences across all splits**

**Time: 5-10 minutes**

## Running the Full Pipeline

### Single Command
```bash
python3 comprehensive_pipeline.py
```

This automatically:
1. Creates `data/the-block/` directory
2. Runs all 4 steps sequentially
3. Validates outputs at each step
4. Prints detailed progress and statistics
5. Handles fallbacks if any step fails

### Timeline
- **Total execution time: ~1-1.5 hours**
  - Step 1 (scrape): 2-5 min
  - Step 2 (chunk): 5-15 min
  - Step 3 (tokenize): 10-30 min (includes CodeBERT download on first run)
  - Step 4 (dataset): 5-10 min

## Expected Output

```
data/the-block/
├── git_history_rich.jsonl           # 513 commits (raw)
├── git_history_rich.json            # Same in JSON format
├── chunks_semantic.jsonl            # 1,500-3,000 code chunks
├── chunks_tokenized.json            # CodeBERT tokenized sequences
└── dataset/
    ├── training_data_train.json     # ~2,500-5,500 sequences (85%)
    ├── training_data_val.json       # ~300-700 sequences (10%)
    └── training_data_test.json      # ~150-350 sequences (5%)
```

**Dataset statistics:**
- Total sequences: 3,000-6,464+
- Total tokens: 1.5M - 3.3M+
- Dataset size: 50-150 MB
- Vocab size: 50,265 (CodeBERT)

## After the Pipeline: Training

### 1. Copy to Training Directory
```bash
cp data/the-block/dataset/training_data_*.json data/scrape-dec23/
```

### 2. Update Config
Edit `training_config_metal_cuda_universal.yaml`:
```yaml
train_path: "data/scrape-dec23/training_data_train.json"
val_path: "data/scrape-dec23/training_data_val.json"
test_path: "data/scrape-dec23/training_data_test.json"
```

### 3. Test Run (1 epoch)
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python3 training/model_trainer_unified.py \
  --config training_config_metal_cuda_universal.yaml \
  --sequences data/scrape-dec23/training_data_train.json \
  --epochs 1 \
  --output models/the-block-IR-comprehensive-test \
  --device cuda
```

**Time: 5-30 minutes** (depends on final sequence count)

### 4. Full Training (200 epochs)
```bash
python3 training/model_trainer_unified.py \
  --config training_config_metal_cuda_universal.yaml \
  --sequences data/scrape-dec23/training_data_train.json \
  --epochs 200 \
  --output models/the-block-IR-comprehensive \
  --device cuda 2>&1 | tee training_comprehensive.log
```

**Time: 16-100 hours** (depends on sequence count and hardware)

## Comparison: Old vs Fresh Start vs Comprehensive

| Aspect | Old Pipeline | Fresh Start | Comprehensive |
|--------|---|---|---|
| Sequences | 6,464 | 136 | **3,000-6,464+** |
| Code Coverage | Full diffs | Metadata only | **Full diffs + snapshots** |
| Context Windows | Yes | 512 tokens | **2048 context + 256 target** |
| Dataset Size | ~100 MB | 0.4 MB | **50-150 MB** |
| Model Params | 3.0B | 125M (CodeBERT) | **125M (CodeBERT)** |
| Training Time/Epoch | 2.5 hours | 19 seconds | **5-30 minutes** |
| Total Training (200 epochs) | ~500 hours | ~63 minutes | **16-100 hours** |
| Reproducibility | Manual | ✅ Deterministic | **✅ Deterministic** |

## Key Advantages

✅ **Full Codebase**: Not just commit messages—actual code changes and diffs
✅ **Semantic Understanding**: CodeBERT tokenization captures code patterns
✅ **Chronological**: Preserves git history, learns progression
✅ **Scalable**: 3,000-6,464+ sequences like the original pipeline
✅ **Efficient**: CodeBERT (125M) vs 3.0B for faster training
✅ **One Command**: No manual steps, fully automated
✅ **Reproducible**: Deterministic with seed=42

## Troubleshooting

### "ModuleNotFoundError: No module named 'semantic_chunker_enhanced_FIXED'"
**Solution:** Verify file exists in repo root:
```bash
ls -la semantic_chunker_enhanced_FIXED.py
```
If missing, check git history or fallback to raw commit data.

### "CUDA out of memory"
**Solution:** The pipeline itself doesn't use CUDA (runs on CPU). This error appears during training (Step 3 onward).
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### "CodeBERT download hangs"
**Solution:** First run downloads ~400 MB. Give it 5-10 minutes and don't interrupt.

### "tokenizers.exceptions.HFTokenizerException"
**Solution:** HuggingFace tokenizers library issue. Reinstall:
```bash
pip install --upgrade tokenizers transformers
```

### Step 2 fails (chunker not found)
**Solution:** Pipeline automatically falls back to using raw commits from Step 1.
Output sequences will be smaller but still valid.

## Next Steps

1. **Run the pipeline:**
   ```bash
   python3 comprehensive_pipeline.py
   ```

2. **Monitor progress** (will take 1-1.5 hours total)

3. **Review output statistics** printed at the end

4. **Copy files and update config** (see Training section above)

5. **Test with 1 epoch** before committing to 200 epochs

6. **Track training** with the log file

## Questions?

See the main [README.md](README.md) or check individual scripts:
- `scrapers/git_scraper_rich.py` - Rich git extraction
- `semantic_chunker_enhanced_FIXED.py` - Code chunking
- `tokenizers/file_snapshot_tokenizer.py` - CodeBERT tokenization
- `dataset_builder_enhanced_v2_optimized.py` - Dataset building

---

**Ready? Run this:**
```bash
python3 comprehensive_pipeline.py
```

🚀
