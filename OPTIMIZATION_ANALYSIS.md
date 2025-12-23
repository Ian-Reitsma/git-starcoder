# Optimization Analysis: Training Dataset Pipeline

## Current Pipeline Flow

```
create_training_dataset.py
  |
  |-- Load CodeBERT tokenizer (400MB download, ~30 sec)
  |
  |-- Scan source files (1,349 files, ~5-10 sec)
  |
  |-- Tokenize each file
  |     * Encode with CodeBERT
  |     * Split into 512-token chunks (overlap=128)
  |     * Pad/truncate
  |     (~5-15 minutes for all files)
  |
  |-- Expand sequences (synthetic augmentation to 11,000+)
  |     (~2-5 minutes)
  |
  |-- Split train/val/test (85/10/5)
  |     (~1-2 minutes)
  |
  +-- Save JSON files (100+ MB)
        (~5-10 minutes for write)

Total Time: 15-40 minutes
```

## Optimization Opportunities Found

### 1. CRITICAL: Tokenization Bottleneck (5-15 min, 40% of pipeline)

**Current approach:**
- Sequential tokenization: one file at a time
- For each file: encode → split → pad → store

**Optimization: Parallel tokenization**
```python
# Before: Sequential
for file in files:
    tokens = tokenizer.encode(file.content)  # ~100ms per file
    # 1,349 files * 100ms = 134 seconds

# After: Parallel (8 workers)
from concurrent.futures import ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=8) as executor:
    futures = [executor.submit(tokenize_file, f) for f in files]
    # 1,349 files / 8 workers = ~16 seconds
```

**Potential speedup: 8x (134 sec → 16 sec)**

**Implementation:**
- Use `ThreadPoolExecutor` (I/O bound, not CPU bound)
- Batch encode multiple files at once
- Queue results as they complete

---

### 2. CRITICAL: JSON Write Bottleneck (5-10 min, 20% of pipeline)

**Current approach:**
- `json.dump()` full 11,000 sequences with indentation
- 100+ MB file written line-by-line

**Optimization: Write without indentation + stream write**
```python
# Before: Full load into memory, pretty-print
json.dump(train, f, indent=2)  # Slow for 100+ MB

# After: Compact JSON, write streaming
json.dump(train, f)  # No indent, ~2x faster

# Even better: Write JSONL (one sequence per line)
with open(f, 'w') as f:
    for seq in train:
        f.write(json.dumps(seq) + '\n')
# JSONL is 10% smaller AND faster to parse
```

**Potential speedup: 2-3x (5-10 min → 2-3 min)**

**Bonus: JSONL format is better for large files**
- Incremental write (no need to load full file in memory)
- Incremental read (trainer can load one sequence at a time)
- Better for streaming architectures

---

### 3. MAJOR: Unnecessary Sequence Expansion (2-5 min, 15% of pipeline)

**Current approach:**
- Create base sequences
- Then create synthetic variations to reach 11,000
- Most variations are just copies with modified metadata

**Analysis:**
- You already get ~3,000-5,000 sequences from actual code chunks
- Adding synthetic duplicates (same tokens, different metadata) has diminishing returns
- Model learns better from UNIQUE tokens than token duplicates

**Optimization options:**

**Option A: Stop at natural sequence count (~3,000-5,000)**
```python
# If you have 3,500 real sequences, use those
# Don't artificially expand to 11,000
# Train longer (more epochs) instead of more sequences
```
**Pro:** Unique data, no redundant training
**Con:** Fewer sequences (but quality > quantity)

**Option B: Create real variations, not synthetic copies**
```python
# Instead of: copy tokens + modify metadata
# Do: back-translation, paraphrase, noise injection
# Takes longer but creates REAL different sequences
```
**Pro:** True augmentation, better learning
**Con:** Slower (need NLP model for paraphrasing)

**Option C: Use smart resampling**
```python
# Weight high-priority files higher during sampling
# More chunks from important directories (src/, crates/)
# Fewer chunks from docs, tests
# Still get 11,000 but with better distribution
```
**Pro:** Fast, smart distribution
**Con:** Hyperparameter tuning needed

**Recommendation: Option A (stop at natural count)**
- Keep 3,000-5,000 actual sequences
- Train for 300-400 epochs instead of 200
- Same total training data, better quality

---

### 4. Tokenizer Caching (one-time 30 sec)

**Current approach:**
- Load tokenizer fresh every run

**Optimization: Cache tokenizer**
```python
# First run: download + cache
tokenizer = AutoTokenizer.from_pretrained(
    "microsoft/codebert-base",
    cache_dir=GIT_STARCODER / ".cache"
)
# Subsequent runs: load from cache (1 sec)
```

**Potential speedup: 30 sec saved on 2nd+ runs**

---

### 5. Chunk Overlap Strategy (affects sequence count)

**Current approach:**
- overlap=128 (25% overlap on 512-token window)
- Creates many overlapping sequences

**Analysis:**
- More overlap = more sequences = more redundancy
- Less overlap = fewer sequences = less redundancy

**Options:**
```python
MAX_TOKENS = 512

# Current: overlap=128 (4x overlap)
seq_count_current = (total_tokens - MAX_TOKENS) // (MAX_TOKENS - 128) + 1

# Option: overlap=256 (2x overlap)
seq_count_medium = (total_tokens - MAX_TOKENS) // (MAX_TOKENS - 256) + 1
# ~2x fewer sequences, less redundancy

# Option: overlap=0 (no overlap)
seq_count_minimal = total_tokens // MAX_TOKENS
# ~4x fewer sequences, independent samples
```

**Recommendation:**
- Use overlap=256 (2x) instead of 128
- Better balance: enough context, less redundancy
- Results in ~5,000-7,000 natural sequences

---

## Comprehensive Optimization Plan

### Phase 1: Quick Wins (Save 10-15 min)

1. **Parallel tokenization** (8x speedup on tokenization)
   - Use ThreadPoolExecutor
   - Time saved: 2-3 minutes

2. **JSONL format instead of JSON** (2-3x speedup on write)
   - Remove indent, use streaming
   - Time saved: 3-7 minutes

3. **Tokenizer caching** (one-time only)
   - Cache in .cache/ directory
   - Time saved: 30 seconds on future runs

**Total time after Phase 1: ~8-15 minutes (down from 15-40 minutes)**

### Phase 2: Quality Improvements (Trade speed for better data)

1. **Reduce synthetic expansion**
   - Stop at 5,000 natural sequences instead of 11,000
   - No quality loss, better training
   - Time saved: 2-5 minutes

2. **Adjust overlap strategy**
   - Use overlap=256 instead of 128
   - Reduces redundancy
   - Happens automatically with step 1

3. **Smart resampling**
   - Weight high-priority files higher
   - Better distribution of sequences
   - No time cost, improves model

### Phase 3: Advanced (If needed)

1. **Batch tokenization**
   - Tokenize 8-16 files at once
   - Better hardware utilization
   - 2-3x additional speedup

2. **GPU tokenization** (if available)
   - Use GPU for encoding if RTX available
   - 5-10x speedup on tokenization
   - Requires CUDA + transformer optimization

---

## Recommended Configuration

**For maximum speed:**
```python
MAX_TOKENS = 512
OVERLAP = 256           # Was 128 (less redundancy)
BATCH_SIZE = 16         # Tokenize in batches
MAX_WORKERS = 8         # Parallel workers
TARGET_SEQUENCES = 5000 # Was 11,000 (natural count)
DISABLE_INDENT = True   # JSONL format
```

**Expected results:**
- **Time:** 5-10 minutes (down from 15-40)
- **Sequences:** 5,000-7,000 high-quality (vs 11,000 with redundancy)
- **File size:** 50-75 MB (vs 100+ MB)
- **Training epochs:** 300-400 (vs 200) with same quality

---

## Before/After Comparison

| Aspect | Current | Optimized | Gain |
|--------|---------|-----------|------|
| **Pipeline time** | 15-40 min | 5-10 min | **3-8x faster** |
| **Tokenization** | 5-15 min | 1-2 min | **8x faster** |
| **File write** | 5-10 min | 2-3 min | **3x faster** |
| **Sequences** | 11,000 | 5,000-7,000 | More unique |
| **File size** | 100+ MB | 50-75 MB | **40% smaller** |
| **Quality** | Good | Better | Less redundancy |
| **Total training time (200 epochs)** | Same | Same (or faster) | Same or better |

---

## Implementation Priority

**Must have (Quick, High Impact):**
1. Parallel tokenization
2. JSONL format
3. Stop at natural sequence count (5K instead of 11K)

**Should have (Medium Impact):**
4. Adjust overlap to 256
5. Smart resampling
6. Tokenizer caching

**Nice to have (Complex):**
7. Batch tokenization
8. GPU tokenization

---

## Next Steps

1. **Update `create_training_dataset.py` with Phase 1 optimizations**
   - Add ThreadPoolExecutor for tokenization
   - Change to JSONL output
   - Reduce target to 5,000 sequences
   - Adjust overlap to 256

2. **Benchmark:** Compare before/after
   - Time each phase
   - Compare file sizes
   - Verify sequence quality

3. **Train with optimized dataset**
   - Run full 200 epochs
   - Monitor training curves
   - Compare to baseline

---

## Questions for Fine-Tuning

1. **How long should the pipeline run?**
   - Target: 5-10 min?
   - Or is 15-40 min acceptable?

2. **Prefer more sequences or better sequences?**
   - 11,000 with redundancy?
   - 5,000 unique?
   - 7,000 smart-resampled?

3. **File size preferences?**
   - Keep 100+ MB (original)?
   - Reduce to 50-75 MB (optimized)?

4. **Training patience?**
   - Run 200 epochs?
   - Run 300+ epochs with fewer sequences?

---

**Ready to implement Phase 1? Update `create_training_dataset.py` with optimizations!**
