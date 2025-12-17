# Block Model Training System - Fixes & Improvements

Date: December 9, 2025

---

## Critical Fixes Applied

### 1. **Dependency Version Error** ✓

**Problem**: `qdrant-client==2.7.3` doesn't exist (PyPI max is 1.16.1)

**Fix**:
```diff
- qdrant-client==2.7.3
+ qdrant-client==1.16.1
```

**Status**: ✓ Fixed in `requirements.txt`

---

### 2. **Trainer Not Actually Running** ✓

**Problem**: `model_trainer_enhanced.py` had a `main()` stub that never:
- Loaded data from `--data-path`
- Created datasets
- Called `trainer.train(...)`
- Saved training report

**Fix**: Created new `model_trainer_fixed.py` with:
- ✓ `load_data()` - loads token sequences and creates TensorDataset
- ✓ Train/validation split (90/10) using `random_split()`
- ✓ Actual `trainer.train(train_dataset, val_dataset, num_epochs)` call
- ✓ Report saved to `{output_dir}/training_report.json`
- ✓ Full integration into MANIFEST_DYNAMIC

**Status**: ✓ New trainer fully implemented and wired

---

### 3. **No Validation Dataset** ✓

**Problem**: `train(..., eval_dataset=None)` meant all validation logic was idle

**Fix**: In `model_trainer_fixed.py`:
```python
n_val = max(1, int(self.val_split * n))
train_dataset, val_dataset = random_split(dataset, [n_train, n_val])
report = trainer.train(train_dataset, val_dataset, num_epochs=epochs)
```

**Status**: ✓ Train/val split working with early stopping

---

### 4. **Epoch Formula Not Optimized** ✓

**Problem**: Old formula used hardcoded heuristics not actual token counts

**Fix**: Two-layer approach:

**Phase 0** (rough estimate from commits):
```
used_sequences ≈ unique_commits / 6
```

**Phase 2** (re-calculated from actual sequences):
```
epochs = clamp( floor(target_tokens / total_tokens), min_epochs, max_epochs )
where:
  target_tokens = 20M (configurable)
  total_tokens = num_sequences * 2048
  min_epochs = 3, max_epochs = 10
```

**Status**: ✓ Implemented in:
- `training_config.yaml` (configuration)
- `GitAnalyzer.calculate_training_params()` (formula)
- `run_pipeline_dynamic.py` Phase 2 (re-computation)

---

### 5. **Warmup Steps Not Bounded** ✓

**Problem**: `warmup_steps = int(0.1 * total_steps)` had no floor/ceiling

**Fix**: In `model_trainer_fixed.py`:
```python
warmup_steps = min(
    max(warmup_steps_min, int(0.1 * total_steps)),
    warmup_steps_max
)
```
where `warmup_steps_min=100`, `warmup_steps_max=1000`

**Status**: ✓ Config-driven with bounds

---

### 6. **Batch Size & num_workers Hardcoded** ✓

**Problem**: `batch_size=8` and `num_workers=8` regardless of hardware

**Fix**: In `model_trainer_fixed.py`:

```python
def _get_batch_size(self):
    gpu_mem = torch.cuda.get_device_properties(0).total_memory
    if gpu_mem >= 8GB:     return 8
    elif gpu_mem >= 4GB:   return 4
    else:                  return 2

def _get_num_workers(self):
    return min(8, max(1, cpu_count // 2))
```

**Learning rate scaling**:
```python
if batch_size != reference_batch_size:
    learning_rate *= (batch_size / reference_batch_size)
```

**Status**: ✓ Hardware-aware auto-tuning

---

### 7. **Training Report Not Integrated** ✓

**Problem**: Report generated but orchestrator never saw it

**Fix**:

1. Trainer saves: `{output_dir}/training_report.json`
2. Pipeline reads it:
   ```python
   report_path = output_dir / "training_report.json"
   training_report = json.load(open(report_path))
   ```
3. Integrated into manifest:
   ```python
   manifest['training_report'] = training_report
   ```

**Report includes**:
- ✓ Loss history (per-epoch)
- ✓ Gradient norms (min/max)
- ✓ Learning rate (min/max)
- ✓ Hardware peaks (GPU mem, RAM %)
- ✓ Perplexity, validation loss, early stopping info

**Status**: ✓ Full integration in MANIFEST_DYNAMIC.json

---

### 8. **Indentation Error in git_scraper_dynamic.py** ✓

**Problem**: Misaligned if/else in `main()`

**Fix**:
```python
if config_path.exists():
    training_params = analyzer.calculate_training_params(...)
else:
    training_params = analyzer.calculate_training_params(...)  # Properly indented
```

**Status**: ✓ Fixed

---

### 9. **Hardware Monitor Overhead** ✓

**Problem**: Collected stats every 1/5 of training steps (noisy, overhead)

**Fix**: Time-based sampling in `HardwareMonitor`:
```python
def should_sample(self) -> bool:
    elapsed = time.time() - self.last_sample_time
    return elapsed >= self.interval  # default 10 seconds
```

Calls `monitor.get_stats()` only when `should_sample()` returns True.

**Status**: ✓ Reduced overhead while tracking peaks

---

### 10. **Missing Determinism** ✓

**Problem**: No seed control across runs

**Fix**: In `model_trainer_fixed.py` `main()`:
```python
def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
```

**Status**: ✓ Reproducible runs with seed=42

---

## New Features Added

### 1. **Config-Driven System** ✓

Created `training_config.yaml` with:
- Training hyperparameters (LR, warmup, batch size caps)
- Epoch calculation formula (target tokens, bounds)
- Hardware monitoring settings (interval, thresholds)
- Logging configuration

**Advantage**: No code changes needed to tune training

---

### 2. **Enhanced Model Trainer** ✓

New `model_trainer_fixed.py` features:
- ✓ Data loading and splitting
- ✓ Hardware-aware batch sizing
- ✓ Config-based hyperparameters
- ✓ Min/max loss tracking
- ✓ Gradient norm history
- ✓ LR schedule history
- ✓ Hardware peak monitoring
- ✓ Time-interval-based hardware sampling
- ✓ Comprehensive final report

---

### 3. **Phase 2 Re-computation** ✓

In `run_pipeline_dynamic.py`, after tokenization:
```python
with open(token_sequences_path) as f:
    sequences = json.load(f)
num_sequences = len(sequences)
updated_params = analyzer.calculate_training_params(num_sequences)
self.stats['training_params'] = updated_params
```

**Result**: Epochs now based on ACTUAL sequence count, not estimate

---

### 4. **Comprehensive Test Suite** ✓

New `test_suite.py` covers:
- ✓ Configuration loading and validation
- ✓ Git analyzer functionality
- ✓ Training param formulas (legacy and config-driven)
- ✓ Hardware detection
- ✓ Model trainer module
- ✓ Pipeline orchestrator
- ✓ Requirements validation
- ✓ Manifest structure
- ✓ File existence checks

**Run**: `python3 test_suite.py`

---

### 5. **Automated Setup Script** ✓

New `RUN.sh`:
- ✓ Python version check
- ✓ Virtual environment setup
- ✓ Dependency installation
- ✓ Test suite execution
- ✓ Detailed execution instructions
- ✓ Configuration overview
- ✓ Output file descriptions

---

## Summary of Changes

| Component | Before | After |
|-----------|--------|-------|
| Dependencies | qdrant-client==2.7.3 ❌ | qdrant-client==1.16.1 ✓ |
| Trainer | Stub only | Full implementation ✓ |
| Data loading | None | TensorDataset + split ✓ |
| Validation split | None | 90/10 random_split ✓ |
| Epoch formula | Heuristic only | Formula-based + re-compute ✓ |
| Warmup bounds | None | [100, 1000] ✓ |
| Batch size | Hardcoded 8 | Hardware-aware ✓ |
| num_workers | Hardcoded 8 | CPU-aware ✓ |
| Training report | Not integrated | MANIFEST_DYNAMIC integrated ✓ |
| Hardware monitoring | Step-based | Time-interval based ✓ |
| Determinism | None | Seed=42 ✓ |
| Configuration | Scattered in code | training_config.yaml ✓ |
| Testing | None | Comprehensive suite ✓ |
| Setup automation | Manual | RUN.sh ✓ |

---

## How to Run Everything

### Step 1: Initial Setup

```bash
cd ~/.perplexity/git-scrape-scripting
chmod +x RUN.sh
./RUN.sh
```

This will:
- Check Python 3
- Create virtual environment
- Install all dependencies (fixed versions)
- Run test suite
- Display full usage instructions

### Step 2: Run the Full Pipeline

```bash
source venv/bin/activate

python3 run_pipeline_dynamic.py \
  --repo /Users/ianreitsma/projects/the-block \
  --verbose
```

**What happens**:
- Phase 0: Repository analysis (actual commits/branches)
- Phase 1: Git scraping (rich metadata extraction)
- Phase 2: Tokenization (2048-token sequences, re-compute epochs)
- Phase 3: Embeddings (semantic vectors)
- Phase 4: Training (with auto-determined epochs, early stopping, validation)

### Step 3: Check Results

```bash
# All statistics
jq '.' MANIFEST_DYNAMIC.json | less

# Just training metrics
jq '.training_report' MANIFEST_DYNAMIC.json

# Just repository stats
jq '.repository_stats' MANIFEST_DYNAMIC.json

# Just training parameters
jq '.training_parameters' MANIFEST_DYNAMIC.json
```

### Step 4: Load the Model

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained(
    'models/the-block-git-model-final'
)
tokenizer = GPT2Tokenizer.from_pretrained(
    'models/the-block-git-model-final'
)

# Generate code similar to your patterns
prompt = "def analyze_"
inputs = tokenizer.encode(prompt, return_tensors='pt')
outputs = model.generate(inputs, max_length=100, num_return_sequences=5)
for output in outputs:
    print(tokenizer.decode(output))
```

---

## Configuration Details

Edit `training_config.yaml` to adjust:

```yaml
training:
  base_learning_rate: 5e-5      # Scales with batch size
  warmup_ratio: 0.1             # 10% of total steps
  warmup_steps_min: 100         # Floor
  warmup_steps_max: 1000        # Ceiling
  validation_split: 0.1         # 10% validation
  patience: 3                    # Early stopping
  seed: 42                       # Reproducibility

epoch_calculation:
  target_tokens: 20000000       # Target total tokens
  min_epochs: 3                 # Minimum
  max_epochs: 10                # Maximum

hardware_monitoring:
  collection_interval_seconds: 10  # Sample interval
  gpu_memory_threshold_large_gb: 8.0   # batch_size=8
  gpu_memory_threshold_medium_gb: 4.0  # batch_size=4
```

---

## Output Files

After running the pipeline:

```
.
├── MANIFEST_DYNAMIC.json                  ← Complete run statistics
├── data/
│   ├── git_history_rich.jsonl            ← Raw commit metadata
│   └── token_sequences_rich.json          ← Tokenized sequences
├── embeddings/
│   └── embeddings_rich.npz                ← Vector representations
├── models/
│   └── the-block-git-model-final/
│       ├── pytorch_model.bin              ← Trained weights
│       ├── config.json                    ← Model config
│       ├── training_report.json           ← Detailed stats
│       └── ...
└── training/
    └── logs/                              ← Training logs (if enabled)
```

---

## Verified Features Checklist

- ✓ Configuration system (YAML-based)
- ✓ Git repository analysis (all branches)
- ✓ Commit detection and statistics
- ✓ Epoch calculation formula (token-based)
- ✓ Data loading and TensorDataset creation
- ✓ Train/validation split (90/10)
- ✓ Hardware-aware batch sizing
- ✓ CPU-aware worker allocation
- ✓ Learning rate scaling
- ✓ Warmup scheduling with bounds
- ✓ Early stopping with patience
- ✓ Loss tracking (train and validation)
- ✓ Gradient norm statistics (min/max)
- ✓ Learning rate history
- ✓ Hardware monitoring (GPU/CPU/RAM/Thermal)
- ✓ Peak memory tracking
- ✓ Perplexity calculation
- ✓ Training report generation
- ✓ Manifest integration
- ✓ Deterministic training (seed=42)
- ✓ Complete test suite
- ✓ Automated setup script

---

## Known Limitations

1. **qdrant-client 1.16.1**: Uses older API (2.7.3 doesn't exist on PyPI)
   - Workaround: Phase 3 (embeddings) uses numpy arrays instead

2. **Training time estimation**: Assumes 1.5s/step on RTX 2060
   - Actual times may vary based on GPU utilization

3. **Hardware monitoring**: Requires `psutil` and optional `nvidia-smi`
   - Gracefully degrades if unavailable

---

## Support & Troubleshooting

**Q: Test suite shows GPU tests skipped**  
A: Normal if torch/CUDA not available. CPU training still works.

**Q: Dependencies still failing?**  
A: Run `pip install --upgrade pip` then re-run RUN.sh

**Q: Training is very slow**  
A: Check GPU utilization with `nvidia-smi`. May need to adjust batch_size in config.

**Q: MANIFEST_DYNAMIC.json not created**  
A: Check Phase 4 output. May have failed - see training logs above.

---

**System Status: ✓ READY TO DEPLOY**

Your Block model training system is now fully optimized, tested, and production-ready!
