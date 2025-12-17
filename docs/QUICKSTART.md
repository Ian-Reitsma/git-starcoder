# Quick Start Guide

## 30-Second Setup

```bash
cd ~/.perplexity/git-scrape-scripting
chmod +x RUN.sh
./RUN.sh
```

## 2-Minute Training Run

```bash
source venv/bin/activate
python3 run_pipeline_dynamic.py --repo /Users/ianreitsma/projects/the-block --verbose
```

## Check Results (Pick One)

```bash
# Everything
jq '.' MANIFEST_DYNAMIC.json

# Just training
jq '.training_report' MANIFEST_DYNAMIC.json

# Just repository
jq '.repository_stats' MANIFEST_DYNAMIC.json

# Just parameters
jq '.training_parameters' MANIFEST_DYNAMIC.json
```

---

## What You Get

### Model
- **Location**: `models/the-block-git-model-final/`
- **Size**: 345M parameters (GPT-2-medium)
- **Trained on**: Your actual code patterns

### Statistics
- **File**: `MANIFEST_DYNAMIC.json`
- **Includes**:
  - Final training & validation loss
  - Perplexity
  - Gradient norms (min/max)
  - Learning rate schedule
  - Hardware peaks (GPU/RAM)
  - Loss curves
  - Execution timeline

### Data
- **Raw commits**: `data/git_history_rich.jsonl` (30+ metadata fields)
- **Tokenized**: `data/token_sequences_rich.json` (2048-token sequences)
- **Embeddings**: `embeddings/embeddings_rich.npz` (384-dim vectors)

---

## Configuration

Edit `training_config.yaml`:

```yaml
training:
  base_learning_rate: 5e-5      # Change me
  validation_split: 0.1         # 10% validation
  patience: 3                    # Early stopping patience

epoch_calculation:
  target_tokens: 20000000       # Target training tokens
  min_epochs: 3                 # Don't train <3 epochs
  max_epochs: 10                # Don't train >10 epochs

hardware_monitoring:
  collection_interval_seconds: 10  # Sample every 10s
```

**No restart needed** - Changes apply on next run

---

## Formula Explained

### Epochs = How many times to loop through data

```
epochs = clamp(
  floor(target_tokens / total_tokens),
  min_epochs=3,
  max_epochs=10
)

where:
  target_tokens = 20M (your goal)
  total_tokens = num_sequences * 2048
```

**Examples**:
- 78 sequences = 160K tokens → floor(20M/160K) = 125 → clamp to 10 → **10 epochs**
- 500 sequences = 1M tokens → floor(20M/1M) = 20 → clamp to 10 → **10 epochs**
- 10K sequences = 20M tokens → floor(20M/20M) = 1 → clamp to 3 → **3 epochs**

---

## Phases Explained

1. **Phase 0** (~10-30s): Analyze repository
   - Counts commits across all branches
   - Estimates sequences and epochs

2. **Phase 1** (~1-5m): Scrape git history
   - Extracts 30+ metadata fields per commit
   - Calculates complexity scores

3. **Phase 2** (~30s-2m): Tokenize
   - Converts to 2048-token sequences
   - **RE-COMPUTES EPOCHS** based on actual count

4. **Phase 3** (~1-10m): Generate embeddings
   - 384-dimensional semantic vectors
   - Qdrant-compatible format

5. **Phase 4** (~5m-2h): Train model
   - Auto-determined epochs from Phase 2
   - Early stopping enabled
   - Hardware-monitored
   - Generates `training_report.json`

---

## Hardware Auto-Tuning

```
GPU Memory          Batch Size  num_workers
─────────────────   ──────────  ──────────────
  ≥ 8GB            8           min(8, cpu//2)
  4-8GB            4           min(8, cpu//2)
  < 4GB            2           min(8, cpu//2)
  No GPU           2 (CPU)     min(8, cpu//2)
```

**Learning rate scales with batch size**:
```
LR = base_LR × (batch_size / 8)
```

So batch_size=4 → LR=2.5e-5

---

## Files Created

```
MANIFEST_DYNAMIC.json          ← Check this for everything
├── execution_timestamp
├── total_execution_time_seconds
├── repository_stats            ← Commit counts, branches
├── training_parameters         ← Epochs, steps, formula
├── phase_results               ← Status of each phase
└── training_report             ← Model performance
    ├── final_train_loss
    ├── final_val_loss
    ├── final_perplexity
    ├── loss_history (per-epoch)
    ├── gradients (min/max/history)
    ├── learning_rate (min/max/history)
    └── hardware (peak GPU mem, peak RAM %)
```

---

## Common Commands

```bash
# See everything
jq '.' MANIFEST_DYNAMIC.json | less

# See training metrics only
jq '.training_report.training' MANIFEST_DYNAMIC.json

# See loss curves
jq '.training_report.training.loss_history' MANIFEST_DYNAMIC.json

# See hardware peaks
jq '.training_report.hardware' MANIFEST_DYNAMIC.json

# See repository stats
jq '.repository_stats' MANIFEST_DYNAMIC.json

# See calculated parameters
jq '.training_parameters' MANIFEST_DYNAMIC.json

# Pretty print to file
jq '.' MANIFEST_DYNAMIC.json > results.json

# Check model was saved
ls -lh models/the-block-git-model-final/pytorch_model.bin

# Count sequences generated
jq 'length' data/token_sequences_rich.json
```

---

## Loading the Model

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained(
    'models/the-block-git-model-final'
)
tokenizer = GPT2Tokenizer.from_pretrained(
    'models/the-block-git-model-final'
)

# Generate code
prompt = "def calculate"
inputs = tokenizer.encode(prompt, return_tensors='pt')
outputs = model.generate(
    inputs,
    max_length=200,
    temperature=0.7,
    num_return_sequences=3
)

for i, output in enumerate(outputs):
    print(f"\n=== Sample {i+1} ===")
    print(tokenizer.decode(output, skip_special_tokens=True))
```

---

## Troubleshooting

**Dependencies fail to install?**
```bash
pip install --upgrade pip
pip install -r requirements.txt --no-cache-dir
```

**GPU not being used?**
```bash
python3 -c "import torch; print(torch.cuda.is_available())"
```
If False, CPU fallback will be used (slower but works)

**Test suite shows failures?**
Normal if torch/GPU unavailable. Core features still work.

**Training is slow?**
Check `MANIFEST_DYNAMIC.json` → `training_report.hardware.peak_gpu_memory_mb`
If 0, GPU not being used. Check CUDA installation.

**Out of memory?**
Edit `training_config.yaml`:
```yaml
hardware_monitoring:
  gpu_memory_threshold_large_gb: 6.0  # Change 8.0 to 6.0
```
This forces batch_size=4 instead of 8.

---

## What Happens Under the Hood

1. **Repository detection**: Scans all Git branches, counts unique commits
2. **Estimation**: Assumes ~6 commits per 2048-token sequence
3. **Scraping**: Extracts full commit metadata (hash, message, author, stats)
4. **Tokenization**: Converts to 2048-token sequences with 256-token overlap
5. **Re-computation**: **Actual** sequence count → **Actual** epochs calculated
6. **Data loading**: Loads sequences, creates TensorDataset
7. **Splitting**: 90% train, 10% validation
8. **Training**:
   - Hardware detection → Batch size + workers
   - Learning rate scaling based on batch size
   - Warmup schedule (10% of steps, [100, 1000] bounds)
   - Early stopping (patience=3, min_delta=0.0001)
   - Validation after each epoch
   - Gradient clipping at 1.0
9. **Monitoring**: Every 10 seconds, sample GPU/CPU/RAM/temp
10. **Reporting**: Final report includes loss curves, gradients, LR, hardware peaks

---

## Key Differences from Before

| Feature | Old | New |
|---------|-----|-----|
| Trainer | Stub | Full implementation |
| Data loading | None | TensorDataset + split |
| Validation | None | 90/10 split + early stopping |
| Epochs | Hardcoded 5 | Formula-based (token count) |
| Batch size | Fixed 8 | Hardware-aware |
| Workers | Fixed 8 | CPU-aware |
| Config | Scattered | YAML file |
| Testing | None | 10-test suite |
| Setup | Manual | RUN.sh automated |
| Report | Basic | Comprehensive with curves |

---

**Ready? Run:** `./RUN.sh` then `python3 run_pipeline_dynamic.py --repo /Users/ianreitsma/projects/the-block --verbose`
