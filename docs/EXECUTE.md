# How to Run Everything - Complete Step-by-Step

## Prerequisites Check

Before running, verify:

```bash
# Check Python 3 is installed
python3 --version
# Should show: Python 3.9.x or higher

# Check git is available
git --version
# Should show: git version 2.x

# Check repository exists and has commits
cd /Users/ianreitsma/projects/the-block
git log --oneline | wc -l
# Should show a number > 0

# Check directory is writable
cd ~/.perplexity/git-scrape-scripting
touch test_write.txt && rm test_write.txt && echo "Writable: OK"
```

---

## STEP 1: Initial Setup (5 minutes, one time only)

### Using Automated Script (Recommended)

```bash
cd ~/.perplexity/git-scrape-scripting
chmod +x RUN.sh
./RUN.sh
```

**What this does:**
1. Checks Python 3
2. Creates virtual environment (venv/)
3. Activates it
4. Upgrades pip
5. Installs dependencies from requirements.txt
6. Runs test suite
7. Shows detailed instructions

**Expected output:**
```
[1/5] Checking Python version...
Python 3.x.x
✓ Python 3 detected

[2/5] Setting up virtual environment...
✓ Virtual environment ready

[3/5] Installing dependencies...
✓ Dependencies installed

[4/5] Running test suite...
  ✓ config_loading: PASS
  ✓ git_analyzer: PASS
  ...
All tests passed! System is ready.

[5/5] Ready to run pipeline
✓ All systems ready
```

If there are errors, see **Troubleshooting** section at the end.

---

## STEP 2: Run the Training Pipeline (5 mins - 2 hours)

### Activate Environment

```bash
cd ~/.perplexity/git-scrape-scripting
source venv/bin/activate
```

You should see `(venv)` in your prompt.

### Run Full Pipeline

```bash
python3 run_pipeline_dynamic.py \
  --repo /Users/ianreitsma/projects/the-block \
  --verbose
```

**What happens (5 phases):**

#### Phase 0: Repository Analysis (10-30 seconds)
```
======================================================================
# PHASE 0: REPOSITORY ANALYSIS
======================================================================

Analyzing Git repository to get ACCURATE commit counts
...
Branch breakdown:
  main: 156 commits
  develop: 342 commits
  feature/energy: 89 commits
  ...

Total unique commits: 467
Training Parameters Calculated:
  Token sequences: 78
  Determined epochs: 6
  Total training steps: 60
  Estimated training time: 1.5 minutes
```

This is the **estimate**. It will be updated after tokenization.

#### Phase 1: Git Scraping (1-5 minutes)
```
======================================================================
# PHASE 1: GIT SCRAPING (467 COMMITS)
======================================================================

Analyzing 467 unique commits
...

✓ PHASE 1 COMPLETE
  Output: git_history_rich.jsonl
    Size: 5.3 MB
    Commits processed: 467
    Rate: 93 commits/second
```

Extracts metadata like hash, author, message, files changed, insertions/deletions.

#### Phase 2: Tokenization (30 seconds - 2 minutes)
```
======================================================================
# PHASE 2: TOKENIZATION
======================================================================

Converting commits to 2048-token sequences
...

✓ PHASE 2 COMPLETE
  Output: token_sequences_rich.json
    Sequences created: 78
    Tokens per sequence: 2048
    Total tokens: 159,744

  Re-computing training parameters based on ACTUAL sequences...
  Updated epochs: 6 (was 6)
```

**IMPORTANT**: Epochs re-computed from actual sequences.

#### Phase 3: Embeddings (1-10 minutes)
```
======================================================================
# PHASE 3: EMBEDDING GENERATION
======================================================================

Generating embeddings for 78 sequences
...

✓ PHASE 3 COMPLETE
  Output: embeddings_rich.npz
    Shape: (78, 384)
    Size: 0.2 MB
```

Generates 384-dimensional vectors for semantic understanding.

#### Phase 4: Model Training (5 minutes - 2 hours)
```
======================================================================
# PHASE 4: MODEL TRAINING (6 EPOCHS CALCULATED)
======================================================================

Training GPT-2-medium on your code patterns
  6 epochs (formula-determined)
  78 steps per epoch
  Total 468 training steps
  Hardware-optimized batch size
  Early stopping + validation enabled

Starting training...
```

Then for each epoch:
```
----------------------------------------------------------------------
EPOCH 1/6
----------------------------------------------------------------------
Training epoch 1: 100%|████████████████| 78/78 [00:25<00:00, 3.1it/s]
  loss: 4.52
  grad: 0.98
  lr: 5.0e-05

✓ Validation loss improved: 3.89
Epoch Summary:
  Train loss: 4.52
  Val loss: 3.89
  Perplexity: 49.23
  Time: 30s
  Total: 30s
  Remaining: 150s
  Hardware:
    GPU: 97% | 7.8GB / 8.0GB
    CPU: 72% | 8 cores
    RAM: 46% | 32GB total

----------------------------------------------------------------------
EPOCH 2/6
----------------------------------------------------------------------
...

EPOCH 6/6
----------------------------------------------------------------------
...
✓ Training complete: 6 epochs, 3m 7s

Final metrics:
  Final train loss: 1.78
  Final val loss: 1.87
  Final perplexity: 6.48
  Best val loss: 1.87
  Peak GPU memory: 7890MB
  Peak RAM: 46.3%
```

Model is saved automatically when validation improves.

### Final Report

```
======================================================================
FINAL REPORT
======================================================================

Phase Status:
  ✓ phase_0_analyze: complete
  ✓ phase_1_scrape: complete
  ✓ phase_2_tokenize: complete
  ✓ phase_3_embeddings: complete
  ✓ phase_4_training: complete

Repository Statistics:
  Commits analyzed: 467
  Branches: 5
  Authors: 12

Training Parameters (Formula-Based):
  Sequences: 78
  Epochs: 6
  Total steps: 468
  Warmup steps: 47
  Target tokens: 20,000,000

Training Results:
  Final train loss: 1.78
  Final val loss: 1.87
  Final perplexity: 6.48
  Best val loss: 1.87

Gradient Statistics:
  Min norm: 0.0024
  Max norm: 2.34

Learning Rate:
  Min: 4.8e-05
  Max: 5.0e-05

Hardware Peaks:
  Peak GPU memory: 7890MB
  Peak RAM: 46.3%

Execution Summary:
  Total time: 1h 3m 42s
  Model location: models/the-block-git-model-final

Manifest saved to: MANIFEST_DYNAMIC.json
======================================================================
```

**SUCCESS!** Training is complete.

---

## STEP 3: Check Results (Immediately after training)

### View Complete Manifest

```bash
jq '.' MANIFEST_DYNAMIC.json | less
```

Press `q` to exit.

### Check Training Metrics

```bash
# Final loss and perplexity
jq '.training_report.training | {final_train_loss, final_val_loss, final_perplexity}' MANIFEST_DYNAMIC.json

# Output example:
# {
#   "final_train_loss": 1.7823,
#   "final_val_loss": 1.8734,
#   "final_perplexity": 6.48
# }
```

### Check Loss Curves

```bash
# Training loss over time
jq '.training_report.training.loss_history' MANIFEST_DYNAMIC.json

# Output example:
# [
#   4.518234,
#   3.782145,
#   2.945123,
#   ...
# ]
```

### Check Gradient Statistics

```bash
jq '.training_report.gradients' MANIFEST_DYNAMIC.json

# Output example:
# {
#   "min_norm": 0.0024,
#   "max_norm": 2.34,
#   "history": [[avg, max], [avg, max], ...]
# }
```

### Check Hardware Usage

```bash
jq '.training_report.hardware' MANIFEST_DYNAMIC.json

# Output example:
# {
#   "peak_gpu_memory_mb": 7890,
#   "peak_ram_percent": 46.3
# }
```

### Check Repository Stats

```bash
jq '.repository_stats' MANIFEST_DYNAMIC.json

# Shows: commits, branches, authors, timestamps
```

### Verify Model Files

```bash
ls -lh models/the-block-git-model-final/

# Should show:
# -rw-r--r--  1 user  group  346M  Dec  9 20:45 pytorch_model.bin
# -rw-r--r--  1 user  group   14K  Dec  9 20:45 config.json
# -rw-r--r--  1 user  group  1.0M  Dec  9 20:45 training_report.json
# ... (other files)
```

---

## STEP 4: Load and Use the Model

### Python Session

```python
# Start Python
python3

# Load model
from transformers import GPT2LMHeadModel, GPT2Tokenizer
model = GPT2LMHeadModel.from_pretrained('models/the-block-git-model-final')
tokenizer = GPT2Tokenizer.from_pretrained('models/the-block-git-model-final')
print("Model loaded!")

# Generate code
prompt = "def analyze_"
inputs = tokenizer.encode(prompt, return_tensors='pt')
outputs = model.generate(inputs, max_length=100)
generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated)

# Exit Python
exit()
```

### Or in a Script

```bash
cat > test_model.py << 'EOF'
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained('models/the-block-git-model-final')
tokenizer = GPT2Tokenizer.from_pretrained('models/the-block-git-model-final')

prompt = "def calculate"
inputs = tokenizer.encode(prompt, return_tensors='pt')
outputs = model.generate(inputs, max_length=150, num_return_sequences=3)

for i, output in enumerate(outputs):
    generated = tokenizer.decode(output, skip_special_tokens=True)
    print(f"\n=== Sample {i+1} ===")
    print(generated)
EOF

python3 test_model.py
```

---

## STEP 5: Optional - Tune Configuration and Re-Run

### View Current Configuration

```bash
cat training_config.yaml
```

### Edit Configuration

```bash
# Use your editor
vim training_config.yaml
# or
code training_config.yaml
```

### Common Adjustments

```yaml
# To train longer (more learning):
epoch_calculation:
  target_tokens: 40000000  # From 20M to 40M

# To train faster (less learning):
epoch_calculation:
  target_tokens: 10000000  # From 20M to 10M

# To use lower batch size (if GPU memory issues):
hardware_monitoring:
  gpu_memory_threshold_large_gb: 6.0  # From 8.0 to 6.0

# To reduce early stopping patience:
training:
  patience: 2  # From 3 to 2
```

### Re-Run Training

After editing config:

```bash
source venv/bin/activate
python3 run_pipeline_dynamic.py --repo /Users/ianreitsma/projects/the-block --verbose
```

Phases 1-3 will re-run from scratch. The config changes only affect Phase 4 (training).

---

## Troubleshooting

### Python 3 Not Found

```bash
# Try python3.11 or higher
python3.11 --version

# Or check PATH
which python3
which python
```

### Virtual Environment Won't Activate

```bash
# Make sure you're in the right directory
cd ~/.perplexity/git-scrape-scripting

# Try absolute path
source /Users/ianreitsma/.perplexity/git-scrape-scripting/venv/bin/activate

# Or create new one
rm -rf venv
python3 -m venv venv
source venv/bin/activate
```

### Dependencies Installation Fails

```bash
# Clear pip cache
pip cache purge

# Try installing without cache
pip install -r requirements.txt --no-cache-dir

# If specific package fails, try installing each manually
pip install torch==2.1.2
pip install transformers==4.36.2
# ... etc
```

### GPU Not Being Used

```bash
# Check CUDA availability
python3 -c "import torch; print('GPU:', torch.cuda.is_available())"

# Check which device
python3 -c "import torch; print('Device:', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))"

# If False, CPU training still works (just slower)
```

### Out of Memory During Training

```bash
# Edit config to reduce batch size
vim training_config.yaml

# Change:
hardware_monitoring:
  gpu_memory_threshold_large_gb: 6.0  # Reduce from 8.0

# Or manually set:
training:
  batch_size_large: 4  # Reduce from 8
  batch_size_medium: 2  # Reduce from 4
```

### Test Suite Failures

```bash
# Some failures are normal if torch/GPU unavailable
python3 test_suite.py

# Look for "PASS" or "SKIP" (both OK)
# Only "FAIL" means real problem

# If fails, check:
python3 -c "import torch; print('Torch OK')"
python3 -c "import yaml; print('YAML OK')"
python3 -c "import psutil; print('psutil OK')"
```

### Training Very Slow

```bash
# Check GPU utilization
nvidia-smi

# Check CPU usage
top

# If GPU underutilized, increase batch size in config:
hardware_monitoring:
  gpu_memory_threshold_large_gb: 10.0  # Force larger batch
```

### Model File Not Created

```bash
# Check if training actually completed
jq '.phase_results.phase_4_training.status' MANIFEST_DYNAMIC.json

# If 'failed', check error:
jq '.phase_results.phase_4_training.error' MANIFEST_DYNAMIC.json

# Check if model directory exists
ls -la models/the-block-git-model-final/
```

---

## Performance Expectations

### On Your System (RTX 2060 + Ryzen 5 3800X)

- **Phase 0**: ~20 seconds (repository analysis)
- **Phase 1**: ~2 minutes (git scraping, 450 commits)
- **Phase 2**: ~30 seconds (tokenization)
- **Phase 3**: ~3 minutes (embeddings, 78 sequences)
- **Phase 4**: ~3 minutes per epoch (training, 6 epochs)

**Total: ~30-40 minutes for complete pipeline**

This is for a medium-sized repo (~450 commits). Larger repos take longer.

---

## Summary

```bash
# 1. One-time setup (5 minutes)
cd ~/.perplexity/git-scrape-scripting
chmod +x RUN.sh
./RUN.sh

# 2. Train (30-40 minutes for typical repo)
source venv/bin/activate
python3 run_pipeline_dynamic.py --repo /Users/ianreitsma/projects/the-block --verbose

# 3. Check results (1 minute)
jq '.' MANIFEST_DYNAMIC.json | less

# 4. Use model (immediately)
python3 << 'EOF'
from transformers import GPT2LMHeadModel, GPT2Tokenizer
model = GPT2LMHeadModel.from_pretrained('models/the-block-git-model-final')
print("Model ready!")
EOF
```

**That's it! You now have a trained model of your codebase.** 🚀
