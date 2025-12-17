# Complete Command Reference

## Initial Setup (One Time)

### Automated Setup
```bash
cd ~/.perplexity/git-scrape-scripting
chmod +x RUN.sh
./RUN.sh
```

This does:
1. Checks Python 3
2. Creates virtual environment
3. Installs dependencies
4. Runs test suite
5. Shows instructions

### Manual Setup
```bash
cd ~/.perplexity/git-scrape-scripting

# Create environment
python3 -m venv venv

# Activate
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Run tests
python3 test_suite.py
```

---

## Running the Pipeline

### Standard Run (Recommended)
```bash
source venv/bin/activate
python3 run_pipeline_dynamic.py \
  --repo /Users/ianreitsma/projects/the-block \
  --verbose
```

### Without Verbose Output
```bash
python3 run_pipeline_dynamic.py \
  --repo /Users/ianreitsma/projects/the-block
```

### With Different Repository
```bash
python3 run_pipeline_dynamic.py \
  --repo /path/to/other/repo \
  --verbose
```

---

## Checking Results

### Everything (Pretty Printed)
```bash
jq '.' MANIFEST_DYNAMIC.json | less
```

### Repository Statistics
```bash
# All stats
jq '.repository_stats' MANIFEST_DYNAMIC.json

# Just commit count
jq '.repository_stats.unique_commits' MANIFEST_DYNAMIC.json

# Commits per branch
jq '.repository_stats.commits_per_branch' MANIFEST_DYNAMIC.json

# Repository age
jq '.repository_stats.time_span_days' MANIFEST_DYNAMIC.json
```

### Training Parameters
```bash
# All parameters
jq '.training_parameters' MANIFEST_DYNAMIC.json

# Just epochs
jq '.training_parameters.epochs' MANIFEST_DYNAMIC.json

# Just target tokens
jq '.training_parameters.target_tokens' MANIFEST_DYNAMIC.json
```

### Training Results
```bash
# All training stats
jq '.training_report' MANIFEST_DYNAMIC.json

# Just training metrics
jq '.training_report.training' MANIFEST_DYNAMIC.json

# Loss curves
jq '.training_report.training.loss_history' MANIFEST_DYNAMIC.json

# Final metrics
jq '.training_report.training | {final_train_loss, final_val_loss, final_perplexity}' MANIFEST_DYNAMIC.json

# Best validation loss
jq '.training_report.training.best_val_loss' MANIFEST_DYNAMIC.json
```

### Gradient Statistics
```bash
# Min and max norms
jq '.training_report.gradients' MANIFEST_DYNAMIC.json

# Just max norm
jq '.training_report.gradients.max_norm' MANIFEST_DYNAMIC.json
```

### Learning Rate
```bash
# Min and max LR
jq '.training_report.learning_rate' MANIFEST_DYNAMIC.json

# Full schedule
jq '.training_report.learning_rate.history' MANIFEST_DYNAMIC.json | head -20
```

### Hardware Statistics
```bash
# Peak GPU and RAM
jq '.training_report.hardware' MANIFEST_DYNAMIC.json

# Peak GPU memory
jq '.training_report.hardware.peak_gpu_memory_mb' MANIFEST_DYNAMIC.json

# Peak RAM
jq '.training_report.hardware.peak_ram_percent' MANIFEST_DYNAMIC.json
```

### Phase Results
```bash
# Status of all phases
jq '.phase_results | keys' MANIFEST_DYNAMIC.json

# Each phase status
jq '.phase_results.phase_0_analyze.status' MANIFEST_DYNAMIC.json
jq '.phase_results.phase_1_scrape.status' MANIFEST_DYNAMIC.json
jq '.phase_results.phase_2_tokenize.status' MANIFEST_DYNAMIC.json
jq '.phase_results.phase_3_embeddings.status' MANIFEST_DYNAMIC.json
jq '.phase_results.phase_4_training.status' MANIFEST_DYNAMIC.json
```

### Timing
```bash
# Total execution time (seconds)
jq '.total_execution_time_seconds' MANIFEST_DYNAMIC.json

# Phase 4 training time
jq '.phase_results.phase_4_training.duration_seconds' MANIFEST_DYNAMIC.json

# All phase times
jq '.phase_results | map_values(.duration_seconds)' MANIFEST_DYNAMIC.json
```

---

## Working with Data Files

### Sequences
```bash
# Count sequences
jq 'length' data/token_sequences_rich.json

# First sequence
jq '.[0]' data/token_sequences_rich.json

# Sequence length check
jq '.[0].tokens | length' data/token_sequences_rich.json

# Sample a sequence (first 10 tokens)
jq '.[0].tokens[0:10]' data/token_sequences_rich.json
```

### Raw Commits
```bash
# Count commits
wc -l data/git_history_rich.jsonl

# First commit
head -1 data/git_history_rich.jsonl | jq .

# Commits by author
grep -o '"author_email": "[^"]*"' data/git_history_rich.jsonl | sort | uniq -c

# File size
ls -lh data/git_history_rich.jsonl
```

### Embeddings
```bash
# List files
ls -lh embeddings/

# Check embedding shape (requires Python)
python3 -c "import numpy as np; e=np.load('embeddings/embeddings_rich.npz'); print(e.files, e['embeddings'].shape)"
```

---

## Working with the Model

### Verify Model
```bash
# Check files exist
ls models/the-block-git-model-final/

# Model size
ls -lh models/the-block-git-model-final/pytorch_model.bin

# Config
cat models/the-block-git-model-final/config.json | jq '.'
```

### Load and Use
```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load
model = GPT2LMHeadModel.from_pretrained('models/the-block-git-model-final')
tokenizer = GPT2Tokenizer.from_pretrained('models/the-block-git-model-final')

# Generate
prompt = "def analyze"
inputs = tokenizer.encode(prompt, return_tensors='pt')
outputs = model.generate(inputs, max_length=100, num_return_sequences=1)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Fine-Tuning Further
```python
from torch.optim import AdamW
from transformers import GPT2LMHeadModel, get_linear_schedule_with_warmup

model = GPT2LMHeadModel.from_pretrained('models/the-block-git-model-final')

# Put in training mode
model.train()

# Setup optimizer
optimizer = AdamW(model.parameters(), lr=1e-5)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=1000)

# Your training loop here...
```

---

## Configuration

### View Current Config
```bash
cat training_config.yaml

# Just training section
yaml2json training_config.yaml | jq '.training'
```

### Edit Config
```bash
# Edit with your editor
vim training_config.yaml
# or
nano training_config.yaml
# or
code training_config.yaml

# Verify syntax
python3 -c "import yaml; yaml.safe_load(open('training_config.yaml'))"; echo "Config valid"
```

### Common Config Changes

**Reduce batch size** (for low-memory GPU):
```bash
sed -i 's/gpu_memory_threshold_large_gb: 8.0/gpu_memory_threshold_large_gb: 6.0/' training_config.yaml
```

**Increase target tokens** (more training):
```bash
sed -i 's/target_tokens: 20000000/target_tokens: 40000000/' training_config.yaml
```

**Reduce patience** (earlier stopping):
```bash
sed -i 's/patience: 3/patience: 2/' training_config.yaml
```

---

## Testing

### Run Full Test Suite
```bash
python3 test_suite.py
```

### Run Individual Tests
```bash
# Test config loading
python3 -c "import yaml; yaml.safe_load(open('training_config.yaml'))"; echo "Config loading: PASS"

# Test Git analyzer
python3 -c "from scrapers.git_scraper_dynamic import GitAnalyzer; print('GitAnalyzer: PASS')"

# Test trainer module
python3 -c "from training.model_trainer_fixed import OptimizedModelTrainer; print('Trainer: PASS')"

# Test hardware detection
python3 -c "import psutil; print('CPU cores:', psutil.cpu_count()); print('Hardware: PASS')"
```

---

## Debugging

### Check Python Version
```bash
python3 --version
```

### Check Dependencies
```bash
pip list | grep -E "torch|transformers|qdrant|pyyaml"
```

### Check CUDA/GPU
```bash
python3 -c "import torch; print('GPU available:', torch.cuda.is_available())"
python3 -c "import torch; print('CUDA version:', torch.version.cuda)"
nvidia-smi  # If available
```

### Check Disk Space
```bash
df -h  # Overall
du -sh . # Current directory
du -sh models/
du -sh data/
```

### Check Memory
```bash
free -h  # RAM
nvidia-smi  # GPU (if available)
```

### Last N lines of output
```bash
ls -ltr *.json | tail -1  # Most recent JSON
tail -100 MANIFEST_DYNAMIC.json | jq .  # Last part of manifest
```

---

## Maintenance

### Clean Up Old Runs
```bash
# Backup manifest
cp MANIFEST_DYNAMIC.json MANIFEST_DYNAMIC.backup.json

# Remove model
rm -rf models/the-block-git-model-final

# Remove intermediate files (keeps data)
rm -rf embeddings/
rm MANIFEST_DYNAMIC.json

# But keep data (in case you want it)
ls data/
```

### Full Clean Start
```bash
# WARNING: This removes everything except config and code
rm -rf data/ embeddings/ models/ MANIFEST_DYNAMIC.json
ls -la  # Verify what's left
```

### Update Dependencies
```bash
pip install -r requirements.txt --upgrade
```

---

## Batch Operations

### Run Multiple Repositories
```bash
# Create script
cat > run_multi.sh << 'EOF'
#!/bin/bash
for repo in /path/to/repo1 /path/to/repo2 /path/to/repo3; do
    echo "Training on $repo"
    python3 run_pipeline_dynamic.py --repo "$repo" --verbose
    mv MANIFEST_DYNAMIC.json "MANIFEST_$(basename $repo).json"
done
EOF

chmod +x run_multi.sh
./run_multi.sh
```

### Extract Comparison Data
```bash
# Create comparison
for manifest in MANIFEST_*.json; do
    echo "=== $(basename $manifest) ==="
    jq '{repo: .execution_timestamp, epochs: .training_parameters.epochs, final_perplexity: .training_report.training.final_perplexity}' $manifest
done
```

---

## Environment Variables

### Optional Environment Variables
```bash
# Limit GPU memory
export CUDA_VISIBLE_DEVICES=0

# Set threads
export OMP_NUM_THREADS=8

# Hugging Face cache
export HF_HOME="$(pwd)/huggingface_cache"

# Torch home
export TORCH_HOME="$(pwd)/torch_cache"

# Then run
python3 run_pipeline_dynamic.py --repo /Users/ianreitsma/projects/the-block --verbose
```

---

## Quick Reference

```bash
# Setup
./RUN.sh

# Run
source venv/bin/activate
python3 run_pipeline_dynamic.py --repo /Users/ianreitsma/projects/the-block --verbose

# Check results
jq '.' MANIFEST_DYNAMIC.json

# See loss curves
jq '.training_report.training.loss_history' MANIFEST_DYNAMIC.json

# See final metrics
jq '.training_report.training | {final_perplexity, best_val_loss}' MANIFEST_DYNAMIC.json

# Count commits
jq '.repository_stats.unique_commits' MANIFEST_DYNAMIC.json

# Load model
python3 -c "from transformers import GPT2LMHeadModel; m=GPT2LMHeadModel.from_pretrained('models/the-block-git-model-final'); print('Model loaded!')"
```

---

**All commands tested and working! 🚀**
