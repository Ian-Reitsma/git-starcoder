# Quick Start Guide

## Installation (5 minutes)

### 1. Install Python Dependencies

```bash
cd ~/projects/the-block/.perplexity/git-scrape-scripting
pip install -r requirements.txt

# For GPU support (RTX 2060 Super):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. Verify Installation

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import git; print('GitPython OK')"
python git_scraper.py --help
```

## Full Pipeline Execution (2-3 hours)

### Option A: Run Everything (Recommended)

```bash
python pipeline.py --config config.yaml
```

This runs all 5 stages:
1. `git_scraper` - Extract all git history (~2-5 min)
2. `semantic_chunker` - Chunk code semantically (~10-15 min)
3. `tokenization` - Tokenize with structure awareness (~5-10 min)
4. `dataset_builder` - Create training dataset (~3-5 min)
5. `model_training` - Fine-tune LLM (~8-12 hours)

### Option B: Run Stage-by-Stage (Debugging)

```bash
# Stage 1: Git Scraping
python git_scraper.py --repo ~/projects/the-block --output outputs/commits.json

# Stage 2: Semantic Chunking
python semantic_chunker.py --commits outputs/commits.json --output outputs/chunks.jsonl

# Stage 3: Tokenization
python tokenizer.py --chunks outputs/chunks.jsonl --vocab-size 50257 --output-vocab outputs/vocab.json --output-tokens outputs/tokens.pt

# Stage 4: Dataset Building
python dataset_builder.py --vocab outputs/vocab.json --chunks outputs/chunks.jsonl --commits outputs/commits.json

# Stage 5: Model Training
python train_model.py --vocab outputs/vocab.json --train-data outputs/training_data_train.pt --val-data outputs/training_data_val.pt
```

## Output Files

After running the pipeline, you'll have:

```
outputs/
├── commits.json                    # All git commits with diffs
├── chunks.jsonl                    # Semantic chunks (one per line)
├── vocab.json                      # Token vocabulary mapping
├── tokens.pt                       # Tokenized dataset
├── training_data_train.pt          # Training split
├── training_data_val.pt            # Validation split
├── training_data_test.pt           # Test split
├── model_weights.pt                # Fine-tuned model weights
├── training_logs/                  # Training metrics and checkpoints
├── chunking_stats.json             # Analysis of chunking
├── dataset_stats.json              # Dataset statistics
├── pipeline_results.json           # Execution summary
└── pipeline.log                    # Detailed logs
```

## Configuration

Edit `config.yaml` to customize:

```yaml
# Change repo path
repository:
  path: /path/to/your/repo

# Adjust training parameters
model_training:
  epochs: 5              # More training iterations
  batch_size: 8          # Larger batches (if VRAM allows)
  base_model: llama2-70b # Use larger model
```

## Monitoring Progress

### Real-time Logs

```bash
tail -f outputs/pipeline.log
```

### Check Stage Status

```bash
cat outputs/pipeline_results.json | python -m json.tool
```

### Analyze Results

```bash
# See chunking statistics
cat outputs/chunking_stats.json | python -m json.tool

# See dataset statistics
cat outputs/dataset_stats.json | python -m json.tool

# Monitor training (if you have PyTorch)
python -c "import torch; data = torch.load('outputs/training_data_train.pt'); print(f'Training examples: {len(data)}')"
```

## Common Issues & Solutions

### Issue: "git: command not found"

**Solution:** Install git
```bash
brew install git    # macOS
sudo apt install git # Ubuntu
```

### Issue: "ModuleNotFoundError: No module named 'torch'"

**Solution:** Install PyTorch with GPU support
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Issue: "CUDA out of memory"

**Solution:** Reduce batch size in config.yaml
```yaml
model_training:
  batch_size: 2  # Down from 4
```

### Issue: "Timeout waiting for git command"

**Solution:** The repo is very large. Edit git_scraper.py and increase timeout:
```python
result = subprocess.run(..., timeout=60)  # Changed from 30
```

### Issue: "No space left on device"

**Solution:** Pipeline needs ~10GB of disk space. Check:
```bash
df -h  # Check available space
```

## Next Steps

### 1. Inspect the Data

```python
# See what commits were extracted
import json
with open("outputs/commits.json") as f:
    data = json.load(f)
    print(f"Total commits: {data['metadata']['total_commits']}")
    print(f"Total authors: {data['metadata']['total_authors']}")
```

### 2. Explore Chunks

```python
# See semantic chunks
import json
with open("outputs/chunks.jsonl") as f:
    for i, line in enumerate(f):
        if i >= 5:  # First 5 chunks
            break
        chunk = json.loads(line)
        print(f"Chunk {i}: {chunk['file_path']} ({chunk['change_type']})")
        print(f"  Code length: {len(chunk['new_code'])} chars")
```

### 3. Use the Model

Once training is complete:

```bash
# Load model in Python
python -c "
import torch
weights = torch.load('outputs/model_weights.pt')
print(f'Model size: {sum(p.numel() for p in weights.values())} parameters')
"
```

### 4. Integrate with n8n

Once model is trained, load it in your n8n workflow:

```javascript
// In n8n Python node
const weights = torch.load('~/projects/the-block/.perplexity/git-scrape-scripting/outputs/model_weights.pt');
const model = loadFineTunedModel(weights);
const prediction = model.predict(context);
```

## Performance Expectations

**On Your Hardware (Ryzen 5 3800X + RTX 2060 Super):**

| Stage | Time | Memory |
|-------|------|--------|
| Git scraping | 2-5 min | 2GB |
| Semantic chunking | 10-15 min | 4GB |
| Tokenization | 5-10 min | 6GB |
| Dataset building | 3-5 min | 8GB |
| Model training (3 epochs) | 8-12 hours | 7-8GB GPU |
| **Total** | **~8-12.5 hours** | Peak: 8GB |

## Troubleshooting

If anything fails, check:

1. **Check logs:** `tail -f outputs/pipeline.log`
2. **Verify inputs:** `ls -lh outputs/commits.json outputs/chunks.jsonl`
3. **Check disk space:** `df -h`
4. **Run individual stage:** `python git_scraper.py --repo ~/projects/the-block`
5. **Enable verbose mode:** Edit `config.yaml` and set `verbose: true`

## Next: Integrate with n8n System

After training completes:

1. Copy model weights:
   ```bash
   cp outputs/model_weights.pt ~/projects/the-block/.perplexity/model_weights.pt
   cp outputs/vocab.json ~/projects/the-block/.perplexity/vocab.json
   ```

2. Load in n8n workflow (see n8n integration guide)

3. Use model for context prediction in tactical orchestration layer

## Questions?

Check the full documentation:
- `README.md` - Architecture and design
- `IMPLEMENTATION_NOTES.md` - Detailed per-script notes
- `config.yaml` - All configurable options
