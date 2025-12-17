# Full Coverage Testing Suite

**Complete test coverage for all untested areas of the system.**

## Overview

Three comprehensive test suites totaling **~35 minutes** of execution:

1. **`test_behavioral_evaluation.py`** (~10 min)
   - Behavioral prompts configuration loading
   - Prompt tokenization and validation
   - Code generation from prompts
   - Output quality validation
   - Evaluation result collection and reporting

2. **`test_pipeline_orchestration.py`** (~10 min)
   - Phase 0: Repository analysis and branch detection
   - Phase 1: Git scraping with full metadata
   - Phase 2: Tokenization pipeline
   - Phase 3: Embeddings configuration validation
   - Phase 4: Training parameter calculation
   - Manifest generation and structure validation

3. **`test_starcoder_lora_quantization.py`** (~20 min)
   - StarCoder2-3B model loading with 4-bit quantization
   - LoRA adapter creation and configuration
   - Data loading and preprocessing
   - Full training loop (2 epochs with real model)
   - Model saving and artifact validation
   - Memory efficiency analysis
   - Hardware monitoring throughout

## Quick Start

### Run All Tests (Master Runner)

```bash
cd ~/.perplexity/git-scrape-scripting

# Option 1: Run with master runner (recommended)
python3 run_full_coverage_test_suite.py --repo /Users/ianreitsma/projects/the-block

# Option 2: Set REPO_PATH environment variable
export REPO_PATH=/Users/ianreitsma/projects/the-block
python3 run_full_coverage_test_suite.py

# Option 3: Skip environment validation (faster)
python3 run_full_coverage_test_suite.py --repo /path/to/repo --skip-validation
```

### Run Individual Tests

**Behavioral Evaluation** (standalone, no dependencies):
```bash
python3 test_behavioral_evaluation.py
```

**Pipeline Orchestration** (requires repo path):
```bash
python3 test_pipeline_orchestration.py /Users/ianreitsma/projects/the-block
# or
export REPO_PATH=/Users/ianreitsma/projects/the-block
python3 test_pipeline_orchestration.py
```

**StarCoder2 + Quantization + LoRA** (full training, requires GPU):
```bash
python3 test_starcoder_lora_quantization.py
```

---

## Test Output Format

All tests produce **verbose, detailed output** at each step:

```
################################################################################
# STARCODER2-3B + 4-BIT + LORA FULL INTEGRATION TEST
################################################################################

[1/6] Model Loading with 4-bit Quantization
--------------------------------------------------------------------------------

Initializing OptimizedModelTrainer...
  ✓ Device: cuda
  ✓ Model: bigcode/starcoder2-3b
  ✓ Use 4-bit: True
  ✓ Use LoRA: True

Loading model and tokenizer...
  ✓ Model loaded: bigcode/starcoder2-3b
  ✓ Model class: PeftModelForCausalLM
  ✓ Tokenizer vocab size: 49152

  LoRA Parameters:
    Trainable: 13,565,568 (2.43%)
    Total: 3,000,000,000

  GPU Memory:
    Allocated: 2.15 GB
    Reserved: 2.30 GB
    Total: 8.00 GB

✓ Model loading test PASSED
```

### Output Sections

Each test produces:

1. **Phase Headers**
   - Clear `[Step/Total]` indicators
   - Descriptive phase names

2. **Detailed Results**
   - ✓ Success indicators
   - Specific metrics and values
   - Nested indentation for hierarchy

3. **Summary Statistics**
   - Epoch-by-epoch training metrics
   - Hardware utilization percentages
   - Success/failure rates
   - Timing information

4. **Final Report**
   - Test-by-test results
   - Pass/fail status with emoji indicators
   - Key findings and achievements
   - Total execution time

---

## Detailed Test Coverage

### 1. Behavioral Evaluation Test

**File:** `test_behavioral_evaluation.py`

**Coverage:**

```
[1/5] Behavioral Prompts Configuration
  ✓ Load from training_config.yaml
  ✓ Load from training_config_rust.yaml
  ✓ Validate prompt counts
  ✓ Detect language-specific prompts
  ✓ Verify Rust patterns (fn, impl, Result<, async fn, #[derive])

[2/5] Behavioral Prompt Tokenization
  ✓ Load tokenizer (GPT2 for demo, StarCoder2 in production)
  ✓ Tokenize test prompts
  ✓ Analyze token distributions
  ✓ Calculate average tokens per prompt
  ✓ Verify tokenizer behavior

[3/5] Code Generation from Prompts
  ✓ Load language model
  ✓ Generate from behavioral prompts
  ✓ Output length analysis
  ✓ Temperature/sampling verification
  ✓ Top-p nucleus sampling

[4/5] Generated Output Validation
  ✓ Check non-empty outputs
  ✓ Verify output starts appropriately
  ✓ Validate output length (reasonable range)
  ✓ Check for common errors
  ✓ Language-specific pattern detection

[5/5] Evaluation Result Reporting
  ✓ Create evaluation report structure
  ✓ Collect per-output metrics
  ✓ Calculate aggregate statistics
  ✓ Language breakdown
  ✓ Perplexity statistics
```

**Verbose Output Includes:**
- Each prompt and its tokenization
- Generated output samples
- Token count statistics
- Validation check results
- Success/failure indicators

---

### 2. Pipeline Orchestration Test

**File:** `test_pipeline_orchestration.py`

**Arguments:**
```bash
python3 test_pipeline_orchestration.py /path/to/repo
```

**Coverage:**

```
[0/6] Repository Verification
  ✓ Check repository exists
  ✓ Verify Git repository
  ✓ Count all commits (all branches)
  ✓ List branches
  ✓ Show branch count and details

[1/6] Phase 0: Repository Analysis
  ✓ Run GitAnalyzer on repository
  ✓ Get repository statistics
  ✓ Calculate training parameters
  ✓ Show commit counts
  ✓ Display estimated epochs

[2/6] Phase 1: Git Scraping
  ✓ Extract commits with metadata
  ✓ Get author information
  ✓ Extract commit dates
  ✓ Process commit messages
  ✓ Handle multiple branches

[3/6] Phase 2: Tokenization
  ✓ Load tokenizer
  ✓ Tokenize sample commits
  ✓ Count tokens per commit
  ✓ Analyze token distribution
  ✓ Estimate for full dataset

[4/6] Phase 3: Embeddings Configuration
  ✓ Verify embedding model
  ✓ Check embedding dimension
  ✓ Validate Qdrant configuration
  ✓ Check storage location

[5/6] Phase 4: Training Configuration
  ✓ Load training_config.yaml
  ✓ Load training_config_rust.yaml
  ✓ Verify all hyperparameters
  ✓ Check LoRA configuration
  ✓ Validate early stopping config

[6/6] Manifest Structure Validation
  ✓ Check manifest exists or create
  ✓ Validate required keys
  ✓ Verify phase results
  ✓ Check timestamps
  ✓ Validate repository statistics
```

**Verbose Output Includes:**
- Branch breakdown with commit counts
- Repository statistics
- Sample commits with full metadata
- Token count analysis
- Config file contents
- Manifest structure validation

---

### 3. StarCoder2-3B + 4-bit + LoRA Test

**File:** `test_starcoder_lora_quantization.py`

**Requirements:**
- GPU with 6GB+ VRAM
- ~15-20 minutes
- Network access for model download

**Coverage:**

```
[1/6] Model Loading with 4-bit Quantization
  ✓ Load config file
  ✓ Initialize OptimizedModelTrainer
  ✓ Load StarCoder2-3B with bitsandbytes 4-bit quantization
  ✓ Load tokenizer
  ✓ Verify model quantization
  ✓ Count trainable parameters
  ✓ Show GPU memory usage
  ✓ Display model class information

[2/6] LoRA Configuration Verification
  ✓ Load LoRA config from YAML
  ✓ Verify rank (r) value
  ✓ Verify alpha value
  ✓ Check target modules
  ✓ Verify dropout rate
  ✓ Confirm PEFT model wrapping
  ✓ Calculate parameter reduction %
  ✓ Show merged model capability

[3/6] Data Loading Pipeline
  ✓ Create Rust sequence samples
  ✓ Load training sequences
  ✓ Create train/val split
  ✓ Verify dataloader batches
  ✓ Check batch shapes
  ✓ Inspect batch data types
  ✓ Sample batch values

[4/6] Full Training Loop (2 epochs)
  ✓ Initialize trainer
  ✓ Run 2 complete epochs
  ✓ Show epoch-by-epoch metrics:
    - Training loss
    - Validation loss
    - Perplexity
    - Gradient norms
    - GPU memory usage
    - Time per epoch
  ✓ Display loss history
  ✓ Show gradient statistics
  ✓ Final model performance

[5/6] Model Saving and Artifacts
  ✓ Check output directory
  ✓ Verify pytorch_model.bin or model.safetensors
  ✓ Check config.json
  ✓ Check tokenizer files
  ✓ Check training_info.json
  ✓ Report total model size
  ✓ Validate artifact structure
  ✓ Verify training metadata

[6/6] 4-bit Quantization Memory Efficiency
  ✓ Show theoretical model sizes:
    - Full FP32: 12.6 GB
    - FP16: 6.3 GB
    - 8-bit: 3.15 GB
    - 4-bit: 2.0 GB ← Used
  ✓ Verify 6GB GPU fit
  ✓ Calculate headroom
  ✓ LoRA adapter overhead
```

**Verbose Output Includes:**
- Real-time epoch progress
- Per-epoch training metrics
- GPU memory tracking
- Gradient statistics
- Loss history
- Training artifacts
- Model sizes
- Memory efficiency analysis

---

## Master Test Runner Output

The master runner (`run_full_coverage_test_suite.py`) provides:

1. **Environment Validation**
   ```
   ✓ Python: 3.10
   ✓ torch
   ✓ transformers
   ✓ peft
   ✓ bitsandbytes
   
   GPU: NVIDIA A10 Tensor Core GPU
     Total memory: 23.0 GB
   ```

2. **Suite Execution**
   - Runs each test sequentially
   - Streams output in real-time
   - Captures timing information
   - Tracks pass/fail status

3. **Final Summary**
   ```
   Test Suite Results:
   
   Suite Name                           Status     Duration
   ────────────────────────────────────────────────────────
   Behavioral Evaluation Test           ✓ PASS     487.3s
   Pipeline Orchestration Test          ✓ PASS     342.1s
   StarCoder2-3B + 4-bit + LoRA Test    ✓ PASS     1247.8s
   
   Summary:
     Total suites run: 3
     Passed: 3
     Failed: 0
     Success rate: 100.0%
   
   Timing:
     Total duration: 2077.2s (34.6 minutes)
     Start time: 2025-12-10 12:00:00
     End time: 2025-12-10 12:34:37
   ```

---

## Expected Performance Metrics

### Behavioral Evaluation
- Config loading: <1 second
- Tokenization: <5 seconds
- Code generation: 3-5 seconds per prompt
- Validation: <2 seconds
- Total: ~10 minutes

### Pipeline Orchestration
- Repository analysis: 10-30 seconds
- Git scraping: 5-15 seconds
- Tokenization: 2-5 seconds
- Config validation: <1 second
- Manifest validation: <1 second
- Total: ~10 minutes (mostly I/O dependent)

### StarCoder2-3B + 4-bit + LoRA
- Model download: 5-10 minutes (first time only)
- Model loading: 15-30 seconds
- Data loading: <5 seconds
- Training (2 epochs): 10-15 minutes
- Model saving: <5 seconds
- Total: ~15-25 minutes

---

## Troubleshooting

### Out of Memory (OOM)

**If:** `torch.cuda.OutOfMemoryError`

**Fix:** Reduce batch size in config
```yaml
training:
  batch_size_large: 2  # instead of 4
  gradient_accumulation_steps: 4  # increase to compensate
```

### Model Download Fails

**If:** `ConnectionError` during model load

**Fix:** 
- Check internet connection
- Use cached model: `HF_HOME=~/.cache/huggingface`
- Pre-download: `huggingface-cli download bigcode/starcoder2-3b`

### Tokenizer Issues

**If:** `Tokenizer not found`

**Fix:**
```bash
pip install --upgrade tokenizers
pip install --upgrade transformers
```

### GPU Not Detected

**If:** `Device: cpu` instead of `cuda`

**Fix:**
```bash
python3 -c "import torch; print(torch.cuda.is_available())"
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## Interpreting Results

### Success Indicators
- ✓ All phases show green checkmark
- Training loss decreases across epochs
- Perplexity improves
- GPU memory stabilizes
- Model artifacts saved successfully

### Warning Signs
- ✗ Red X symbols indicate failures
- Training loss increases or plateaus
- GPU memory grows continuously
- Tokenization takes >30 seconds
- Model loading >2 minutes

---

## Output Storage

Test outputs are not automatically saved, but you can capture them:

```bash
# Capture master runner output
python3 run_full_coverage_test_suite.py --repo /path/to/repo 2>&1 | tee full_test_results.log

# Capture individual test
python3 test_starcoder_lora_quantization.py 2>&1 | tee starcoder_test.log

# Run with timestamps
python3 test_behavioral_evaluation.py 2>&1 | tee -a test_results_$(date +%Y%m%d_%H%M%S).log
```

---

## Summary of Coverage

✅ **Areas Fully Tested:**
- StarCoder2-3B with 4-bit quantization
- LoRA adapter creation and training
- Full pipeline orchestration (all 5 phases)
- Behavioral evaluation system
- Config-driven behavior
- Real model training
- Hardware monitoring
- Output validation
- Manifest generation

✅ **Metrics Captured:**
- Training loss/perplexity across epochs
- GPU memory usage
- Gradient norms
- Learning rate tracking
- Token counts
- Commit counts
- Execution timing
- Success rates

✅ **Languages Covered:**
- Default Python-focused configs
- Rust-optimized configs
- Multi-language prompt handling

---

**Run the master suite to get full test coverage in ~35 minutes!**

```bash
python3 run_full_coverage_test_suite.py --repo /Users/ianreitsma/projects/the-block
```
