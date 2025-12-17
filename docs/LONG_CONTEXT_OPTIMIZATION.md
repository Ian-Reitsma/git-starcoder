# Long-Context Optimization Guide

This guide explains the long-context optimization strategy implemented in this codebase for training a code model (StarCoder2-3B) with maximum effective context on an 8GB Mac.

## Table of Contents
1. [Strategy Overview](#strategy-overview)
2. [Hardware Limits](#hardware-limits)
3. [Multi-Phase Training](#multi-phase-training)
4. [Sequence Building](#sequence-building)
5. [Training Commands](#training-commands)
6. [Monitoring & Evaluation](#monitoring--evaluation)
7. [Troubleshooting](#troubleshooting)

---

## Strategy Overview

**Goal**: Maximize the model's effective context window (ability to understand and reason over long code spans) within 8GB RAM, trading training time for quality.

**Key Decisions**:
- **Sequence length**: Start with 2048 tokens (phase 1-2), extend to 4096 (phase 3, if stable)
- **LoRA + 4-bit quantization**: Fit 3B model + gradients in ~6GB, leaving room for batch + activations
- **Multi-phase curriculum**: Progressively increase sequence difficulty and length
- **Hierarchical packing**: Multi-file sequences to teach cross-file reasoning

**Effective Context** ≠ max_position_embeddings. It's the range over which gradients flow effectively during training.

---

## Hardware Limits

Your system: **8GB RAM, 8-core CPU, no GPU**

### Memory Budget Breakdown (8GB total)
- **Model weights (4-bit StarCoder2-3B)**: ~750 MB
- **LoRA adapters**: ~200-300 MB (rank=32, 3 modules)
- **Optimizer state** (AdamW): ~600-800 MB
- **Gradients (checkpointed)**: ~500 MB
- **Batch activations**: ~1-2 GB (depends on batch size, sequence length)
- **PyTorch overhead**: ~500 MB
- **Buffer (safety margin)**: ~1 GB

**Result**: Comfortable at:
- Sequence length: 2048 tokens
- Per-GPU batch: 1
- Gradient accumulation: 16 (effective batch ~16)
- Precision: bfloat16 (float32 would be too tight)

### If you hit OOM:
1. Reduce `gradient_accumulation_steps` (16 → 8 or 4)
2. Reduce `sequence_length` (2048 → 1536 or 1024)
3. Reduce `LoRA rank` (32 → 16)
4. Enable more aggressive gradient checkpointing

---

## Multi-Phase Training

Training is split into **3 phases**. Each phase has specific goals, sequence properties, and learning rate schedules.

### Phase 1: General Adaptation (epochs 1-5)
**Goal**: Establish general code understanding at reasonable sequence lengths

- **Sequence length**: 512-1024 tokens
- **Curriculum difficulty**: easy → medium
- **LR multiplier**: 1.0x (normal base_learning_rate)
- **Typical sequences**: Single files, simple patterns
- **Batch**: mixed difficulties to prevent overfitting

**What's happening**: The model learns fundamental tokenization patterns, syntax, and short-range dependencies.

### Phase 2: Long-Context Specialization (epochs 6-13)
**Goal**: Adapt to long sequences and learn cross-file patterns

- **Sequence length**: 1536-2048 tokens
- **Curriculum difficulty**: medium → hard
- **LR multiplier**: 0.5x (slower, more stable learning)
- **Typical sequences**: Multi-file spans, commit evolution chains
- **Batch**: predominantly longer sequences

**What's happening**: Lower LR prevents gradient instability at longer ranges. Model learns to track state across 2K tokens.

### Phase 3: Extended Context (epochs 14-21, optional)
**Goal**: Experiment with even longer sequences (3K-4K tokens)

- **Sequence length**: 3072-4096 tokens
- **Curriculum difficulty**: hard → very_hard
- **LR multiplier**: 0.1x (extreme stability)
- **Typical sequences**: Full files + related contexts, repo-wide spans
- **Batch**: long-only, rarest patterns

**What's happening**: At 10% LR, gradient flow becomes very conservative. Only activates if Phase 2 converges stably.

#### Configuration (training_config.yaml)
```yaml
training:
  long_context:
    # Phase 1
    phase1_max_length: 1024
    phase1_epochs: 5
    
    # Phase 2
    phase2_min_length: 1536
    phase2_max_length: 2048
    phase2_epochs: 8
    phase2_lr_multiplier: 0.5
    
    # Phase 3 (disabled by default)
    phase3_min_length: 3072
    phase3_max_length: 4096
    phase3_epochs: 8
    phase3_lr_multiplier: 0.1
    phase3_enabled: false  # Enable after Phase 2 succeeds
```

---

## Sequence Building

### Sequence Types

#### 1. Hierarchical Multi-File (Phase 2-3)
```
<|FILE_START|> path/to/core.rs
  [... 800 tokens ...]
<|FILE_END|>

<|FILE_START|> path/to/utils.rs
  [... 800 tokens ...]
<|FILE_END|>

<|FILE_START|> path/to/tests.rs
  [... 400 tokens ...]
<|FILE_END|>
```

**Why**: Teaches the model:
- Cross-file dependencies
- How imports/functions flow across files
- Realistic code organization

#### 2. Commit Evolution Chain (Phase 2-3)
```
<|COMMIT_START|> a1b2c3d
  [file content from commit a1b2c3d]
<|COMMIT_END|>

<|COMMIT_START|> e4f5g6h
  [file content from commit e4f5g6h - modified]
<|COMMIT_END|>

<|COMMIT_START|> i7j8k9l
  [file content from commit i7j8k9l - refactored]
<|COMMIT_END|>
```

**Why**: Teaches the model:
- Code evolution and refactoring patterns
- How to read diffs implicitly (before/after)
- Long-range causality (bug fixes, architectural changes)

#### 3. Synthetic Long-Context QA (Phase 2, eval only)
```
<|SNIPPET_0|> [imports and type definitions]
<|SNIPPET_1|> [function A implementation]
<|SNIPPET_2|> [function B implementation]
<|SNIPPET_3|> [test code]

QUESTION: What does function B depend on?
ANSWER: [function A's signature from SNIPPET_1]
```

**Why**: Evaluates if the model can attend to and reason over specific distant content.

### Building Sequences

```python
from dataset_builder_long_context import LongContextSequenceBuilder
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bigcode/starcoder2-3b")
builder = LongContextSequenceBuilder(
    tokenizer=tokenizer,
    max_length=2048,
    curriculum_mode="hierarchical_long_context"
)

# Build multi-file sequence
seq = builder.build_hierarchical_sequence(
    file_segments=[
        ("core.rs", open("core.rs").read()),
        ("utils.rs", open("utils.rs").read()),
        ("tests.rs", open("tests.rs").read()),
    ],
    metadata_base={"source": "repo_analysis"},
)

if seq:
    print(f"Sequence: {len(seq.tokens)} tokens")
    print(f"Difficulty: {seq.metadata['curriculum_difficulty']}")
    print(f"Files: {seq.metadata['files_included']}")
```

---

## Training Commands

### 1. Prepare Long-Context Data
```bash
python prepare_long_context.py \
    --repo /path/to/the-block \
    --output-dir ./long_context_prep \
    --max-sequence-length 2048 \
    --phases 3 \
    --verbose
```

Outputs:
- `preparation_summary.json`: Analysis + curriculum
- Statistics on large files, dependencies

### 2. Tokenize with Long Sequences
```bash
python tokenizers/git_tokenizer_rich.py \
    --repo /path/to/the-block \
    --model bigcode/starcoder2-3b \
    --output sequences_2048_tokens.json \
    --sequence-length 2048 \
    --overlap 512
```

### 3. Run Training with Multi-Phase Curriculum
```bash
python training/model_trainer_unified.py \
    --config training_config.yaml \
    --sequences sequences_2048_tokens.json \
    --output ./trained_long_context_model \
    --epochs 20 \
    --verbose
```

The trainer will:
1. Load `LongContextPhaseScheduler` from config
2. Per epoch, filter sequences by phase curriculum
3. Adjust LR based on phase
4. Log phase transitions

### 4. Enable Phase 3 (After Phase 2 Succeeds)
```yaml
# training_config.yaml
long_context:
  phase3_enabled: true  # Set to true
```

```bash
python training/model_trainer_unified.py \
    --config training_config.yaml \
    --sequences sequences_4096_tokens.json \
    --checkpoint ./trained_long_context_model/best \
    --output ./trained_long_context_model_extended \
    --epochs 8 \
    --verbose
```

---

## Monitoring & Evaluation

### Metrics to Track

1. **Per-Phase Loss & Perplexity**
   - Should decrease within each phase
   - May spike at phase transitions (expected; LR change)
   - Phase 2 loss > Phase 1 is normal (longer sequences)

2. **Gradient Flow** (logged per phase)
   - Max gradient norm should stay <1.0 (clipped at 1.0)
   - Min gradient norm should not be too close to 0 (layer saturation)
   - Watch for divergence at phase transitions

3. **Attention Pattern Analysis** (if enabled)
   - Histogram of attention weights per position
   - Should spread across sequence (not just recent tokens)
   - Extended context phases should show longer-range attention

4. **Validation Loss on Phase-Specific Sequences**
   - Phase 1 val: easy/medium sequences
   - Phase 2 val: hard sequences
   - Phase 3 val: very_hard sequences

### Eval Harness (Optional)

Create a test set of long-context coding tasks:
```python
# test_long_context.py
tasks = [
    {
        "context": "[2K tokens of imports + main struct definition]",
        "question": "What are the lifetime parameters of MainStruct?",
        "expected": "'a, 'b",
    },
    # More tasks...
]

for task in tasks:
    # Generate from model
    output = model.generate(task["context"] + task["question"])
    # Check if output matches expected
```

---

## Troubleshooting

### OOM During Training

1. **Immediate fix**: Reduce gradient_accumulation_steps
   ```yaml
   training:
     gradient_accumulation_steps: 8  # was 16
   ```

2. **Sequence length fix**: Reduce max_position_embeddings
   ```yaml
   model:
     max_position_embeddings: 1024  # was 2048
   ```

3. **LoRA rank fix**: Reduce LoRA rank
   ```yaml
   lora:
     r: 16  # was 32
   ```

### Loss Not Decreasing

1. Check if **LR is too high**: Reduce base_learning_rate
   ```yaml
   training:
     base_learning_rate: 2.5e-5  # was 5e-5
   ```

2. Check if **sequences are too hard**: Stay in Phase 1 longer
   ```yaml
   long_context:
     phase1_epochs: 10  # was 5
   ```

3. **Gradient clipping**: If max_grad_norm is hit frequently, something is unstable
   - Reduce LR further
   - Reduce sequence length
   - Increase warmup_ratio

### Phase Transition Spike

Loss may spike when moving from Phase 1 → Phase 2. **This is expected** because:
- Sequences are suddenly longer (1024 → 1536+)
- LR drops to 0.5x (gradients flow differently)
- Data distribution changes

**What to do**: Let it stabilize (usually recovers in 2-3 epochs). If it doesn't recover:
1. Reduce the LR multiplier drop (0.5 → 0.7)
2. Increase phase 1 epochs (more base stability)
3. Use a learning rate scheduler (e.g., cosine annealing within phases)

---

## Summary: Optimization Checklist

- [ ] Config: max_position_embeddings = 2048
- [ ] Config: LoRA rank = 32, target_modules includes c_fc
- [ ] Config: gradient_accumulation_steps = 16
- [ ] Config: use_gradient_checkpointing = true
- [ ] Config: warmup_ratio = 0.15
- [ ] Data: Build hierarchical multi-file sequences
- [ ] Data: Build commit evolution chains
- [ ] Training: Use LongContextPhaseScheduler
- [ ] Training: Monitor per-phase metrics
- [ ] Eval: Test long-context reasoning on validation set
- [ ] Phase 3: Enable only after Phase 2 converges

---

## References

- **Long-Context LLMs**: https://huggingface.co/blog/long-context-llms
- **LoRA**: Lora: Low-rank adaptation of large language models (https://arxiv.org/abs/2106.09685)
- **Flash Attention**: https://github.com/HazyResearch/flash-attention
- **Gradient Checkpointing**: https://pytorch.org/docs/stable/generated/torch.utils.checkpoint.checkpoint.html
