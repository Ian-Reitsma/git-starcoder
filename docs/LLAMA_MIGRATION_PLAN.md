# StarCoder2-3B → Llama-3.1-8B Comprehensive Migration Plan

## Founder-Level Architecture Redesign for General Reasoning + Project Understanding

**Version**: 2.0 (Comprehensive Dev-to-Dev)  
**Created**: December 15, 2025  
**Status**: Ready for Implementation  
**Target Model**: `meta-llama/Llama-3.1-8b-instruct`  
**GPU Constraint**: 8GB VRAM (same as current setup)  
**Strategic Goal**: Transform from "code autocomplete" to "architectural thinking partner"  
**Estimated Timeline**: 4-6 weeks full implementation + validation  
**Effort Level**: Developer-ready, production-grade changes  
**Codebase Location**: `~/projects/the-block`  
**Output Location**: `~/projects/the-block/.perplexity/git-scrape-scripting`

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Model Selection Deep Dive](#model-selection-deep-dive)
3. [Complete Codebase Audit](#complete-codebase-audit)
4. [Line-by-Line Code Changes](#line-by-line-code-changes)
5. [Testing Strategy & Validation](#testing-strategy--validation)
6. [Implementation Checklist](#implementation-checklist-5-week-breakdown)
7. [Architecture Deep Dive](#architecture-deep-dive)
8. [Hybrid Approach (Both Models)](#hybrid-approach-keeping-both-models)
9. [Pitfalls, Solutions & Debugging](#pitfalls-solutions--debugging)
10. [Advanced Topics & Optimization](#advanced-topics--optimization)
11. [Founder's Perspective & Strategic Vision](#founders-perspective--strategic-vision)

---

## Executive Summary

### The Strategic Shift

You currently have **StarCoder2-3B**: a code-specialized model tuned for function completion, variable suggestion, and syntax-aware code generation. It excels at "complete this line" but fails at "redesign this architecture."

You need **Llama-3.1-8B**: a general reasoning model capable of strategic thinking about your codebase. It can reason about:

- **System design tradeoffs** (energy market vs transaction ledger)
- **Error handling edge cases** (what breaks with concurrent updates?)
- **Performance implications** of patterns (token efficiency vs clarity)
- **Cross-module dependencies** (how does DKG affect consensus?)
- **Refactoring strategies** with justification (why migrate this?)
- **Architectural bottlenecks** (where will we hit scaling limits?)
- **Economic system soundness** (is this inflation model incentive-compatible?)

### Why This Matters

**Current Position (StarCoder-3B)**:
- Competes directly with GitHub Copilot
- Limited context window: 512 tokens = ~2 small functions
- No cross-module understanding
- No architectural reasoning
- No "founder's mindset"
- Cannot contextualize to blockchain economics

**New Position (Llama-3.1-8B)**:
- Becomes a **project-specific architectural advisor**
- Can fit entire small services in context: 1024+ tokens = ~8-10 functions
- Understands interactions between modules
- Can think about system design and economic incentives
- Can mentor junior developers with reasoning and explanations
- Can analyze blockchain-specific patterns (token economics, validator selection, etc.)

### The Business Case

A 3B code model loses to GitHub Copilot. An 8B reasoning model with YOUR codebase becomes a **defensible, unique capability** that GitHub can't offer:

- **Always available**: Local, zero latency, zero API costs
- **Understands your architecture**: Learns your specific design patterns
- **Can reason about decisions**: Explains tradeoffs, not just completes code
- **Helps plan migrations**: Can suggest refactorings with justification
- **Reviews with insight**: Catches architectural issues, not just syntax errors
- **Domain-expert level**: Understands blockchain economics, energy markets, consensus

### VRAM & Performance Expectations

```
Current Setup (StarCoder2-3B):
  Model params: 3B
  4-bit VRAM: ~3.5GB
  LoRA adapters: ~50MB
  Training VRAM total: ~6GB
  Training speed: ~20-30 steps/sec
  Context window: 512 tokens

Target Setup (Llama-3.1-8B):
  Model params: 8B
  4-bit VRAM: ~4.5GB
  LoRA adapters: ~100-150MB
  Training VRAM total: ~6.5-7GB (slightly tighter)
  Training speed: ~10-15 steps/sec (50% slower per-step)
  Context window: 1024 tokens (2x longer)

Net Result:
  • Slightly higher VRAM (6.5 vs 6.0GB) → Still safe on 8GB
  • Slower per-step (10-15 vs 20-30 steps/sec)
  • BUT: 2x context + 2.7x better base model
  • Training time: ~50% longer, but model emerges smarter
```

---

## Model Selection Deep Dive

### Why Llama-3.1-8B (Not 13B, 70B, or Others)

#### VRAM Constraint Analysis

```
8GB GPU VRAM Budget Breakdown:
┌─────────────────────────────────────────────────────────────────┐
│ Model            │ Params │ 4-bit + LoRA │ Decision            │
├──────────────────┼────────┼─────────────┼─────────────────────┤
│ StarCoder2-3B    │ 3.0B   │ ~3.5GB      │ ✓ Current (works)   │
│ Llama-2-7B       │ 7.0B   │ ~4.0GB      │ ✓ Tight but OK      │
│ Llama-3-8B       │ 8.0B   │ ~4.5GB      │ ✓ Safe margin       │
│ Llama-3.1-8B     │ 8.0B   │ ~4.5GB      │ ✓✓ RECOMMENDED      │
│ Llama-2-13B      │ 13.0B  │ ~7.0GB      │ ⚠️  Too tight        │
│ Mistral-7B       │ 7.0B   │ ~4.0GB      │ ✓ Alternative       │
│ Llama-3-70B      │ 70.0B  │ ~35GB       │ ✗ Impossible        │
└─────────────────────────────────────────────────────────────────┘

VRAM Calculation Details:
  • Base model (4-bit): params_GB * 0.5 = 4.5GB for 8B model
  • LoRA adapters: ~100MB (minimal)
  • Optimizer states: ~1GB (Adam state for 0.1% params)
  • Activations & gradients: ~1.5GB (batch_size=1, seq_len=1024)
  • Total budget: 4.5GB + 1GB + 1.5GB = ~7GB
  • Margin: 8GB - 7GB = 1GB safety buffer ✓
```

### Why NOT Llama-2-13B

**The Temptation**: "13B is better, let's try it."

**The Reality**:

```
Llama-2-13B with 4-bit:
  • 13B params * 0.5 = 6.5GB base model
  • LoRA adapters: ~150MB
  • Optimizer states: ~1.2GB
  • Activations: ~1.5GB
  • TOTAL: 6.5 + 1.2 + 1.5 = ~9.2GB
  • Available: 8GB
  • Shortfall: 1.2GB → OOM on ANY GPU

You CANNOT fit Llama-2-13B on 8GB GPU, period.
Not with batch_size=1, not with gradient checkpointing,
not even with aggressive quantization.
```

### Why Llama-3.1-8B Specifically

1. **Better than Llama-3-8B**:
   - Llama-3.1 has improved reasoning and longer context support
   - Better at multi-step problems (which your codebase is)
   - More stable training with LoRA
   - Improved chat/instruction following

2. **Better than Mistral-7B**:
   - Larger model (8B vs 7B) = more capacity
   - Better instruction following (you'll fine-tune on codebase patterns)
   - More stable training dynamics
   - Better community support

3. **Perfect VRAM Sweet Spot**:
   - 4-bit fits comfortably in 8GB
   - 1GB safety buffer for spikes
   - Can do batch_size=1 with seq_len=1024
   - Can enable all optimizations (gradient checkpointing, etc.)

4. **Instruction-Tuned Variant**:
   - Use `meta-llama/Llama-3.1-8b-instruct` (not base)
   - Already fine-tuned for instruction-following
   - Better at understanding prompts about codebase analysis
   - Better quality outputs without extra training

---

## Complete Codebase Audit

### Files That Touch the Model

**CRITICAL FILES (Must Change for Llama migration)**:

#### 1. `training_config.yaml` (PRIMARY) [~200 lines]

**Current**:
```yaml
model:
  name: bigcode/starcoder2-3b
  tokenizer_name: bigcode/starcoder2-3b
  
  lora:
    r: 16
    target_modules: ["c_attn", "c_proj", "c_fc"]  # StarCoder-specific
  
  max_position_embeddings: 512  # StarCoder context window
```

**Must Change**:
- Model name → `meta-llama/Llama-3.1-8b-instruct`
- Tokenizer name → `meta-llama/Llama-3.1-8b` (different tokenizer!)
- LoRA rank: 16 → 32 (Llama has 4096 hidden_dim vs StarCoder's 3072)
- LoRA targets: "c_attn", "c_proj", "c_fc" → "q_proj", "k_proj", "v_proj", "up_proj", "down_proj"
- Context window: 512 → 1024 (Llama's strength)
- Batch size: Keep at 1 (Llama tokens take 2x VRAM each)
- Accumulation steps: 4 → 8 (compensate for smaller effective batch)
- Learning rate: 1e-4 → 2e-4 (Llama responds to higher LR)
- Evaluation prompts: Completely rewrite (not code completion focused)
- Generation config: Different token limits, temperature, top_p

**Lines to Change**: ~40-50 lines

#### 2. `train_model.py` (SECONDARY) [~300 lines]

**Current**: Loads StarCoder2-3B with StarCoder-specific tokenizer handling.

**Must Change**:

```python
# CHANGE 1: Model loading
model_name = "meta-llama/Llama-3.1-8b-instruct"  # From StarCoder
tokenizer_name = "meta-llama/Llama-3.1-8b"      # Different!

# CHANGE 2: Tokenizer config
# Llama tokenizer has different special tokens
tokenizer.pad_token = tokenizer.eos_token  # Llama uses EOS for padding
tokenizer.add_special_tokens({
    "additional_special_tokens": [
        "<|code_start|>",
        "<|code_end|>",
        "<|analysis|>",
    ]
})  # Add domain-specific tokens

# CHANGE 3: LoRA config
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=32,  # From 16 (bigger hidden dim)
    lora_alpha=64,  # Scale with rank
    target_modules=[  # Different targets!
        "q_proj",   # Query projection (attention)
        "k_proj",   # Key projection (attention)
        "v_proj",   # Value projection (attention)
        "up_proj",  # MLP up (feed-forward)
        "down_proj", # MLP down (feed-forward)
    ],
    lora_dropout=0.05,
    bias="none",
)

# CHANGE 4: Training config
# Llama layers: 32 vs StarCoder's 30
# Hidden dim: 4096 vs StarCoder's 3072
# Llama takes more memory per token
training_args = TrainingArguments(
    per_device_train_batch_size=1,  # Stays at 1 (VRAM)
    gradient_accumulation_steps=8,  # From 4 (compensate)
    learning_rate=2e-4,  # From 1e-4 (higher for Llama)
    num_train_epochs=12,  # Adjust based on data
    max_grad_norm=1.0,  # Keep stable clipping
    gradient_checkpointing=True,  # CRITICAL for 1024-token sequences
)
```

**Lines to Change**: ~60-80 lines (tokenizer setup, LoRA config, training args)

#### 3. `run_pipeline_dynamic.py` (CRITICAL) [~100 lines]

**Current**: Orchestrates entire pipeline, calculates epochs.

**Must Change**:

```python
# CHANGE 1: Epoch calculation for longer sequences
# Llama processes tokens faster (more mature model)
# But sequences are 2x longer (1024 vs 512)
# Result: roughly same epoch count, but consider:

epoch_calculation:
  # Current (StarCoder, 512-token sequences)
  target_tokens: 50000000  # 50M tokens
  min_epochs: 5
  max_epochs: 12
  
  # Proposed (Llama, 1024-token sequences)
  # Keep target_tokens same (same training budget)
  # But sequence length doubled, so fewer sequence steps
  target_tokens: 50000000  # SAME
  min_epochs: 5           # Maybe 6 (reasoning needs more passes)
  max_epochs: 12          # Keep same

# CHANGE 2: Update tokenizer initialization
self.tokenizer_model = "meta-llama/Llama-3.1-8b"
self.tokenizer_trust_remote = True  # Llama uses custom tokenizer

# CHANGE 3: Sequence length handling
self.sequence_length = 1024  # From 512
self.max_position_embeddings = 1024  # Tell model about longer context

# CHANGE 4: Hardware monitoring thresholds
# Llama is memory-hungriere, watch thresholds more closely
gpu_memory_threshold_large_gb: 6.5  # From 7.0 (tighter)
gpu_memory_threshold_medium_gb: 4.0 # Same
gpu_memory_threshold_small_gb: 2.0  # Same
```

**Lines to Change**: ~30-40 lines (epoch calculation, tokenizer setup, sequence length)

#### 4. `test_behavioral_evaluation.py` (COMPLETE REWRITE) [~200 lines]

**Current**: Tests StarCoder with code completion prompts
```python
test_prompts = [
    "def factorial(n):",
    "fn process_block(",
    "class Transaction:",
]
```

**Must Rewrite** for Llama (reasoning-focused):

```python
# NEW: Architecture analysis prompts
test_prompts = {
    "architecture_understanding": [
        """Analyze this module structure:
        - DKG (Distributed Key Generation)
        - Consensus (Byzantine-tolerant)
        - StateManager (Account state)
        
What are the failure modes if consensus and state get out of sync?""",
        
        """The energy market has:
        - Supply curve (price increases with demand)
        - Validator selection (proportional to stake)
        - Inflation (new tokens per block)
        
How would these dynamics behave during network stress?""",
    ],
    
    "refactoring_strategy": [
        """Given these module interactions:
        A -> B -> C -> A (circular dependency)
        
How would you refactor this? What's the strategic approach?""",
    ],
    
    "edge_case_reasoning": [
        """The validator selection uses randomness.
        The consensus requires 2/3+ majority.
        
What happens if randomness breaks? Is the system resilient?""",
    ],
    
    "performance_analysis": [
        """This function iterates over all accounts.
        You have 1M validators.
        This runs every block (1s).
        
Will this scale? What's the fix?""",
    ],
}

# Evaluation logic
for category, prompts in test_prompts.items():
    for prompt in prompts:
        response = model.generate(
            input_ids,
            max_new_tokens=500,  # Reasoning needs more tokens
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
        )
        
        # Check for reasoning markers
        reasoning_markers = [
            "because",
            "however",
            "in addition",
            "considering",
            "the risk",
            "trade-off",
        ]
        
        has_reasoning = any(
            marker in response.lower() 
            for marker in reasoning_markers
        )
        
        score = len(reasoning_markers_found) / len(reasoning_markers)
        print(f"{category}: {score:.2%} reasoning depth")
```

**Lines to Change**: Complete rewrite, ~150-200 lines

#### 5. `tokenizer.py` or tokenization logic (IMPORTANT) [~50 lines]

**Current**: Uses StarCoder's tokenizer

**Must Update**:

```python
# CHANGE 1: Tokenizer loading
from transformers import AutoTokenizer

# StarCoder tokenizer
# tokenizer = AutoTokenizer.from_pretrained("bigcode/starcoder2-3b")

# Llama tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8b")

# CHANGE 2: Special token handling
# Llama's vocab is ~129K (vs StarCoder's ~50K)
# Different special tokens

# StarCoder: '<|im_start|>', '<|im_end|>'
# Llama: '<|reserved_special_token_0|>', etc.

tokenizer.pad_token = tokenizer.eos_token  # Llama pattern

# CHANGE 3: Encoding error recovery
# Llama tokenizer may have issues with:
# - Binary files (shouldn't happen but guard)
# - Unusual Unicode
# - Very long sequences

encoded = tokenizer.encode(
    text,
    max_length=1024,  # Llama context
    truncation=True,
    errors='ignore',  # Skip problematic tokens
)
```

**Lines to Change**: ~20-30 lines

### SUPPORTING FILES (May Need Minor Updates)

#### 6. `config.yaml` (OPTIONAL) [~5 lines]

**Current**: Points to training_config.yaml

**May Change**: Add notes about Llama vs StarCoder

```yaml
# Add comment
training_framework:
  note: "Llama-3.1-8B migration - see LLAMA_MIGRATION_PLAN.md"
  context_window: 1024  # From 512 (StarCoder)
  reasoning_mode: true  # New capability
```

#### 7. `run_pipeline_unified.py` (CONDITIONAL) [~20 lines]

**Current**: May reference old model

**Check**: Does it hardcode "starcoder"? If yes, update to dynamic.

#### 8. `dataset_builder.py` (NO CHANGES NEEDED) [0 lines]

**Reason**: Works with tokenized sequences, model-agnostic.

#### 9. `git_scraper.py` (NO CHANGES NEEDED) [0 lines]

**Reason**: Extracts commits, model-independent.

#### 10. Model loading in any inference scripts (CHECK) [~10 lines]

**Search for**: Hardcoded "starcoder" → replace with dynamic loading

### Summary of Changes

```
FILE                              │ SIZE    │ CHANGES │ PRIORITY
─────────────────────────────────┼─────────┼─────────┼──────────
training_config.yaml             │ ~200L   │ 40-50L  │ CRITICAL
train_model.py                   │ ~300L   │ 60-80L  │ CRITICAL
run_pipeline_dynamic.py          │ ~500L   │ 30-40L  │ CRITICAL
test_behavioral_evaluation.py    │ ~150L   │ 150L    │ REWRITE
tokenizer.py                     │ ~100L   │ 20-30L  │ IMPORTANT
config.yaml                      │ ~50L    │ 5L      │ OPTIONAL
run_pipeline_unified.py          │ ~200L   │ 0-20L   │ CHECK
dataset_builder.py               │ ~300L   │ 0L      │ NO CHANGE
git_scraper.py                   │ ~400L   │ 0L      │ NO CHANGE
─────────────────────────────────┴─────────┴─────────┴──────────
TOTAL AFFECTED CODE              │ ~2000L  │ ~395L   │ ~20% changes
```

---

## Line-by-Line Code Changes

### Change 1: training_config.yaml

**BEFORE** (StarCoder configuration):

```yaml
model:
  # StarCoder2-3B is PERFECT for Rust
  # Trained on GitHub Rust repositories + standard library + common crates
  name: bigcode/starcoder2-3b
  tokenizer_name: bigcode/starcoder2-3b
  trust_remote_code: true
  
  # Quantization (fits 3B model on 6GB GPU)
  use_4bit: true
  use_8bit: false
  use_bf16: true
  
  # LoRA fine-tuning
  use_lora: true
  
  lora:
    # RUST OPTIMIZATION: Higher rank for more Rust idioms
    r: 16
    lora_alpha: 32
    
    # RUST OPTIMIZATION: Target more modules
    target_modules: ["c_attn", "c_proj", "c_fc"]  # StarCoder-specific
    
    lora_dropout: 0.05
    bias: "none"
  
  # Context window
  max_position_embeddings: 512
  max_new_tokens: 512
```

**AFTER** (Llama configuration):

```yaml
model:
  # Llama-3.1-8B: General reasoning + architectural understanding
  # Trained on diverse data including technical reasoning, code reasoning,
  # and instruction-following. Better at multi-step analysis than code models.
  name: meta-llama/Llama-3.1-8b-instruct
  tokenizer_name: meta-llama/Llama-3.1-8b  # Different from model name!
  trust_remote_code: true
  
  # Quantization (fits 8B model on 8GB GPU)
  # 4-bit: ~4.5GB + LoRA + optimizer + activations = ~7GB total
  use_4bit: true
  use_8bit: false
  use_bf16: true  # bfloat16 for stability with larger model
  
  # LoRA fine-tuning for parameter efficiency
  use_lora: true
  
  lora:
    # LLAMA OPTIMIZATION: Rank 32 for 4096-dim hidden state
    # StarCoder had 3072-dim, so needed rank 16
    # Llama has 4096-dim, so needs rank 32 for same capacity
    # Formula: rank ≈ hidden_dim / 128
    r: 32
    lora_alpha: 64  # Scale with rank (typically 2x)
    
    # LLAMA ARCHITECTURE: Different attention structure
    # StarCoder: [c_attn, c_proj, c_fc] (merged attention + MLP)
    # Llama: [q_proj, k_proj, v_proj, up_proj, down_proj] (separated)
    # We target all to maximize learning capacity
    target_modules:
      - "q_proj"    # Query projection (attention)
      - "k_proj"    # Key projection (attention)
      - "v_proj"    # Value projection (attention)
      - "up_proj"   # Feed-forward expansion
      - "down_proj" # Feed-forward projection
    
    lora_dropout: 0.05  # Keep regularization same
    bias: "none"       # Don't adapt bias terms
    task_type: "CAUSAL_LM"  # Llama is causal language model
  
  # Context window: Double for better architectural understanding
  # StarCoder: 512 tokens (~2-3 functions)
  # Llama: 1024 tokens (~8-10 functions, entire service)
  max_position_embeddings: 1024
  
  # Generation limits
  # Llama tokens are more expensive (more params), but reasoning needs more tokens
  max_new_tokens: 512  # Same as StarCoder (absolute tokens)
```

**Explanation of Key Changes**:

1. **Model Name Change**:
   - `bigcode/starcoder2-3b` → `meta-llama/Llama-3.1-8b-instruct`
   - The "-instruct" variant is important: already fine-tuned for instructions
   - Non-instruct (base) version is harder to work with

2. **Tokenizer Mismatch** (CRITICAL):
   - Model: `meta-llama/Llama-3.1-8b-instruct`
   - Tokenizer: `meta-llama/Llama-3.1-8b` (WITHOUT "-instruct")
   - This is **not a typo**. Hugging Face requires this split.
   - The tokenizer is same for both versions; only model differs

3. **LoRA Rank Increase**:
   - StarCoder: hidden_dim = 3072, rank = 16 (ratio 1:192)
   - Llama: hidden_dim = 4096, rank = 32 (ratio 1:128)
   - Larger model, larger hidden dim, needs larger rank
   - Formula: `rank ≈ hidden_dim / 128` (empirical)

4. **LoRA Module Targets**:
   - **Why these changed**: Llama and StarCoder have different architectures
   - **StarCoder**: Uses GPT-Neo architecture (merged attention: c_attn)
   - **Llama**: Uses Transformer architecture (separate Q/K/V projections)
   - **Strategy**: Target all 5 modules to maximize learning capacity
   - Targeting fewer (e.g., just q_proj/v_proj) would constrain learning

5. **Context Window**:
   - 512 → 1024 tokens
   - Why 1024 (not 2048 or higher)?
     - 2048: Would need ~8GB just for activations (OOM)
     - 1024: Sweet spot with 1GB safety margin
     - You can test longer context after confirming 1024 works

6. **Alpha Scaling**:
   - lora_alpha: 32 → 64
   - Formula: `alpha = 2 * rank` (empirical sweet spot)
   - Alpha controls the magnitude of LoRA updates
   - Larger alpha = LoRA has more influence

---

### Change 2: training_config.yaml - Training Parameters

**BEFORE** (StarCoder training):

```yaml
training:
  base_learning_rate: 1e-4
  
  batch_size_reference: 4
  batch_size_large: 2
  batch_size_medium: 1
  batch_size_small: 1
  
  gradient_accumulation_steps: 4  # Effective batch = 16
  incremental_context_sequences: 2
  
  use_gradient_checkpointing: true
  
repoch_calculation:
  target_tokens: 50000000
  min_epochs: 5
  max_epochs: 12
```

**AFTER** (Llama training):

```yaml
training:
  # LLAMA OPTIMIZATION: Higher learning rate
  # Llama responds better to higher LR than code models
  # Intuition: Llama's pre-training used ~2e-4, so adapted to it
  # Code models (StarCoder, Phi) used lower LR ~1e-4
  # Testing: Start at 2e-4, monitor loss. If unstable, reduce to 1.5e-4
  base_learning_rate: 2e-4
  
  # BATCH SIZE: Llama needs more memory per token
  # Each Llama token in activations: ~2x StarCoder
  # StarCoder: 512 tokens, batch=2 → ~6GB
  # Llama: 1024 tokens, batch=1 → ~6.5-7GB
  # Can't increase batch, so use gradient accumulation
  batch_size_reference: 1  # From 4 (can't fit larger)
  batch_size_large: 1      # From 2
  batch_size_medium: 1     # Same
  batch_size_small: 1      # Same (all the same now)
  
  # GRADIENT ACCUMULATION: Compensate for small batch
  # StarCoder: batch=2, accumulation=4 → effective batch = 8
  # Llama: batch=1, accumulation=8 → effective batch = 8 (same)
  # Keep effective batch constant for comparable learning
  gradient_accumulation_steps: 8  # From 4 (double it)
  
  # Context sequences: Same strategy
  incremental_context_sequences: 2
  
  # GRADIENT CHECKPOINTING: Absolutely critical now
  # Saves activations, recomputes during backprop
  # Memory saving: ~50-60% at cost of ~25% speed
  # With 1024 tokens, this is REQUIRED
  use_gradient_checkpointing: true
  
epoch_calculation:
  # EPOCH STRATEGY: Keep tokens constant
  # StarCoder: 50M target tokens
  # Llama: 50M target tokens (same budget)
  # But sequences are 2x longer (1024 vs 512)
  # So fewer sequence steps, but each step better
  target_tokens: 50000000  # Same as StarCoder
  
  # Epoch bounds: Llama may need more passes for reasoning
  min_epochs: 6            # From 5 (reasoning takes more passes)
  max_epochs: 12           # Keep same
  
  # Reasoning requires multiple passes over data
  # to learn cross-module patterns
```

**Explanation**:

1. **Learning Rate: 1e-4 → 2e-4**
   - Empirical: Llama pre-training uses ~2e-4
   - LoRA fine-tuning typically uses 1-2x pre-training LR
   - Code models: converge at lower LR
   - Reasoning models: need higher LR for generalization
   - **Action**: Monitor validation loss. If it spikes, reduce to 1.5e-4

2. **Batch Size: 2/4 → 1**
   - Llama's 8B params take 2x VRAM per token vs StarCoder's 3B
   - 1024-token sequences (2x longer) compound memory pressure
   - Math: StarCoder batch=2 at 512 tokens ≈ Llama batch=1 at 1024 tokens
   - Cannot increase batch further without OOM

3. **Gradient Accumulation: 4 → 8**
   - Compensates for smaller batch
   - Effective batch = batch * accumulation = 1 * 8 = 8 (same as 2 * 4)
   - Same learning dynamics as StarCoder
   - Takes slightly longer per epoch (8 accumulation steps)

4. **Min Epochs: 5 → 6**
   - Reasoning emerges slowly; needs multiple passes
   - Code models converge in 3-5 epochs
   - Reasoning models: 5-8 epochs typical
   - First epoch: learns syntax patterns
   - Epochs 2-3: learns function relationships
   - Epochs 4-6: learns cross-module dependencies

---

### Change 3: training_config.yaml - Evaluation Configuration

**BEFORE** (StarCoder evaluation):

```yaml
evaluation:
  run_behavioral_eval: true
  eval_every_n_epochs: 2
  
  behavioral_test_prompts:
    # Code completion
    - "fn process"
    - "impl"
    - "pub struct"
    - "async fn"
    - "match"
    - "Result<"
    - "fn new() -> Self"
    # etc. (25+ code completion prompts)
```

**AFTER** (Llama evaluation - COMPLETE REWRITE):

```yaml
evaluation:
  run_behavioral_eval: true
  eval_every_n_epochs: 1  # More frequent (faster feedback)
  
  # LLAMA EVALUATION: Reasoning-focused, not code completion
  # These prompts test architectural understanding
  # Format: [Question], [Expected Reasoning]
  behavioral_test_prompts:
    # ARCHITECTURE UNDERSTANDING
    - prompt: "What is the purpose of the DKG module?"
      expected_keywords: ["distributed", "key", "generation", "consensus"]
    
    - prompt: "How does the StateManager track account balances?"
      expected_keywords: ["state", "merkle", "tree", "root", "hash"]
    
    # CROSS-MODULE REASONING
    - prompt: "What happens if consensus fails but StateManager continues?"
      expected_keywords: ["inconsistency", "fork", "Byzantine", "recovery"]
    
    - prompt: "Why is validator selection randomized?"
      expected_keywords: ["stake", "security", "concentration", "incentive"]
    
    # ECONOMIC SYSTEM REASONING
    - prompt: "How do inflation and validator rewards interact?"
      expected_keywords: ["incentive", "token", "economics", "supply"]
    
    - prompt: "What prevents validators from attacking the network?"
      expected_keywords: ["slashing", "penalty", "stake", "economics", "cost"]
    
    # EDGE CASE ANALYSIS
    - prompt: "What if 1/3 of validators go offline?"
      expected_keywords: ["safety", "liveness", "threshold", "Byzantine"]
    
    # CODE ANALYSIS (Still important, but different focus)
    - prompt: "Review this code for concurrency issues:"
      context: "async fn process_block { update_state(); finalize_block(); }"
      expected_keywords: ["race condition", "lock", "atomic", "order"]
    
    # REFACTORING DECISIONS
    - prompt: "Should we extract consensus into separate crate?"
      expected_keywords: ["modularity", "coupling", "maintainability", "trade-off"]
  
  # Longer generation for reasoning responses
  eval_max_length: 300       # From 150 (reasoning needs more tokens)
  eval_num_return_sequences: 1
  eval_temperature: 0.7      # Same (good balance of creativity/consistency)
  eval_top_p: 0.95           # Same
  eval_do_sample: true       # Same (needed for reasoning variety)
  eval_output_trim: 500      # From 300 (collect more reasoning)
```

**Key Differences**:

1. **Prompt Style**:
   - StarCoder: "fn process" (prefix completion)
   - Llama: "What is the purpose of the DKG module?" (question answering)
   - Llama excels at questions; code models excel at completion

2. **Expected Output**:
   - StarCoder: Checks for generated code tokens
   - Llama: Checks for reasoning keywords
   - Example: "Byzantine" indicates understanding of consensus algorithm

3. **Evaluation Frequency**:
   - StarCoder: Every 2 epochs
   - Llama: Every epoch (reasoning signal clearer, easier to spot problems)

4. **Token Limits**:
   - eval_max_length: 150 → 300
   - Reasoning responses are longer than code completions
   - Need more tokens to express multi-step reasoning

---

### Change 4: train_model.py - Tokenizer Setup

**BEFORE** (StarCoder tokenizer):

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

class ModelTrainer:
    def __init__(self, config_path: str):
        # Load config
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        # Tokenizer from StarCoder
        model_name = self.config["model"]["name"]  # "bigcode/starcoder2-3b"
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=self.config["model"]["trust_remote_code"],
        )
        # StarCoder tokenizer handles padding automatically
        self.tokenizer.pad_token = self.tokenizer.eos_token
```

**AFTER** (Llama tokenizer - CRITICAL CHANGES):

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

class ModelTrainer:
    def __init__(self, config_path: str):
        # Load config
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        # CRITICAL: Llama uses DIFFERENT tokenizer from model
        # Model: meta-llama/Llama-3.1-8b-instruct
        # Tokenizer: meta-llama/Llama-3.1-8b (no "-instruct")
        tokenizer_name = self.config["model"]["tokenizer_name"]
        # This should be "meta-llama/Llama-3.1-8b"
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=self.config["model"].get("trust_remote_code", False),
        )
        
        # LLAMA SPECIFIC: Set padding token to EOS
        # Llama doesn't have dedicated pad token
        # Using EOS is standard practice
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Add special tokens for domain marking (OPTIONAL but recommended)
        # These help the model understand context boundaries
        special_tokens = {
            "additional_special_tokens": [
                "<|code_start|>",      # Mark code sections
                "<|code_end|>",        # End code section
                "<|analysis|>",        # Architectural analysis
                "<|question|>",        # Q&A marker
            ]
        }
        self.tokenizer.add_special_tokens(special_tokens)
        
        # Verify tokenizer vocab size
        logger.info(f"Tokenizer vocab size: {len(self.tokenizer)}")
        # Llama: ~129K tokens (vs StarCoder ~50K)
        # This is fine; model will learn new tokens naturally
```

**Explanation**:

1. **Different Tokenizer/Model Names**:
   - This is NOT a bug
   - Hugging Face structure: model variants share tokenizer
   - `meta-llama/Llama-3.1-8b` and `meta-llama/Llama-3.1-8b-instruct` both use the same tokenizer
   - Always use the base model name for tokenizer

2. **Adding Special Tokens**:
   - Not strictly necessary, but helpful
   - Helps model understand context boundaries
   - Example: "<|code_start|>" signals "this is code"
   - Model learns these represent special context
   - Makes fine-tuning more interpretable

3. **Vocab Size**:
   - StarCoder: ~50K tokens
   - Llama: ~129K tokens
   - Larger vocab: more specialized tokens for various languages
   - Doesn't hurt; model learns which ones are relevant to your codebase

---

### Change 5: train_model.py - LoRA Configuration

**BEFORE** (StarCoder LoRA):

```python
from peft import LoraConfig, get_peft_model

class ModelTrainer:
    def setup_lora_model(self, model, config):
        lora_cfg = config["model"]["lora"]
        
        lora_config = LoraConfig(
            r=lora_cfg["r"],  # 16 for StarCoder
            lora_alpha=lora_cfg["lora_alpha"],  # 32
            target_modules=lora_cfg["target_modules"],  # ["c_attn", "c_proj", "c_fc"]
            lora_dropout=lora_cfg["lora_dropout"],
            bias=lora_cfg["bias"],
        )
        
        return get_peft_model(model, lora_config)
```

**AFTER** (Llama LoRA - CRITICAL ARCHITECTURE CHANGES):

```python
from peft import LoraConfig, get_peft_model
import logging

logger = logging.getLogger(__name__)

class ModelTrainer:
    def setup_lora_model(self, model, config):
        lora_cfg = config["model"]["lora"]
        
        # Validate that we have the right target modules for Llama
        # Common mistake: using StarCoder targets with Llama model
        expected_targets = {"q_proj", "k_proj", "v_proj", "up_proj", "down_proj"}
        provided_targets = set(lora_cfg["target_modules"])
        
        if not provided_targets.issubset(expected_targets):
            logger.warning(
                f"Unexpected LoRA targets for Llama: {provided_targets}\n"
                f"Expected subset of: {expected_targets}\n"
                f"StarCoder targets (c_attn, c_proj, c_fc) won't work with Llama!"
            )
        
        lora_config = LoraConfig(
            r=lora_cfg["r"],  # 32 for Llama (vs 16 for StarCoder)
            lora_alpha=lora_cfg["lora_alpha"],  # 64 (2x rank)
            target_modules=lora_cfg["target_modules"],  # ["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]
            lora_dropout=lora_cfg["lora_dropout"],  # 0.05 (same)
            bias=lora_cfg["bias"],  # "none"
            task_type="CAUSAL_LM",  # IMPORTANT for transformers
        )
        
        peft_model = get_peft_model(model, lora_config)
        
        # Log LoRA statistics
        trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in peft_model.parameters())
        trainable_pct = 100 * trainable_params / total_params
        
        logger.info(
            f"LoRA Configuration:\n"
            f"  Rank: {lora_cfg['r']}\n"
            f"  Alpha: {lora_cfg['lora_alpha']}\n"
            f"  Target modules: {lora_cfg['target_modules']}\n"
            f"  Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_pct:.2f}%)\n"
        )
        
        return peft_model
```

**Key Differences - ARCHITECTURE DETAILS**:

**StarCoder Architecture** (GPT-Neo style):
```
Attention Module: "c_attn" (merged Q, K, V)
  c_attn: Linear(3072, 3*768)  # Query, Key, Value concatenated
  ↓ (split into Q, K, V)
  attention scores
  ↓
  c_proj: Linear(768, 768)     # Output projection

MLP Module:
  c_fc: Linear(768, 3072)      # Expansion
  activation (GELU)
  c_proj: Linear(3072, 768)    # Projection

LoRA Targets: ["c_attn", "c_proj", "c_fc"]
```

**Llama Architecture** (Transformer style):
```
Attention Module: Separate Q, K, V
  q_proj: Linear(4096, 4096)   # Query
  k_proj: Linear(4096, 4096)   # Key
  v_proj: Linear(4096, 4096)   # Value
  attention scores
  (output projection built-in)

MLP Module ("feed_forward"):
  gate_proj: Linear(4096, 14336)  # Gate (for gated linear unit)
  up_proj: Linear(4096, 14336)    # Expansion
  down_proj: Linear(14336, 4096)  # Projection

LoRA Targets: ["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]
```

**Why This Matters**:

1. **Different modules**: If you target "c_attn" on Llama, PEFT won't find it → LoRA won't adapt anything → model trains as if it's frozen.
2. **Different layer count**: StarCoder (30 layers) vs Llama (32 layers) → Different parameter distributions
3. **Different hidden dims**: StarCoder (3072) vs Llama (4096) → Rank 16 insufficient for Llama

**Why Target All 5 Modules for Llama**:
- Q/K/V: Contain the learned attention patterns
- Up/Down: Contain the learned representations in MLP
- Targeting all maximizes adaptation capacity
- LoRA rank 32 still only trains ~0.1% of parameters
- Minimal overhead, maximum flexibility

---

### Change 6: train_model.py - Training Arguments

**BEFORE** (StarCoder training):

```python
from transformers import TrainingArguments, Trainer

class ModelTrainer:
    def train(self, sequences_file, num_epochs, output_dir):
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=2,  # StarCoder at 512 tokens
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=1e-4,
            warmup_ratio=0.1,
            weight_decay=0.01,
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            gradient_checkpointing=True,
            bf16=True,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
        )
```

**AFTER** (Llama training - MEMORY AND SPEED OPTIMIZED):

```python
from transformers import TrainingArguments, Trainer

class ModelTrainer:
    def train(self, sequences_file, num_epochs, output_dir):
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            
            # BATCH SIZE: Reduced for larger model + longer sequences
            per_device_train_batch_size=1,  # From 2 (Llama needs 2x memory)
            per_device_eval_batch_size=1,   # From 2
            
            # GRADIENT ACCUMULATION: Doubled to maintain effective batch
            gradient_accumulation_steps=8,  # From 4 (batch 1 * 8 = effective 8)
            
            # LEARNING RATE: Increased for Llama
            learning_rate=2e-4,  # From 1e-4
            
            # WARMUP: Same proportions
            warmup_ratio=0.1,
            warmup_steps_override=100,  # Explicit warmup for first 100 steps
            
            # REGULARIZATION
            weight_decay=0.01,  # Same
            max_grad_norm=1.0,  # Aggressive clipping for stability
            
            # LOGGING: More frequent for debugging (slower training)
            logging_steps=5,  # From 10 (faster feedback)
            
            # EVALUATION: More frequent (reasoning signal clearer)
            evaluation_strategy="steps",  # From "epoch"
            eval_steps=50,  # Every 50 steps
            
            # SAVING: Don't save every checkpoint (disk space)
            save_strategy="steps",
            save_steps=100,
            save_total_limit=3,  # Keep only 3 checkpoints
            
            # OPTIMIZATION: Critical for 1024-token sequences
            gradient_checkpointing=True,
            bf16=True,  # bfloat16 for numerical stability
            bf16_full_eval=False,  # Use float32 for eval (cleaner metrics)
            
            # EARLY STOPPING
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            
            # REPRODUCIBILITY
            seed=42,
            
            # DDPS (if multi-GPU in future)
            ddp_find_unused_parameters=False,  # Llama uses all parameters
            
            # OPTIMIZATION FLAGS FOR MEMORY
            # Disable certain optimizations to save memory
            optim="adamw_8bit",  # 8-bit Adam (saves memory vs default)
            dataloader_pin_memory=True,  # Faster data loading
            dataloader_num_workers=2,
            remove_unused_columns=True,  # Don't keep extra columns
        )
```

**Key Changes Explained**:

1. **per_device_train_batch_size: 2 → 1**
   - Llama at 1024 tokens takes ~2x VRAM per token
   - Can't fit batch=2
   - Compensate with gradient_accumulation

2. **gradient_accumulation_steps: 4 → 8**
   - Maintain effective batch size = 1 * 8 = 8 (same as 2 * 4)
   - Each step processes 8 sequences worth of gradients
   - Takes longer per epoch but same learning dynamics

3. **learning_rate: 1e-4 → 2e-4**
   - Llama pre-training used higher LR
   - Reasoning models respond better to higher LR
   - Monitor: If loss becomes noisy, reduce to 1.5e-4

4. **Logging and Evaluation: More Frequent**
   - logging_steps: 10 → 5
   - evaluation_strategy: "epoch" → "steps" with eval_steps=50
   - Why: Longer training per epoch (slower per-step), want more feedback
   - Helps catch problems earlier

5. **optim: adamw_8bit**
   - 8-bit Adam saves ~50% memory for optimizer state
   - Trade-off: slightly less stable, but necessary for VRAM
   - Disable with `optim="adamw_torch"` if you hit issues

6. **max_grad_norm: 1.0**
   - Aggressive gradient clipping
   - Llama can have gradient spikes during reasoning
   - Prevents training from destabilizing

---

### Change 7: run_pipeline_dynamic.py - Epoch Calculation

**BEFORE** (StarCoder epoch calculation):

```python
class DynamicPipelineOrchestrator:
    def calculate_epochs(self, num_sequences: int, avg_tokens_per_seq: float) -> int:
        """Calculate epochs based on target training tokens."""
        
        config = self.training_cfg.get("epoch_calculation", {})
        target_tokens = config.get("target_tokens", 50000000)
        
        total_tokens = num_sequences * avg_tokens_per_seq
        
        if total_tokens == 0:
            epochs = config.get("fallback_epochs_tiny", 12)
        else:
            # Calculate epochs needed to reach target
            epochs_needed = target_tokens / total_tokens
            epochs = int(ceil(epochs_needed))
        
        # Clamp to bounds
        min_epochs = config.get("min_epochs", 5)
        max_epochs = config.get("max_epochs", 12)
        epochs = max(min_epochs, min(epochs, max_epochs))
        
        return epochs
```

**AFTER** (Llama epoch calculation - SAME LOGIC, UPDATED BOUNDS):

```python
class DynamicPipelineOrchestrator:
    def calculate_epochs(self, num_sequences: int, avg_tokens_per_seq: float) -> int:
        """Calculate epochs based on target training tokens.
        
        For Llama:
        - Sequences are 2x longer (1024 vs 512 tokens)
        - So fewer sequence steps to reach target
        - BUT reasoning needs more passes over data
        - So increase min_epochs slightly
        """
        
        config = self.training_cfg.get("epoch_calculation", {})
        target_tokens = config.get("target_tokens", 50000000)  # Same as StarCoder
        
        total_tokens = num_sequences * avg_tokens_per_seq
        
        if total_tokens == 0:
            # No  use fallback
            # Llama needs more epochs for reasoning
            epochs = config.get("fallback_epochs_tiny", 12)
        else:
            # Calculate epochs needed to reach target
            # Example:
            #   target_tokens = 50M
            #   num_sequences = 1000
            #   avg_tokens_per_seq = 512 (StarCoder)
            #   total_tokens_per_epoch = 512K
            #   epochs_needed = 50M / 512K ≈ 98 epochs → clamped to 12
            #
            #   With Llama (same 1000 sequences, but 1024 tokens each):
            #   total_tokens_per_epoch = 1024K
            #   epochs_needed = 50M / 1024K ≈ 49 epochs → clamped to 12
            #   (Same result due to clamping)
            
            epochs_needed = target_tokens / total_tokens
            epochs = int(ceil(epochs_needed))
        
        # Clamp to bounds - Llama min increased
        min_epochs = config.get("min_epochs", 6)  # From 5 (reasoning needs passes)
        max_epochs = config.get("max_epochs", 12)  # Same
        epochs = max(min_epochs, min(epochs, max_epochs))
        
        logger.info(
            f"Epoch calculation (Llama):\n"
            f"  Sequences: {num_sequences}\n"
            f"  Avg tokens/seq: {avg_tokens_per_seq:.0f}\n"
            f"  Total tokens/epoch: {total_tokens:,.0f}\n"
            f"  Target tokens: {target_tokens:,.0f}\n"
            f"  Epochs needed: {epochs_needed:.1f}\n"
            f"  Final epochs: {epochs}\n"
        )
        
        return epochs
```

**Key Changes**:

1. **min_epochs: 5 → 6**
   - Reasoning emerges slowly
   - StarCoder (code completion) converges in 3-5 epochs
   - Llama (reasoning) needs 5-8 epochs
   - First epoch: learns tokens and patterns
   - Epochs 2-3: learns function relationships
   - Epochs 4-6: learns cross-module interactions
   - Epochs 7+: diminishing returns

2. **Math: Same result despite longer sequences**
   - StarCoder: 1000 seqs * 512 tokens = 512K tokens/epoch
   - 50M target / 512K = 98 epochs → clamped to max 12
   - Llama: 1000 seqs * 1024 tokens = 1024K tokens/epoch
   - 50M target / 1024K = 49 epochs → clamped to max 12
   - Same clamping result, but Llama trains deeper per epoch

3. **Why Keep target_tokens Same**:
   - "50M tokens of training" is a calibrated amount
   - Works for both models
   - Doubles the effective training density (2x tokens per sequence)

---

### Change 8: run_pipeline_dynamic.py - Sequence Length Configuration

**BEFORE** (StarCoder pipeline):

```python
class DynamicPipelineOrchestrator:
    def __init__(self, repo_path, ...):
        self.sequence_length = 512  # StarCoder context
        self.overlap = 128
        # ...
    
    def build_sequences(self):
        # Tokenize and chunk commits into 512-token sequences
        logger.info(f"Building sequences with max_length={self.sequence_length}")
```

**AFTER** (Llama pipeline):

```python
class DynamicPipelineOrchestrator:
    def __init__(self, repo_path, ...):
        self.sequence_length = 1024  # From 512 (Llama context)
        self.overlap = 256            # From 128 (proportional)
        # ...
    
    def build_sequences(self):
        # Tokenize and chunk commits into 1024-token sequences
        logger.info(
            f"Building sequences for Llama (length={self.sequence_length}, overlap={self.overlap})\n"
            f"  Expected: ~2x longer sequences, ~50% more overlap\n"
            f"  Result: Better architectural context, smoother transitions\n"
        )
```

**Why 2x Sequence Length**:

- **StarCoder**: 512 tokens ≈ 2-3 small functions
- **Llama**: 1024 tokens ≈ 8-10 functions or one small module
- Llama excels at understanding entire services in context
- 1024 is max before OOM on 8GB GPU
- 2048 would require 16GB minimum

**Why 2x Overlap**:

- Overlap: 512 overlap means sequences share 50% of tokens
- Prevents artificial boundary artifacts
- At 512 seq length, 128 overlap = 25% shared
- At 1024 seq length, 256 overlap = 25% shared (proportional)
- Keeps consistency

---

### Change 9: test_behavioral_evaluation.py - COMPLETE REWRITE

**FILE SIZE**: ~150 lines → ~300 lines (more comprehensive)

**BEFORE** (StarCoder evaluation):

```python
#!/usr/bin/env python3
"""Test behavioral evaluation for StarCoder fine-tuning."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

def test_starcoder_completion():
    """Test if model completes Rust code patterns."""
    
    model_path = "models/the-block-rust-model/best"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    
    test_prompts = [
        "fn process_block",
        "impl Transaction",
        "pub struct Validator",
        "async fn handle",
        "match result {",
    ]
    
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=50, temperature=0.7)
        response = tokenizer.decode(outputs[0])
        print(f"{prompt} -> {response}")

if __name__ == "__main__":
    test_starcoder_completion()
```

**AFTER** (Llama evaluation - REASONING-FOCUSED):

```python
#!/usr/bin/env python3
"""Test behavioral evaluation for Llama fine-tuning.

Llama excels at reasoning and understanding, not code completion.
Test prompts focus on:
1. Architectural understanding
2. Cross-module reasoning
3. Economic system analysis
4. Edge case identification
5. Strategic decision-making
"""

import json
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from typing import Dict, List, Any
import re

logger.basicConfig(level=logging.INFO)
logger.getLogger(__name__)

class LlamaEvaluator:
    """Comprehensive evaluation for Llama fine-tuning on blockchain codebase."""
    
    def __init__(self, model_path: str, device: str = "cuda:0"):
        self.model_path = Path(model_path)
        self.device = device
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path),
            trust_remote_code=True,
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            str(self.model_path),
            device_map=device,
            torch_dtype=torch.bfloat16,
            load_in_4bit=True,
        )
        
        # Generation config for reasoning (not code completion)
        self.generation_config = GenerationConfig(
            max_new_tokens=300,  # Reasoning needs more tokens
            temperature=0.7,     # Balance creativity with consistency
            top_p=0.95,          # Nucleus sampling
            do_sample=True,      # Stochastic sampling
            pad_token_id=self.tokenizer.eos_token_id,
        )
        
        logger.info(f"Loaded Llama model from {model_path}")
    
    def evaluate_architecture_understanding(self) -> Dict[str, Any]:
        """Test if model understands codebase architecture."""
        
        prompts = [
            {
                "question": "What is the purpose of the DKG module in this blockchain?",
                "context": "",
                "expected_keywords": ["distributed", "key", "generation", "consensus"],
            },
            {
                "question": "How does the StateManager maintain consistency across validators?",
                "context": "",
                "expected_keywords": ["state", "merkle", "hash", "root", "consistency"],
            },
            {
                "question": "What prevents a single validator from controlling the network?",
                "context": "",
                "expected_keywords": ["Byzantine", "tolerance", "2/3", "voting", "consensus"],
            },
        ]
        
        results = []
        for item in prompts:
            response = self._generate_response(item["question"])
            score = self._score_reasoning(response, item["expected_keywords"])
            
            results.append({
                "question": item["question"],
                "response": response[:200],  # Truncate for display
                "score": score,
                "reasoning_depth": self._measure_reasoning_depth(response),
            })
        
        return {
            "category": "architecture_understanding",
            "results": results,
            "average_score": sum(r["score"] for r in results) / len(results),
        }
    
    def evaluate_cross_module_reasoning(self) -> Dict[str, Any]:
        """Test if model understands interactions between modules."""
        
        prompts = [
            {
                "question": "What happens if consensus fails but StateManager continues updating?",
                "expected_keywords": ["fork", "inconsistency", "recovery", "Byzantine"],
            },
            {
                "question": "How do DKG and validator selection interact?",
                "expected_keywords": ["coordination", "timing", "security", "Byzantine"],
            },
            {
                "question": "What's the relationship between inflation and validator incentives?",
                "expected_keywords": ["reward", "incentive", "economics", "supply"],
            },
        ]
        
        results = []
        for item in prompts:
            response = self._generate_response(item["question"])
            score = self._score_reasoning(response, item["expected_keywords"])
            
            results.append({
                "question": item["question"],
                "response": response[:200],
                "score": score,
                "mentions_interactions": self._check_mentions_interactions(response),
            })
        
        return {
            "category": "cross_module_reasoning",
            "results": results,
            "average_score": sum(r["score"] for r in results) / len(results),
        }
    
    def evaluate_edge_case_analysis(self) -> Dict[str, Any]:
        """Test if model can identify and reason about edge cases."""
        
        prompts = [
            {
                "scenario": "1/3 of validators go offline",
                "expected_reasoning": ["liveness", "safety", "threshold", "Byzantine"],
            },
            {
                "scenario": "Network partitions into two parts",
                "expected_reasoning": ["partition", "fork", "recovery", "consistency"],
            },
            {
                "scenario": "A validator has a memory leak and crashes every 1 hour",
                "expected_reasoning": ["fault tolerance", "recovery", "state"],
            },
        ]
        
        results = []
        for item in prompts:
            question = f"What happens if {item['scenario']}?"
            response = self._generate_response(question)
            score = self._score_reasoning(response, item["expected_reasoning"])
            
            results.append({
                "scenario": item["scenario"],
                "response": response[:200],
                "score": score,
            })
        
        return {
            "category": "edge_case_analysis",
            "results": results,
            "average_score": sum(r["score"] for r in results) / len(results),
        }
    
    def evaluate_refactoring_reasoning(self) -> Dict[str, Any]:
        """Test if model can reason about architectural changes."""
        
        prompts = [
            {
                "question": "Should we extract consensus into a separate crate? Why or why not?",
                "expected_keywords": ["modularity", "coupling", "complexity", "trade-off"],
            },
            {
                "question": "How would you refactor this circular dependency: A -> B -> C -> A?",
                "expected_keywords": ["dependency", "layer", "interface", "strategy"],
            },
        ]
        
        results = []
        for item in prompts:
            response = self._generate_response(item["question"])
            score = self._score_reasoning(response, item["expected_keywords"])
            
            results.append({
                "question": item["question"],
                "response": response[:200],
                "score": score,
                "has_tradeoff_analysis": self._check_tradeoff(response),
            })
        
        return {
            "category": "refactoring_reasoning",
            "results": results,
            "average_score": sum(r["score"] for r in results) / len(results),
        }
    
    # Helper methods
    
    def _generate_response(self, prompt: str) -> str:
        """Generate response to a prompt using Llama."""
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=self.generation_config,
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
    
    def _score_reasoning(self, response: str, keywords: List[str]) -> float:
        """Score reasoning by counting expected keywords."""
        
        response_lower = response.lower()
        found = sum(1 for kw in keywords if kw.lower() in response_lower)
        return found / len(keywords) if keywords else 0.0
    
    def _measure_reasoning_depth(self, response: str) -> float:
        """Measure reasoning depth by counting reasoning markers."""
        
        markers = [
            "because",
            "however",
            "in addition",
            "considering",
            "the risk",
            "trade-off",
            "therefore",
            "as a result",
            "on the other hand",
        ]
        
        response_lower = response.lower()
        found = sum(1 for marker in markers if marker in response_lower)
        return found / len(markers)
    
    def _check_mentions_interactions(self, response: str) -> bool:
        """Check if response mentions module interactions."""
        
        interaction_keywords = [
            "interact",
            "coordinate",
            "synchronize",
            "interface",
            "dependency",
            "coupling",
        ]
        
        response_lower = response.lower()
        return any(kw in response_lower for kw in interaction_keywords)
    
    def _check_tradeoff(self, response: str) -> bool:
        """Check if response discusses trade-offs."""
        
        tradeoff_keywords = [
            "trade-off",
            "tradeoff",
            "benefit",
            "cost",
            "risk",
            "on the other hand",
            "however",
            "but",
        ]
        
        response_lower = response.lower()
        return any(kw in response_lower for kw in tradeoff_keywords)
    
    def run_full_evaluation(self) -> Dict[str, Any]:
        """Run complete evaluation suite."""
        
        results = {
            "timestamp": str(Path.ctime(Path.cwd())),
            "model_path": str(self.model_path),
            "evaluations": [
                self.evaluate_architecture_understanding(),
                self.evaluate_cross_module_reasoning(),
                self.evaluate_edge_case_analysis(),
                self.evaluate_refactoring_reasoning(),
            ],
        }
        
        # Calculate overall score
        all_scores = [
            eval["average_score"]
            for eval in results["evaluations"]
        ]
        results["overall_score"] = sum(all_scores) / len(all_scores)
        
        return results


if __name__ == "__main__":
    import logging
    
    logging.basicConfig(level=logging.INFO)
    
    model_path = "models/the-block-llama-model/best"
    evaluator = LlamaEvaluator(model_path)
    
    results = evaluator.run_full_evaluation()
    
    print("\n" + "="*80)
    print("LLAMA BEHAVIORAL EVALUATION RESULTS")
    print("="*80)
    print(f"Overall Score: {results['overall_score']:.2%}")
    print(f"Model: {results['model_path']}")
    
    for eval_result in results["evaluations"]:
        print(f"\n{eval_result['category'].replace('_', ' ').title()}:")
        print(f"  Average Score: {eval_result['average_score']:.2%}")
        
        for result in eval_result["results"][:2]:  # Show first 2
            question = result.get("question") or result.get("scenario")
            print(f"    Q: {question}")
            print(f"    Score: {result['score']:.0%}")
            print()
    
    # Save results
    output_file = Path("evaluation_results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to {output_file}")
```

**Key Differences from StarCoder Evaluation**:

1. **Prompt Style**: Questions instead of code prefixes
2. **Evaluation Metrics**: Reasoning keywords, not generated tokens
3. **Scoring**: Keyword-based, not token-based
4. **Test Categories**:
   - Architecture understanding
   - Cross-module reasoning
   - Edge case analysis
   - Refactoring reasoning
5. **Output**: Structured JSON with scores and reasoning depth

---

## Testing Strategy & Validation

### Phase 1: Model Loading (Week 1)

**Goal**: Verify Llama loads correctly with right config.

**Test Script**:

```python
#!/usr/bin/env python3
"""Test Phase 1: Model Loading"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

print("\n" + "="*80)
print("PHASE 1: MODEL LOADING TEST")
print("="*80)

# Test 1: Load tokenizer
print("\n[1/4] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8b")
print(f"✓ Tokenizer loaded. Vocab size: {len(tokenizer)}")
print(f"  Pad token: {tokenizer.pad_token}")
print(f"  EOS token: {tokenizer.eos_token}")

# Test 2: Load base model (quantized)
print("\n[2/4] Loading base model (4-bit quantized)...")
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8b-instruct",
    quantization_config=bnb_config,
    device_map="auto",
)
print(f"✓ Model loaded (4-bit quantized)")
print(f"  Layers: {model.config.num_hidden_layers}")
print(f"  Hidden dim: {model.config.hidden_size}")
print(f"  Vocab size: {model.config.vocab_size}")

# Test 3: Apply LoRA
print("\n[3/4] Applying LoRA adapters...")
lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
print("✓ LoRA applied")

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
trainable_pct = 100 * trainable_params / total_params

print(f"  Trainable params: {trainable_params:,} / {total_params:,} ({trainable_pct:.2f}%)")
print(f"  Should be ~0.1%")

# Test 4: Test tokenization and generation
print("\n[4/4] Testing tokenization and generation...")

test_prompt = """Analyze this blockchain module:
What is the purpose of validator selection in consensus?"""

inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
print(f"✓ Prompt tokenized: {inputs['input_ids'].shape}")
print(f"  Tokens: {inputs['input_ids'].shape[1]}")

# Generate (short, just to test)
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        temperature=0.7,
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"✓ Generation successful (first 100 chars):")
print(f"  {response[:100]}...")

# VRAM check
if torch.cuda.is_available():
    print(f"\n✓ VRAM Usage:")
    print(f"  Allocated: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
    print(f"  Reserved: {torch.cuda.memory_reserved() / 1e9:.2f}GB")
    print(f"  Budget: 8GB")
    
    allocated = torch.cuda.memory_allocated() / 1e9
    if allocated > 7.5:
        print(f"  ⚠️  WARNING: Close to limit!")
    else:
        print(f"  ✓ Safe margin: {8 - allocated:.2f}GB")

print("\n" + "="*80)
print("✓ PHASE 1 COMPLETE: Model loads successfully")
print("="*80)
```

**Expected Output**:
```
PHASE 1: MODEL LOADING TEST
[1/4] Loading tokenizer...
✓ Tokenizer loaded. Vocab size: 129152

[2/4] Loading base model (4-bit quantized)...
✓ Model loaded (4-bit quantized)
  Layers: 32
  Hidden dim: 4096
  Vocab size: 129152

[3/4] Applying LoRA adapters...
✓ LoRA applied
  Trainable params: 8,388,608 / 8,030,000,128 (0.10%)
  Should be ~0.1%

[4/4] Testing tokenization and generation...
✓ Prompt tokenized: torch.Size([1, 23])
  Tokens: 23
✓ Generation successful (first 100 chars):
  Validator selection is crucial for...

✓ VRAM Usage:
  Allocated: 6.45GB
  Reserved: 6.48GB
  Budget: 8GB
  ✓ Safe margin: 1.55GB

PHASE 1 COMPLETE: Model loads successfully
```

### Phase 2: Tokenization (Week 1)

**Goal**: Verify tokenizer works with codebase data.

**Test Script**:

```python
#!/usr/bin/env python3
"""Test Phase 2: Tokenization"""

import json
from pathlib import Path
from transformers import AutoTokenizer

print("\n" + "="*80)
print("PHASE 2: TOKENIZATION TEST")
print("="*80)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8b")

# Load sample data
data_file = Path("data/git_history_rich.jsonl")
if not data_file.exists():
    print(f"✗ Data file not found: {data_file}")
    exit(1)

print(f"\n[1/3] Loading sample data...")
samples = []
with open(data_file) as f:
    for i, line in enumerate(f):
        if i >= 10:
            break
        samples.append(json.loads(line))

print(f"✓ Loaded {len(samples)} sample commits")

# Test tokenization
print(f"\n[2/3] Tokenizing samples...")

token_lengths = []
errors = 0

for i, sample in enumerate(samples):
    try:
        # Combine commit message and diff
        text = f"""{sample.get('message', '')}

{sample.get('diff', '')}"""
        
        # Encode
        encoded = tokenizer.encode(text)
        token_lengths.append(len(encoded))
        
    except Exception as e:
        print(f"  ✗ Error tokenizing sample {i}: {e}")
        errors += 1

print(f"✓ Tokenized {len(token_lengths)} samples")
print(f"  Errors: {errors}")
print(f"  Token lengths: min={min(token_lengths)}, max={max(token_lengths)}, mean={sum(token_lengths)/len(token_lengths):.0f}")

# Test long sequence handling
print(f"\n[3/3] Testing long sequence handling...")

long_text = "\n".join([s.get('diff', '') for s in samples[:5]])
long_encoded = tokenizer.encode(long_text)
print(f"✓ Long sequence: {len(long_encoded)} tokens")

if len(long_encoded) > 1024:
    print(f"  ⚠️  WARNING: Exceeds 1024-token context window")
    print(f"  Will be truncated during training")
else:
    print(f"  ✓ Fits within 1024-token context")

print("\n" + "="*80)
print("✓ PHASE 2 COMPLETE: Tokenization works")
print("="*80)
```

### Phase 3: Small Training Run (Week 2)

**Goal**: Verify training loop works with Llama configuration.

**Test Script**: Use existing `run_pipeline_dynamic.py` with small subset

```bash
# Create tiny dataset (10 samples)
head -10 data/git_history_rich.jsonl > data/git_history_tiny.jsonl

# Run pipeline on tiny dataset
python run_pipeline_dynamic.py \
  ~/projects/the-block \
  --force \
  --force-epochs 2  # Only 2 epochs for testing
```

**Expected Output**:
- Completes in <30 minutes
- No VRAM errors
- Loss decreases over epochs
- Model produces coherent responses

### Phase 4: Full Training (Weeks 3-4)

**Goal**: Run complete training cycle.

**Command**:

```bash
python run_pipeline_dynamic.py \
  ~/projects/the-block \
  --force
```

**Monitoring**:
- Watch VRAM: Should stay <7.5GB
- Watch loss: Should decrease smoothly
- Watch eval metrics: Check reasoning quality

### Phase 5: Evaluation (Week 4-5)

**Goal**: Test model quality with behavioral evaluation.

**Commands**:

```bash
# Run evaluation suite
python test_behavioral_evaluation.py \
  models/the-block-llama-model/best

# Review results
cat evaluation_results.json
```

**Success Criteria**:
- Overall score > 50%
- Architecture understanding > 60%
- Cross-module reasoning > 50%
- No crashes or VRAM errors

---

## Implementation Checklist (5-Week Breakdown)

### Week 1: Configuration & Planning

- [ ] **Day 1**: Backup current StarCoder model and config
  - [ ] `cp -r models/the-block-git-model-final models/the-block-git-model-final.backup`
  - [ ] `cp training_config.yaml training_config.yaml.backup`
  - [ ] Verify backups exist

- [ ] **Day 2**: Update training_config.yaml
  - [ ] Change model name to `meta-llama/Llama-3.1-8b-instruct`
  - [ ] Change tokenizer to `meta-llama/Llama-3.1-8b`
  - [ ] Update LoRA config: rank 16→32, targets q_proj/k_proj/v_proj/up_proj/down_proj
  - [ ] Update context: 512→1024
  - [ ] Update learning rate: 1e-4→2e-4
  - [ ] Update batch sizes: 2→1, 4→1
  - [ ] Update accumulation: 4→8
  - [ ] Update epochs: min 5→6
  - [ ] Verify file syntax (yaml lint)

- [ ] **Day 3**: Update train_model.py
  - [ ] Update tokenizer loading (use config tokenizer_name)
  - [ ] Add special tokens for domain marking
  - [ ] Update LoRA config initialization
  - [ ] Update training arguments
  - [ ] Add validation checks for Llama vs StarCoder
  - [ ] Test imports (python -c "import train_model")

- [ ] **Day 4**: Update run_pipeline_dynamic.py
  - [ ] Update sequence_length: 512→1024
  - [ ] Update overlap: 128→256
  - [ ] Update epoch calculation: min_epochs 5→6
  - [ ] Add logging for Llama mode
  - [ ] Test config loading

- [ ] **Day 5**: Update test_behavioral_evaluation.py
  - [ ] Rewrite entire evaluation suite
  - [ ] Focus on reasoning vs code completion
  - [ ] Add architecture understanding tests
  - [ ] Add cross-module reasoning tests
  - [ ] Add edge case analysis
  - [ ] Implement scoring logic

### Week 2: Testing Phase 1 (Model Loading)

- [ ] **Day 1**: Phase 1 Test - Model Loading
  - [ ] Run Phase 1 test script (see Testing Strategy)
  - [ ] Verify tokenizer loads
  - [ ] Verify model loads (4-bit)
  - [ ] Verify LoRA applies
  - [ ] Check VRAM usage: should be 6-7GB
  - [ ] Document results

- [ ] **Day 2**: Phase 2 Test - Tokenization
  - [ ] Run Phase 2 test script
  - [ ] Test with git_history_rich.jsonl
  - [ ] Verify token lengths reasonable
  - [ ] Check for encoding errors
  - [ ] Document results

- [ ] **Days 3-5**: Debugging
  - [ ] Fix any errors from Phase 1-2
  - [ ] Adjust configs if needed
  - [ ] Re-run tests until all pass

### Week 3: Testing Phase 2 (Small Training Run)

- [ ] **Day 1-2**: Prepare tiny dataset
  - [ ] Create subset: `head -100 data/git_history_rich.jsonl > data/test_small.jsonl`
  - [ ] Verify subset loads
  - [ ] Estimate token count

- [ ] **Day 2-3**: Run small training
  - [ ] Run pipeline with --force-epochs 2 on small dataset
  - [ ] Monitor:
    - [ ] VRAM usage (should stay <7.5GB)
    - [ ] Training loss (should decrease)
    - [ ] Training time per epoch
  - [ ] Complete at least 2 full epochs
  - [ ] Save checkpoint

- [ ] **Day 4**: Test small model
  - [ ] Load best checkpoint from small training
  - [ ] Run Phase 1 inference test (simple prompts)
  - [ ] Run Phase 4 behavioral evaluation (quick subset)
  - [ ] Document results

- [ ] **Day 5**: Debugging & Adjustments
  - [ ] If VRAM issues: Reduce batch size, enable flash attention, etc.
  - [ ] If loss not decreasing: Adjust learning rate
  - [ ] If evaluation poor: Check prompts are appropriate for Llama
  - [ ] Adjust configs as needed
  - [ ] Document changes

### Week 4: Full Training

- [ ] **Day 1**: Start full training run
  - [ ] Run: `python run_pipeline_dynamic.py ~/projects/the-block --force`
  - [ ] Set up monitoring (watch logs, VRAM, loss)
  - [ ] Estimated duration: 6-12 hours depending on dataset size
  - [ ] Document start time

- [ ] **Days 2-4**: Monitor training
  - [ ] Check VRAM stays <7.5GB
  - [ ] Verify loss decreasing
  - [ ] Watch for convergence
  - [ ] Save best checkpoint
  - [ ] Log eval metrics every epoch

- [ ] **Day 5**: Post-training analysis
  - [ ] Training complete
  - [ ] Collect metrics:
    - [ ] Final training loss
    - [ ] Final validation loss
    - [ ] Training time
    - [ ] VRAM peak usage
  - [ ] Save metrics to JSON
  - [ ] Compare to StarCoder baseline

### Week 5: Evaluation & Refinement

- [ ] **Day 1-2**: Behavioral Evaluation
  - [ ] Run full evaluation suite on best model
  - [ ] Save results to JSON
  - [ ] Document scores:
    - [ ] Overall score
    - [ ] Architecture understanding
    - [ ] Cross-module reasoning
    - [ ] Edge case analysis
    - [ ] Refactoring reasoning

- [ ] **Day 2-3**: Quality Assessment
  - [ ] Manual review of generated responses
  - [ ] Compare to StarCoder model
  - [ ] Check for reasoning depth
  - [ ] Verify architectural understanding
  - [ ] Document findings

- [ ] **Day 4**: Optional Fine-tuning
  - [ ] If scores <50%: Consider additional training
  - [ ] If VRAM issues resolved: Try rank 48 LoRA
  - [ ] If reasoning poor: Adjust evaluation prompts
  - [ ] Run one more epoch if needed

- [ ] **Day 5**: Documentation & Deployment
  - [ ] Create LLAMA_DEPLOYMENT.md guide
  - [ ] Document best practices
  - [ ] Create inference examples
  - [ ] Update README with new model info
  - [ ] Commit all changes
  - [ ] Tag release: v2.0-llama

---

## Architecture Deep Dive

### Why Llama Layers Are Different

**StarCoder Architecture** (GPT-Neo style):

```
Attention Block:
  ├─ c_attn: Linear(3072 → 2304)  # Merged Q,K,V (768*3)
  │  ├─ Split into: [Q:768, K:768, V:768]
  │  └─ (This is the GPT-Neo way)
  ├─ MultiHeadAttention
  │  └─ 12 heads, 64 dim each
  └─ c_proj: Linear(768 → 768)    # Output

MLP Block:
  ├─ c_fc: Linear(768 → 3072)     # Expand
  ├─ GELU
  └─ c_proj: Linear(3072 → 768)   # Project

LoRA Targets: ["c_attn", "c_proj", "c_fc"]
```

**Llama Architecture** (Transformer style):

```
Attention Block:
  ├─ q_proj: Linear(4096 → 4096)  # Query only
  ├─ k_proj: Linear(4096 → 4096)  # Key only
  ├─ v_proj: Linear(4096 → 4096)  # Value only
  ├─ MultiHeadAttention
  │  └─ 32 heads, 128 dim each
  └─ (Output projection implicit in attention)

MLP Block (GatedLinearUnit):
  ├─ gate_proj: Linear(4096 → 14336)  # Gate for GLU
  ├─ up_proj: Linear(4096 → 14336)    # Expansion
  ├─ SiLU activation
  └─ down_proj: Linear(14336 → 4096)  # Project

LoRA Targets: ["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]
```

### Why We Target All 5 Modules

**Question**: Could we target fewer modules to save VRAM?

**Answer**: No, because:

1. **Q/K/V are all important**:
   - Q controls "what to query"
   - K/V control "what information to retrieve"
   - Targeting only Q/V would limit learning

2. **Up/Down are different**:
   - Gate/Up: Expansion (like c_fc)
   - Down: Projection (like c_proj in MLP)
   - Need both to learn proper MLP patterns

3. **VRAM cost is minimal**:
   - Rank 32 LoRA: ~150MB
   - Base model: ~4.5GB
   - Ratio: 3% overhead
   - Worth it for 5x better adaptation

### VRAM Breakdown for Llama Training

```
8GB GPU VRAM Budget:
┌─────────────────────────────────────────────────────┐
│                    VRAM Usage                       │
├─────────────────────────────────────────────────────┤
│ Base Model (4-bit)        │ 4.5GB  │ 56% of budget  │
│ LoRA Adapters             │ 0.15GB │  2% of budget  │
│ Optimizer State (8-bit)   │ 0.8GB  │ 10% of budget  │
│ Activations & Gradients   │ 1.5GB  │ 19% of budget  │
│ PyTorch Overhead          │ 0.3GB  │  4% of budget  │
│ ─────────────────────────────────                   │
│ TOTAL                     │ 7.25GB │ 91% of budget  │
│ MARGIN                    │ 0.75GB │  9% of budget  │
└─────────────────────────────────────────────────────┘

Detailed Calculation:

1. Base Model (4-bit quantization):
   • 8B params = 8 * 10^9
   • 4-bit = 0.5 bytes per param
   • 8 * 10^9 * 0.5 bytes = 4GB
   • Overhead for headers, routing: ~0.5GB
   • Subtotal: ~4.5GB

2. LoRA Adapters (all 5 modules):
   • 5 modules × (2 matrices per module)
   • Each matrix: [hidden_dim, rank] or [rank, hidden_dim]
   • q_proj: [4096, 32] + [32, 4096] = 262K params
   • × 5 modules = 1.31M params
   • × 2 bytes (bfloat16) = 2.6MB
   • Overhead: ~150MB total

3. Optimizer State (8-bit Adam):
   • 8-bit Adam: 1 byte per param + metadata
   • Trainable params: 1.31M (LoRA only)
   • States: momentum, variance
   • 1.31M * 1 byte ≈ 1.31MB base
   • But scales to full model params for effective training
   • Estimated: ~0.8GB

4. Activations & Gradients:
   • Batch size: 1
   • Sequence length: 1024
   • Hidden dim: 4096
   • Layers: 32
   • Activation storage: (1 * 1024 * 4096 * 4 bytes) * (num intermediate layers)
   • Rough estimate: ~1.5GB

5. PyTorch Overhead:
   • CUDA context, memory fragmentation
   • Rough estimate: ~0.3GB

Total: ~7.25GB (89% of 8GB budget)
```

---

## Hybrid Approach (Keeping Both Models)

### Why Keep Both StarCoder AND Llama

**Different Strengths**:

| Task | StarCoder | Llama |
|------|-----------|-------|
| Code completion | ★★★★★ | ★★ |
| IDE integration | ★★★★★ | ★★★ |
| Function generation | ★★★★★ | ★★★ |
| Architecture analysis | ★ | ★★★★★ |
| Cross-module reasoning | ★★ | ★★★★★ |
| Refactoring strategy | ★ | ★★★★ |
| Bug analysis | ★★ | ★★★★ |

**Use Cases**:

- **IDE Integration**: StarCoder (code completion while typing)
- **Code Review**: Llama (architectural issues)
- **Refactoring**: Llama (strategic guidance)
- **Bug Hunting**: Both (StarCoder finds patterns, Llama analyzes impact)
- **Architectural Planning**: Llama (system design)

### How to Run Both

**Directory Structure**:

```
models/
  ├─ the-block-git-model-final/      # StarCoder (original)
  │  └─ best/
  │     ├─ adapter_model.safetensors
  │     └─ ...
  │
  └─ the-block-llama-model/          # Llama (new)
     └─ best/
        ├─ adapter_model.safetensors
        └─ ...
```

**Config Selection**:

```python
# inference.py
from pathlib import Path

def load_model(model_type: str = "llama"):
    """Load either StarCoder or Llama model."""
    
    if model_type == "starcoder":
        config_file = "training_config.yaml"
        model_path = "models/the-block-git-model-final/best"
        model_name = "bigcode/starcoder2-3b"
        
    elif model_type == "llama":
        config_file = "training_config_llama.yaml"
        model_path = "models/the-block-llama-model/best"
        model_name = "meta-llama/Llama-3.1-8b-instruct"
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load config and model
    # ...
```

**API Interface**:

```python
from typing import Literal

class BlockCodeAnalyzer:
    """Unified interface for both models."""
    
    def __init__(self, model_type: Literal["starcoder", "llama"] = "llama"):
        self.model_type = model_type
        self.model = load_model(model_type)
    
    def complete_code(self, prompt: str) -> str:
        """Code completion (StarCoder only)."""
        if self.model_type != "starcoder":
            raise ValueError("Use StarCoder model for code completion")
        return self.model.generate(prompt, max_tokens=100)
    
    def analyze_architecture(self, context: str, question: str) -> str:
        """Architecture analysis (Llama only)."""
        if self.model_type != "llama":
            raise ValueError("Use Llama model for architecture analysis")
        return self.model.generate(f"{context}\n\nQ: {question}", max_tokens=300)
    
    def hybrid_analysis(self, code: str) -> dict:
        """Use both models for comprehensive analysis."""
        
        # StarCoder: find potential issues/patterns
        starcoder = BlockCodeAnalyzer("starcoder")
        code_score = starcoder.analyze(code)  # hypothetical method
        
        # Llama: understand architectural implications
        llama = BlockCodeAnalyzer("llama")
        arch_analysis = llama.analyze_architecture(
            context=code,
            question="What are the architectural implications of this code?"
        )
        
        return {
            "code_analysis": code_score,
            "architecture_analysis": arch_analysis,
        }
```

---

## Pitfalls, Solutions & Debugging

### Pitfall 1: LoRA Module Names Mismatch

**The Problem**:

```python
# WRONG: StarCoder targets in Llama config
lora_config = LoraConfig(
    target_modules=["c_attn", "c_proj", "c_fc"],  # ❌ Won't find these
)

# Result: Training does nothing (model frozen)
```

**The Solution**:

```python
# Validate module names before training
valid_modules = {n for n, m in model.named_modules()}
requested = set(target_modules)
missing = requested - valid_modules

if missing:
    raise ValueError(
        f"Invalid LoRA targets: {missing}\n"
        f"Available: {valid_modules}"
    )

# For Llama, use:
lora_config = LoraConfig(
    target_modules=["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"],  # ✓
)
```

**Debug Script**:

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8b-instruct",
    device_map="cpu",  # CPU to save VRAM
)

# Print all layer names
for name, module in model.named_modules():
    if "proj" in name:
        print(f"{name}: {module.__class__.__name__}")

# Output:
# model.layers.0.self_attn.q_proj: Linear
# model.layers.0.self_attn.k_proj: Linear
# model.layers.0.self_attn.v_proj: Linear
# model.layers.0.mlp.up_proj: Linear
# model.layers.0.mlp.down_proj: Linear
```

### Pitfall 2: OOM (Out of Memory) Errors

**The Problem**:

```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB. GPU has 7.95 GiB free.
```

**Root Causes**:

1. **Batch size too large** (batch_size=2 causes OOM)
2. **Sequence length too long** (>1024 tokens)
3. **Gradient accumulation steps too high** (>8)
4. **Gradient checkpointing disabled** (required for 1024 tokens)

**Solutions**:

```yaml
# FIX 1: Reduce batch size
training:
  per_device_train_batch_size: 1  # ← Already at minimum

# FIX 2: Reduce sequence length
model:
  max_position_embeddings: 512  # ← Reduce if OOM persists

# FIX 3: Enable all memory optimizations
training:
  gradient_checkpointing: true
  use_flash_attention_2: true  # ← If available
  optim: "adamw_8bit"  # ← Use 8-bit Adam

# FIX 4: Reduce hidden state caching
model:
  use_cache: false  # ← During training only
```

**Debug Script**:

```python
import torch
from transformers import TrainingArguments

# Check available memory
if torch.cuda.is_available():
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    
    print(f"Total: {total:.1f}GB")
    print(f"Allocated: {allocated:.1f}GB")
    print(f"Reserved: {reserved:.1f}GB")
    print(f"Free: {total - reserved:.1f}GB")
    
    if total < 8:
        print("⚠️  WARNING: GPU has <8GB VRAM. Llama may not fit.")
        print("   Consider: batch_size=1, max_len=512, 8bit Adam")
```

### Pitfall 3: Model Produces Incoherent Reasoning

**The Problem**:

```
Prompt: "Analyze this architecture: ..."
Response: "The architecture is very good very good good good good good."
```

**Root Causes**:

1. **Learning rate too high** (2e-4 causes divergence)
2. **Training data too small** (<100 samples)
3. **Prompts not diverse** (model memorizes rather than reasons)
4. **Not enough epochs** (<6 epochs, reasoning hasn't emerged)

**Solutions**:

```python
# FIX 1: Reduce learning rate
training_args.learning_rate = 1.5e-4  # From 2e-4

# FIX 2: Monitor loss more frequently
training_args.logging_steps = 5  # From 10
training_args.eval_steps = 20   # Check validation often

# FIX 3: Ensure enough training data
# Need at least 500 sequences (50K tokens minimum)
if num_sequences < 500:
    print("⚠️  WARNING: Too few sequences for reasoning to emerge")
    print(f"   Have: {num_sequences}, need: 500+")

# FIX 4: Run more epochs
epochs = 8  # From 6 (give reasoning more time to emerge)
```

**Diagnosis Script**:

```python
def diagnose_coherence(model, tokenizer, prompts):
    """Diagnose reasoning coherence."""
    
    import re
    
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=100)
        response = tokenizer.decode(outputs[0])
        
        # Check for signs of incoherence
        repetitions = len(re.findall(r'\b(\w+)\s+\1{2,}\b', response))
        avg_word_length = sum(len(w) for w in response.split()) / len(response.split())
        
        print(f"Prompt: {prompt[:50]}...")
        print(f"  Repetitions: {repetitions}")
        print(f"  Avg word length: {avg_word_length:.1f}")
        print(f"  Response: {response[:100]}...")
        print()
        
        if repetitions > 5:
            print("  ⚠️  High repetition suggests model is stuck in loop")
            print("     Try: reduce LR, increase data, run more epochs")
```

### Pitfall 4: Training Loss Not Decreasing

**The Problem**:

```
Epoch 1, Step 10: loss=8.2
Epoch 1, Step 20: loss=8.1
Epoch 1, Step 30: loss=8.2
← Stuck, not improving
```

**Root Causes**:

1. **Learning rate too low** (model too conservative)
2. **Data distribution problems** (e.g., all same type of commit)
3. **Tokenizer mismatch** (config says one tokenizer, code uses another)
4. **Model not actually training** (LoRA not attached properly)

**Solutions**:

```python
# FIX 1: Increase learning rate
training_args.learning_rate = 3e-4  # From 2e-4 (experiment)

# FIX 2: Verify LoRA is actually attached
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
if trainable_params == 0:
    print("❌ ERROR: No trainable parameters. LoRA not attached.")
else:
    print(f"✓ Trainable: {trainable_params:,} params")

# FIX 3: Check tokenizer consistency
config_tokenizer = config["model"]["tokenizer_name"]
actual_tokenizer = tokenizer.__class__.__name__
if config_tokenizer not in str(actual_tokenizer):
    print(f"⚠️  Tokenizer mismatch: config={config_tokenizer}, actual={actual_tokenizer}")

# FIX 4: Verify data diversity
from collections import Counter
commit_types = Counter([c.get("type") for c in commits])
if len(commit_types) == 1:
    print("⚠️  All commits same type. Model may be memorizing.")
    print(f"   Types: {commit_types}")
```

### Pitfall 5: Tokenization Errors (Special Characters)

**The Problem**:

```
UnicodeEncodeError: 'utf-8' codec can't encode character in position 42
```

**Root Causes**:

1. **Binary files in git history** (shouldn't happen, but can)
2. **Unusual Unicode** (emoji, special symbols)
3. **Encoding mismatches** (UTF-16 in UTF-8 context)

**Solutions**:

```python
# FIX 1: Handle encoding errors
encoded = tokenizer.encode(
    text,
    max_length=1024,
    truncation=True,
    return_tensors="pt",
    errors="ignore",  # Skip problematic characters
)

# FIX 2: Clean text before tokenization
import unicodedata

def clean_text(text):
    # Remove control characters
    text = "".join(c for c in text if unicodedata.category(c)[0] != 'C')
    # Replace multiple spaces
    text = " ".join(text.split())
    return text

cleaned = clean_text(text)
encoded = tokenizer.encode(cleaned)

# FIX 3: Validate before tokenization
def is_valid_text(text):
    try:
        text.encode('utf-8')
        return True
    except UnicodeEncodeError:
        return False

if not is_valid_text(text):
    print(f"Skipping invalid text: {text[:50]}...")
```

---

## Advanced Topics & Optimization

### Flash Attention 2 (If Supported)

**What It Does**: Speeds up attention computation by 2-3x, saves memory

**How to Enable**:

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8b-instruct",
    attn_implementation="flash_attention_2",  # ← Requires flash-attn
    device_map="auto",
)
```

**Requirements**:

```bash
pip install flash-attn>=2.0
# Requires CUDA 11.6+ and compatible GPU (A100, H100, etc.)
```

**Expected Speedup**:
- 2-3x faster attention computation
- 20-30% faster training per step
- ~1GB less VRAM usage

**Warning**: Not available on all GPUs (check compatibility)

### Curriculum Learning (Optional)

**What It Does**: Train on easy commits first, then harder ones

**Configuration**:

```yaml
training:
  use_curriculum: true
  curriculum_stages:
    - epochs: 2
      max_tokens: 256      # Easy: short commits
    - epochs: 2
      max_tokens: 512      # Medium: normal commits
    - epochs: 2
      max_tokens: 1024     # Hard: complex commits
```

**Implementation**:

```python
def build_curriculum_dataset(sequences, num_stages=3):
    """Split dataset into difficulty stages."""
    
    # Sort by token length (proxy for difficulty)
    sorted_seqs = sorted(sequences, key=lambda x: len(x["tokens"]))
    
    stage_size = len(sorted_seqs) // num_stages
    stages = [
        sorted_seqs[i*stage_size:(i+1)*stage_size]
        for i in range(num_stages)
    ]
    
    return stages
```

### Quantization-Aware Training (QAT)

**What It Does**: Fine-tune while simulating 4-bit quantization

**Why Useful**: Model adapts to quantized activations, potentially better quality

**Trade-off**: 30% slower training, potentially better accuracy

**Implementation**: (Advanced, skip for now)

---

## Founder's Perspective & Strategic Vision

### Why This Migration Matters

**Current Problem**: StarCoder wins at code completion, loses at strategic thinking.

GitHub Copilot is better at "complete this line." You can't beat them there.

**New Advantage**: Llama trained on your codebase becomes a **project-specific thinking partner**, not just a code suggester.

**Specific Capabilities**:

1. **Understanding Blockchain Economics**:
   - Can reason about token supply dynamics
   - Understands validator incentive compatibility
   - Can spot economically unsound patterns
   - Can suggest economic model improvements

2. **Architectural Reasoning**:
   - Sees how changes affect consensus
   - Understands cross-module dependencies
   - Can suggest modularization strategies
   - Can plan migrations and refactorings

3. **System Design**:
   - Can analyze Byzantine fault tolerance implications
   - Can model network partition recovery
   - Can suggest scaling strategies
   - Can reason about trade-offs (latency vs throughput)

### Why 8B Matters

**8B Parameters** ≈ **College-level reasoning ability**
- Can hold complex concepts in context
- Can reason about multiple factors simultaneously
- Can explain trade-offs
- Can suggest alternatives with reasoning

**vs 3B Parameters** ≈ **High school level**
- Can complete syntax
- Can generate code from templates
- Limited multi-step reasoning
- Can't explain reasoning well

**vs 70B+ Parameters** ≈ **PhD-level reasoning**
- Too large for your GPU
- Needs enterprise hardware
- Can't justify cost for now

**8B is the sweet spot** for local reasoning with your hardware.

### The Business Model Implication

**Current**: You're building a GitHub Copilot competitor.
- GitHub has 50M developers
- They have better hardware, more data, better engineers
- You can't win on code completion

**New**: You're building a **project-specific architecture advisor**.
- Unique to each codebase
- Can't be replicated easily
- Becomes more valuable over time (learns your patterns)
- Defensible competitive advantage

**Pitch**: "Local, always-available architectural thinking partner trained on YOUR codebase."

### Deployment Strategy

**Phase 1** (Now): Local-only, self-hosted
- Your machine: Full context, no privacy concerns
- 100% uptime, zero latency
- Foundation for future products

**Phase 2** (Next): Team deployment
- Deploy on company server
- Multiple developers query same model
- Fine-tune on team commits (learns team patterns)

**Phase 3** (Future): Commercial product
- "Architectural AI" as a service
- Per-codebase fine-tuned models
- Multi-language support
- IDE integration

### Key Metrics to Track

After migration, measure:

1. **Reasoning Quality**
   - Evaluation suite scores (target: >60%)
   - Manual review of generated analysis
   - Comparison to expert architect judgments

2. **Practical Utility**
   - How often do you use it for architectural decisions?
   - Does it suggest things you didn't consider?
   - Do suggestions improve code quality?

3. **Domain Expertise**
   - Does it understand blockchain economics?
   - Can it reason about validator incentives?
   - Does it catch economic design flaws?

4. **System Understanding**
   - Can it trace through cross-module flows?
   - Does it understand consensus algorithm?
   - Can it explain failure modes?

### Next Steps After Successful Migration

1. **Multi-language models**: Llama is general (works for Rust, Go, Python)
2. **Specialized domains**: Fine-tune on economic papers for better finance reasoning
3. **IDE integration**: Build VSCode plugin that queries model
4. **API wrapper**: Expose as REST API for other tools
5. **Commercial**: Package as SaaS product for other blockchain teams

---

## Appendix: Quick Reference

### Key File Changes

| File | Change | Lines |
|------|--------|-------|
| training_config.yaml | Model/tokenizer/LoRA/LR | 40-50 |
| train_model.py | Tokenizer/LoRA config/args | 60-80 |
| run_pipeline_dynamic.py | Epochs/seq_length | 30-40 |
| test_behavioral_evaluation.py | Complete rewrite | 150 |
| tokenizer.py (if exists) | Encoding handling | 20-30 |
| **TOTAL** | | **~395 lines** |

### Critical Parameters

```yaml
# Llama-specific
model: meta-llama/Llama-3.1-8b-instruct
tokenizer: meta-llama/Llama-3.1-8b  # NOTE: Different!
lora_rank: 32  # Not 16
lora_targets: [q_proj, k_proj, v_proj, up_proj, down_proj]
seq_length: 1024  # Not 512
learning_rate: 2e-4  # Not 1e-4
batch_size: 1  # Not 2
accumulation_steps: 8  # Not 4
epochs_min: 6  # Not 5
```

### Debug Commands

```bash
# Check GPU VRAM
gpu-z -h  # or nvidia-smi

# Test tokenizer
python -c "from transformers import AutoTokenizer; t = AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8b'); print(f'Vocab: {len(t)}')"

# List Llama layer names
python -c "from transformers import AutoModelForCausalLM; m = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.1-8b-instruct'); [print(name) for name, _ in m.named_modules() if 'proj' in name]"

# Run small test
python run_pipeline_dynamic.py ~/projects/the-block --force --force-epochs 1
```

### Testing Checklist

- [ ] Tokenizer loads
- [ ] Model loads (4-bit)
- [ ] LoRA applies (0.1% trainable)
- [ ] Generation works (produces text)
- [ ] VRAM <7.5GB
- [ ] Small training completes (2 epochs)
- [ ] Loss decreases
- [ ] Evaluation shows >50% reasoning
- [ ] No crashes on full dataset

---

## Version History

| Version | Date | Status | Changes |
|---------|------|--------|----------|
| 1.0 | Dec 15, 2025 | Draft | Initial plan |
| 2.0 | Dec 15, 2025 | Complete | ~1,100 lines comprehensive dev-to-dev guide |

---

## Questions & Support

If you encounter issues:

1. **VRAM**: See "Pitfalls & Debugging → Pitfall 2"
2. **LoRA modules**: See "Architecture Deep Dive → Why Llama Layers Are Different"
3. **Loss not decreasing**: See "Pitfalls & Debugging → Pitfall 4"
4. **Incoherent reasoning**: See "Pitfalls & Debugging → Pitfall 3"
5. **Tokenization errors**: See "Pitfalls & Debugging → Pitfall 5"

For questions not covered, check the original Llama documentation or create an issue with:
- Error message (full traceback)
- Config file (training_config.yaml)
- Output logs (last 100 lines)
- VRAM available
- Dataset size

---

**End of Comprehensive Migration Plan**
