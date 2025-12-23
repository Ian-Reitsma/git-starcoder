# Effectiveness Optimization: Building the BEST Model

## Goal: Maximum Model Quality (Not Speed)

Forget pipeline speed. Focus: **What dataset produces the smartest, most capable model?**

---

## Critical Questions for Model Quality

### 1. **Sequence Diversity vs Sequence Count**

Current approach:
- 11,000 sequences via synthetic duplication
- Many sequences have identical tokens, different metadata
- Model sees same code patterns repeatedly

**For BEST model:**
```
Option A: 11,000 sequences with duplication
  Pro: More training steps per epoch
  Con: Model overfits to repeated patterns
  Result: Memorization > generalization

Option B: 5,000 unique sequences, no duplication
  Pro: Every sequence teaches something new
  Con: Fewer total training steps
  Result: Better generalization, less overfitting
  
Option C: 15,000+ sequences with REAL variations
  Pro: Maximum diversity + volume
  Con: Requires advanced augmentation (back-translation, etc.)
  Result: Best of both worlds (if done right)
```

**Recommendation for BEST model: Option C (if possible) or B**
- Unique data > duplicate data
- Train longer (more epochs) to compensate for fewer sequences
- **Quality of exposure matters more than quantity of repetition**

---

### 2. **Code Coverage: What Code Should We Include?**

Current approach:
- All source files equally weighted
- Tests, docs, configs all treated same as core logic

**For BEST model - Smart Prioritization:**

```python
# High-Priority Code (3x weight):
- Core business logic (src/, crates/core/)
- Complex algorithms
- API definitions
- Data structures

# Medium-Priority Code (1x weight):
- Utility functions
- Helper modules
- Standard implementations

# Low-Priority Code (0.3x weight):
- Tests (repetitive patterns)
- Configuration files (mostly templates)
- Generated code
- Documentation (natural language, not code)
```

**Why this matters:**
- Tests teach model to write repetitive assertions
- Core logic teaches model to write actual algorithms
- Model learns from WHAT you emphasize

**Implementation:**
```python
def get_file_weight(file_path):
    if 'test' in file_path or 'spec' in file_path:
        return 0.3  # Less emphasis on tests
    elif file_path.startswith('src/') or 'core' in file_path:
        return 3.0  # Heavy emphasis on core logic
    elif file_path.endswith(('.md', '.txt', '.json')):
        return 0.5  # Config/docs less important
    else:
        return 1.0  # Standard weight

# Sample sequences proportional to weight
weighted_sequences = []
for seq in base_sequences:
    weight = get_file_weight(seq['source_file'])
    num_copies = int(weight)  # 3x for high-priority, 0 or 1 for low
    for _ in range(num_copies):
        weighted_sequences.append(seq)
```

**Result: Model becomes expert at core logic, not test boilerplate**

---

### 3. **Context Window Size: How Much Code Per Sequence?**

Current: 512 tokens per sequence

**Analysis:**
```
512 tokens ≈ 30-40 lines of code
  Pro: Fast training, fits in GPU memory easily
  Con: Model never sees full function context
  
1024 tokens ≈ 60-80 lines of code
  Pro: Captures full function + surrounding context
  Con: 2x slower training, 2x memory usage
  
2048 tokens ≈ 120-150 lines of code
  Pro: Captures full file context, relationships
  Con: 4x slower training, 4x memory usage
```

**For BEST model:**
- **Use 1024 tokens** (sweet spot for Rust/complex code)
- Captures full function definitions
- Sees inter-function dependencies
- Still trainable on RTX 2060 (8GB VRAM)

**Trade-off:**
- Fewer sequences (half as many)
- Richer context per sequence
- Better understanding of code structure

---

### 4. **Overlap Strategy: How Much Context Sharing?**

Current: overlap=128 tokens (25% overlap)

**Analysis:**
```
Overlap=0 (no overlap):
  Pro: Maximum unique data
  Con: Model never sees inter-chunk dependencies
  Example: Sees function A and function B separately
  
Overlap=256 (50% overlap):
  Pro: Model sees how functions relate
  Con: More redundancy
  Example: Sees function A→B transition in one chunk,
          B→C transition in next chunk
  
Overlap=384 (75% overlap):
  Pro: Maximum context continuity
  Con: Heavy redundancy
  Example: Learns long-range dependencies
```

**For BEST model:**
- **Use overlap=384** (75%) with 1024-token windows
- Model learns how code flows between sections
- Understands long-range dependencies
- Critical for Rust (lifetime annotations, borrow checker)

**Trade-off:**
- Fewer unique chunks
- But each chunk teaches more about relationships

---

### 5. **Training Data Composition: What Mix of Code Types?**

Current: Random sampling from all files

**For BEST model - Curriculum Learning:**

```python
# Phase 1: Foundation (epochs 1-50)
- Simple utility functions
- Data structure definitions
- Basic algorithms
- Goal: Learn syntax, patterns

# Phase 2: Intermediate (epochs 51-100)
- Complex functions
- API implementations
- Error handling
- Goal: Learn architecture

# Phase 3: Advanced (epochs 101-200)
- Full modules
- Interdependent code
- Optimization techniques
- Goal: Learn system design
```

**Implementation:**
```python
# Order sequences by complexity
sequences_by_complexity = sorted(
    sequences,
    key=lambda s: (
        s['file_lines'],  # Longer = more complex
        s['directory'] == 'src',  # Core = more complex
        'test' not in s['source_file']  # Not tests
    )
)

# Train in curriculum order (simple → complex)
# OR randomly sample with complexity-based probability
```

**Why this matters:**
- Humans learn simple→complex
- Models learn better the same way
- Prevents overwhelming model with complexity early

---

### 6. **Code Augmentation: Create Real Variations**

Current: Copy sequences with modified metadata (fake variations)

**For BEST model - Real Code Augmentation:**

**Technique 1: Variable Renaming**
```rust
// Original
fn calculate_sum(numbers: Vec<i32>) -> i32 {
    numbers.iter().sum()
}

// Augmented
fn compute_total(values: Vec<i32>) -> i32 {
    values.iter().sum()
}
```
**Result: Model learns semantic equivalence**

**Technique 2: Comment Insertion/Removal**
```rust
// Original (no comments)
fn process(data: &str) -> Result<String> {
    let cleaned = data.trim();
    Ok(cleaned.to_string())
}

// Augmented (with comments)
fn process(data: &str) -> Result<String> {
    // Remove whitespace
    let cleaned = data.trim();
    // Convert to owned string
    Ok(cleaned.to_string())
}
```
**Result: Model learns to work with/without docs**

**Technique 3: Code Style Variations**
```rust
// Original (compact)
if x > 0 { return true; } else { return false; }

// Augmented (verbose)
if x > 0 {
    return true;
} else {
    return false;
}
```
**Result: Model learns multiple valid styles**

**Technique 4: Partial Masking (like BERT)**
```rust
// Original
fn add(a: i32, b: i32) -> i32 { a + b }

// Masked (model must predict [MASK])
fn add(a: i32, b: [MASK]) -> i32 { a + b }
```
**Result: Model learns to fill in gaps**

**Implementation:**
```python
def augment_code(code_content):
    augmentations = []
    
    # Variation 1: Rename variables
    augmentations.append(rename_variables(code_content))
    
    # Variation 2: Add/remove comments
    augmentations.append(toggle_comments(code_content))
    
    # Variation 3: Reformat
    augmentations.append(reformat_code(code_content))
    
    # Variation 4: Mask tokens (10% random)
    augmentations.append(mask_random_tokens(code_content, mask_prob=0.1))
    
    return augmentations

# Create 3-5 REAL variations per file
for file in files:
    base_seq = create_sequence(file)
    sequences.append(base_seq)
    
    for aug in augment_code(file.content):
        aug_seq = create_sequence(aug)
        sequences.append(aug_seq)

# Result: 5,000 files * 4 variations = 20,000 UNIQUE sequences
```

---

### 7. **Git History Integration: Learn Evolution Patterns**

Current: Only current file snapshots

**For BEST model - Include Diffs:**

```python
# Sequence format:
{
    'before_code': '...',  # Code before commit
    'after_code': '...',   # Code after commit
    'diff': '...',         # What changed
    'commit_message': '...', # Why it changed
}

# Model learns:
- How code evolves over time
- Common refactoring patterns
- Bug fix patterns
- How commit messages relate to changes
```

**Example:**
```
Before:
def process(data):
    return data.strip().lower()

After:
def process(data: str) -> str:
    return data.strip().lower()

Commit: "Add type hints for better clarity"

Model learns: When to add type annotations
```

**Implementation:**
```python
import git

repo = git.Repo(THE_BLOCK)
for commit in repo.iter_commits():
    for file_path in commit.stats.files.keys():
        # Get before/after versions
        before = get_file_at_commit(commit.parents[0], file_path)
        after = get_file_at_commit(commit, file_path)
        
        sequence = {
            'tokens': tokenize(before + after),
            'metadata': {
                'type': 'evolution',
                'commit_message': commit.message,
                'author': commit.author.name,
                'timestamp': commit.committed_date
            }
        }
        sequences.append(sequence)
```

**Result: Model learns temporal patterns, not just static code**

---

## Optimal Configuration for BEST Model

```python
# Core Parameters
MAX_TOKENS = 1024          # Larger context (vs 512)
OVERLAP = 768              # 75% overlap (vs 25%)
TARGET_SEQUENCES = 20000   # More REAL variations (vs 11K synthetic)

# Weighting Strategy
HIGH_PRIORITY_WEIGHT = 3.0  # Core logic
MEDIUM_PRIORITY_WEIGHT = 1.0  # Utilities
LOW_PRIORITY_WEIGHT = 0.3   # Tests/configs

# Augmentation Strategy
AUGMENTATIONS_PER_FILE = 4  # Real variations
augmentation_techniques = [
    'variable_renaming',
    'comment_toggling',
    'style_reformatting',
    'partial_masking'
]

# Data Composition
include_git_diffs = True     # Learn evolution
include_commit_context = True # Learn why changes happen
curriculum_learning = True   # Simple → complex

# Training Strategy
epochs = 300                 # Longer training with richer data
batch_size = 8               # Smaller batch, richer gradients
learning_rate = 2e-5         # Lower LR for complex data
```

**Expected Results:**
- **More sequences:** 20,000 (vs 11,000 or 5,000)
- **Higher quality:** All unique, real variations
- **Richer context:** 1024 tokens (vs 512)
- **Better understanding:** Learns evolution + static patterns
- **Training time:** 2-3x longer (but worth it)
- **Model capability:** SIGNIFICANTLY better

---

## Implementation Priority for BEST Model

### Must Have (Critical for Quality):
1. **Increase context to 1024 tokens**
2. **Add smart weighting** (prioritize core code)
3. **Include git diffs + evolution**
4. **Real code augmentation** (4 variations per file)

### Should Have (Improves Quality):
5. **75% overlap** for relationship learning
6. **Curriculum learning** (simple → complex)
7. **Partial masking** (BERT-style)

### Nice to Have (Fine-tuning):
8. **Multi-language support** (if repo has multiple)
9. **Cross-file relationships** (import tracking)
10. **Complexity-based sampling**

---

## Key Insight

> "The best model doesn't come from the most sequences.
> It comes from the most INFORMATIVE sequences.
>
> 20,000 unique, weighted, context-rich sequences with evolution
> beats
> 50,000 redundant, flat, snapshot-only sequences."

---

## Next Step

Should I implement the **EFFECTIVENESS-OPTIMIZED** version of `create_training_dataset.py`?

It will:
- Take longer to run (30-60 min)
- Produce larger files (200-300 MB)
- Create better training data
- Result in SMARTER model

**Trade time for quality?**
