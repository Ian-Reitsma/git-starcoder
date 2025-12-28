#!/usr/bin/env python3
"""
Fresh Start: Extract the-block commits → Tokenize → Split → Ready to Train

This unified script handles:
1. Git commit extraction from the-block repo
2. Tokenization with git_tokenizer_rich.py
3. Train/Val/Test splitting
4. Data validation
"""

import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import random

# Configuration
THE_BLOCK_PATH = Path.home() / "projects" / "the-block"
GIT_STARCODER_PATH = Path.home() / "projects" / "git-starcoder"
DATA_OUTPUT_DIR = GIT_STARCODER_PATH / "data" / "scrape-dec23"
RANDOM_SEED = 42

print(f"""
═══════════════════════════════════════════════════════════════
  FRESH START: Extract & Train on the-block Repository
═══════════════════════════════════════════════════════════════
""")

# ============================================================================
# PHASE 1: Extract Commits from the-block
# ============================================================================

print(f"\n[PHASE 1] Extracting commits from {THE_BLOCK_PATH}...")

if not THE_BLOCK_PATH.exists():
    print(f"❌ ERROR: {THE_BLOCK_PATH} not found")
    sys.exit(1)

commits = []

try:
    # Get all commits with full info, chronologically
    result = subprocess.run(
        ['git', 'log', '--format=%H%n%an%n%ae%n%aI%n%s%n%b%n---COMMIT_END---', '--reverse'],
        cwd=THE_BLOCK_PATH,
        capture_output=True,
        text=True,
        timeout=60
    )
    
    if result.returncode != 0:
        print(f"❌ Git log failed: {result.stderr}")
        sys.exit(1)
    
    commit_blocks = result.stdout.split('---COMMIT_END---')
    
    for block in commit_blocks:
        lines = [line.rstrip() for line in block.split('\n')]
        lines = [l for l in lines if l]  # Remove empty lines but preserve structure
        
        if len(lines) < 4:
            continue
        
        commit_hash = lines[0].strip()
        author_name = lines[1].strip()
        author_email = lines[2].strip()
        timestamp = lines[3].strip()
        subject = lines[4].strip() if len(lines) > 4 else ""
        
        # Body is lines 5 onwards (up to 5 lines)
        body_lines = []
        for i in range(5, min(10, len(lines))):
            body_lines.append(lines[i])
        body = '\n'.join(body_lines)
        
        # Get files changed in this commit
        try:
            files_result = subprocess.run(
                ['git', 'show', '--name-only', '--format=', commit_hash],
                cwd=THE_BLOCK_PATH,
                capture_output=True,
                text=True,
                timeout=5
            )
            files_changed = [f.strip() for f in files_result.stdout.split('\n') if f.strip()]
        except:
            files_changed = []
        
        commits.append({
            'hash': commit_hash[:40],
            'abbrev_hash': commit_hash[:7],
            'author_name': author_name,
            'author_email': author_email,
            'commit_timestamp': timestamp,
            'subject': subject,
            'body': body,
            'files_modified': files_changed,
            'files_added': [],
            'files_deleted': [],
            'insertions': 0,
            'deletions': 0,
            'is_merge': 'merge' in subject.lower(),
            'parents': [],
            'branches': [],
            'complexity_score': len(files_changed) * 0.5,
        })
        
        if len(commits) % 10 == 0:
            print(f"  → Extracted {len(commits)} commits...", end='\r')
    
    print(f"✓ Extracted {len(commits)} commits                    ")
    
except Exception as e:
    print(f"❌ Extraction failed: {e}")
    sys.exit(1)

if len(commits) == 0:
    print("❌ No commits found in the-block repo")
    sys.exit(1)

# Save raw JSONL
raw_jsonl_path = THE_BLOCK_PATH / "commits_raw.jsonl"
try:
    with open(raw_jsonl_path, 'w') as f:
        for commit in commits:
            f.write(json.dumps(commit) + '\n')
    print(f"✓ Saved raw commits to {raw_jsonl_path}")
except Exception as e:
    print(f"❌ Failed to save JSONL: {e}")
    sys.exit(1)

# ============================================================================
# PHASE 2: Tokenize with git_tokenizer_rich.py
# ============================================================================

print(f"\n[PHASE 2] Tokenizing with git_tokenizer_rich.py...")

tokenizer_path = GIT_STARCODER_PATH / "tokenizers" / "git_tokenizer_rich.py"
if not tokenizer_path.exists():
    print(f"❌ {tokenizer_path} not found")
    sys.exit(1)

# Create output directory
DATA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

tokenized_path = DATA_OUTPUT_DIR / "commits_rich_tokenized.json"

try:
    result = subprocess.run(
        [
            'python3',
            str(tokenizer_path),
            '--input', str(raw_jsonl_path),
            '--sequences', str(tokenized_path),
            '--model', 'microsoft/codebert-base',
            '--sequence-length', '512',
            '--overlap', '128',
            '--stats'
        ],
        cwd=GIT_STARCODER_PATH,
        capture_output=True,
        text=True,
        timeout=600
    )
    
    print(result.stdout)
    
    if result.returncode != 0:
        print(f"❌ Tokenization failed:")
        print(result.stderr)
        sys.exit(1)
    
    if not tokenized_path.exists():
        print(f"❌ Tokenized file not created: {tokenized_path}")
        sys.exit(1)
    
    print(f"✓ Tokenization complete: {tokenized_path}")
    
except Exception as e:
    print(f"❌ Tokenization error: {e}")
    sys.exit(1)

# ============================================================================
# PHASE 3: Load and Split
# ============================================================================

print(f"\n[PHASE 3] Creating train/val/test splits...")

try:
    with open(tokenized_path, 'r') as f:
        data = json.load(f)
    
    token_sequences = data.get('token_sequences', [])
    metadata = data.get('metadata', {})
    
    print(f"  Loaded {len(token_sequences)} token sequences")
    
    # Convert to objects with metadata
    sequence_objects = []
    for i, tokens in enumerate(token_sequences):
        seq_obj = {
            'tokens': tokens,
            'metadata': metadata.get(str(i), {})
        }
        sequence_objects.append(seq_obj)
    
    # Shuffle with fixed seed for reproducibility
    random.seed(RANDOM_SEED)
    random.shuffle(sequence_objects)
    
    # Split: 85% train, 10% val, 5% test
    n = len(sequence_objects)
    split1 = int(n * 0.85)
    split2 = int(n * 0.95)
    
    train_data = sequence_objects[:split1]
    val_data = sequence_objects[split1:split2]
    test_data = sequence_objects[split2:]
    
    print(f"  Train: {len(train_data)} sequences")
    print(f"  Val:   {len(val_data)} sequences")
    print(f"  Test:  {len(test_data)} sequences")
    
    # Save splits
    splits = {
        'training_data_train.json': train_data,
        'training_data_val.json': val_data,
        'training_data_test.json': test_data,
    }
    
    for filename, data_split in splits.items():
        output_path = DATA_OUTPUT_DIR / filename
        with open(output_path, 'w') as f:
            json.dump(data_split, f)
        
        file_size_mb = output_path.stat().st_size / 1024 / 1024
        print(f"✓ {filename}: {len(data_split)} sequences ({file_size_mb:.1f} MB)")
    
except Exception as e:
    print(f"❌ Split creation failed: {e}")
    sys.exit(1)

# ============================================================================
# PHASE 4: Validation
# ============================================================================

print(f"\n[PHASE 4] Data Integrity Validation...")

try:
    for split_name in ['train', 'val', 'test']:
        path = DATA_OUTPUT_DIR / f'training_data_{split_name}.json'
        with open(path, 'r') as f:
            data = json.load(f)
        
        if len(data) == 0:
            print(f"  {split_name.upper()}: (empty - will be skipped during training)")
            continue
        
        first_item = data[0]
        tokens = first_item['tokens'] if isinstance(first_item, dict) else first_item
        metadata = first_item.get('metadata', {}) if isinstance(first_item, dict) else {}
        
        print(f"  {split_name.upper()}:")
        print(f"    ✓ Count: {len(data)}")
        print(f"    ✓ First seq tokens: {len(tokens)}")
        print(f"    ✓ Metadata keys: {list(metadata.keys())}")
    
except Exception as e:
    print(f"❌ Validation failed: {e}")
    sys.exit(1)

# ============================================================================
# NEXT STEPS
# ============================================================================

print(f"""
═══════════════════════════════════════════════════════════════
  ✓ EXTRACTION COMPLETE - Ready to Train!
═══════════════════════════════════════════════════════════════

Next Steps:

1. Update config file:
   Edit training_config_metal_cuda_universal.yaml:
   
   train_path: "data/scrape-dec23/training_data_train.json"
   val_path: "data/scrape-dec23/training_data_val.json"
   test_path: "data/scrape-dec23/training_data_test.json"

2. Run quick test (1 epoch):
   cd {GIT_STARCODER_PATH}
   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
   
   python3 training/model_trainer_unified.py \\
     --config training_config_metal_cuda_universal.yaml \\
     --sequences data/scrape-dec23/training_data_train.json \\
     --epochs 1 \\
     --output models/the-block-IR-test \\
     --device cuda

3. If test passes, run full training (200 epochs):
   python3 training/model_trainer_unified.py \\
     --config training_config_metal_cuda_universal.yaml \\
     --sequences data/scrape-dec23/training_data_train.json \\
     --epochs 200 \\
     --output models/the-block-IR-fresh \\
     --device cuda 2>&1 | tee training_fresh.log

Data location: {DATA_OUTPUT_DIR}
═══════════════════════════════════════════════════════════════
""")

print("✓ All extraction phases complete!")
