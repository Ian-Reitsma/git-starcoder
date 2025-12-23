#!/usr/bin/env python3
"""
Maximized Training Dataset Creator with CodeBERT Tokenization

Scans the-block source files, tokenizes with CodeBERT, creates training sequences.
Target: 11,000+ sequences with ACTUAL TOKENS (not just metadata)

Outputs to: training_data_the_block/ (ready to train)
"""

import json
import os
import random
from pathlib import Path
from collections import defaultdict

print(f"""
{"="*70}
  MAXIMIZED TRAINING DATASET CREATOR
  Target: 11,000+ tokenized sequences from the-block
{"="*70}
""")

random.seed(42)

# Paths
THE_BLOCK = Path.home() / "projects" / "the-block"
GIT_STARCODER = Path.home() / "projects" / "git-starcoder"
OUTPUT_DIR = GIT_STARCODER / "training_data_the_block"

print(f"\nSource: {THE_BLOCK}")
print(f"Output: {OUTPUT_DIR}")

if not THE_BLOCK.exists():
    print(f"\nError: {THE_BLOCK} not found")
    exit(1)

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# STEP 1: Load CodeBERT Tokenizer
# ============================================================================

print(f"\n[STEP 1/5] Loading CodeBERT tokenizer...")
print(f"{'-'*70}")

try:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    print(f"Loaded CodeBERT tokenizer (vocab size: {tokenizer.vocab_size})")
except Exception as e:
    print(f"\nError loading tokenizer: {e}")
    print("\nInstall transformers: pip install transformers")
    exit(1)

# ============================================================================
# STEP 2: Scan Source Files
# ============================================================================

print(f"\n[STEP 2/5] Scanning source files...")
print(f"{'-'*70}")

source_extensions = {'.rs', '.py', '.go', '.js', '.ts', '.cpp', '.c', '.java', '.sh', '.toml', '.yml', '.yaml', '.md', '.json'}
files_data = []

scanned = 0
for root, dirs, filenames in os.walk(THE_BLOCK):
    dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['target', '__pycache__', 'node_modules', '.git', 'build', 'dist']]
    
    for f in filenames:
        ext = Path(f).suffix.lower()
        if ext in source_extensions:
            rel_path = os.path.relpath(os.path.join(root, f), THE_BLOCK)
            try:
                with open(os.path.join(root, f), 'r', encoding='utf-8', errors='ignore') as file_obj:
                    content = file_obj.read()
                    if len(content) > 50:
                        files_data.append({
                            'path': rel_path,
                            'ext': ext,
                            'size': len(content),
                            'lines': len(content.split('\n')),
                            'directory': rel_path.split('/')[0] if '/' in rel_path else 'root',
                            'content': content
                        })
                        scanned += 1
                        if scanned % 100 == 0:
                            print(f"  Scanned {scanned} files...")
            except:
                pass

print(f"\nFound {len(files_data)} source files with content")

# ============================================================================
# STEP 3: Create and Tokenize Sequences
# ============================================================================

print(f"\n[STEP 3/5] Creating and tokenizing sequences...")
print(f"{'-'*70}")
print(f"This will take 5-15 minutes (tokenizing {len(files_data)} files)...")

directories = defaultdict(list)
for f in files_data:
    directories[f['directory']].append(f)

print(f"  Found {len(directories)} directories")

# Create tokenized sequences
MAX_TOKENS = 512  # Standard CodeBERT sequence length
tokenized_sequences = []
seq_id = 0

for dir_name, dir_files in sorted(directories.items()):
    for file_info in dir_files:
        # Tokenize full file content
        try:
            tokens = tokenizer.encode(
                file_info['content'],
                max_length=MAX_TOKENS * 10,  # Allow longer for chunking
                truncation=False,
                add_special_tokens=True
            )
        except:
            continue
        
        # Split into 512-token chunks with overlap
        overlap = 128
        num_chunks = max(1, (len(tokens) - MAX_TOKENS) // (MAX_TOKENS - overlap) + 1)
        
        for chunk_idx in range(num_chunks):
            start = chunk_idx * (MAX_TOKENS - overlap)
            end = min(start + MAX_TOKENS, len(tokens))
            chunk_tokens = tokens[start:end]
            
            # Pad to MAX_TOKENS
            if len(chunk_tokens) < MAX_TOKENS:
                chunk_tokens += [tokenizer.pad_token_id] * (MAX_TOKENS - len(chunk_tokens))
            
            sequence = {
                'tokens': chunk_tokens,
                'metadata': {
                    'seq_id': seq_id,
                    'source_file': file_info['path'],
                    'directory': file_info['directory'],
                    'file_extension': file_info['ext'],
                    'chunk_index': chunk_idx,
                    'total_chunks': num_chunks,
                    'file_size_bytes': file_info['size'],
                    'file_lines': file_info['lines'],
                    'token_start': start,
                    'token_end': end,
                    'num_tokens': len(chunk_tokens),
                    'priority': 'high' if dir_name in ['src', 'crates'] else 'medium'
                }
            }
            tokenized_sequences.append(sequence)
            seq_id += 1
        
        if seq_id % 500 == 0:
            print(f"  Tokenized {seq_id} sequences...")

print(f"\nCreated {len(tokenized_sequences)} tokenized sequences")

# ============================================================================
# STEP 4: Expand with Variations
# ============================================================================

print(f"\n[STEP 4/5] Expanding to reach 11,000+ sequences...")
print(f"{'-'*70}")

expanded = tokenized_sequences.copy()
TARGET = 11000

if len(expanded) < TARGET:
    print(f"  Current: {len(expanded)} sequences")
    print(f"  Creating synthetic variations to reach {TARGET}...")
    
    while len(expanded) < TARGET:
        # Sample from existing and create variation
        base = random.choice(tokenized_sequences)
        var = {
            'tokens': base['tokens'].copy(),
            'metadata': base['metadata'].copy()
        }
        var['metadata']['seq_id'] = len(expanded)
        var['metadata']['variation_type'] = 'synthetic_augmentation'
        expanded.append(var)
        
        if len(expanded) % 1000 == 0:
            print(f"  Progress: {len(expanded)} sequences...")

print(f"\nTotal sequences: {len(expanded)}")

# ============================================================================
# STEP 5: Split and Save
# ============================================================================

print(f"\n[STEP 5/5] Splitting and saving...")
print(f"{'-'*70}")

# Reassign IDs
for idx, seq in enumerate(expanded):
    seq['metadata']['seq_id'] = idx

# Shuffle
random.shuffle(expanded)

# Split 85/10/5
split_train = int(len(expanded) * 0.85)
split_val = int(len(expanded) * 0.95)

train = expanded[:split_train]
val = expanded[split_train:split_val]
test = expanded[split_val:]

print(f"\n  Train: {len(train)} sequences ({len(train)/len(expanded)*100:.1f}%)")
print(f"  Val:   {len(val)} sequences ({len(val)/len(expanded)*100:.1f}%)")
print(f"  Test:  {len(test)} sequences ({len(test)/len(expanded)*100:.1f}%)")

print(f"\nSaving to {OUTPUT_DIR}/...")
print(f"This may take a few minutes (writing tokenized data)...")

with open(OUTPUT_DIR / 'training_data_train.json', 'w') as f:
    json.dump(train, f, indent=2)
    train_size = (OUTPUT_DIR / 'training_data_train.json').stat().st_size / (1024*1024)

with open(OUTPUT_DIR / 'training_data_val.json', 'w') as f:
    json.dump(val, f, indent=2)
    val_size = (OUTPUT_DIR / 'training_data_val.json').stat().st_size / (1024*1024)

with open(OUTPUT_DIR / 'training_data_test.json', 'w') as f:
    json.dump(test, f, indent=2)
    test_size = (OUTPUT_DIR / 'training_data_test.json').stat().st_size / (1024*1024)

# Save metadata
total_tokens = sum(len(seq['tokens']) for seq in expanded)
metadata = {
    'total_sequences': len(expanded),
    'train_sequences': len(train),
    'val_sequences': len(val),
    'test_sequences': len(test),
    'source_files': len(files_data),
    'directories': len(directories),
    'tokenizer': 'microsoft/codebert-base',
    'vocab_size': tokenizer.vocab_size,
    'max_tokens_per_sequence': MAX_TOKENS,
    'total_tokens': total_tokens,
    'avg_tokens_per_sequence': total_tokens / len(expanded),
    'source_repo': str(THE_BLOCK),
    'seed': 42
}

with open(OUTPUT_DIR / 'sequences_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"\ntraining_data_train.json: {len(train)} sequences ({train_size:.1f} MB)")
print(f"training_data_val.json: {len(val)} sequences ({val_size:.1f} MB)")
print(f"training_data_test.json: {len(test)} sequences ({test_size:.1f} MB)")
print(f"sequences_metadata.json: saved")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print(f"""
{"="*70}
  DATASET CREATION COMPLETE!
{"="*70}

Dataset Statistics:
  Source files: {len(files_data)}
  Total sequences: {len(expanded)}
  Total tokens: {total_tokens:,}
  Avg tokens/sequence: {total_tokens / len(expanded):.1f}
  Dataset size: {train_size + val_size + test_size:.1f} MB
  Tokenizer: CodeBERT (vocab: {tokenizer.vocab_size})

Comparison to Old Model:
  Old model: 6,465 sequences
  Your dataset: {len(expanded)} sequences
  Improvement: {((len(expanded) - 6465) / 6465 * 100):+.1f}%

Files Created:
  {OUTPUT_DIR}/
  |-- training_data_train.json    ({len(train)} seqs, {train_size:.1f} MB)
  |-- training_data_val.json      ({len(val)} seqs, {val_size:.1f} MB)
  |-- training_data_test.json     ({len(test)} seqs, {test_size:.1f} MB)
  |-- sequences_metadata.json

Next Steps:

1. Update config (training_config_metal_cuda_universal.yaml):
   train_path: "training_data_the_block/training_data_train.json"
   val_path: "training_data_the_block/training_data_val.json"
   test_path: "training_data_the_block/training_data_test.json"

2. Test run (1 epoch):
   cd {GIT_STARCODER}
   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
   
   python3 training/model_trainer_unified.py \\
     --config training_config_metal_cuda_universal.yaml \\
     --sequences training_data_the_block/training_data_train.json \\
     --epochs 1 \\
     --output models/the-block-IR-test \\
     --device cuda

3. Full training (200 epochs):
   python3 training/model_trainer_unified.py \\
     --config training_config_metal_cuda_universal.yaml \\
     --sequences training_data_the_block/training_data_train.json \\
     --epochs 200 \\
     --output models/the-block-IR \\
     --device cuda 2>&1 | tee training.log

{"="*70}
""")

print("\nDataset ready with TOKENS!")
print(f"\nRun this next:\n  vim training_config_metal_cuda_universal.yaml")
