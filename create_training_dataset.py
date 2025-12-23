#!/usr/bin/env python3
"""
Simplified Training Dataset Creator

Scans the-block source files and creates training dataset directly.
Outputs to: training_data_the_block/ (ready to train)

No complex pipelines - just scan, chunk, and split.
"""

import json
import os
import random
from pathlib import Path
from collections import defaultdict

print(f"""
{"="*70}
  TRAINING DATASET CREATOR: Scan the-block Source Files
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
    print(f"\n\u274c Error: {THE_BLOCK} not found")
    exit(1)

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# STEP 1: Scan Source Files
# ============================================================================

print(f"\n[STEP 1/3] Scanning source files...")
print(f"{'-'*70}")

source_extensions = {'.rs', '.py', '.go', '.js', '.ts', '.cpp', '.c', '.java', '.sh', '.toml', '.yml', '.yaml'}
files_data = []

scanned = 0
for root, dirs, filenames in os.walk(THE_BLOCK):
    # Skip hidden dirs and build artifacts
    dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['target', '__pycache__', 'node_modules', '.git', 'build', 'dist']]
    
    for f in filenames:
        ext = Path(f).suffix.lower()
        if ext in source_extensions:
            rel_path = os.path.relpath(os.path.join(root, f), THE_BLOCK)
            try:
                with open(os.path.join(root, f), 'r', encoding='utf-8', errors='ignore') as file_obj:
                    content = file_obj.read()
                    if len(content) > 50:  # Skip tiny files
                        files_data.append({
                            'path': rel_path,
                            'ext': ext,
                            'size': len(content),
                            'lines': len(content.split('\n')),
                            'directory': rel_path.split('/')[0] if '/' in rel_path else 'root',
                            'content': content  # Keep content for later
                        })
                        scanned += 1
                        if scanned % 100 == 0:
                            print(f"  Scanned {scanned} files...")
            except:
                pass

print(f"\n\u2713 Found {len(files_data)} source files with content")

# ============================================================================
# STEP 2: Create Training Sequences
# ============================================================================

print(f"\n[STEP 2/3] Creating training sequences...")
print(f"{'-'*70}")

directories = defaultdict(list)
for f in files_data:
    directories[f['directory']].append(f)

print(f"  Found {len(directories)} directories")

# Create sequences: ~1 per 100 lines + variations
sequences = []
seq_id = 0

for dir_name, dir_files in sorted(directories.items()):
    for file_info in dir_files:
        # Base sequences: 1 per ~100 lines
        num_base_sequences = max(1, file_info['lines'] // 100)
        
        for chunk_idx in range(num_base_sequences):
            # Calculate chunk boundaries
            lines = file_info['content'].split('\n')
            chunk_size = max(1, len(lines) // num_base_sequences)
            start_line = chunk_idx * chunk_size
            end_line = min(start_line + chunk_size, len(lines))
            chunk_content = '\n'.join(lines[start_line:end_line])
            
            sequence = {
                'seq_id': seq_id,
                'source_file': file_info['path'],
                'directory': file_info['directory'],
                'file_extension': file_info['ext'],
                'chunk_index': chunk_idx,
                'total_chunks': num_base_sequences,
                'file_size_bytes': file_info['size'],
                'file_lines': file_info['lines'],
                'chunk_start_line': start_line,
                'chunk_end_line': end_line,
                'chunk_content': chunk_content[:2000],  # Limit to 2K chars per chunk
                'context_metadata': {
                    'sequence_index': seq_id,
                    'directory_context': dir_name,
                    'file_context': file_info['path'],
                    'priority': 'high' if dir_name in ['src', 'crates'] else 'medium'
                }
            }
            sequences.append(sequence)
            seq_id += 1
        
        # Add variations for multi-chunk files
        if num_base_sequences > 1:
            for offset in range(1, min(2, num_base_sequences)):
                var = sequences[-1].copy()
                var['seq_id'] = seq_id
                var['chunk_index'] = (chunk_idx + offset) % num_base_sequences
                var['context_metadata'] = var['context_metadata'].copy()
                var['context_metadata']['variation'] = f"chunk_offset_{offset}"
                sequences.append(var)
                seq_id += 1
        
        if seq_id % 500 == 0:
            print(f"  Created {seq_id} sequences...")

print(f"\n\u2713 Created {len(sequences)} training sequences")

# ============================================================================
# STEP 3: Split and Save
# ============================================================================

print(f"\n[STEP 3/3] Splitting into train/val/test...")
print(f"{'-'*70}")

# Shuffle
random.shuffle(sequences)

# Split 85/10/5
split_train = int(len(sequences) * 0.85)
split_val = int(len(sequences) * 0.95)

train = sequences[:split_train]
val = sequences[split_train:split_val]
test = sequences[split_val:]

print(f"\n  Train: {len(train)} sequences ({len(train)/len(sequences)*100:.1f}%)")
print(f"  Val:   {len(val)} sequences ({len(val)/len(sequences)*100:.1f}%)")
print(f"  Test:  {len(test)} sequences ({len(test)/len(sequences)*100:.1f}%)")

# Save to files
print(f"\nSaving to {OUTPUT_DIR}/...")

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
metadata = {
    'total_sequences': len(sequences),
    'train_sequences': len(train),
    'val_sequences': len(val),
    'test_sequences': len(test),
    'source_files': len(files_data),
    'directories': len(directories),
    'source_repo': str(THE_BLOCK),
    'created': str(Path.cwd()),
    'seed': 42
}

with open(OUTPUT_DIR / 'sequences_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"\n\u2713 training_data_train.json: {len(train)} sequences ({train_size:.1f} MB)")
print(f"\u2713 training_data_val.json: {len(val)} sequences ({val_size:.1f} MB)")
print(f"\u2713 training_data_test.json: {len(test)} sequences ({test_size:.1f} MB)")
print(f"\u2713 sequences_metadata.json: saved")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print(f"""
{"="*70}
  \u2713 DATASET CREATION COMPLETE!
{"="*70}

Dataset Statistics:
  \u2022 Source files scanned: {len(files_data)}
  \u2022 Directories: {len(directories)}
  \u2022 Total sequences: {len(sequences)}
  \u2022 Dataset size: {train_size + val_size + test_size:.1f} MB

Files Created:
  {OUTPUT_DIR}/
  \u251c\u2500 training_data_train.json    ({len(train)} sequences)
  \u251c\u2500 training_data_val.json      ({len(val)} sequences)
  \u251c\u2500 training_data_test.json     ({len(test)} sequences)
  \u2514\u2500 sequences_metadata.json     (metadata)

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

Timeline:
  \u2022 Dataset creation: DONE
  \u2022 Config update: 1 minute
  \u2022 Test run: 5-30 minutes
  \u2022 Full training: 10-50 hours

{"="*70}
""")

print("\nDataset ready! \ud83d\ude80")
print(f"\nRun this next:\n  vim training_config_metal_cuda_universal.yaml")
