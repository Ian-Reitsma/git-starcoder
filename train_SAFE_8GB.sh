#!/bin/bash
# SAFE TRAINING SCRIPT FOR 8GB GPUs
# Uses ultra-conservative settings guaranteed to work on RTX 2060 Super

set -e  # Exit on error

echo "================================================================"
echo "  SAFE 8GB GPU TRAINING MODE"
echo "================================================================"
echo ""

# Step 1: Clean GPU memory
echo "Step 1: Cleaning GPU memory..."
pkill -9 -f "python.*training" 2>/dev/null || true
pkill -9 -f "deepspeed" 2>/dev/null || true
sleep 2

python3 -c "
import torch
import gc
if torch.cuda.is_available():
    gc.collect()
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info()
    print(f'  GPU Memory: {free/(1024**3):.2f} GB free / {total/(1024**3):.2f} GB total')
else:
    print('  WARNING: CUDA not available!')
"
echo ""

# Step 2: Show what we're using
echo "Step 2: Configuration"
echo "  Config: training_config_SAFE_8GB.yaml"
echo "  DeepSpeed: ds_config_SAFE_8GB.json"
echo "  Settings:"
echo "    - Context: 512 tokens"
echo "    - Target: 64 tokens"
echo "    - LoRA rank: 8"
echo "    - Batch size: 1"
echo "    - Gradient accumulation: 256"
echo "    - Gradient checkpointing: ENABLED"
echo "    - Expected VRAM usage: ~2-3 GB"
echo ""

# Step 3: Start training
echo "Step 3: Starting training..."
echo ""

# Use the safe configs
deepspeed --num_gpus=1 training/model_trainer_unified.py \
    --config training_config_SAFE_8GB.yaml \
    --deepspeed ds_config_SAFE_8GB.json \
    --sequences models/the-block_jan1_68k/training_data_ELITE/training_data_train.jsonl \
    --output models/the-block_jan1_68k \
    --epochs 15 \
    --val-sequences models/the-block_jan1_68k/training_data_ELITE/training_data_val.jsonl

echo ""
echo "================================================================"
echo "  TRAINING COMPLETE!"
echo "================================================================"
