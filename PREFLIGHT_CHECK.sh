#!/bin/bash
# PRE-FLIGHT CHECK - Run this before starting training
# Validates everything is ready to go

set -e

echo "================================================================================"
echo "  ✈️  PRE-FLIGHT CHECK - TIER 4 (32K Context Training)"
echo "  Validating everything before starting full training"
echo "================================================================================"
echo ""

# Run comprehensive tests
echo "[1/3] Running comprehensive test suite..."
python3 test_extreme_optimizations.py || {
    echo ""
    echo "❌ Tests failed! Please fix errors above before proceeding."
    exit 1
}

echo ""
echo "[2/3] Checking dataset..."
if [ -f "training_data_ELITE/training_data_train.jsonl" ]; then
    # Check if dataset has large sequences
    echo "  Checking sequence sizes in dataset..."
    python3 << 'EOF'
import json

with open('training_data_ELITE/training_data_train.jsonl', 'r') as f:
    first_line = f.readline()
    data = json.loads(first_line)
    seq_len = len(data['tokens'])

    print(f"  Sample sequence length: {seq_len} tokens")

    if seq_len >= 8192:
        print(f"  ✓ Dataset has large sequences (ready for TIER 4+)")
    elif seq_len >= 1024:
        print(f"  ⚠️  Dataset has medium sequences (ready for TIER 2-3)")
        print(f"  ⚠️  For TIER 4 (32K), regenerate with larger windows")
    else:
        print(f"  ❌ Dataset has small sequences (only ready for TIER 1)")
        print(f"  ❌ MUST regenerate dataset for TIER 4!")
        exit(1)
EOF
else
    echo "  ❌ Dataset not found!"
    echo "  Run: python3 create_training_dataset_ELITE.py"
    exit 1
fi

echo ""
echo "[3/3] Checking VRAM availability..."
nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | python3 -c "
import sys
free_mb = int(sys.stdin.read().strip())
free_gb = free_mb / 1024

print(f'  Free VRAM: {free_gb:.2f} GB')

if free_gb < 7.0:
    print(f'  ⚠️  Warning: Low free VRAM ({free_gb:.2f} GB)')
    print(f'  Close other GPU applications before training')
    sys.exit(1)
else:
    print(f'  ✓ Sufficient VRAM available')
"

echo ""
echo "================================================================================"
echo "  ✅ PRE-FLIGHT CHECK PASSED!"
echo "================================================================================"
echo ""
echo "Everything is ready. You can now start training:"
echo ""
echo "  TIER 4 (32K context - RECOMMENDED):"
echo "  ────────────────────────────────────"
echo "  deepspeed --num_gpus=1 model_trainer_metal_cuda.py \\"
echo "    --config training_config_TIER4_32k.yaml \\"
echo "    --deepspeed ds_config_tier4_32k.json \\"
echo "    --sequences training_data_ELITE/training_data_train.jsonl \\"
echo "    --epochs 20 \\"
echo "    --output models/the-block-ELITE-TIER4-32kctx"
echo ""
echo "  OR without DeepSpeed (uses more VRAM):"
echo "  ──────────────────────────────────────"
echo "  python3 model_trainer_metal_cuda.py \\"
echo "    --config training_config_TIER4_32k.yaml \\"
echo "    --sequences training_data_ELITE/training_data_train.jsonl \\"
echo "    --epochs 20 \\"
echo "    --output models/the-block-ELITE-TIER4-32kctx"
echo ""
echo "Happy training! 🚀"
echo ""
