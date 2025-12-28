#!/bin/bash
# EXTREME CONTEXT OPTIMIZATION - Installation Script
# Installs all dependencies for 32K-256K context training

set -e  # Exit on error

echo "================================================================================"
echo "  🚀 EXTREME CONTEXT OPTIMIZATION - Installation"
echo "  Installing dependencies for 32K-256K token contexts"
echo "================================================================================"
echo ""

# Check CUDA
echo "[1/6] Checking CUDA availability..."
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; print(f'✓ CUDA {torch.version.cuda} available')"
echo ""

# Install build dependencies
echo "[2/6] Installing build dependencies..."
pip install packaging ninja
echo "✓ Build dependencies installed"
echo ""

# Install FlashAttention-2
echo "[3/6] Installing FlashAttention-2 (this will compile, takes 10-15 min)..."
echo "   Compiling custom CUDA kernels..."
pip install flash-attn --no-build-isolation || {
    echo "⚠ FlashAttention-2 installation failed"
    echo "   Continuing anyway - will fall back to SDPA (60% of benefits)"
}
echo ""

# Verify FlashAttention-2
python3 -c "import flash_attn; print(f'✓ FlashAttention {flash_attn.__version__} installed!')" || {
    echo "⚠ FlashAttention-2 not available, will use PyTorch SDPA fallback"
}
echo ""

# Install DeepSpeed
echo "[4/6] Installing DeepSpeed for CPU offloading..."
pip install deepspeed
echo "✓ DeepSpeed installed"
echo ""

# Verify DeepSpeed
echo "[5/6] Verifying DeepSpeed installation..."
python3 -c "import deepspeed; print(f'✓ DeepSpeed {deepspeed.__version__}')"
echo ""

# Verify all dependencies
echo "[6/6] Verifying all dependencies..."
python3 << 'EOF'
import sys

deps = {}

# Check bitsandbytes
try:
    import bitsandbytes as bnb
    deps['bitsandbytes'] = f'✓ v{bnb.__version__}'
except:
    deps['bitsandbytes'] = '❌ NOT INSTALLED'

# Check flash-attn
try:
    import flash_attn
    deps['flash-attn'] = f'✓ v{flash_attn.__version__}'
except:
    deps['flash-attn'] = '⚠ Not available (will use SDPA fallback)'

# Check deepspeed
try:
    import deepspeed
    deps['deepspeed'] = f'✓ v{deepspeed.__version__}'
except:
    deps['deepspeed'] = '❌ NOT INSTALLED'

# Check transformers
try:
    import transformers
    deps['transformers'] = f'✓ v{transformers.__version__}'
except:
    deps['transformers'] = '❌ NOT INSTALLED'

# Check peft
try:
    import peft
    deps['peft'] = f'✓ v{peft.__version__}'
except:
    deps['peft'] = '❌ NOT INSTALLED'

# Check torch
try:
    import torch
    deps['torch'] = f'✓ v{torch.__version__}'
    deps['CUDA'] = f'✓ v{torch.version.cuda}' if torch.cuda.is_available() else '❌ NOT AVAILABLE'
except:
    deps['torch'] = '❌ NOT INSTALLED'

print("\n" + "="*80)
print("  DEPENDENCY CHECK")
print("="*80)
for name, status in deps.items():
    print(f"  {name:20s} {status}")
print("="*80)

# Determine what's possible
has_flash = 'flash_attn' in sys.modules
has_deepspeed = 'deepspeed' in sys.modules
has_bnb = 'bitsandbytes' in sys.modules

print("\n" + "="*80)
print("  WHAT YOU CAN RUN")
print("="*80)

if has_flash and has_deepspeed and has_bnb:
    print("  ✓ TIER 1-3: 4K-16K contexts (easy)")
    print("  ✓ TIER 4-5: 32K-64K contexts (with all optimizations)")
    print("  ✓ TIER 6: 128K contexts (with Ring Attention - may need additional setup)")
    print("  ✓ TIER 7: 256K contexts (experimental)")
    print("\n  🎯 RECOMMENDED: Start with TIER 4 (32K context)")
elif has_bnb:
    print("  ✓ TIER 1-2: 4K-8K contexts")
    print("  ✓ TIER 3: 16K contexts (with SDPA fallback)")
    print("  ⚠ TIER 4+: Requires FlashAttention-2 and DeepSpeed")
    print("\n  🎯 RECOMMENDED: Start with TIER 2 (8K context)")
else:
    print("  ✓ TIER 1: 4K contexts (basic)")
    print("  ⚠ Higher tiers require bitsandbytes, FlashAttention-2, and DeepSpeed")
    print("\n  🎯 RECOMMENDED: Install missing dependencies")

print("="*80)
EOF

echo ""
echo "================================================================================"
echo "  ✅ INSTALLATION COMPLETE!"
echo "================================================================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Modify dataset creator for your target tier:"
echo "   vim create_training_dataset_ELITE.py"
echo "   (Update CONTEXT_WINDOWS around line 79)"
echo ""
echo "2. Regenerate dataset:"
echo "   python3 create_training_dataset_ELITE.py"
echo ""
echo "3. Start training (example for TIER 4):"
echo "   deepspeed --num_gpus=1 model_trainer_metal_cuda.py \\"
echo "     --config training_config_TIER4_32k.yaml \\"
echo "     --deepspeed ds_config_tier4_32k.json \\"
echo "     --sequences training_data_ELITE/training_data_train.jsonl \\"
echo "     --epochs 20 \\"
echo "     --output models/the-block-ELITE-TIER4-32kctx"
echo ""
echo "4. Read the complete guide:"
echo "   cat EXTREME_CONTEXT_GUIDE.md"
echo ""
echo "Good luck! 🚀"
echo ""
