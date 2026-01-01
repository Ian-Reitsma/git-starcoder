#!/bin/bash
# GPU Memory Cleanup Script
# Run this before training to free up VRAM

echo "🧹 Cleaning up GPU memory..."

# Kill any stale Python training processes
echo "Terminating stale training processes..."
pkill -f "python.*training" 2>/dev/null || true
pkill -f "deepspeed" 2>/dev/null || true
sleep 2

# Show GPU status before cleanup
echo ""
echo "GPU status BEFORE cleanup:"
nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv,noheader,nounits | \
  awk '{printf "  Used: %.2f GB | Free: %.2f GB | Total: %.2f GB\n", $1/1024, $2/1024, $3/1024}'

# Run Python to clean CUDA cache
python3 -c "
import torch
import gc
if torch.cuda.is_available():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.reset_accumulated_memory_stats()
    print('✓ CUDA cache cleared')
else:
    print('⚠ CUDA not available')
"

# Show GPU status after cleanup
echo ""
echo "GPU status AFTER cleanup:"
nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv,noheader,nounits | \
  awk '{printf "  Used: %.2f GB | Free: %.2f GB | Total: %.2f GB\n", $1/1024, $2/1024, $3/1024}'

echo ""
echo "✓ GPU cleanup complete!"
echo ""
echo "Detailed GPU info:"
nvidia-smi
