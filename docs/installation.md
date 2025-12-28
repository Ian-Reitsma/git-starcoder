# 📦 Installation Guide

## Prerequisites

### System Requirements
- **OS**: Linux (tested on Fedora, Ubuntu)
- **Python**: 3.8+
- **CUDA**: 11.8+ (for GPU support)
- **GPU**: NVIDIA GPU with 8GB+ VRAM recommended
- **RAM**: 16GB+ recommended
- **Disk**: 20GB+ free space

### Check CUDA
```bash
nvidia-smi  # Should show your GPU
nvcc --version  # Should show CUDA version
```

---

## Installation Steps

### 1. Clone Repository
```bash
git clone <repository-url>
cd git-starcoder
```

### 2. Install Base Dependencies
```bash
pip install -r requirements.txt
```

**Base requirements.txt includes:**
- torch>=2.0.0
- transformers>=4.30.0
- peft>=0.4.0
- bitsandbytes>=0.40.0
- pyyaml
- tqdm
- psutil

### 3. Install Advanced Optimizations (Recommended)
```bash
./install_extreme_optimizations.sh
```

**This installs:**
- FlashAttention-2 (10-15 min compile time)
- DeepSpeed (for CPU offloading)
- Build dependencies

**Note**: FlashAttention-2 compilation requires:
- CUDA 11.8+
- GPU compute capability 7.0+ (Turing/Ampere/Ada/Hopper)
- 8GB+ RAM during compilation

---

## Verification

### Test Your Installation
```bash
python3 test_extreme_optimizations.py
```

**Should show:**
```
✅ PASS: CUDA Availability
✅ PASS: 8-bit AdamW Optimizer
✅ PASS: FlashAttention-2 / SDPA
✅ PASS: YAML Config Loading
✅ PASS: DeepSpeed Config Loading
✅ PASS: Gradient Checkpointing
✅ PASS: VRAM Estimation
✅ PASS: Trainer Code Modifications

🎉 ALL TESTS PASSED!
```

---

## Optional Components

### For Maximum Performance
```bash
# Install Triton (for fused kernels)
pip install triton>=2.0.0

# Install ninja (faster compilation)
pip install ninja

# Install apex (NVIDIA optimizations)
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir ./
```

---

## Troubleshooting

### FlashAttention-2 Fails to Install
**Solution 1**: Use SDPA fallback (60% of Flash benefits, no installation needed)
**Solution 2**: Install from wheel (faster):
```bash
pip install flash-attn --no-build-isolation
```

### CUDA Out of Memory During Install
**Solution**: Close other GPU applications, or compile with less parallelism:
```bash
MAX_JOBS=2 pip install flash-attn
```

### DeepSpeed Installation Fails
**Solution**: Install from source:
```bash
git clone https://github.com/microsoft/DeepSpeed
cd DeepSpeed
pip install -e .
```

---

## Next Steps

After installation:
1. Read [Quickstart Guide](quickstart.md)
2. Run `python3 elite_train.py` to start training
3. See [Configuration](configuration.md) for advanced options

---

## Hardware-Specific Notes

### RTX 2060 Super (8GB)
- ✅ Supports TIER 4-8 (32K-512K)
- ✅ All optimizations work
- Recommended: Install FlashAttention-2 for TIER 4+

### RTX 3090 (24GB)
- ✅ Supports TIER 1-10 (4K-2M)
- ✅ Best price/performance
- Recommended: All optimizations

### RTX 4090 (24GB)
- ✅ Supports TIER 1-11 (4K-4M+)
- ✅ Fastest training
- Recommended: All optimizations + Triton

### A100 (80GB)
- ✅ Supports ALL tiers with maximum batch sizes
- ✅ Enterprise-grade performance
- Recommended: Multi-GPU for even larger contexts

---

**Installation complete!** See [Quickstart](quickstart.md) to begin training.
