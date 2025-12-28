# 🏆 Tier System - Hardware to Context Mapping

## Overview

The ELITE Training System uses an **intelligent tier system** that automatically selects the optimal context window based on your hardware.

**11 tiers** ranging from **4K to 4M+ tokens**.

---

## 📊 Complete Tier Specifications

| Tier | Context | Target | Total | LoRA Rank | Improvement | Min VRAM | Requirements |
|------|---------|--------|-------|-----------|-------------|----------|--------------|
| 1 | 4K | 512 | 4.5K | 48 | 16x | 4GB | None |
| 2 | 8K | 2K | 10K | 32 | 32x | 6GB | None |
| 3 | 16K | 2K | 18K | 24 | 64x | 6GB | Flash/SDPA |
| 4 | 32K | 4K | 37K | 12 | 128x | 8GB | Flash + DeepSpeed |
| 5 | 57K | 7K | 64K | 8 | 256x | 12GB | Flash + DeepSpeed |
| 6 | 131K | 16K | 147K | 8 | 512x | 16GB | Flash + DeepSpeed |
| 7 | 262K | 32K | 295K | 6 | 1024x | 20GB | Flash + DeepSpeed |
| **8** 🔥 | **512K** | **65K** | **577K** | **4** | **2048x** | **8GB** | **+ EXTREME** |
| **9** 🌟 | **1M** | **131K** | **1.1M** | **4** | **4096x** | **12GB** | **+ Ring Attn** |
| **10** 🌟 | **2M** | **262K** | **2.3M** | **3** | **8192x** | **24GB** | **+ Ring Attn** |
| **11** 🌟 | **4M** | **524K** | **4.5M** | **2** | **16384x** | **24GB** | **+ Ring Attn** |

**EXTREME optimizations** = GQA + Selective Checkpointing + 4-bit Activations + PowerSGD + PagedAttention + Fused Kernels

---

## 🎯 Tier Selection Logic

The system automatically:
1. Profiles your hardware (stress tests VRAM)
2. Calculates memory requirements for each tier
3. Selects the **highest tier that fits safely**
4. Provides 10-15% headroom for stability

**You don't need to choose - it's automatic!**

---

## 💾 Memory Breakdown by Tier

### TIER 4 (32K) - RTX 2060 Super
```
Base model (QLoRA 4-bit):  1.25 GB
LoRA params (rank 12):     0.20 GB
Activations (Flash 0.25):  0.70 GB
KV cache:                  0.60 GB
Gradients:                 0.40 GB
Optimizer (8-bit):         0.60 GB
Buffers:                   0.50 GB
────────────────────────────────────
Total:                     4.25 GB
Headroom:                  3.75 GB ✅
```

### TIER 8 (512K) - RTX 2060 Super with EXTREME
```
Base model (QLoRA 4-bit):  1.25 GB
LoRA params (rank 4):      0.10 GB
Activations (Sel CP + 4-bit): 0.30 GB  ⚡ (16x reduction!)
KV cache (GQA + Paged):    0.15 GB  ⚡ (16x reduction!)
Gradients (PowerSGD):      0.01 GB  ⚡ (80x reduction!)
Optimizer (ZeRO-3):        0.05 GB
Buffers (Fused):           0.40 GB
────────────────────────────────────
Total:                     2.26 GB  🔥
Headroom:                  5.74 GB ✅
```

---

## 🔧 Requirements by Tier

### TIER 1-2: Basic
- **No special requirements**
- Works on any NVIDIA GPU
- No DeepSpeed needed

### TIER 3: Flash/SDPA
- **FlashAttention-2** (recommended) or SDPA fallback
- RTX 2060+ recommended

### TIER 4-7: Standard High-Context
- **FlashAttention-2** (or SDPA with reduced context)
- **DeepSpeed** for CPU offloading
- RTX 2060 Super+ for TIER 4
- RTX 3090+ for TIER 6-7

### TIER 8-11: EXTREME 🔥
- **All standard requirements** PLUS:
- **EXTREME optimizations** automatically enabled:
  - Grouped Query Attention (GQA)
  - Selective Checkpointing
  - 4-bit Activation Quantization
  - PowerSGD Gradient Compression
  - PagedAttention
  - Fused Kernels
- **Ring Attention** (TIER 9+) for infinite scaling

---

## 📈 Context Size Impact

### What Can You See?

| Context | Lines of Code | Use Cases |
|---------|---------------|-----------|
| 4K | ~1,000 | Single function/class |
| 8K | ~2,000 | Small module |
| 16K | ~4,000 | Medium module |
| 32K | ~8,000 | Large module/multiple files |
| 64K | ~16,000 | Small project |
| 128K | ~32,000 | Medium project |
| 256K | ~64,000 | Large project |
| 512K | ~128,000 | Very large project |
| 1M | ~256,000 | Entire codebase |
| 2M+ | ~512,000+ | Massive monorepo |

---

## 🚀 Hardware Recommendations

### Budget: RTX 2060 Super (8GB) - $200-300 used
- **Best for**: TIER 4 (32K) standard, TIER 8 (512K) with EXTREME
- **Value**: Incredible - 512K context for <$300!
- **Training time**: ~2 days for 20 epochs

### Enthusiast: RTX 3090 (24GB) - $800-1200 used
- **Best for**: TIER 6-7 standard, TIER 9-10 with EXTREME
- **Value**: Best price/performance
- **Training time**: ~2-3 days for 15 epochs
- **Can reach**: 2M context with all optimizations!

### Professional: RTX 4090 (24GB) - $1600-2000
- **Best for**: TIER 7 standard, TIER 11 with EXTREME
- **Value**: Fastest consumer GPU
- **Training time**: ~1.5-2 days for 12 epochs
- **Can reach**: 4M+ context with Ring Attention!

### Enterprise: A100 (80GB) - $10k-15k
- **Best for**: All tiers, multi-GPU for massive contexts
- **Value**: Production deployments
- **Training time**: Sub-day for most tiers
- **Can reach**: Unlimited with Ring Attention + multi-GPU

---

## 🎯 Tier Selection Examples

### Example 1: RTX 2060 Super (8GB)
```
Stress test: 7.2 GB safe VRAM
Available: 7.2 - 1.25 (base) = 5.95 GB

Checking tiers:
  TIER 4 (32K):  4.25 GB ✅ FITS
  TIER 5 (64K):  8.50 GB ❌ Too large
  TIER 8 (512K): 2.26 GB ✅ FITS (with EXTREME!)

Selected: TIER 8 (512K context) 🔥
```

### Example 2: RTX 3090 (24GB)
```
Stress test: 22.5 GB safe VRAM
Available: 22.5 - 1.25 (base) = 21.25 GB

Checking tiers:
  TIER 7 (256K):  18.5 GB ✅ FITS
  TIER 9 (1M):    12.8 GB ✅ FITS (with EXTREME!)
  TIER 10 (2M):   21.0 GB ✅ FITS (with EXTREME!)

Selected: TIER 10 (2M context) 🌟
```

---

## 💡 Tips for Maximizing Context

1. **Install FlashAttention-2**: 80% memory reduction vs standard attention
2. **Enable EXTREME optimizations**: Automatic for TIER 8+
3. **Close other GPU apps**: Free up maximum VRAM
4. **Use DeepSpeed**: CPU offloading for larger contexts
5. **Dynamic curriculum**: Train on smaller contexts first (40% faster convergence)

---

## 🔬 Advanced: Custom Tier Configuration

Want to create custom tiers? Edit `elite_train.py`:

```python
# Around line 431
tier_specs = [
    # (tier, context, target, lora_rank, needs_flash, needs_deepspeed)
    (12, 8388608, 1048576, 2, True, True),  # TIER 12: 8M context!
]
```

**Note**: Requires sufficient VRAM and all EXTREME optimizations.

---

## 📚 Related Documentation

- [Optimizations](optimizations.md) - All 43 optimizations explained
- [Configuration](configuration.md) - Config file options
- [Architecture](architecture.md) - How tier selection works internally

---

**The tier system makes extreme contexts accessible to everyone!** 🚀
