# 🔧 Troubleshooting Guide

Common issues and solutions for the ELITE Training System.

---

## 🚨 Installation Issues

### FlashAttention-2 Won't Install

**Error**: `Compilation failed` or `CUDA error`

**Solutions**:
1. **Use SDPA fallback** (automatic, 60% of Flash benefits):
   - System automatically falls back to SDPA if Flash not available
   - No action needed!

2. **Install prebuilt wheel**:
   ```bash
   pip install flash-attn --no-build-isolation
   ```

3. **Reduce parallelism during compile**:
   ```bash
   MAX_JOBS=2 pip install flash-attn
   ```

4. **Check CUDA version**:
   ```bash
   nvcc --version  # Must be 11.8+
   ```

---

### DeepSpeed Installation Fails

**Error**: `ModuleNotFoundError: No module named 'deepspeed'`

**Solutions**:
1. **Install from PyPI**:
   ```bash
   pip install deepspeed
   ```

2. **Install from source**:
   ```bash
   git clone https://github.com/microsoft/DeepSpeed
   cd DeepSpeed
   DS_BUILD_OPS=1 pip install -e .
   ```

3. **Skip DeepSpeed** (limits to TIER 1-3):
   - System works without DeepSpeed for smaller contexts
   - TIER 4+ requires DeepSpeed for CPU offloading

---

## 💾 Memory Issues

### Out of Memory During Training

**Error**: `CUDA out of memory`

**Solutions**:
1. **System auto-selects lower tier** - check logs for selected tier
2. **Close other GPU applications**:
   ```bash
   nvidia-smi  # Check what's using GPU
   pkill -f <process_name>  # Kill GPU hogs
   ```

3. **Reduce batch size** (system does this automatically)

4. **Enable gradient checkpointing** (already enabled by default)

5. **Free GPU memory manually**:
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

---

### Out of Memory During Profiling

**Error**: `OOM during stress test`

**Solution**: Profiler uses conservative safety margin (85%). This is expected - system will select appropriate tier.

---

## 🐛 Training Issues

### Training Starts Then Crashes

**Symptoms**: Begins training, then OOM after a few steps

**Solutions**:
1. **Check actual batch size**:
   - System uses binary search to find optimal batch size
   - May need manual override if automatic detection fails

2. **Disable dynamic batch finding**:
   ```python
   # In elite_train.py, force batch_size=1
   optimal_batch_size = 1
   ```

3. **Enable more aggressive memory management**:
   - System already uses memory defragmentation
   - Run `nvidia-smi` to check for memory leaks

---

### Loss Spikes / Training Diverges

**Symptoms**: Loss suddenly increases dramatically

**Solutions**:
- **Automatic rollback enabled** - system detects spikes and rolls back
- Check `loss_spike_detection` config
- May indicate LR too high - system uses scaling laws to prevent this

---

### Slow Training Speed

**Symptoms**: Training is slower than expected

**Checks**:
1. **GPU utilization**:
   ```bash
   nvidia-smi -l 1  # Monitor GPU usage
   ```
   Should be 90-100% during training

2. **Batch size**:
   - Larger batch = faster training
   - System finds maximum safe batch automatically

3. **Optimizations active**:
   ```bash
   # Check config shows all optimizations
   cat <output_path>/training_config.yaml
   ```

4. **Enable sequence packing** (if not already):
   - Check `sequence_packing_enabled` in config
   - Provides 5-6x speedup

---

## 📁 Dataset Issues

### Dataset Generation Fails

**Error**: `No valid sequences found`

**Solutions**:
1. **Check repository path** is correct
2. **Ensure repository has code files** (.py, .js, etc.)
3. **Check disk space**:
   ```bash
   df -h  # Need 5-10 GB free
   ```

---

### Dataset Too Small

**Warning**: `Only X sequences generated`

**Solutions**:
1. **Point to larger repository**
2. **Include more file types** in dataset generator
3. **Lower context window** temporarily to generate more sequences

---

## 🔧 Configuration Issues

### Config File Won't Load

**Error**: `YAML parse error`

**Solutions**:
1. **Regenerate config**:
   ```bash
   rm training_config_*.yaml
   python3 elite_train.py  # Will regenerate
   ```

2. **Check YAML syntax**:
   ```bash
   python3 -c "import yaml; yaml.safe_load(open('config.yaml'))"
   ```

---

### Wrong Tier Selected

**Issue**: System selects lower tier than expected

**Checks**:
1. **Run stress test manually**:
   ```bash
   python3 -c "from elite_train import HardwareProfiler; HardwareProfiler().profile_hardware()"
   ```

2. **Check VRAM actually free**:
   ```bash
   nvidia-smi  # Should show minimal usage
   ```

3. **Close background apps** using GPU

---

## 🚀 Performance Issues

### TIER 8+ Not Available

**Issue**: System selects TIER 7 or lower even with EXTREME optimizations

**Checks**:
1. **EXTREME optimizations enabled**:
   - Should see "🔥 EXTREME optimizations initialized!" during startup
   - Check `extreme_optimizations` section in config

2. **All dependencies installed**:
   ```bash
   python3 test_extreme_optimizations.py
   ```

3. **Sufficient VRAM available**:
   - TIER 8 needs ~2.3 GB with optimizations
   - TIER 9 needs ~12 GB
   - TIER 10+ needs 20+ GB

---

## 🔬 Advanced Debugging

### Enable Debug Logging

```python
# In elite_train.py, add at top:
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Check Memory Breakdown

```python
# After tier selection, check:
print(optimal_config['memory_breakdown'])
```

### Verify Optimizations Active

```bash
# Check YAML config for:
grep -A 5 "extreme_optimizations" training_config_*.yaml
```

---

## 📊 System Requirements Not Met

### GPU Too Old

**Issue**: GPU doesn't support required features

**Minimum Requirements**:
- CUDA Compute Capability 7.0+ (Turing/Ampere/Ada/Hopper)
- 6GB+ VRAM for TIER 3+
- 8GB+ VRAM for TIER 4+

**Solutions**:
- Use TIER 1-2 (works on older GPUs)
- Upgrade GPU (RTX 2060 Super for ~$250 used)

---

### Insufficient RAM

**Issue**: System RAM too low

**Minimum**: 16GB RAM recommended
**For DeepSpeed**: 32GB+ RAM recommended (CPU offloading)

**Solutions**:
- Close other applications
- Disable DeepSpeed (limits to TIER 3)
- Add swap space (slower but works):
  ```bash
  sudo fallocate -l 32G /swapfile
  sudo mkswap /swapfile
  sudo swapon /swapfile
  ```

---

## 🆘 Getting Help

### Before Asking for Help

1. **Run tests**:
   ```bash
   python3 test_extreme_optimizations.py
   ```

2. **Check logs**:
   ```bash
   tail -100 <output_path>/training_monitor.log
   ```

3. **Verify hardware**:
   ```bash
   nvidia-smi
   free -h
   df -h
   ```

### Include in Bug Report

- GPU model and VRAM
- CUDA version (`nvcc --version`)
- Python version
- Output of test suite
- Full error message
- Config files (if applicable)

---

## 📚 Related Documentation

- [Installation](installation.md) - Setup guide
- [Tiers](tiers.md) - Tier requirements
- [Optimizations](optimizations.md) - How optimizations work

---

**Most issues are resolved by ensuring all dependencies are installed and GPU is free!** ✅
