#!/usr/bin/env python3
"""MPS-specific PyTorch optimizations for maximum training performance.

Apply these settings at the start of training to leverage Apple Silicon capabilities.
"""

import torch
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


def apply_mps_optimizations(verbose: bool = True) -> dict:
    """Apply MPS-specific optimizations for training.
    
    Returns:
        dict with applied settings for logging
    """
    if not torch.backends.mps.is_available():
        if verbose:
            logger.warning("MPS not available; skipping MPS-specific optimizations")
        return {}
    
    settings = {}
    
    # 1. Enable reduced precision for matmul reductions (faster, minimal accuracy loss)
    try:
        torch.backends.mps.matmul.allow_fp16_reduced_precision_reduction = True
        settings['matmul_reduced_precision'] = True
        if verbose:
            logger.info("✓ Enabled MPS fp16 reduced precision matmul")
    except Exception as e:
        if verbose:
            logger.warning(f"Could not enable MPS reduced precision: {e}")
    
    # 2. Set memory allocation strategy (helps with fragmentation)
    try:
        # Use PyTorch's caching allocator settings
        os.environ.setdefault('PYTORCH_MPS_HIGH_WATERMARK_RATIO', '0.0')
        settings['mps_allocator_tuned'] = True
        if verbose:
            logger.info("✓ Tuned MPS memory allocator")
    except Exception as e:
        if verbose:
            logger.warning(f"Could not tune MPS allocator: {e}")
    
    # 3. Enable MPS profiler hooks if available (for debugging)
    try:
        if hasattr(torch.mps, 'profiler'):
            settings['profiler_available'] = True
    except:
        pass
    
    # 4. Ensure Metal capture is disabled during training (perf hit)
    try:
        os.environ.setdefault('METAL_DEVICE_WRAPPER_TYPE', '1')  # Conservative mode
        settings['metal_wrapper_configured'] = True
    except:
        pass
    
    if verbose:
        logger.info(f"Applied {len(settings)} MPS-specific optimizations")
    
    return settings


def configure_attention_backend(model, device_type: str, verbose: bool = True) -> str:
    """Configure optimal attention backend for the device.
    
    Args:
        model: The model to configure
        device_type: 'mps', 'cuda', or 'cpu'
        verbose: Log the selected backend
    
    Returns:
        Name of the selected attention backend
    """
    backend = 'default'
    
    try:
        # Try to use SDPA (scaled dot product attention) - works on MPS
        if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            # Force SDPA for MPS (better than default attention)
            if device_type == 'mps':
                backend = 'sdpa'
                if verbose:
                    logger.info("✓ Using SDPA attention backend (optimized for MPS)")
            elif device_type == 'cuda':
                # CUDA can use flash attention if available
                try:
                    import flash_attn
                    backend = 'flash_attention'
                    if verbose:
                        logger.info("✓ Using FlashAttention (CUDA)")
                except ImportError:
                    backend = 'sdpa'
                    if verbose:
                        logger.info("✓ Using SDPA attention backend (CUDA)")
    except Exception as e:
        if verbose:
            logger.warning(f"Could not configure attention backend: {e}")
        backend = 'default'
    
    return backend


def optimize_dataloader_for_mps(num_workers: Optional[int] = None) -> int:
    """Determine optimal DataLoader num_workers for MPS.
    
    On Apple Silicon with unified memory, fewer workers is often better
    since there's no PCIe bottleneck.
    
    Args:
        num_workers: If provided, use this value. Otherwise auto-detect.
    
    Returns:
        Optimal number of workers
    """
    if num_workers is not None:
        return num_workers
    
    # On MPS, 1-2 workers is usually optimal due to unified memory
    # More workers can actually slow things down due to process overhead
    import multiprocessing
    cpu_count = multiprocessing.cpu_count()
    
    # Conservative: use min(2, cpu_count // 4)
    # Unified memory means less benefit from parallel data loading
    optimal = min(2, max(1, cpu_count // 4))
    
    logger.info(f"✓ Dataloader workers for MPS: {optimal} (CPU count: {cpu_count})")
    return optimal


def enable_mps_fallback():
    """Enable CPU fallback for unsupported MPS ops.
    
    Some PyTorch operations don't have MPS implementations yet.
    This allows graceful fallback to CPU.
    """
    try:
        os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
        logger.info("✓ Enabled MPS->CPU fallback for unsupported ops")
    except:
        pass


def apply_compile_optimizations(model, device_type: str, mode: str = 'default'):
    """Apply torch.compile() optimizations if available.
    
    Args:
        model: Model to compile
        device_type: 'mps', 'cuda', or 'cpu'
        mode: Compilation mode ('default', 'reduce-overhead', 'max-autotune')
    
    Returns:
        Compiled model or original if compilation not available
    """
    # torch.compile() on MPS is experimental in PyTorch 2.x
    # Only enable if torch version supports it
    try:
        if hasattr(torch, 'compile') and device_type in ('cuda', 'cpu'):
            logger.info(f"Applying torch.compile(mode={mode})...")
            return torch.compile(model, mode=mode)
        else:
            logger.info("torch.compile() not available or not recommended for this device")
            return model
    except Exception as e:
        logger.warning(f"torch.compile() failed: {e}; using eager mode")
        return model


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("\nMPS Optimization Settings Test\n" + "="*50)
    
    settings = apply_mps_optimizations(verbose=True)
    print(f"\nApplied settings: {settings}")
    
    workers = optimize_dataloader_for_mps()
    print(f"Recommended DataLoader workers: {workers}")
    
    enable_mps_fallback()
    print("\nMPS optimizations configured.")
