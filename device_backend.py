#!/usr/bin/env python3
"""
Unified Device Backend: Auto-detect and configure CUDA (Linux) or Metal (macOS).

This module provides:
- Platform detection (Darwin/Linux)
- Device backend selection (mps/cuda/cpu)
- Attention backend wiring (Metal FlashAttention vs SDPA vs xFormers)
- Unified environment setup

Usage:
    from device_backend import DeviceBackend
    backend = DeviceBackend()
    backend.setup()  # Configure environment, patch models if needed
    device = backend.device
    attention_backend = backend.attention_backend
"""

import os
import sys
import platform
import logging
import math
from typing import Dict, List, Literal, Optional
from dataclasses import dataclass
import warnings

try:
    import torch
except ImportError:
    torch = None

logger = logging.getLogger(__name__)


@dataclass
class BackendConfig:
    """Configuration for device backend."""
    device: str  # "cuda", "mps", "cpu"
    device_type: str  # "cuda", "metal", "cpu"
    is_metal: bool  # True if macOS + metal-enabled
    is_cuda: bool  # True if Linux + CUDA-enabled
    attention_backend: str  # "metal", "sdpa", "xformers", "native"
    supports_bf16: bool
    max_vram_gb: float
    platform_name: str  # "Darwin", "Linux"


class MetalFlashAttentionPatcher:
    """Patch PyTorch models to use Metal FlashAttention when available."""

    @staticmethod
    def enable_metal_flash_attn() -> bool:
        """Load and enable Metal FlashAttention dylib via local orchard_bridge.

        Sets Orchard runtime flags (USE_FLASH_ATTN) and verifies kernels.
        """
        try:
            # Prefer local orchard_bridge package
            try:
                from orchard_bridge import enable_flash, is_metal_flash_attn_available
            except ImportError:
                logger.warning("orchard_bridge not available; Metal FlashAttention disabled")
                return False

            # Ensure runtime flags and dylib are configured
            ok = enable_flash(verbose=False, strict=False)
            if not ok:
                logger.warning("enable_flash did not complete successfully; falling back to standard attention")
                return False

            # Verify kernels
            if not is_metal_flash_attn_available():
                logger.warning("Metal FlashAttention kernels not available after enable_flash; falling back")
                return False

            logger.info("Metal FlashAttention enabled successfully via orchard_bridge")
            return True
        except Exception as e:
            logger.debug(f"Metal FlashAttention not available: {e}")
            return False

    @staticmethod
    def patch_model_for_metal(model) -> None:
        """Patch model to use Metal-optimized attention if available.

        If orchard_bridge.flash_attn is available and kernels are loaded,
        wraps the model's attention modules to call Metal FlashAttention
        when running on 'mps'. Otherwise, only applies dtype safety fixes.
        """
        try:
            # Try to import orchard_bridge.flash_attn
            try:
                from orchard_bridge import flash_attn, is_metal_flash_attn_available
            except ImportError:
                flash_attn = None
                is_metal_flash_attn_available = lambda: False

            use_orchard = flash_attn is not None and is_metal_flash_attn_available()

            # Try to patch attention modules
            if hasattr(model, "transformer"):
                # For GPT-like models (e.g., StarCoder)
                for module in model.transformer.modules():
                    if hasattr(module, "_attn"):
                        MetalFlashAttentionPatcher._patch_attn_module(module, flash_attn, use_orchard)
            elif hasattr(model, "model"):
                # For other models
                for module in model.model.modules():
                    if hasattr(module, "_attn") or hasattr(module, "attn"):
                        MetalFlashAttentionPatcher._patch_attn_module(module, flash_attn, use_orchard)

            logger.info(
                "Model patched for Metal optimization (orchard_flash_attn=%s)",
                use_orchard,
            )
        except Exception as e:
            logger.debug(f"Could not patch model for Metal: {e}")

    @staticmethod
    def _patch_attn_module(module, orchard_flash_attn=None, use_orchard: bool = False) -> None:
        """Patch attention module to use Metal kernels when available.

        If orchard_flash_attn is provided and use_orchard is True, calls into
        orchard_bridge.flash_attn on MPS. Otherwise, falls back to original _attn
        with dtype safety fixes.
        """
        try:
            if hasattr(module, "_attn"):
                original_attn = module._attn

                def metal_aware_attn(*args, **kwargs):
                    """Metal-aware attention wrapper (signature-agnostic).

                    Safety rules:
                    - Only use Orchard FlashAttention when running on MPS AND there is no
                      explicit attention mask (padding mask / arbitrary mask).
                    - Causality is taken from kwargs (is_causal/causal) or module attribute.
                    - Otherwise, fall back to original attention.
                    """
                    try:
                        import torch

                        # Extract q/k/v from positional args or kwargs
                        if len(args) >= 3:
                            query, key, value = args[0], args[1], args[2]
                        else:
                            query = kwargs.get("query") or kwargs.get("q")
                            key = kwargs.get("key") or kwargs.get("k")
                            value = kwargs.get("value") or kwargs.get("v")

                        if query is None or key is None or value is None:
                            return original_attn(*args, **kwargs)

                        attn_mask = (
                            kwargs.get("attn_mask")
                            if "attn_mask" in kwargs
                            else kwargs.get("attention_mask")
                        )
                        dropout_p = kwargs.get("dropout_p", kwargs.get("dropout", 0.0))
                        is_causal = kwargs.get(
                            "is_causal",
                            kwargs.get("causal", getattr(module, "is_causal", False)),
                        )

                        # For Metal, ensure proper dtype handling
                        if getattr(query, "device", None) is not None and query.device.type == "mps":
                            # Metal has specific dtype requirements
                            if query.dtype == torch.float16:
                                query = query.to(torch.float32)
                            if key.dtype == torch.float16:
                                key = key.to(torch.float32)
                            if value.dtype == torch.float16:
                                value = value.to(torch.float32)

                            def _mask_is_empty(m):
                                if m is None:
                                    return True
                                if isinstance(m, dict):
                                    return len(m) == 0
                                if isinstance(m, tuple):
                                    return len(m) == 0
                                return False

                            # Orchard path does not support arbitrary masks.
                            mask_is_empty = _mask_is_empty(attn_mask)

                            if use_orchard and orchard_flash_attn is not None and mask_is_empty:
                                head_dim = query.shape[-1]
                                scale = 1.0 / math.sqrt(head_dim)
                                out, _ = orchard_flash_attn(
                                    query,
                                    key,
                                    value,
                                    scale,
                                    dropout_p=float(dropout_p),
                                    causal=bool(is_causal),
                                )
                                return out, None

                        return original_attn(*args, **kwargs)
                    except Exception:
                        return original_attn(*args, **kwargs)

                module._attn = metal_aware_attn
        except Exception:
            pass


class DeviceBackend:
    """Unified device backend for CUDA/Metal training."""

    def __init__(self, force_device: Optional[str] = None, verbose: bool = False):
        self.force_device = force_device
        self.verbose = verbose
        self.config = self._detect_backend()
        self._metal_flash_enabled = False

    def _detect_backend(self) -> BackendConfig:
        """Detect and configure device backend."""
        platform_name = platform.system()
        is_darwin = platform_name == "Darwin"
        is_linux = platform_name == "Linux"

        # Device priority: force_device > platform default
        if self.force_device:
            device = self.force_device
        elif is_darwin:
            device = "mps" if self._metal_available() else "cpu"
        elif is_linux:
            device = "cuda" if self._cuda_available() else "cpu"
        else:
            device = "cpu"

        is_cuda = device == "cuda" and self._cuda_available()
        is_metal = device == "mps" and self._metal_available()

        # Attention backend selection
        if is_metal:
            attention_backend = "metal"  # Will try Metal FlashAttention first
        elif is_cuda:
            attention_backend = self._select_cuda_attention_backend()
        else:
            attention_backend = "native"

        # VRAM estimation
        if is_cuda:
            max_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        elif is_metal:
            max_vram = self._estimate_metal_vram()
        else:
            max_vram = 0.0

        supports_bf16 = is_cuda or is_metal  # Both support bf16 natively

        config = BackendConfig(
            device=device,
            device_type="metal" if is_metal else ("cuda" if is_cuda else "cpu"),
            is_metal=is_metal,
            is_cuda=is_cuda,
            attention_backend=attention_backend,
            supports_bf16=supports_bf16,
            max_vram_gb=max_vram,
            platform_name=platform_name,
        )

        if self.verbose:
            logger.info(f"Device Config: {config}")

        return config

    @staticmethod
    def _metal_available() -> bool:
        """Check if Metal (Apple GPU) is available."""
        try:
            # Always try to import torch fresh for this check
            import torch as torch_check
            is_available = torch_check.backends.mps.is_available()
            is_built = torch_check.backends.mps.is_built()
            logger.debug(f"Metal availability check - available: {is_available}, built: {is_built}")
            return is_available and is_built
        except Exception as e:
            logger.debug(f"Metal availability check failed: {e}")
            return False

    @staticmethod
    def _cuda_available() -> bool:
        """Check if CUDA is available."""
        if torch is None:
            return False
        try:
            return torch.cuda.is_available()
        except Exception:
            return False

    @staticmethod
    def _select_cuda_attention_backend() -> str:
        """Select best attention backend for CUDA."""
        try:
            import torch

            # FlashAttention v2 requires Ampere+; RTX 2060 is Turing, so check
            device_properties = torch.cuda.get_device_properties(0)
            capability = device_properties.major

            if capability >= 8:  # Ampere+
                return "xformers"  # Use xFormers (has FA2 support)
            else:
                return "sdpa"  # PyTorch SDPA for Turing
        except Exception:
            return "native"

    @staticmethod
    def _estimate_metal_vram() -> float:
        """Estimate Metal VRAM (Apple Silicon)."""
        try:
            import subprocess

            result = subprocess.run(
                ["sysctl", "-a"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            for line in result.stdout.split("\n"):
                if "hw.memsize" in line:
                    try:
                        memsize = int(line.split(":")[1].strip())
                        return memsize / 1e9 * 0.8  # Estimate 80% available for compute
                    except ValueError:
                        pass
        except Exception:
            pass
        return 16.0  # Conservative default for Apple Silicon

    def setup(self) -> None:
        """Configure environment and patch models if needed."""
        # Set environment variables
        if self.config.is_cuda:
            self._setup_cuda_env()
        elif self.config.is_metal:
            self._setup_metal_env()

        if self.verbose:
            logger.info(f"Device backend setup complete. Device: {self.config.device}")

    def _setup_cuda_env(self) -> None:
        """Configure CUDA environment."""
        # Enable memory fragmentation reduction
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        # Suppress warnings
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

        logger.info("CUDA environment configured")

    def _setup_metal_env(self) -> None:
        """Configure Metal environment and load FlashAttention."""
        # Metal-specific settings
        os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        # Try to enable Metal FlashAttention
        self._metal_flash_enabled = MetalFlashAttentionPatcher.enable_metal_flash_attn()

        # If Orchard kernels are available, patch PyTorch SDPA globally (safe gating).
        # This catches models that don't expose a stable `_attn` hook.
        if self._metal_flash_enabled and torch is not None:
            try:
                import torch.nn.functional as F
                from orchard_bridge import flash_attn as orchard_flash_attn
                from orchard_bridge import is_metal_flash_attn_available

                if is_metal_flash_attn_available():
                    if not hasattr(F, "_star_orig_sdpa"):
                        F._star_orig_sdpa = F.scaled_dot_product_attention

                    def _orchard_sdpa(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
                        try:
                            if (
                                getattr(q, "device", None) is not None
                                and q.device.type == "mps"
                                and (attn_mask is None or (isinstance(attn_mask, dict) and len(attn_mask) == 0) or (isinstance(attn_mask, tuple) and len(attn_mask) == 0))
                            ):
                                head_dim = q.shape[-1]
                                s = float(scale) if scale is not None else (1.0 / math.sqrt(head_dim))
                                out, _ = orchard_flash_attn(
                                    q, k, v, s, dropout_p=float(dropout_p), causal=bool(is_causal)
                                )
                                return out
                        except Exception:
                            pass
                        return F._star_orig_sdpa(
                            q,
                            k,
                            v,
                            attn_mask=attn_mask,
                            dropout_p=dropout_p,
                            is_causal=is_causal,
                            scale=scale,
                        )

                    F.scaled_dot_product_attention = _orchard_sdpa
                    logger.info("Patched torch.nn.functional.scaled_dot_product_attention to use Orchard on MPS when safe")
            except Exception as e:
                logger.warning(f"Could not install global SDPA patch for Orchard: {e}")

        logger.info(
            f"Metal environment configured. FlashAttention: {self._metal_flash_enabled}"
        )

    @property
    def device(self) -> str:
        """Get device string for model.to(device)."""
        return self.config.device

    @property
    def device_type(self) -> str:
        """Get device type for low-level operations."""
        return self.config.device_type

    @property
    def attention_backend(self) -> str:
        """Get attention backend selector."""
        return self.config.attention_backend

    @property
    def supports_bf16(self) -> bool:
        """Check if device supports bfloat16."""
        return self.config.supports_bf16

    @property
    def is_metal(self) -> bool:
        """Check if using Metal backend."""
        return self.config.is_metal

    @property
    def is_cuda(self) -> bool:
        """Check if using CUDA backend."""
        return self.config.is_cuda

    @property
    def max_vram_gb(self) -> float:
        """Get estimated max VRAM in GB."""
        return self.config.max_vram_gb

    def patch_model(self, model) -> None:
        """Patch model for device-specific optimizations."""
        if self.config.is_metal:
            MetalFlashAttentionPatcher.patch_model_for_metal(model)

    def get_model_config_overrides(self) -> Dict[str, any]:
        """Get model config overrides for device."""
        overrides = {}

        # Metal requires specific dtype handling
        if self.config.is_metal:
            # Metal doesn't fully support fp16; prefer bf16 or fp32
            overrides["torch_dtype"] = "bfloat16" if self.supports_bf16 else "float32"
            # Disable gradient checkpointing on Metal (can cause issues)
            overrides["gradient_checkpointing"] = False
        elif self.config.is_cuda:
            # CUDA can handle everything
            overrides["torch_dtype"] = "auto"
            overrides["gradient_checkpointing"] = True

        return overrides

    def log_summary(self) -> None:
        """Log device backend summary."""
        logger.info("=" * 70)
        logger.info("Device Backend Summary")
        logger.info("=" * 70)
        logger.info(f"Platform: {self.config.platform_name}")
        logger.info(f"Device: {self.config.device}")
        logger.info(f"Attention Backend: {self.config.attention_backend}")
        logger.info(f"Max VRAM: {self.config.max_vram_gb:.1f} GB")
        logger.info(f"Supports bf16: {self.config.supports_bf16}")
        if self.config.is_metal:
            logger.info(f"Metal FlashAttention: {self._metal_flash_enabled}")
        logger.info("=" * 70)


def get_device_backend(force_device: Optional[str] = None, verbose: bool = False) -> DeviceBackend:
    """Factory function to create device backend."""
    return DeviceBackend(force_device=force_device, verbose=verbose)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    backend = DeviceBackend(verbose=True)
    backend.setup()
    backend.log_summary()
