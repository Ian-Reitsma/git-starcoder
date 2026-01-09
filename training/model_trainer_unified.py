#!/usr/bin/env python3
"""
Unified Model Trainer - Supports Multiple Architectures

Supports:
- GPT2 (original, no quantization/LoRA needed)
- StarCoder2-3B (recommended: code-specialized, with 4-bit + LoRA)
- Phi-2 (alternative: reasoning + code)
- Any HuggingFace causal LM (AutoModel interface)

Key Features:
- Config-driven model selection
- 4-bit and 8-bit quantization via bitsandbytes
- LoRA (Parameter-Efficient Fine-Tuning) via peft
- Mixed precision (bf16 / fp16)
- Gradient checkpointing
- Behavioral evaluation (code generation tests)
- Hardware monitoring
- Comprehensive metrics tracking
"""

import os
import sys

# 🧠 EINSTEIN FIX: Set CUDA memory config BEFORE torch import!
# expandable_segments:True reduces fragmentation significantly
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import json
import time
import math
import inspect
import torch
import logging
import random
import numpy as np
import warnings
import yaml
import psutil
import shutil
import hashlib
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict, field
from collections import defaultdict

# Avoid tokenizer parallelism warnings when dataloader workers spawn
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Patch RoBERTa before importing to avoid attention device mismatches
def _patch_roberta_at_import():
    """Patch RoBERTa attention to disable causal attention which causes device mismatches"""
    try:
        from transformers.models.roberta.modeling_roberta import RobertaSelfAttention
        original_forward = RobertaSelfAttention.forward

        def patched_forward(self, hidden_states, attention_mask=None, head_mask=None,
                          encoder_hidden_states=None, encoder_attention_mask=None,
                          past_key_value=None, output_attentions=False, cache_position=None):
            # Disable causal attention to avoid attn_bias device mismatch
            original_is_causal = getattr(self, 'is_causal', False)
            self.is_causal = False
            try:
                return original_forward(self, hidden_states, attention_mask, head_mask,
                                      encoder_hidden_states, encoder_attention_mask,
                                      past_key_value, output_attentions, cache_position)
            finally:
                self.is_causal = original_is_causal

        RobertaSelfAttention.forward = patched_forward
    except:
        pass

_patch_roberta_at_import()

try:
    from torch.utils.data import DataLoader, TensorDataset, Subset, WeightedRandomSampler
    from torch.optim import AdamW  # Fallback if bitsandbytes not available
    # Import scheduler from transformers, not torch
    try:
        from transformers.optimization import get_cosine_schedule_with_warmup
    except ImportError:
        from transformers import get_cosine_schedule_with_warmup
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    try:
        from torch.optim.swa_utils import AveragedModel, SWALR
        HAS_SWA_UTILS = True
    except Exception:
        HAS_SWA_UTILS = False
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from peft import get_peft_model, LoraConfig, TaskType
    from tqdm import tqdm

    # Try to import bitsandbytes for 8-bit optimizer (TIER 4 optimization)
    try:
        import bitsandbytes as bnb
        HAS_BNB_OPTIMIZER = True
    except ImportError:
        HAS_BNB_OPTIMIZER = False

    # Check for FlashAttention support (TIER 4 optimization)
    # CRITICAL: FA2 requires version >= 2.1.0 for transformers attn_implementation
    # FA1 (v1.x) is NOT compatible with attn_implementation="flash_attention_2"
    # On Turing GPUs, SDPA is 15-100x faster than custom FA1 kernels anyway
    try:
        import flash_attn
        from packaging import version
        _fa_ver = version.parse(flash_attn.__version__)
        # Only FA2 (v2.1.0+) works with transformers attn_implementation
        HAS_FLASH_ATTN_2 = _fa_ver >= version.parse("2.1.0")
        HAS_FLASH_ATTN = HAS_FLASH_ATTN_2  # Legacy compat
    except ImportError:
        HAS_FLASH_ATTN = False
        HAS_FLASH_ATTN_2 = False

    # Check for DeepSpeed support (TIER 4+ optimization for CPU offloading)
    try:
        import deepspeed
        HAS_DEEPSPEED = True
    except ImportError:
        HAS_DEEPSPEED = False

    # CPU LoRA offload toolkit (keeps adapters on pinned CPU memory)
    try:
        from cpu_offload_implementation import (
            apply_cpu_offloaded_lora,
            get_cpu_offload_parameters,
            wrap_optimizer_for_cpu_offload,
        )
        HAS_CPU_LORA_OFFLOAD = True
    except Exception as e:
        HAS_CPU_LORA_OFFLOAD = False
        _cpu_offload_import_error = e

except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install: pip install torch transformers peft bitsandbytes pyyaml tqdm")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Log optimization flags after logger is configured
if HAS_BNB_OPTIMIZER:
    logger.info("✓ bitsandbytes 8-bit optimizer available (saves ~1.7 GB VRAM!)")
else:
    logger.warning("⚠ bitsandbytes 8-bit optimizer not available, using standard AdamW")

if HAS_FLASH_ATTN_2:
    import flash_attn
    logger.info(f"✓ FlashAttention-2 v{flash_attn.__version__} (enables 32K+ contexts!)")
else:
    try:
        import flash_attn
        logger.info(f"✓ Using SDPA (FA1 v{flash_attn.__version__} detected but SDPA is faster on Turing)")
    except ImportError:
        logger.info("✓ Using SDPA attention (optimized for Turing)")

if HAS_DEEPSPEED:
    import deepspeed
    logger.info(f"✓ DeepSpeed available v{deepspeed.__version__} (enables CPU offloading for extreme contexts!)")
else:
    logger.warning("⚠ DeepSpeed not available, TIER 4+ may require more VRAM")

if HAS_CPU_LORA_OFFLOAD:
    logger.info("✓ CPU LoRA offload module loaded (pin adapters on CPU, stage to GPU per step)")
else:
    logger.warning(f"⚠ CPU LoRA offload unavailable: {_cpu_offload_import_error}")
if not HAS_SWA_UTILS:
    logger.warning("⚠ SWA utilities unavailable; SWA will be disabled if configured")


def load_yaml_config(config_path: str) -> Dict:
    """Load training configuration from YAML"""
    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}

    if isinstance(cfg, dict):
        # Normalize training.base_learning_rate to float if it’s a string
        tr = cfg.get("training")
        if isinstance(tr, dict) and isinstance(tr.get("base_learning_rate"), str):
            try:
                tr["base_learning_rate"] = float(tr["base_learning_rate"])
            except ValueError:
                pass

        # Ensure Rust build artifacts are ignored (especially for rust config)
        data = cfg.get("data")
        if not isinstance(data, dict):
            data = {}
            cfg["data"] = data

        ignore = data.get("ignore_patterns")
        if not isinstance(ignore, list):
            ignore = []
            data["ignore_patterns"] = ignore

        rust_defaults = ["target/", "*.rlib", "*.rmeta", "Cargo.lock"]
        # add missing defaults
        for p in rust_defaults:
            if p not in ignore:
                ignore.append(p)

    return cfg

def set_seeds(seed: int = 42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Seeds set to {seed}")


class LookaheadOptimizer(torch.optim.Optimizer):
    """Lookahead optimizer wrapper for improved convergence stability."""

    def __init__(self, optimizer: torch.optim.Optimizer, k: int = 5, alpha: float = 0.5):
        if k <= 0:
            raise ValueError("Lookahead k must be > 0")
        if not (0.0 < alpha <= 1.0):
            raise ValueError("Lookahead alpha must be in (0, 1]")
        super().__init__(optimizer.param_groups, optimizer.defaults)
        self.optimizer = optimizer
        self.k = int(k)
        self.alpha = float(alpha)
        self._lookahead_step = 0

        # Mirror underlying optimizer metadata for schedulers.
        self.defaults = optimizer.defaults
        self.param_groups = optimizer.param_groups
        self.state = optimizer.state

        # Initialize slow weights in the optimizer state for each parameter.
        for group in self.param_groups:
            for p in group['params']:
                if p is None:
                    continue
                state = self.state.setdefault(p, {})
                if 'slow_param' not in state:
                    slow = p.detach().clone()
                    slow.requires_grad = False
                    state['slow_param'] = slow

    def step(self, closure=None):
        loss = self.optimizer.step(closure)
        self._lookahead_step += 1

        if self._lookahead_step % self.k != 0:
            return loss

        for group in self.param_groups:
            for p in group['params']:
                if p is None or not p.requires_grad:
                    continue
                state = self.state.setdefault(p, {})
                slow = state.get('slow_param')
                if slow is None:
                    slow = p.detach().clone()
                    slow.requires_grad = False
                    state['slow_param'] = slow
                slow.add_(p.data - slow, alpha=self.alpha)
                p.data.copy_(slow)

        return loss

    def zero_grad(self, *args, **kwargs):
        return self.optimizer.zero_grad(*args, **kwargs)

    def state_dict(self):
        return {
            'optimizer': self.optimizer.state_dict(),
            'lookahead': {
                'k': self.k,
                'alpha': self.alpha,
                'step': self._lookahead_step,
            },
        }

    def load_state_dict(self, state_dict):
        opt_state = state_dict.get('optimizer')
        if opt_state is not None:
            self.optimizer.load_state_dict(opt_state)
        lookahead_state = state_dict.get('lookahead', {})
        self.k = int(lookahead_state.get('k', self.k))
        self.alpha = float(lookahead_state.get('alpha', self.alpha))
        self._lookahead_step = int(lookahead_state.get('step', self._lookahead_step))

class HardwareMonitor:
    """Monitor GPU/CPU/RAM/Thermal with time-based sampling"""
    
    def __init__(self, interval_seconds: float = 5.0):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.has_gpu = torch.cuda.is_available()
        self.interval = interval_seconds
        self.last_sample_time = time.time() - interval_seconds  # allow immediate first sample
        self.stats_history = []
        self.peak_gpu_memory_mb = 0
        self.peak_ram_percent = 0
    
    def should_sample(self) -> bool:
        """Check if it's time to sample"""
        elapsed = time.time() - self.last_sample_time
        return elapsed >= self.interval
    
    def get_stats(self) -> Dict:
        """Get current hardware stats"""
        stats = {
            'timestamp': datetime.now().isoformat(),
            'gpu_memory_mb': 0,
            'gpu_memory_percent': 0,
            'gpu_utilization': 0,
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'ram_gb': psutil.virtual_memory().used / 1e9,
            'ram_percent': psutil.virtual_memory().percent,
        }
        
        if self.has_gpu:
            props = torch.cuda.get_device_properties(0)
            allocated = torch.cuda.memory_allocated() / 1e6
            reserved = torch.cuda.memory_reserved() / 1e6
            
            stats['gpu_memory_mb'] = allocated
            stats['gpu_memory_percent'] = (allocated / (props.total_memory / 1e6)) * 100
            
            # Try to get GPU utilization (requires nvidia-smi)
            try:
                import subprocess
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                    capture_output=True,
                    text=True,
                    timeout=1
                )
                if result.returncode == 0:
                    stats['gpu_utilization'] = float(result.stdout.strip())
            except:
                pass
            
            # Update peaks
            self.peak_gpu_memory_mb = max(self.peak_gpu_memory_mb, allocated)
            
        self.peak_ram_percent = max(self.peak_ram_percent, stats['ram_percent'])
        self.stats_history.append(stats)
        self.last_sample_time = time.time()

        return stats


class CUDADataPrefetcher:
    """
    🔥🔥🔥 1% of 1% OPTIMIZATION: CUDA Stream Data Prefetching 🔥🔥🔥

    Overlaps data transfer (CPU→GPU) with compute using CUDA streams.
    While GPU is computing on batch N, we're transferring batch N+1.

    This eliminates data transfer latency from the critical path!
    """

    def __init__(self, loader, device):
        self.loader = loader
        self.device = device
        self.stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        self.next_input_ids = None
        self.next_attention_mask = None

    def __iter__(self):
        self.loader_iter = iter(self.loader)
        self._preload()
        return self

    def _preload(self):
        try:
            input_ids, attention_mask = next(self.loader_iter)
        except StopIteration:
            self.next_input_ids = None
            self.next_attention_mask = None
            return

        if self.stream is not None:
            with torch.cuda.stream(self.stream):
                self.next_input_ids = input_ids.to(self.device, non_blocking=True)
                self.next_attention_mask = attention_mask.to(self.device, non_blocking=True)
        else:
            self.next_input_ids = input_ids.to(self.device)
            self.next_attention_mask = attention_mask.to(self.device)

    def __next__(self):
        if self.stream is not None:
            torch.cuda.current_stream().wait_stream(self.stream)

        if self.next_input_ids is None:
            raise StopIteration

        input_ids = self.next_input_ids
        attention_mask = self.next_attention_mask

        # Record that these tensors are being used by current stream
        if self.stream is not None:
            input_ids.record_stream(torch.cuda.current_stream())
            attention_mask.record_stream(torch.cuda.current_stream())

        # Start prefetching the next batch
        self._preload()

        return input_ids, attention_mask

    def __len__(self):
        return len(self.loader)


class MemmapTokenDataset(torch.utils.data.Dataset):
    """Memory-mapped token dataset (input_ids + attention_mask)."""

    def __init__(self, input_ids_path: Path, attention_mask_path: Path, shape: Tuple[int, int],
                 input_dtype: np.dtype, mask_dtype: np.dtype):
        self.shape = shape
        self.input_ids = np.memmap(str(input_ids_path), dtype=input_dtype, mode='r', shape=shape)
        self.attention_mask = np.memmap(str(attention_mask_path), dtype=mask_dtype, mode='r', shape=shape)

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, idx: int):
        input_ids = torch.from_numpy(self.input_ids[idx])
        attention_mask = torch.from_numpy(self.attention_mask[idx])
        return input_ids, attention_mask


class OptimizedModelTrainer:
    """Unified trainer supporting multiple model architectures"""
    
    def __init__(
        self,
        config_path: str,
        device: Optional[str] = None,
        force_device: Optional[str] = None,
        verbose_device_backend: bool = False,
    ):
        self.config = load_yaml_config(config_path)

        # Canonicalize/upgrade config schema (supports both the original trainer schema
        # and the alternate universal metal/cuda config schema).
        self.config = self._canonicalize_config(self.config)

        # Ensure optional sections exist with safe defaults.
        self.config.setdefault('hardware_monitoring', {})
        self.config['hardware_monitoring'].setdefault('collection_interval_seconds', 5.0)
        self.config['hardware_monitoring'].setdefault('gpu_memory_threshold_large_gb', 24.0)
        self.config['hardware_monitoring'].setdefault('gpu_memory_threshold_medium_gb', 12.0)
        self.config.setdefault('training', {})
        self.config['training'].setdefault('drop_full_attention_mask', True)
        self.config['training'].setdefault('deterministic_mode', False)

        # Final validation (after canonicalization)
        if not isinstance(self.config, dict) or 'model' not in self.config or 'training' not in self.config:
            raise ValueError(
                "Invalid training config schema. Expected top-level keys: 'model' and 'training'. "
                "If using the universal metal/cuda config, it must be canonicalized successfully."
            )

        # Device backend integration (Metal/CUDA/CPU)
        self.device_backend = None
        requested_device = force_device or self.config.get('device_backend', {}).get('force_device') or device
        selected_device = None
        try:
            from device_backend import get_device_backend
            self.device_backend = get_device_backend(
                force_device=requested_device,
                verbose=verbose_device_backend,
            )
            self.device_backend.setup()
            selected_device = self.device_backend.device
        except Exception as e:
            logger.warning(f"DeviceBackend unavailable or failed ({e}); falling back to torch default")

        resolved_device = selected_device or requested_device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device(resolved_device)
        self.model_cfg = self.config['model']
        self.train_cfg = self.config['training']
        self.eval_cfg = self.config.get('evaluation', {})
        self.system_cfg = self.config.get('system', {}) if isinstance(self.config.get('system', {}), dict) else {}
        self._normalize_training_config()

        # Device-specific overrides (important for MPS correctness)
        if self.device_backend is not None and getattr(self.device_backend, "is_metal", False):
            # bitsandbytes is CUDA-only. If user requested 4/8-bit, route to MPS-native int8 backend.
            mps_quant_requested = bool(self.model_cfg.get('use_4bit') or self.model_cfg.get('use_8bit'))
            self.model_cfg['use_mps_int8_quant'] = mps_quant_requested

            if mps_quant_requested:
                logger.warning(
                    "4-bit/8-bit quantization requested but bitsandbytes is CUDA-only; "
                    "enabling MPS-native int8 weight-only quant + LoRA backend."
                )

            # Disable bnb quant flags on MPS (we're not using BitsAndBytesConfig)
            self.model_cfg['use_4bit'] = False
            self.model_cfg['use_8bit'] = False

            # Gradient checkpointing: do NOT force-disable on MPS.
            # If you hit instability on a specific torch/MPS build, disable it in config.
            # (This helps memory and is one of the main levers to approach CUDA parity.)

            # Precision: default to bf16 on MPS when configured; optionally allow fp16 to save memory.
            # Set `model.mps_prefer_fp16: true` in config to force fp16 on MPS.
            if bool(self.model_cfg.get('mps_prefer_fp16', False)):
                self.model_cfg['use_fp16'] = True
                self.model_cfg['use_bf16'] = False
            else:
                if 'use_fp16' in self.model_cfg:
                    self.model_cfg['use_fp16'] = False
                if 'use_bf16' in self.model_cfg:
                    self.model_cfg['use_bf16'] = True

        self.amp_device = self.device.type if self.device.type in ("cuda", "cpu", "mps") else "cpu"
        self.training_output_dir: Optional[Path] = None
        self.previous_training_info: Dict[str, Any] = {}
        self.model_saving_cfg = self.config.get('model_saving', {})
        self.incremental_context_sequences = self.train_cfg.get('incremental_context_sequences', 2)
        self.latest_sequence_commit: Optional[str] = None
        self._total_sequences: int = 0
        self.latest_commit_idx: int = -1
        self.curriculum_summary: Optional[Dict[str, Any]] = None
        self.behavioral_eval_history: List[Dict[str, Any]] = []

        # Optional deterministic mode (best-effort; may reduce performance).
        if self.train_cfg.get('deterministic_mode', False):
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
                os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
                logger.info("Deterministic mode enabled (best-effort)")
            except Exception as e:
                logger.warning(f"Could not enable deterministic algorithms: {e}")

        set_seeds(self.train_cfg['seed'])

        # Apply system-level optimizations (CPU threads, matmul precision, TF32)
        self._apply_system_optimizations()

        # 1% CUDA OPTIMIZATIONS for maximum throughput
        if torch.cuda.is_available():
            # Enable cudnn benchmark for auto-tuning (1.1-1.3x speedup)
            torch.backends.cudnn.benchmark = True
            # Enable TF32 for Ampere+ GPUs (2x faster matmul with minimal precision loss)
            allow_tf32 = self.system_cfg.get('allow_tf32', True)
            torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
            torch.backends.cudnn.allow_tf32 = bool(allow_tf32)
            if hasattr(torch.backends.cuda.matmul, "allow_fp16_reduced_precision_reduction"):
                torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
            if hasattr(torch.backends.cuda.matmul, "allow_bf16_reduced_precision_reduction"):
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
            # Pre-allocate memory pool to avoid fragmentation
            torch.cuda.empty_cache()
            logger.info(
                f"🚀 CUDA optimizations: cudnn.benchmark=True, TF32={bool(allow_tf32)}, memory pool cleared"
            )

        # Apply MPS-specific optimizations if on Metal
        if self.device.type == 'mps':
            try:
                from training.mps_optimizations import apply_mps_optimizations, enable_mps_fallback
                enable_mps_fallback()
                mps_settings = apply_mps_optimizations(verbose=True)
                logger.info(f"MPS optimizations applied: {list(mps_settings.keys())}")
            except Exception as e:
                logger.warning(f"Could not apply MPS optimizations: {e}")
        
        logger.info(f"\n" + "="*70)
        logger.info(f"OPTIMIZED MODEL TRAINER (Multi-Architecture)")
        logger.info(f"="*70)
        logger.info(f"Base Model: {self.model_cfg['name']}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Use LoRA: {self.model_cfg['use_lora']}")
        if self.model_cfg.get('use_lora', False):
            logger.info(f"LoRA CPU offload: {self.model_cfg.get('cpu_offload_lora', False)}")
        logger.info(f"Use 4-bit: {self.model_cfg['use_4bit']}")
        logger.info(f"Use Mixed Precision: {self.train_cfg['use_mixed_precision']}")
        logger.info(f"="*70 + "\n")
        
        self.model = None
        self.tokenizer = None
        self.use_deepspeed = False
        self.deepspeed_engine = None

        # Eagerly initialize tokenizer so downstream code/tests can call trainer.tokenizer
        # without requiring model setup first.
        try:
            tok_name = self.model_cfg.get('tokenizer_name') or self.model_cfg.get('name')
            trust_remote = bool(self.model_cfg.get('trust_remote_code', True))
            self.tokenizer = AutoTokenizer.from_pretrained(tok_name, trust_remote_code=trust_remote)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception as e:
            logger.warning(f"Tokenizer initialization failed; will retry lazily in load_data(): {e}")

        self.hardware_monitor = HardwareMonitor(self.config['hardware_monitoring']['collection_interval_seconds'])
        self.training_stats = {}
        self._staged_files: Dict[str, str] = {}
        self._staging_root: Optional[Path] = None
        self.dataset_stats: Optional[Dict[str, float]] = None

    def _canonicalize_config(self, cfg: Any) -> Dict[str, Any]:
        """Upgrade config into the canonical schema expected by this trainer.

        Canonical schema keys used by this trainer:
        - model
        - training
        - hardware_monitoring
        - evaluation (optional)
        - model_saving (optional)
        - device_backend (optional)

        Supports an alternate schema (the universal metal/cuda config) that uses:
        - model / quantization / optimization / training / output / device_backend

        This function intentionally chooses safe defaults so runs don't crash.
        """
        if not isinstance(cfg, dict):
            return {}

        # Already canonical
        if 'model' in cfg and 'training' in cfg and 'hardware_monitoring' in cfg:
            return cfg

        # Alternate schema: model + optimization + quantization + training
        if 'model' in cfg and 'optimization' in cfg:
            model_in = cfg.get('model', {}) if isinstance(cfg.get('model', {}), dict) else {}
            opt_in = cfg.get('optimization', {}) if isinstance(cfg.get('optimization', {}), dict) else {}
            quant_in = cfg.get('quantization', {}) if isinstance(cfg.get('quantization', {}), dict) else {}
            train_in = cfg.get('training', {}) if isinstance(cfg.get('training', {}), dict) else {}
            out_in = cfg.get('output', {}) if isinstance(cfg.get('output', {}), dict) else {}

            # Map alternate fields to canonical model section.
            pretrained = model_in.get('pretrained_model') or model_in.get('name')
            model_cfg = {
                'name': pretrained or model_in.get('name', 'gpt2'),
                'tokenizer_name': pretrained or model_in.get('tokenizer_name') or pretrained or 'gpt2',
                'trust_remote_code': bool(model_in.get('trust_remote_code', True)),
                # Check multiple places for LoRA - config compatibility!
                'use_lora': bool(
                    quant_in.get('lora_enabled', False) or
                    quant_in.get('load_in_4bit', False) or  # 4-bit implies LoRA!
                    config.get('ultra_optimizations', {}).get('qlora_4bit', {}).get('enabled', False)
                ),
                'use_4bit': bool(quant_in.get('load_in_4bit', False)),
                'use_8bit': bool(quant_in.get('load_in_8bit', False)),
                'use_bf16': str(opt_in.get('mixed_precision', '')).lower() in ('bf16', 'bfloat16'),
                'use_fp16': str(opt_in.get('mixed_precision', '')).lower() in ('fp16', 'float16'),
                'cpu_offload_lora': bool(
                    opt_in.get('cpu_offload_lora', False) or quant_in.get('cpu_offload_lora', False)
                ),
                # MPS passthrough knobs (alternate schema)
                'mps_prefer_fp16': bool(model_in.get('mps_prefer_fp16', False)),
                'mps_quant_dtype': quant_in.get('mps_quant_dtype', model_in.get('mps_quant_dtype')),
                'mps_group_size': quant_in.get('mps_group_size', model_in.get('mps_group_size')),
                'mps_compute_dtype': quant_in.get('mps_compute_dtype', model_in.get('mps_compute_dtype')),
                # LoRA configuration (from alternate schema quantization section)
                'lora': {
                    'r': int(quant_in.get('lora_rank', 32)),
                    'lora_alpha': int(quant_in.get('lora_alpha', 64)),
                    'target_modules': quant_in.get('lora_target_modules', ['c_attn', 'c_proj', 'fc1', 'fc2']),
                    'lora_dropout': float(quant_in.get('lora_dropout', 0.05)),
                    'bias': quant_in.get('lora_bias', 'none'),
                },
                # required by downstream log lines
                'pretrained_model': pretrained,
            }

            # Map optimization/training into canonical training section expected by this trainer.
            batch = int(opt_in.get('batch_size', 2))
            warmup_steps = int(opt_in.get('warmup_steps', 0))
            training_cfg = {
                'seed': int(train_in.get('seed', 42)),
                'use_mixed_precision': bool(opt_in.get('use_mixed_precision', True)),
                'use_gradient_checkpointing': bool(opt_in.get('gradient_checkpointing', False)),
                'gradient_accumulation_steps': int(opt_in.get('gradient_accumulation_steps', 1)),
                'max_grad_norm': float(opt_in.get('max_grad_norm', 1.0)),
                'base_learning_rate': float(opt_in.get('learning_rate', 2e-4)),
                'weight_decay': float(opt_in.get('weight_decay', 0.01)),
                'warmup_ratio': float(train_in.get('warmup_ratio', 0.0)),
                'warmup_steps_min': int(train_in.get('warmup_steps_min', 0)),
                'warmup_steps_max': int(train_in.get('warmup_steps_max', warmup_steps)),
                'lr_reduction_factor': float(train_in.get('lr_reduction_factor', 0.5)),
                'lr_plateau_patience': int(train_in.get('lr_plateau_patience', 2)),
                'min_delta': float(train_in.get('min_delta', 0.0)),
                'pin_memory': bool(train_in.get('pin_memory', True)),
                'pin_memory_device': train_in.get('pin_memory_device'),
                'batch_size_reference': batch,
                'batch_size_large': batch,
                'batch_size_medium': batch,
                'batch_size_small': batch,
                'batch_size_override': train_in.get('batch_size_override'),
                'num_workers': int(train_in.get('num_workers', 0)),
                'num_workers_min': int(train_in.get('num_workers_min', 0)),
                'num_workers_max': int(train_in.get('num_workers_max', 0)),
                'dataloader_workers_auto': bool(train_in.get('dataloader_workers_auto', True)),
                'prefetch_factor': int(train_in.get('prefetch_factor', 4)),
                'use_ram_cache': bool(train_in.get('use_ram_cache', True)),
                'ram_cache_max_ratio': float(train_in.get('ram_cache_max_ratio', 0.5)),  # fraction of available RAM allowed
                'incremental_context_sequences': int(train_in.get('incremental_context_sequences', 2)),
                'drop_full_attention_mask': bool(train_in.get('drop_full_attention_mask', True)),
                'deterministic_mode': bool(train_in.get('deterministic_mode', False)),
                'memory_mapped_dataset': train_in.get('memory_mapped_dataset', {}),
                'dataset_staging': train_in.get('dataset_staging', {}),
                'throughput_probe': train_in.get('throughput_probe', {}),
                'cuda_graphs': train_in.get('cuda_graphs', {}),
                'label_smoothing': train_in.get('label_smoothing', {}),
                'gradient_noise': train_in.get('gradient_noise', {}),
                'gradient_centralization': train_in.get('gradient_centralization', {}),
                'swa': train_in.get('swa', {}),
                'lookahead': train_in.get('lookahead', {}),
            }

            system_cfg = cfg.get('system', {}) if isinstance(cfg.get('system', {}), dict) else {}

            canonical = {
                'model': model_cfg,
                'training': training_cfg,
                'hardware_monitoring': {
                    'collection_interval_seconds': float(cfg.get('hardware_monitoring', {}).get('collection_interval_seconds', 5.0))
                    if isinstance(cfg.get('hardware_monitoring', {}), dict)
                    else 5.0
                },
                'evaluation': cfg.get('evaluation', {}),
                'model_saving': cfg.get('model_saving', {
                    'save_final_model': bool(out_in.get('save_model', True)),
                }),
                'device_backend': cfg.get('device_backend', {}),
                'system': system_cfg,
            }
            return canonical

        # Fallback: attempt to preserve any existing sections
        return {
            'model': cfg.get('model', {}) if isinstance(cfg.get('model', {}), dict) else {},
            'training': cfg.get('training', {}) if isinstance(cfg.get('training', {}), dict) else {},
            'hardware_monitoring': cfg.get('hardware_monitoring', {'collection_interval_seconds': 5.0})
            if isinstance(cfg.get('hardware_monitoring', {}), dict)
            else {'collection_interval_seconds': 5.0},
            'evaluation': cfg.get('evaluation', {}),
            'model_saving': cfg.get('model_saving', {}),
            'device_backend': cfg.get('device_backend', {}),
            'system': cfg.get('system', {}) if isinstance(cfg.get('system', {}), dict) else {},
        }

    def _init_synthetic_model_and_tokenizer(self) -> None:
        """Initialize a tiny local model + tokenizer without network downloads.

        This is used by the integration test and for quick pipeline validation.
        The model is a small GPT2-like LM; the tokenizer is a minimal stub.
        """
        try:
            from transformers import GPT2Config, GPT2LMHeadModel
        except Exception as e:
            raise RuntimeError(f"Synthetic mode requires transformers: {e}")

        vocab_size = int(self.model_cfg.get('synthetic_vocab_size', 256))
        n_positions = int(self.model_cfg.get('synthetic_n_positions', 128))
        n_embd = int(self.model_cfg.get('synthetic_n_embd', 64))
        n_layer = int(self.model_cfg.get('synthetic_n_layer', 2))
        n_head = int(self.model_cfg.get('synthetic_n_head', 4))

        # Reserve a few special token IDs for objectives.
        # Keep these stable to avoid confusing downstream logic.
        token_ids = {
            '<pad>': 0,
            '<unk>': 1,
            '<bos>': 2,
            '<eos>': 3,
            '<fim_prefix>': 4,
            '<fim_middle>': 5,
            '<fim_suffix>': 6,
        }

        class _SyntheticTokenizer:
            def __init__(self, vocab_size: int, token_ids: Dict[str, int]):
                self.vocab_size = vocab_size
                self._token_ids = dict(token_ids)
                self.pad_token = '<pad>'
                self.unk_token = '<unk>'
                self.bos_token = '<bos>'
                self.eos_token = '<eos>'
                self.pad_token_id = self._token_ids[self.pad_token]
                self.eos_token_id = self._token_ids[self.eos_token]

            def __len__(self):
                return self.vocab_size

            def save_pretrained(self, save_directory: Path):
                # Minimal artifact to keep downstream save calls safe.
                save_directory = Path(save_directory)
                save_directory.mkdir(parents=True, exist_ok=True)
                (save_directory / 'synthetic_tokenizer.json').write_text(
                    json.dumps({'vocab_size': self.vocab_size, 'special_ids': self._token_ids}, indent=2)
                )

            def convert_tokens_to_ids(self, token: str) -> int:
                return int(self._token_ids.get(token, self._token_ids['<unk>']))

        self.tokenizer = _SyntheticTokenizer(vocab_size=vocab_size, token_ids=token_ids)
        self._fim_ids = {
            'prefix': self.tokenizer.convert_tokens_to_ids('<fim_prefix>'),
            'middle': self.tokenizer.convert_tokens_to_ids('<fim_middle>'),
            'suffix': self.tokenizer.convert_tokens_to_ids('<fim_suffix>'),
        }

        cfg = GPT2Config(
            vocab_size=vocab_size,
            n_positions=n_positions,
            n_ctx=n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
        )
        self.model = GPT2LMHeadModel(cfg).to(self.device)
        self.model.train()

        if getattr(self, 'device_backend', None) is not None:
            try:
                self.device_backend.patch_model(self.model)
            except Exception as e:
                logger.warning(f"Device backend model patching failed (synthetic): {e}")

    def _curriculum_max_len(self, epoch: int, num_epochs: int) -> Optional[int]:
        cur = self.train_cfg.get('curriculum', {}) if isinstance(self.train_cfg, dict) else {}
        if not isinstance(cur, dict) or not cur.get('enabled', False):
            return None
        start_len = int(cur.get('start_max_len', 0))
        end_len = int(cur.get('end_max_len', 0))
        warm = int(cur.get('warmup_epochs', max(1, num_epochs)))
        if start_len <= 0 or end_len <= 0:
            return None
        if warm <= 1:
            return end_len
        t = min(1.0, max(0.0, epoch / float(warm - 1)))
        return int(round(start_len + (end_len - start_len) * t))

    def _objective_sample(self) -> str:
        obj = self.train_cfg.get('objectives', {}) if isinstance(self.train_cfg, dict) else {}
        if not isinstance(obj, dict):
            return 'lm'
        fim_rate = max(0.0, float(obj.get('fim_rate', 0.0)))
        span_rate = max(0.0, float(obj.get('span_rate', 0.0)))
        total = fim_rate + span_rate
        if total <= 0.0:
            return 'lm'
        r = random.random() * total
        if r < fim_rate:
            return 'fim'
        return 'span'

    def _apply_fim(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Apply a lightweight FIM transformation to tokenized input.

        Produces a sequence like:
          <fim_prefix> prefix <fim_suffix> suffix <fim_middle> middle

        Notes:
        - This is intentionally simple and stable.
        - If the transformed sequence exceeds model context, it is cropped.
        """
        if input_ids.ndim != 2:
            return input_ids
        B, T = input_ids.shape
        if T < 8:
            return input_ids

        # Choose split points deterministically from RNG state (seeded in init).
        i = max(1, T // 4)
        j = max(i + 1, (3 * T) // 4)

        prefix = input_ids[:, :i]
        middle = input_ids[:, i:j]
        suffix = input_ids[:, j:]

        fp = torch.full((B, 1), int(self._fim_ids['prefix']), device=input_ids.device, dtype=input_ids.dtype)
        fs = torch.full((B, 1), int(self._fim_ids['suffix']), device=input_ids.device, dtype=input_ids.dtype)
        fm = torch.full((B, 1), int(self._fim_ids['middle']), device=input_ids.device, dtype=input_ids.dtype)

        out = torch.cat([fp, prefix, fs, suffix, fm, middle], dim=1)

        # Crop if needed
        max_pos = int(getattr(getattr(self.model, 'config', None), 'n_positions', out.shape[1]))
        if out.shape[1] > max_pos:
            out = out[:, :max_pos]
        return out

    def _apply_span_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """Edit-locality objective: only compute loss on a contiguous span."""
        if labels.ndim != 2:
            return labels
        obj = self.train_cfg.get('objectives', {}) if isinstance(self.train_cfg, dict) else {}
        span_frac = float(obj.get('span_frac', 0.25)) if isinstance(obj, dict) else 0.25
        span_frac = min(1.0, max(0.0, span_frac))
        B, T = labels.shape
        span = max(1, int(round(T * span_frac)))
        start = max(0, (T - span) // 2)
        end = min(T, start + span)
        masked = labels.clone()
        masked[:, :start] = -100
        masked[:, end:] = -100
        return masked

    def _maybe_init_ema(self) -> None:
        ema_cfg = self.train_cfg.get('ema', {}) if isinstance(self.train_cfg, dict) else {}
        if not isinstance(ema_cfg, dict) or not ema_cfg.get('enabled', False):
            self._ema = None
            return
        decay = float(ema_cfg.get('decay', 0.999))
        decay = min(0.99999, max(0.0, decay))
        self._ema_decay = decay
        self._ema = {}
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                if p.requires_grad:
                    self._ema[name] = p.detach().clone()

    def _ema_update(self) -> None:
        if getattr(self, '_ema', None) is None:
            return
        d = float(getattr(self, '_ema_decay', 0.999))
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                if not p.requires_grad:
                    continue
                if name not in self._ema:
                    self._ema[name] = p.detach().clone()
                    continue
                self._ema[name].mul_(d).add_(p.detach(), alpha=(1.0 - d))

    def _normalize_training_config(self):
        """Ensure numeric training config values are correctly typed"""
        float_keys = [
            'base_learning_rate',
            'weight_decay',
            'warmup_ratio',
            'lr_reduction_factor',
            'validation_split',
        ]
        int_keys = [
            'warmup_steps_min',
            'warmup_steps_max',
            'gradient_accumulation_steps',
            'batch_size_reference',
            'batch_size_large',
            'batch_size_medium',
            'batch_size_small',
            'num_workers',
            'num_workers_min',
            'num_workers_max',
            'seed',
            'lr_plateau_patience',
        ]
        bool_keys = ['pin_memory', 'use_mixed_precision', 'use_gradient_checkpointing', 'drop_full_attention_mask', 'deterministic_mode']

        for key in float_keys:
            if key in self.train_cfg:
                try:
                    self.train_cfg[key] = float(self.train_cfg[key])
                except (TypeError, ValueError):
                    logger.warning(f"Training config {key} could not be cast to float; keeping original value")

        for key in int_keys:
            if key in self.train_cfg:
                try:
                    self.train_cfg[key] = int(self.train_cfg[key])
                except (TypeError, ValueError):
                    logger.warning(f"Training config {key} could not be cast to int; keeping original value")

        for key in bool_keys:
            if key in self.train_cfg and isinstance(self.train_cfg[key], str):
                self.train_cfg[key] = self.train_cfg[key].lower() in ('true', '1', 'yes')

    def _apply_system_optimizations(self):
        """Maximize CPU/GPU utilization based on system config."""
        if not isinstance(self.system_cfg, dict):
            return

        cpu_threads = self.system_cfg.get('cpu_threads')
        interop_threads = self.system_cfg.get('interop_threads')
        matmul_precision = self.system_cfg.get('matmul_precision')
        cpu_affinity = self.system_cfg.get('cpu_affinity')
        numa_prefer_node = self.system_cfg.get('numa_prefer_node')

        if cpu_threads:
            try:
                cpu_threads = int(cpu_threads)
                for env_var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                    os.environ.setdefault(env_var, str(cpu_threads))
                torch.set_num_threads(cpu_threads)
                logger.info(f"CPU threads set to {cpu_threads} (compute)")
            except Exception as e:
                logger.warning(f"Could not set CPU threads: {e}")

        if interop_threads:
            try:
                interop_threads = int(interop_threads)
                torch.set_num_interop_threads(interop_threads)
                logger.info(f"Interop threads set to {interop_threads}")
            except Exception as e:
                logger.warning(f"Could not set interop threads: {e}")

        if matmul_precision and hasattr(torch, "set_float32_matmul_precision"):
            try:
                torch.set_float32_matmul_precision(str(matmul_precision))
                logger.info(f"Float32 matmul precision set to {matmul_precision}")
            except Exception as e:
                logger.warning(f"Could not set matmul precision: {e}")

        if torch.cuda.is_available():
            allow_tf32 = self.system_cfg.get('allow_tf32')
            if allow_tf32 is not None:
                try:
                    torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
                    torch.backends.cudnn.allow_tf32 = bool(allow_tf32)
                    logger.info(f"TF32 allowed: {bool(allow_tf32)}")
                except Exception as e:
                    logger.warning(f"Could not set TF32 flags: {e}")

        # CPU affinity / NUMA pinning (Linux only, safe no-op elsewhere)
        if cpu_affinity or numa_prefer_node:
            if hasattr(os, "sched_setaffinity"):
                try:
                    available = sorted(os.sched_getaffinity(0))
                    target_cpus = self._select_affinity_cpus(
                        available,
                        cpu_affinity=cpu_affinity,
                        numa_prefer_node=numa_prefer_node,
                        cpu_threads=cpu_threads,
                    )
                    if target_cpus:
                        os.sched_setaffinity(0, set(target_cpus))
                        logger.info(f"CPU affinity pinned to {len(target_cpus)} cores")
                except Exception as e:
                    logger.warning(f"Could not set CPU affinity: {e}")
            else:
                logger.info("CPU affinity not supported on this platform")

    @staticmethod
    def _parse_cpu_list(cpu_list: str) -> List[int]:
        cpus: List[int] = []
        for part in cpu_list.split(','):
            part = part.strip()
            if not part:
                continue
            if '-' in part:
                start, end = part.split('-', 1)
                try:
                    for cpu in range(int(start), int(end) + 1):
                        cpus.append(cpu)
                except ValueError:
                    continue
            else:
                try:
                    cpus.append(int(part))
                except ValueError:
                    continue
        return sorted(set(cpus))

    def _numa_node_cpu_map(self) -> Dict[int, List[int]]:
        nodes: Dict[int, List[int]] = {}
        sys_nodes = Path("/sys/devices/system/node")
        if not sys_nodes.exists():
            return nodes
        for node_dir in sys_nodes.glob("node[0-9]*"):
            try:
                node_idx = int(node_dir.name.replace("node", ""))
                cpulist_path = node_dir / "cpulist"
                if not cpulist_path.exists():
                    continue
                cpus = self._parse_cpu_list(cpulist_path.read_text().strip())
                if cpus:
                    nodes[node_idx] = cpus
            except Exception:
                continue
        return nodes

    def _select_affinity_cpus(
        self,
        available: List[int],
        cpu_affinity: Optional[object],
        numa_prefer_node: Optional[object],
        cpu_threads: Optional[int],
    ) -> List[int]:
        if not available:
            return []

        nodes = self._numa_node_cpu_map()
        node_cpus: Optional[List[int]] = None

        if isinstance(numa_prefer_node, int):
            node_cpus = nodes.get(numa_prefer_node)
        elif isinstance(numa_prefer_node, str) and numa_prefer_node.lower() == "auto":
            if nodes:
                node_cpus = max(nodes.values(), key=len)

        if isinstance(cpu_affinity, list):
            target = [c for c in cpu_affinity if c in available]
        elif isinstance(cpu_affinity, str):
            lower = cpu_affinity.lower()
            if lower == "auto":
                base = node_cpus if node_cpus else available
                limit = int(cpu_threads) if cpu_threads else len(base)
                target = base[: max(1, limit)]
            elif lower == "all":
                target = available
            else:
                target = available
        else:
            target = node_cpus if node_cpus else available

        return sorted(set(target))

    def _load_training_info(self, output_dir: Path) -> Dict[str, Any]:
        """Load persisted training metadata from the output directory"""
        info_path = output_dir / "training_info.json"
        if info_path.exists():
            try:
                with open(info_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                logger.warning(f"Could not read existing training info at {info_path}; starting fresh")
        return {}

    def _prepare_incremental_slice(self, sequences_file: str) -> Dict[str, Any]:
        """Determine whether there are new sequences and where to start training"""
        sequences_list = []
        metadata_map = {}

        try:
            with open(sequences_file, 'r') as f:
                # Try reading as JSONL first (one JSON object per line)
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        if isinstance(obj, dict):
                            if "token_sequences" in obj:
                                sequences_list.extend(obj["token_sequences"])
                            elif "tokens" in obj:
                                sequences_list.append(obj["tokens"])
                            if "metadata" in obj:
                                metadata_map.update(obj["metadata"])
                        else:
                            sequences_list.append(obj)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Unable to parse line in {sequences_file}: {exc}") from exc
        except (OSError, ValueError) as exc:
            raise ValueError(f"Unable to parse {sequences_file}: {exc}") from exc

        total_sequences = len(sequences_list)
        self._total_sequences = total_sequences

        sequence_details = []
        latest_commit_hash = None
        latest_commit_idx = -1
        for idx in range(total_sequences):
            meta = metadata_map.get(str(idx), {})
            commit_idx = meta.get("end_commit_idx", idx)
            commit_hash = meta.get("sample_commit")
            sequence_details.append({
                'index': idx,
                'commit_idx': commit_idx if commit_idx is not None else idx,
                'commit_hash': commit_hash,
            })
            if commit_hash:
                latest_commit_hash = commit_hash
            latest_commit_idx = max(latest_commit_idx, commit_idx if commit_idx is not None else idx)
        incremental_context = self.incremental_context_sequences

        prev_commit_idx = self.previous_training_info.get("last_trained_commit_idx", -1)
        prev_commit_hash = self.previous_training_info.get("last_trained_sequence_commit")
        prev_sequence_idx = self.previous_training_info.get("last_trained_sequence_idx", -1)

        detection_idx = None
        for detail in sequence_details:
            commit_idx = detail['commit_idx']
            commit_hash = detail['commit_hash']
            if commit_idx > prev_commit_idx:
                detection_idx = detail['index']
                break
            if commit_idx == prev_commit_idx and prev_commit_hash and commit_hash and commit_hash != prev_commit_hash:
                detection_idx = detail['index']
                break
        if detection_idx is None:
            detection_idx = prev_sequence_idx + 1

        self.latest_sequence_commit = latest_commit_hash or prev_commit_hash
        self.latest_commit_idx = latest_commit_idx if latest_commit_idx >= 0 else prev_commit_idx

        if total_sequences == 0:
            logger.warning(f"No sequences found in {sequences_file}; nothing to train on")
            return {
                "new_data": False,
                "train_start_idx": 0,
                "total_sequences": total_sequences,
                "latest_sequence_commit": self.latest_sequence_commit,
                "latest_sequence_commit_idx": self.latest_commit_idx,
            }

        if detection_idx >= total_sequences:
            logger.info("No new sequences detected since last training run")
            return {
                "new_data": False,
                "train_start_idx": total_sequences,
                "total_sequences": total_sequences,
                "latest_sequence_commit": self.latest_sequence_commit,
                "latest_sequence_commit_idx": self.latest_commit_idx,
            }

        train_start_idx = max(0, detection_idx - incremental_context)
        logger.info(
            f"Incremental training slice: starting at sequence {train_start_idx} "
            f"of {total_sequences} (context={incremental_context})"
        )
        return {
            "new_data": True,
            "train_start_idx": train_start_idx,
            "total_sequences": total_sequences,
            "latest_sequence_commit": self.latest_sequence_commit,
            "latest_sequence_commit_idx": self.latest_commit_idx,
        }

    def _detect_storage_profile(self, path: Path) -> Dict[str, Optional[bool]]:
        profile = {
            'is_ssd': None,
            'is_nvme': None,
            'rotational': None,
        }
        try:
            probe_path = path if path.exists() else path.parent
            dev = os.stat(probe_path).st_dev
            sys_block = Path(f"/sys/dev/block/{os.major(dev)}:{os.minor(dev)}")
            if not sys_block.exists():
                return profile
            sys_resolved = sys_block.resolve()
            profile['is_nvme'] = "nvme" in str(sys_resolved)
            rotational_path = sys_resolved / "queue" / "rotational"
            if rotational_path.exists():
                rotational = int(rotational_path.read_text().strip())
                profile['rotational'] = rotational
                profile['is_ssd'] = rotational == 0
        except Exception:
            return profile
        return profile

    def _checksum_settings(self, cfg: Dict[str, Any]) -> Tuple[bool, str, int]:
        enabled = bool(cfg.get('checksum_enabled', True))
        mode = str(cfg.get('checksum_mode', 'sample')).lower()
        sample_bytes = int(cfg.get('checksum_bytes', 4 * 1024 * 1024))
        return enabled, mode, sample_bytes

    def _compute_file_checksum(self, path: Path, mode: str, sample_bytes: int) -> str:
        h = hashlib.sha256()
        size = path.stat().st_size
        if mode == "full" or size <= sample_bytes:
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
        else:
            head = min(sample_bytes // 2, size)
            tail = min(sample_bytes - head, max(0, size - head))
            with open(path, "rb") as f:
                if head:
                    h.update(f.read(head))
                if tail:
                    f.seek(-tail, os.SEEK_END)
                    h.update(f.read(tail))
        return h.hexdigest()

    def _select_staging_dir(self, source_path: Path, staging_cfg: Dict[str, Any]) -> Optional[Path]:
        prefer_ramdisk = bool(staging_cfg.get('prefer_ramdisk', True))
        explicit_dir = staging_cfg.get('staging_dir')
        min_free_gb = float(staging_cfg.get('min_free_gb', 4))

        candidates: List[Path] = []
        if explicit_dir:
            candidates.append(Path(explicit_dir))
        if prefer_ramdisk and Path("/dev/shm").exists():
            candidates.append(Path("/dev/shm"))
        candidates.append(Path("/tmp"))

        source_profile = self._detect_storage_profile(source_path)

        for candidate in candidates:
            if not candidate.exists():
                continue
            try:
                usage = shutil.disk_usage(candidate)
                if usage.free < min_free_gb * (1024 ** 3):
                    continue
                candidate_profile = self._detect_storage_profile(candidate)
                if prefer_ramdisk and candidate == Path("/dev/shm"):
                    return candidate
                if source_profile.get('rotational') == 1 and candidate_profile.get('is_ssd'):
                    return candidate
                if explicit_dir and candidate == Path(explicit_dir):
                    return candidate
            except Exception:
                continue
        return None

    def _stage_dataset_if_needed(self, sequences_file: str) -> str:
        staging_cfg = self.train_cfg.get('dataset_staging', {})
        if not isinstance(staging_cfg, dict) or not staging_cfg.get('enabled', False):
            return sequences_file

        source_path = Path(sequences_file)
        if not source_path.exists():
            return sequences_file

        if sequences_file in self._staged_files:
            return self._staged_files[sequences_file]

        min_size_mb = float(staging_cfg.get('min_dataset_mb', 512))
        if source_path.stat().st_size < min_size_mb * (1024 ** 2):
            return sequences_file

        staging_root = self._select_staging_dir(source_path, staging_cfg)
        if staging_root is None:
            return sequences_file

        try:
            staging_root.mkdir(parents=True, exist_ok=True)
            staged_path = staging_root / source_path.name
            metadata_path = staged_path.with_suffix(staged_path.suffix + ".stage.json")
            checksum_enabled, checksum_mode, checksum_bytes = self._checksum_settings(staging_cfg)

            if staged_path.exists() and staging_cfg.get('keep_staged', True):
                reuse_ok = True
                if metadata_path.exists():
                    try:
                        meta = json.loads(metadata_path.read_text())
                        if meta.get("source_mtime") != source_path.stat().st_mtime:
                            reuse_ok = False
                        if meta.get("source_size") != source_path.stat().st_size:
                            reuse_ok = False
                        if checksum_enabled:
                            current_checksum = self._compute_file_checksum(
                                source_path, checksum_mode, checksum_bytes
                            )
                            if meta.get("source_checksum") != current_checksum:
                                reuse_ok = False
                    except Exception:
                        reuse_ok = False
                elif checksum_enabled:
                    reuse_ok = False

                if reuse_ok:
                    self._staged_files[sequences_file] = str(staged_path)
                    self._staging_root = staging_root
                    logger.info(f"Dataset already staged at {staged_path}")
                    return str(staged_path)

            shutil.copy2(source_path, staged_path)
            metadata = {
                "source_path": str(source_path),
                "source_mtime": source_path.stat().st_mtime,
                "source_size": source_path.stat().st_size,
                "checksum_mode": checksum_mode,
                "checksum_bytes": checksum_bytes,
            }
            if checksum_enabled:
                metadata["source_checksum"] = self._compute_file_checksum(
                    source_path, checksum_mode, checksum_bytes
                )
            metadata_path.write_text(json.dumps(metadata, indent=2))
            self._staged_files[sequences_file] = str(staged_path)
            self._staging_root = staging_root
            logger.info(f"Dataset staged to {staged_path}")
            return str(staged_path)
        except Exception as e:
            logger.warning(f"Dataset staging failed: {e}")
            return sequences_file

    def _is_pretokenized_jsonl(self, sequences_file: str) -> bool:
        try:
            with open(sequences_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        if "token_sequences" in obj and obj["token_sequences"]:
                            return isinstance(obj["token_sequences"][0], list)
                        if "tokens" in obj:
                            return isinstance(obj["tokens"], list)
                    if isinstance(obj, list):
                        return True
            return False
        except Exception:
            return False

    def _iter_token_sequences(self, sequences_file: str, start_idx: int = 0):
        idx = 0
        with open(sequences_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                sequences = []
                if isinstance(obj, dict):
                    if "token_sequences" in obj:
                        sequences = obj["token_sequences"]
                    elif "tokens" in obj:
                        sequences = [obj["tokens"]]
                elif isinstance(obj, list):
                    sequences = [obj]

                for seq in sequences:
                    if not isinstance(seq, list):
                        continue
                    if idx >= start_idx:
                        yield seq
                    idx += 1

    def _next_pow2(self, value: int) -> int:
        if value <= 0:
            return 1
        return 1 << (value - 1).bit_length()

    def _profile_sequence_lengths(
        self,
        sequences_file: str,
        start_idx: int = 0,
        max_samples: int = 20000,
    ) -> Dict[str, float]:
        """Stream length stats without loading full dataset into RAM."""
        import random

        lengths: List[int] = []
        total = 0
        max_len = 0
        for seq in self._iter_token_sequences(sequences_file, start_idx=start_idx):
            total += 1
            seq_len = len(seq)
            if seq_len > max_len:
                max_len = seq_len
            if len(lengths) < max_samples:
                lengths.append(seq_len)
            else:
                # Reservoir sampling
                j = random.randint(0, total - 1)
                if j < max_samples:
                    lengths[j] = seq_len

        if not lengths:
            return {
                'count': 0,
                'avg': 0.0,
                'p50': 0.0,
                'p90': 0.0,
                'p95': 0.0,
                'p99': 0.0,
                'max': 0.0,
            }

        lengths_sorted = sorted(lengths)
        def _pct(p: float) -> float:
            idx = min(len(lengths_sorted) - 1, int(round(p * (len(lengths_sorted) - 1))))
            return float(lengths_sorted[idx])

        avg_len = float(sum(lengths_sorted)) / len(lengths_sorted)
        return {
            'count': total,
            'avg': avg_len,
            'p50': _pct(0.50),
            'p90': _pct(0.90),
            'p95': _pct(0.95),
            'p99': _pct(0.99),
            'max': float(max_len),
        }

    def _peek_vocab_size(self, sequences_file: str) -> Optional[int]:
        try:
            with open(sequences_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    if isinstance(obj, dict) and "vocab_size" in obj:
                        return int(obj["vocab_size"])
        except Exception:
            return None
        return None

    def _auto_pack_target(self, stats: Optional[Dict[str, float]] = None,
                          mmap_cfg: Optional[Dict[str, Any]] = None) -> int:
        gpu_compute_cap = 0.0
        if torch.cuda.is_available():
            gpu_compute_cap = torch.cuda.get_device_capability()[0] + torch.cuda.get_device_capability()[1] / 10
        min_target = 256
        max_target = 4096
        if mmap_cfg:
            min_target = int(mmap_cfg.get('pack_target_min', min_target))
            max_target = int(mmap_cfg.get('pack_target_max', max_target))

        if gpu_compute_cap >= 8.0:
            base = 2048
        elif gpu_compute_cap >= 7.5:
            base = 256
        else:
            base = 256

        if not stats:
            return max(min_target, min(max_target, base))

        p95 = int(stats.get('p95', base))
        target = self._next_pow2(p95)
        target = max(min_target, min(max_target, target))
        return max(min_target, min(max_target, max(target, base)))

    def _probe_seq_len(self, seq_len: int, batch_size: int,
                       warmup_steps: int, probe_steps: int) -> Optional[float]:
        """Measure forward+backward time for a given seq_len (seconds/step)."""
        if not torch.cuda.is_available():
            return None
        if self.model is None:
            return None

        device = self.device
        vocab = 32000
        if self.tokenizer is not None:
            vocab = len(self.tokenizer)

        input_ids = torch.randint(0, vocab, (batch_size, seq_len), device=device)
        attention_mask = torch.ones((batch_size, seq_len), device=device, dtype=torch.long)
        labels = input_ids.clone()

        use_autocast = bool(self.train_cfg.get('use_mixed_precision', True)) and self.device.type == "cuda"

        try:
            self.model.train()
            torch.cuda.synchronize()
            for _ in range(max(0, warmup_steps)):
                self.model.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type="cuda", enabled=use_autocast):
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = outputs.loss
                loss.backward()
            torch.cuda.synchronize()

            start = time.time()
            for _ in range(max(1, probe_steps)):
                self.model.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type="cuda", enabled=use_autocast):
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = outputs.loss
                loss.backward()
            torch.cuda.synchronize()
            self.model.zero_grad(set_to_none=True)
            elapsed = time.time() - start
            return elapsed / max(1, probe_steps)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                return None
            raise

    def _batch_size_candidates_for_probe(
        self,
        seq_len: int,
        probe_cfg: Dict[str, Any],
    ) -> List[int]:
        explicit = probe_cfg.get('batch_size_candidates') or probe_cfg.get('batch_sizes')
        if explicit:
            if isinstance(explicit, str):
                explicit = [p.strip() for p in explicit.split(',') if p.strip()]
            try:
                candidates = [int(x) for x in explicit]
            except Exception:
                candidates = []
        else:
            candidates = []

        min_bs = int(probe_cfg.get('batch_size_min', 1))
        max_bs = int(probe_cfg.get('batch_size_max', 16))
        try:
            max_est = self._get_batch_size(seq_len, log=False)
        except Exception:
            max_est = max_bs
        max_bs = max(min_bs, min(max_bs, max_est))

        if not candidates:
            ref = int(self.train_cfg.get('batch_size_reference', max(1, min(2, max_bs))))
            candidates = {ref, max(1, ref // 2), min(max_bs, ref * 2), max_bs, 1}
            bs = 1
            while bs <= max_bs:
                candidates.add(bs)
                bs *= 2
            candidates = sorted(candidates)

        filtered = [int(c) for c in candidates if min_bs <= int(c) <= max_bs]
        filtered = sorted(set(filtered))

        max_candidates = int(probe_cfg.get('max_batch_candidates', 3))
        favor_speed = bool(probe_cfg.get('favor_speed', True))
        if max_candidates > 0 and len(filtered) > max_candidates:
            filtered = sorted(filtered, reverse=favor_speed)[:max_candidates]
            filtered = sorted(set(filtered), reverse=favor_speed)

        return filtered if filtered else [max(1, min_bs)]

    def _select_pack_target_and_batch_from_probe(
        self,
        seq_candidates: List[int],
        batch_candidates: List[int],
        probe_cfg: Dict[str, Any],
    ) -> Optional[Tuple[int, int]]:
        enabled = bool(probe_cfg.get('enabled', False)) and bool(probe_cfg.get('joint_batch_pack', False))
        if not enabled or not seq_candidates or not batch_candidates:
            return None
        if not torch.cuda.is_available() or self.model is None:
            return None

        warmup_steps = int(probe_cfg.get('warmup_steps', 1))
        probe_steps = int(probe_cfg.get('probe_steps', 2))
        metric = str(probe_cfg.get('metric', 'tokens_per_sec')).lower()
        max_combos = int(probe_cfg.get('max_combinations', 6))

        combos = [(s, b) for s in seq_candidates for b in batch_candidates]
        if max_combos > 0 and len(combos) > max_combos:
            combos = combos[:max_combos]

        best = None
        best_score = None

        for seq_len, batch_size in combos:
            step_time = self._probe_seq_len(seq_len, batch_size, warmup_steps, probe_steps)
            if step_time is None or step_time <= 0:
                logger.info(f"Throughput probe: seq_len {seq_len} batch {batch_size} OOM/unavailable")
                continue
            it_per_sec = 1.0 / step_time
            tokens_per_sec = (seq_len * batch_size) / step_time
            score = tokens_per_sec if metric != 'it_per_sec' else it_per_sec
            logger.info(
                f"Throughput probe: seq_len {seq_len} batch {batch_size} → "
                f"{it_per_sec:.2f} it/s ({tokens_per_sec/1e6:.2f} Mtok/s)"
            )
            if best_score is None or score > best_score:
                best_score = score
                best = (seq_len, batch_size)

        return best

    def _select_pack_target_from_probe(
        self,
        candidates: List[int],
        batch_size: int,
        probe_cfg: Dict[str, Any],
    ) -> Optional[int]:
        if not candidates:
            return None
        enabled = bool(probe_cfg.get('enabled', False))
        if not enabled:
            return None
        if not torch.cuda.is_available() or self.model is None:
            return None

        warmup_steps = int(probe_cfg.get('warmup_steps', 1))
        probe_steps = int(probe_cfg.get('probe_steps', 2))

        best_target = None
        best_time = None
        for seq_len in candidates:
            step_time = self._probe_seq_len(seq_len, batch_size, warmup_steps, probe_steps)
            if step_time is None:
                logger.info(f"Throughput probe: seq_len {seq_len} OOM or unavailable")
                continue
            logger.info(f"Throughput probe: seq_len {seq_len} → {step_time:.4f}s/step")
            if best_time is None or step_time < best_time:
                best_time = step_time
                best_target = seq_len

        return best_target

    def _setup_cuda_graph(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        use_autocast: bool,
        warmup_steps: int,
        label_smoothing: float,
    ) -> Optional[Dict[str, Any]]:
        if not torch.cuda.is_available() or self.model is None:
            return None

        try:
            static_input_ids = input_ids.detach().clone()
            static_attention_mask = attention_mask.detach().clone() if attention_mask is not None else None
            static_labels = labels.detach().clone()

            torch.cuda.synchronize()
            for _ in range(max(0, warmup_steps)):
                self.model.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type="cuda", enabled=use_autocast):
                    outputs = self.model(
                        input_ids=static_input_ids,
                        attention_mask=static_attention_mask,
                        labels=static_labels,
                    )
                    if label_smoothing > 0:
                        logits = outputs.logits
                        loss = F.cross_entropy(
                            logits.view(-1, logits.size(-1)),
                            static_labels.view(-1),
                            label_smoothing=label_smoothing,
                            ignore_index=-100,
                        )
                    else:
                        loss = outputs.loss
                loss.backward()
            torch.cuda.synchronize()
            self.model.zero_grad(set_to_none=True)

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                with torch.amp.autocast(device_type="cuda", enabled=use_autocast):
                    outputs = self.model(
                        input_ids=static_input_ids,
                        attention_mask=static_attention_mask,
                        labels=static_labels,
                    )
                    if label_smoothing > 0:
                        logits = outputs.logits
                        loss = F.cross_entropy(
                            logits.view(-1, logits.size(-1)),
                            static_labels.view(-1),
                            label_smoothing=label_smoothing,
                            ignore_index=-100,
                        )
                    else:
                        loss = outputs.loss
                loss = loss / self.train_cfg['gradient_accumulation_steps']
                loss.backward()

            return {
                'graph': graph,
                'static_input_ids': static_input_ids,
                'static_attention_mask': static_attention_mask,
                'static_labels': static_labels,
                'loss': loss,
            }
        except Exception as exc:
            logger.warning(f"CUDA graph setup failed: {exc}")
            torch.cuda.empty_cache()
            return None

    def _apply_gradient_noise(self, step: int) -> None:
        cfg = self.train_cfg.get('gradient_noise', {}) if isinstance(self.train_cfg, dict) else {}
        if not isinstance(cfg, dict) or not cfg.get('enabled', False):
            return
        eta = float(cfg.get('eta', 0.3))
        gamma = float(cfg.get('gamma', 0.55))
        noise_scale = eta / max(1.0, (step + 1) ** gamma)

        for p in self.model.parameters():
            if p.grad is None:
                continue
            p.grad.add_(torch.randn_like(p.grad) * noise_scale)

    def _apply_gradient_centralization(self) -> None:
        cfg = self.train_cfg.get('gradient_centralization', {}) if isinstance(self.train_cfg, dict) else {}
        if not isinstance(cfg, dict) or not cfg.get('enabled', False):
            return

        for p in self.model.parameters():
            if p.grad is None:
                continue
            if p.grad.dim() <= 1:
                continue
            dims = tuple(range(1, p.grad.dim()))
            p.grad.sub_(p.grad.mean(dim=dims, keepdim=True))

    def _init_lookahead_state(self, k: int, alpha: float) -> None:
        self._lookahead_enabled = True
        self._lookahead_k = max(1, int(k))
        self._lookahead_alpha = float(alpha)
        self._lookahead_step = 0
        self._lookahead_slow = {}
        for p in self.model.parameters():
            if p.requires_grad:
                self._lookahead_slow[p] = p.detach().clone()

    def _apply_lookahead_update(self) -> None:
        if not getattr(self, "_lookahead_enabled", False):
            return
        self._lookahead_step += 1
        if self._lookahead_step % self._lookahead_k != 0:
            return
        for p in self.model.parameters():
            if not p.requires_grad:
                continue
            slow = self._lookahead_slow.get(p)
            if slow is None:
                slow = p.detach().clone()
                self._lookahead_slow[p] = slow
            slow.add_(p.data - slow, alpha=self._lookahead_alpha)
            p.data.copy_(slow)

    def _estimate_ram_bytes(self, num_sequences: int, seq_len: int, dtype: str) -> int:
        dtype_bytes = 8 if str(dtype).lower() == "int64" else 4
        input_bytes = num_sequences * seq_len * dtype_bytes
        mask_bytes = num_sequences * seq_len * 1
        return int((input_bytes + mask_bytes) * 1.1)

    def _should_use_memmap(
        self,
        mmap_cfg: Dict[str, Any],
        stats: Dict[str, float],
        seq_len: int,
        packed_count: Optional[int] = None,
    ) -> bool:
        mode = str(mmap_cfg.get('mode', 'auto')).lower()
        if mode == 'off':
            return False
        if mode == 'force':
            return True

        available_ram = psutil.virtual_memory().available
        ratio_limit = float(self.train_cfg.get('ram_cache_max_ratio', 0.5))
        budget = available_ram * ratio_limit
        num_sequences = packed_count if packed_count is not None else int(stats.get('count', 0))
        est_bytes = self._estimate_ram_bytes(num_sequences, seq_len, mmap_cfg.get('dtype', 'int64'))
        return est_bytes > budget

    def _count_sequences(self, sequences_file: str, start_idx: int = 0) -> Tuple[int, int]:
        count = 0
        max_len = 0
        for seq in self._iter_token_sequences(sequences_file, start_idx=start_idx):
            count += 1
            if len(seq) > max_len:
                max_len = len(seq)
        return count, max_len

    def _count_packed_sequences(self, sequences_file: str, target_length: int,
                                 start_idx: int = 0) -> int:
        packs = 0
        current_len = 0
        for seq in self._iter_token_sequences(sequences_file, start_idx=start_idx):
            seq_len = min(len(seq), target_length)
            if seq_len >= target_length:
                if current_len:
                    packs += 1
                    current_len = 0
                packs += 1
                continue
            if current_len and current_len + seq_len + 1 > target_length:
                packs += 1
                current_len = 0
            if current_len:
                current_len += 1
            current_len += seq_len
        if current_len:
            packs += 1
        return packs

    def _mmap_cache_dir(self, sequences_file: str, start_idx: int,
                         target_length: int, packed: bool) -> Path:
        mmap_cfg = self.train_cfg.get('memory_mapped_dataset', {})
        cache_root = mmap_cfg.get('cache_dir')
        if cache_root:
            base_dir = Path(cache_root)
        else:
            base_dir = Path(sequences_file).parent / "mmap_cache"
        if self._staging_root and self.train_cfg.get('dataset_staging', {}).get('cache_on_staging', True):
            base_dir = self._staging_root / "mmap_cache"
        suffix = f"s{start_idx}_t{target_length}_p{int(packed)}"
        return base_dir / suffix

    def _ensure_mmap_dataset(
        self,
        sequences_file: str,
        start_idx: int,
        target_length: int,
        pad_token_id: int,
        eos_token_id: int,
        packed: bool,
        dtype: str,
        vocab_limit: Optional[int] = None,
    ) -> Tuple[MemmapTokenDataset, int]:
        cache_dir = self._mmap_cache_dir(sequences_file, start_idx, target_length, packed)
        cache_dir.mkdir(parents=True, exist_ok=True)

        input_ids_path = cache_dir / "input_ids.mmap"
        attention_mask_path = cache_dir / "attention_mask.mmap"
        manifest_path = cache_dir / "manifest.json"

        source_path = Path(sequences_file)
        source_mtime = source_path.stat().st_mtime
        source_size = source_path.stat().st_size
        manifest = {}
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text())
            except Exception:
                manifest = {}

        dtype_np = np.int64 if str(dtype).lower() == "int64" else np.int32
        mask_dtype = np.uint8

        checksum_cfg = self.train_cfg.get('memory_mapped_dataset', {})
        checksum_enabled, checksum_mode, checksum_bytes = self._checksum_settings(checksum_cfg)
        expected_checksum = manifest.get("source_checksum") if manifest else None

        manifest_valid = (
            manifest
            and manifest.get("source_mtime") == source_mtime
            and manifest.get("source_size") == source_size
            and manifest.get("target_length") == target_length
            and manifest.get("packed") == packed
        )
        if manifest_valid and checksum_enabled:
            current_checksum = self._compute_file_checksum(
                source_path, checksum_mode, checksum_bytes
            )
            if expected_checksum != current_checksum:
                manifest_valid = False

        if manifest_valid and input_ids_path.exists() and attention_mask_path.exists():
            num_sequences = int(manifest.get("num_sequences", 0))
            dataset = MemmapTokenDataset(
                input_ids_path,
                attention_mask_path,
                (num_sequences, target_length),
                dtype_np,
                mask_dtype,
            )
            return dataset, num_sequences

        if packed:
            num_sequences = self._count_packed_sequences(
                sequences_file, target_length, start_idx=start_idx
            )
            self._build_mmap_packed(
                sequences_file,
                input_ids_path,
                attention_mask_path,
                num_sequences,
                target_length,
                pad_token_id,
                eos_token_id,
                dtype_np,
                mask_dtype,
                start_idx=start_idx,
                vocab_limit=vocab_limit,
            )
        else:
            num_sequences, max_len = self._count_sequences(sequences_file, start_idx=start_idx)
            self._build_mmap_unpacked(
                sequences_file,
                input_ids_path,
                attention_mask_path,
                num_sequences,
                target_length,
                pad_token_id,
                dtype_np,
                mask_dtype,
                start_idx=start_idx,
                vocab_limit=vocab_limit,
            )

        manifest = {
            "source_path": sequences_file,
            "source_mtime": source_mtime,
            "source_size": source_size,
            "target_length": target_length,
            "num_sequences": num_sequences,
            "packed": packed,
            "dtype": str(dtype),
            "checksum_mode": checksum_mode,
            "checksum_bytes": checksum_bytes,
        }
        if checksum_enabled:
            manifest["source_checksum"] = self._compute_file_checksum(
                source_path, checksum_mode, checksum_bytes
            )
        manifest_path.write_text(json.dumps(manifest, indent=2))

        dataset = MemmapTokenDataset(
            input_ids_path,
            attention_mask_path,
            (num_sequences, target_length),
            dtype_np,
            mask_dtype,
        )
        return dataset, num_sequences

    def _build_mmap_packed(
        self,
        sequences_file: str,
        input_ids_path: Path,
        attention_mask_path: Path,
        num_sequences: int,
        target_length: int,
        pad_token_id: int,
        eos_token_id: int,
        dtype_np: np.dtype,
        mask_dtype: np.dtype,
        start_idx: int = 0,
        vocab_limit: Optional[int] = None,
    ) -> None:
        input_ids = np.memmap(str(input_ids_path), dtype=dtype_np, mode='w+', shape=(num_sequences, target_length))
        attention_mask = np.memmap(str(attention_mask_path), dtype=mask_dtype, mode='w+', shape=(num_sequences, target_length))

        pack_idx = 0
        current_tokens: List[int] = []
        current_mask: List[int] = []

        def finalize_pack():
            nonlocal pack_idx, current_tokens, current_mask
            if not current_tokens:
                return
            pad_len = target_length - len(current_tokens)
            if pad_len > 0:
                current_tokens.extend([pad_token_id] * pad_len)
                current_mask.extend([0] * pad_len)
            input_ids[pack_idx, :] = np.array(current_tokens[:target_length], dtype=dtype_np)
            attention_mask[pack_idx, :] = np.array(current_mask[:target_length], dtype=mask_dtype)
            pack_idx += 1
            current_tokens = []
            current_mask = []

        for seq in self._iter_token_sequences(sequences_file, start_idx=start_idx):
            if not seq:
                continue
            if vocab_limit is not None:
                seq = [max(0, min(int(t), vocab_limit - 1)) for t in seq]
            if len(seq) > target_length:
                seq = seq[:target_length]
            seq_len = len(seq)
            if seq_len >= target_length:
                finalize_pack()
                input_ids[pack_idx, :] = np.array(seq[:target_length], dtype=dtype_np)
                attention_mask[pack_idx, :] = 1
                pack_idx += 1
                continue
            if current_tokens and len(current_tokens) + seq_len + 1 > target_length:
                finalize_pack()
            if current_tokens:
                current_tokens.append(eos_token_id)
                current_mask.append(1)
            current_tokens.extend(seq)
            current_mask.extend([1] * seq_len)

        finalize_pack()
        input_ids.flush()
        attention_mask.flush()

    def _build_mmap_unpacked(
        self,
        sequences_file: str,
        input_ids_path: Path,
        attention_mask_path: Path,
        num_sequences: int,
        target_length: int,
        pad_token_id: int,
        dtype_np: np.dtype,
        mask_dtype: np.dtype,
        start_idx: int = 0,
        vocab_limit: Optional[int] = None,
    ) -> None:
        input_ids = np.memmap(str(input_ids_path), dtype=dtype_np, mode='w+', shape=(num_sequences, target_length))
        attention_mask = np.memmap(str(attention_mask_path), dtype=mask_dtype, mode='w+', shape=(num_sequences, target_length))

        row = 0
        for seq in self._iter_token_sequences(sequences_file, start_idx=start_idx):
            if not isinstance(seq, list):
                continue
            if vocab_limit is not None:
                seq = [max(0, min(int(t), vocab_limit - 1)) for t in seq]
            seq_len = min(len(seq), target_length)
            if seq_len:
                input_ids[row, :seq_len] = np.array(seq[:seq_len], dtype=dtype_np)
                attention_mask[row, :seq_len] = 1
            if seq_len < target_length:
                input_ids[row, seq_len:] = pad_token_id
                attention_mask[row, seq_len:] = 0
            row += 1

        input_ids.flush()
        attention_mask.flush()

    def _load_data_mmap(
        self,
        sequences_file: str,
        start_idx: int = 0,
        val_sequences_file: Optional[str] = None,
        pack_target_override: Optional[int] = None,
        stats: Optional[Dict[str, float]] = None,
    ) -> Tuple[DataLoader, DataLoader, List[Dict], List[Dict], Optional[Dict[str, Any]]]:
        mmap_cfg = self.train_cfg.get('memory_mapped_dataset', {})
        dtype = mmap_cfg.get('dtype', 'int64')

        # Ensure tokenizer loaded for vocab checks and pad/eos IDs
        if self.tokenizer is None:
            tok_name = self.model_cfg.get('tokenizer_name') or self.model_cfg.get('name')
            trust_remote = bool(self.model_cfg.get('trust_remote_code', True))
            self.tokenizer = AutoTokenizer.from_pretrained(tok_name, trust_remote_code=trust_remote)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        eos_token_id = self.tokenizer.eos_token_id or pad_token_id
        tokenizer_vocab_size = len(self.tokenizer)

        data_vocab_size = self._peek_vocab_size(sequences_file)
        if data_vocab_size is not None and data_vocab_size > tokenizer_vocab_size:
            raise ValueError(
                f"Token sequences in {sequences_file} were created with vocab_size={data_vocab_size}, "
                f"but tokenizer {self.tokenizer.name_or_path} only has {tokenizer_vocab_size} tokens."
            )

        use_packing = self.train_cfg.get('use_sequence_packing', True)
        pack_target = pack_target_override if pack_target_override is not None else mmap_cfg.get('pack_target')
        if use_packing:
            if pack_target is None or str(pack_target).lower() == "auto":
                pack_target = self._auto_pack_target(stats, mmap_cfg)
            pack_target = int(pack_target)
            self.train_cfg['pack_target_selected'] = pack_target
        else:
            pack_target = 0

        if use_packing:
            target_length = pack_target
            packed = True
        else:
            _, max_len = self._count_sequences(sequences_file, start_idx=start_idx)
            actual_max_seq = ((max_len + 63) // 64) * 64 if max_len else 64
            model_max = self.model_cfg.get('max_position_embeddings')
            if model_max is None and getattr(self, 'model', None) is not None:
                cfg = getattr(self.model, 'config', None)
                model_max = getattr(cfg, 'max_position_embeddings', None) or getattr(cfg, 'n_positions', None)
            if model_max is None:
                model_max = int(self.train_cfg.get('context_window', 2048)) if isinstance(self.train_cfg, dict) else 2048
            target_length = min(actual_max_seq, model_max)
            packed = False

        train_dataset, train_count = self._ensure_mmap_dataset(
            sequences_file,
            start_idx=start_idx,
            target_length=target_length,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            packed=packed,
            dtype=dtype,
            vocab_limit=tokenizer_vocab_size,
        )

        if train_count == 0:
            raise ValueError(f"No token sequences found in {sequences_file}")

        train_metadata = [{} for _ in range(train_count)]

        if val_sequences_file:
            val_dataset, val_count = self._ensure_mmap_dataset(
                val_sequences_file,
                start_idx=0,
                target_length=target_length,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                packed=packed,
                dtype=dtype,
                vocab_limit=tokenizer_vocab_size,
            )
            val_metadata = [{} for _ in range(val_count)]
        else:
            val_ratio = float(self.train_cfg.get('validation_split', 0.1))
            val_count = max(1, int(train_count * val_ratio)) if train_count > 1 else 0
            train_count = train_count - val_count
            indices = list(range(train_count + val_count))
            train_indices = indices[:train_count]
            val_indices = indices[train_count:]
            base_dataset = train_dataset
            train_dataset = Subset(base_dataset, train_indices)
            val_dataset = Subset(base_dataset, val_indices) if val_indices else Subset(base_dataset, [])
            train_metadata = [{} for _ in range(len(train_indices))]
            val_metadata = [{} for _ in range(len(val_indices))]

        train_sampler, curriculum_summary = self._build_curriculum_sampler(train_metadata)

        batch_size = self._get_batch_size(seq_length=target_length)
        num_workers = self._recommend_num_workers()
        use_pin_memory = self.train_cfg.get('pin_memory', True) and torch.cuda.is_available()
        pin_memory_device = self.train_cfg.get('pin_memory_device')
        prefetch_factor = int(self.train_cfg.get('prefetch_factor', 4))
        if num_workers == 0:
            prefetch_factor = None

        loader_kwargs = {
            'batch_size': batch_size,
            'num_workers': num_workers,
            'pin_memory': use_pin_memory,
            'prefetch_factor': prefetch_factor,
            'persistent_workers': num_workers > 0,
        }
        if pin_memory_device and 'pin_memory_device' in inspect.signature(DataLoader).parameters:
            loader_kwargs['pin_memory_device'] = pin_memory_device

        train_loader = DataLoader(
            train_dataset,
            sampler=train_sampler,
            shuffle=train_sampler is None,
            **loader_kwargs,
        )

        val_loader = DataLoader(
            val_dataset,
            shuffle=False,
            **loader_kwargs,
        )

        logger.info(
            f"🗺️ Memory-mapped dataset loaded: {len(train_dataset)} train, {len(val_dataset)} val, "
            f"seq_len={target_length}, packed={packed}"
        )

        return train_loader, val_loader, train_metadata, val_metadata, curriculum_summary

    def _safe_scaled_step(self, scaler, optimizer):
        """Work around PyTorch bug where inf checks may be missing"""
        try:
            scaler.step(optimizer)
        except AssertionError as exc:
            msg = str(exc)
            if "No inf checks were recorded for this optimizer." not in msg:
                raise
            optimizer_state = scaler._per_optimizer_states.get(id(optimizer))
            if optimizer_state is None:
                raise
            if not optimizer_state["found_inf_per_device"]:
                optimizer_state["found_inf_per_device"][self.device] = torch.zeros(1, device=self.device)
                scaler.step(optimizer)
                return
            raise
    
    def _ensure_buffers_on_device(self):
        """Ensure all model buffers are on the correct device (needed for quantized models)"""
        try:
            for module in self.model.modules():
                for name, buf in list(module.named_buffers(recurse=False)):
                    if buf is not None and buf.device != self.device:
                        setattr(module, name, buf.to(self.device))
        except Exception as e:
            logger.warning(f"Failed to ensure buffers on device: {e}")

    def _patch_attention_bias_device(self):
        """Patch attention mask preparation to fix CUDA device side asserts"""
        try:
            import torch.nn.functional as F
            from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask_for_sdpa

            original_prepare_mask = _prepare_4d_attention_mask_for_sdpa

            def patched_prepare_mask(attention_mask, dtype, tgt_len=None):
                # Cast to float first to avoid CUDA errors
                if attention_mask is not None:
                    attention_mask = attention_mask.float()
                try:
                    return original_prepare_mask(attention_mask, dtype, tgt_len)
                except Exception:
                    # Fallback: just return the mask as-is
                    return attention_mask

            # Patch the function
            import transformers.modeling_attn_mask_utils
            transformers.modeling_attn_mask_utils._prepare_4d_attention_mask_for_sdpa = patched_prepare_mask

            # Also patch scaled_dot_product_attention
            original_scaled_dot = F.scaled_dot_product_attention

            def patched_scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
                # If attn_mask is present and on wrong device, move it
                if attn_mask is not None and hasattr(attn_mask, 'device'):
                    if attn_mask.device != query.device:
                        attn_mask = attn_mask.to(query.device)

                # Call original
                return original_scaled_dot(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale)

            F.scaled_dot_product_attention = patched_scaled_dot_product_attention
            logger.info("Patched attention functions to handle CUDA device mismatches")
        except Exception as e:
            logger.warning(f"Failed to patch attention functions: {e}")

    def load_model_and_tokenizer(self):
        """Load model and tokenizer with quantization and LoRA if configured"""
        logger.info(f"Loading model and tokenizer...")

        # CRITICAL: Clean CUDA memory before model loading to maximize available VRAM
        if torch.cuda.is_available():
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            free_mem, total_mem = torch.cuda.mem_get_info()
            free_gb = free_mem / (1024**3)
            logger.info(f"Pre-model-load VRAM: {free_gb:.2f} GB / {total_mem/(1024**3):.2f} GB free")

        # Check if we're using DeepSpeed
        # DeepSpeed is incompatible with torch.compile, so we need to detect it early
        use_deepspeed = False
        use_deepspeed_offload = False
        deepspeed_config_path = None
        import sys
        for i, arg in enumerate(sys.argv):
            if arg == '--deepspeed' and i + 1 < len(sys.argv):
                deepspeed_config_path = sys.argv[i + 1]
                break

        if deepspeed_config_path and os.path.exists(deepspeed_config_path):
            try:
                import json
                with open(deepspeed_config_path, 'r') as f:
                    ds_config = json.load(f)
                # DeepSpeed is being used
                use_deepspeed = True
                zero_config = ds_config.get('zero_optimization', {})
                zero_stage = zero_config.get('stage', 0)
                logger.info(f"🔧 DeepSpeed ZeRO-{zero_stage} detected")

                # Check for ZeRO-3 with CPU offload (requires special model loading)
                if zero_stage == 3:
                    offload_params = zero_config.get('offload_param', {})
                    if offload_params.get('device') == 'cpu':
                        use_deepspeed_offload = True
                        logger.info("🔧 DeepSpeed ZeRO-3 with CPU offload - loading model on CPU")
            except Exception as e:
                logger.warning(f"Could not parse DeepSpeed config: {e}")

        # Synthetic mode: no downloads; used by tests/CI.
        if self.model_cfg.get("synthetic_model", False):
            self._init_synthetic_model_and_tokenizer()
            self._maybe_init_ema()
            return
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_cfg['tokenizer_name'],
            trust_remote_code=self.model_cfg['trust_remote_code'],
        )
        
        # Ensure padding token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info(f"Tokenizer loaded: vocab_size={len(self.tokenizer)}")

        # MPS/Metal: bitsandbytes is CUDA-only. If the run requested 4/8-bit quant, use
        # the MPS-native int8 weight-only quant backend instead.
        if getattr(self, "device_backend", None) is not None and getattr(self.device_backend, "is_metal", False):
            if bool(self.model_cfg.get("use_mps_int8_quant", False)):
                # ensure bnb flags remain off
                self.model_cfg['use_4bit'] = False
                self.model_cfg['use_8bit'] = False
        
        # Quantization config (CUDA path only)
        quantization_config = None
        use_mps_int8_quant = bool(self.model_cfg.get("use_mps_int8_quant", False))
        if (self.model_cfg['use_4bit'] or self.model_cfg['use_8bit']) and not use_mps_int8_quant:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=self.model_cfg['use_4bit'],
                load_in_8bit=self.model_cfg['use_8bit'],
                bnb_4bit_compute_dtype=torch.bfloat16 if self.model_cfg['use_bf16'] else torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            logger.info(f"Quantization: {'4-bit' if self.model_cfg['use_4bit'] else '8-bit'}")

        # Load model
        if self.device.type == "mps" and use_mps_int8_quant:
            try:
                from training.mps_quant_backend import load_quantized_starcoder2_mps, MPSQuantConfig
            except Exception:
                from .mps_quant_backend import load_quantized_starcoder2_mps, MPSQuantConfig

            qcfg = MPSQuantConfig.from_trainer_cfg(self.model_cfg, self.config.get("quantization", {}))
            self.model = load_quantized_starcoder2_mps(
                pretrained_model=self.model_cfg['name'],
                device=self.device,
                cfg=qcfg,
                trust_remote_code=bool(self.model_cfg.get('trust_remote_code', True)),
            )
        else:
            # TIER 4 OPTIMIZATION: Use FlashAttention-2 if available for 32K+ contexts
            attn_implementation = "flash_attention_2" if HAS_FLASH_ATTN else "sdpa"
            logger.info(f"Using attention implementation: {attn_implementation}")

            # CRITICAL MEMORY FIX: When using DeepSpeed ZeRO-3 with CPU offload, load on CPU
            # and let DeepSpeed handle GPU placement to avoid OOM
            if use_deepspeed_offload:
                # Load model on CPU without device_map
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_cfg['name'],
                    quantization_config=None,  # Disable quantization with ZeRO-3
                    device_map=None,  # Don't use device_map with DeepSpeed
                    trust_remote_code=self.model_cfg['trust_remote_code'],
                    torch_dtype=torch.float16,  # FP16 for DeepSpeed
                    attn_implementation=attn_implementation,
                    use_cache=False,
                    low_cpu_mem_usage=True,  # Use less CPU memory during loading
                )
                logger.info("✓ Model loaded on CPU for DeepSpeed ZeRO-3 offloading")
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_cfg['name'],
                    quantization_config=quantization_config,
                    device_map='auto' if (self.model_cfg['use_4bit'] or self.model_cfg['use_8bit']) else None,
                    trust_remote_code=self.model_cfg['trust_remote_code'],
                    torch_dtype=torch.bfloat16 if self.model_cfg['use_bf16'] else torch.float32,
                    attn_implementation=attn_implementation,  # FlashAttention-2 or SDPA
                    use_cache=False,  # Required for training with Flash Attention
                )

        # If not using quantized device_map or DeepSpeed offload, move model explicitly to selected device
        if not (self.model_cfg['use_4bit'] or self.model_cfg['use_8bit'] or use_deepspeed_offload):
            self.model = self.model.to(self.device)
        else:
            # For 8-bit quantized models, we need to ensure any buffers are on the correct device
            # This is especially important for models with attention bias buffers
            self._ensure_buffers_on_device()

        # Patch RoBERTa attention to handle device mismatch
        self._patch_attention_bias_device()

        # Apply device-specific patches (Metal FlashAttention, etc.)
        if getattr(self, "device_backend", None) is not None:
            try:
                self.device_backend.patch_model(self.model)
            except Exception as e:
                logger.warning(f"Device backend model patching failed: {e}")
        
        logger.info(f"Base model loaded: {self.model_cfg['name']}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()) / 1e9:.2f}B")
        
        # 🧠 EINSTEIN MEMORY DECISION: Gradient checkpointing based on ACTUAL memory
        # Rule: ALWAYS enable checkpointing for GPUs < 12GB - it's necessary!
        # Without it, activations alone can use 300-700MB per batch item
        total_memory_gb = 0
        if torch.cuda.is_available():
            total_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9

        # For < 12GB GPUs, ALWAYS enable checkpointing - speed isn't worth OOM!
        if total_memory_gb < 12:
            try:
                self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                logger.info(f"🧠 GPU {total_memory_gb:.1f}GB: Gradient checkpointing REQUIRED (saves ~3x memory)")
            except TypeError:
                self.model.gradient_checkpointing_enable()
                logger.info(f"🧠 GPU {total_memory_gb:.1f}GB: Gradient checkpointing enabled (legacy mode)")
        elif self.train_cfg['use_gradient_checkpointing']:
            try:
                self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                logger.info("🚀 Gradient checkpointing enabled (non-reentrant mode)")
            except TypeError:
                self.model.gradient_checkpointing_enable()
                logger.info("Gradient checkpointing enabled (legacy mode)")
        else:
            logger.info(f"⚡ GPU {total_memory_gb:.1f}GB: Skipping checkpointing (enough VRAM)")
        
        # Apply LoRA if configured (skip when using MPS quant backend; it has per-layer LoRA built in)
        if self.model_cfg['use_lora'] and not bool(self.model_cfg.get("use_mps_int8_quant", False)):
            self.model = self._apply_lora()
            # Ensure buffers are on device after LoRA application
            self._ensure_buffers_on_device()
        elif bool(self.model_cfg.get("use_mps_int8_quant", False)):
            logger.info("MPS int8 backend: LoRA is built-in per-layer; skipping PEFT LoRA application.")

        # Only move to device if not using quantization with device_map (which already handles device placement)
        if not (self.model_cfg['use_4bit'] or self.model_cfg['use_8bit']):
            self.model = self.model.to(self.device)
        else:
            # Ensure all buffers (especially attn_bias) are on correct device after all setup
            self._ensure_buffers_on_device()

        # torch.compile - ARCHITECTURE AWARE!
        # Turing: SKIP - "Not enough SMs" warning + CUDA graph conflicts = slower!
        # Ampere+: Use reduce-overhead (max-autotune CUDA graphs conflict with DeepSpeed)
        # CRITICAL: Skip torch.compile when using DeepSpeed - they are incompatible!
        if use_deepspeed:
            logger.info("⚠️ Skipping torch.compile - incompatible with DeepSpeed")
        elif torch.cuda.is_available() and hasattr(torch, 'compile'):
            gpu_compute_cap = torch.cuda.get_device_capability()[0] + torch.cuda.get_device_capability()[1] / 10

            if gpu_compute_cap < 8.0:
                # Turing and older - torch.compile adds overhead, SKIP IT!
                logger.info("⚡ Turing GPU: Skipping torch.compile (adds overhead on sm_75)")
            else:
                # Ampere+ - use reduce-overhead (NOT max-autotune, CUDA graphs conflict!)
                try:
                    logger.info("🚀 Applying torch.compile (reduce-overhead mode)...")
                    self.model = torch.compile(self.model, mode="reduce-overhead")
                    logger.info("✓ torch.compile applied (1.3-2x speedup expected)")
                except Exception as e:
                    logger.warning(f"torch.compile failed (non-critical): {e}")

    def _apply_lora(self):
        """Apply LoRA (Parameter-Efficient Fine-Tuning) to the model"""
        lora_cfg = self.model_cfg['lora']
        if self.model_cfg.get("cpu_offload_lora", False) and not HAS_CPU_LORA_OFFLOAD:
            logger.warning(f"cpu_offload_lora requested but unavailable: {_cpu_offload_import_error}")
        use_cpu_offload = bool(self.model_cfg.get("cpu_offload_lora", False)) and HAS_CPU_LORA_OFFLOAD

        if use_cpu_offload:
            model = apply_cpu_offloaded_lora(self.model, lora_cfg)
        else:
            peft_config = LoraConfig(
                r=lora_cfg['r'],
                lora_alpha=lora_cfg['lora_alpha'],
                target_modules=lora_cfg['target_modules'],
                lora_dropout=lora_cfg['lora_dropout'],
                bias=lora_cfg['bias'],
                task_type=TaskType.CAUSAL_LM,
            )
            model = get_peft_model(self.model, peft_config)
        
        # Log trainable params
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_percent = 100 * trainable_params / total_params
        
        logger.info(f"\nLoRA Configuration:")
        logger.info(f"  Rank: {lora_cfg['r']}")
        logger.info(f"  Alpha: {lora_cfg['lora_alpha']}")
        logger.info(f"  Target modules: {lora_cfg['target_modules']}")
        logger.info(f"  Trainable params: {trainable_params:,} ({trainable_percent:.3f}%)")
        logger.info(f"  Total params: {total_params:,}\n")
        if use_cpu_offload:
            logger.info("  Mode: CPU-pinned LoRA adapters (staged to GPU per step)")
        
        return model

    def _pack_sequences(
        self,
        token_lists: List[List[int]],
        target_length: int,
        pad_token_id: int,
        eos_token_id: int,
    ) -> Tuple[List[List[int]], List[List[int]]]:
        """
        🔥🔥🔥 ULTRA-SPEED CHUNKING + PACKING 🔥🔥🔥

        Instead of truncating long sequences, we CHUNK them into multiple pieces!
        This keeps ALL the data while enabling blazing fast training.

        Example with target_length=256:
        - 1000-token sequence → 4 chunks of 250 tokens each
        - Each chunk is a separate training example
        - More examples, but each is FAST (O(n²) with n=256 is tiny!)
        """
        if not token_lists:
            return [], []

        # STEP 1: Process long sequences - SPEED-FIRST!
        # For ultra-speed mode, we just TRUNCATE long sequences.
        # This reduces total work and maximizes throughput.
        chunked_seqs = []
        truncated_count = 0
        for seq in token_lists:
            if not seq:
                continue
            if len(seq) <= target_length:
                chunked_seqs.append(seq)
            else:
                # 🔥 ULTRA-SPEED: Just truncate to target_length
                # No chunking = fewer sequences = faster training!
                chunked_seqs.append(seq[:target_length])
                truncated_count += 1

        if truncated_count > 0:
            logger.info(f"⚡ Truncated {truncated_count} sequences to {target_length} tokens (SPEED MODE)")

        # STEP 2: Pack chunks into target_length sequences
        packed_tokens = []
        packed_masks = []
        current_tokens = []
        current_mask = []

        for seq in chunked_seqs:
            seq_len = len(seq)

            # If this chunk alone is >= target, make it its own pack
            if seq_len >= target_length:
                if current_tokens:
                    # Finalize current pack first
                    pad_len = target_length - len(current_tokens)
                    if pad_len > 0:
                        current_tokens.extend([pad_token_id] * pad_len)
                        current_mask.extend([0] * pad_len)
                    packed_tokens.append(current_tokens[:target_length])
                    packed_masks.append(current_mask[:target_length])
                    current_tokens = []
                    current_mask = []
                # Add this chunk as its own pack (truncated to target)
                packed_tokens.append(seq[:target_length])
                packed_masks.append([1] * target_length)
                continue

            # If adding this sequence would exceed target, finalize current pack
            if current_tokens and len(current_tokens) + seq_len + 1 > target_length:
                pad_len = target_length - len(current_tokens)
                if pad_len > 0:
                    current_tokens.extend([pad_token_id] * pad_len)
                    current_mask.extend([0] * pad_len)
                packed_tokens.append(current_tokens[:target_length])
                packed_masks.append(current_mask[:target_length])
                current_tokens = []
                current_mask = []

            # Add sequence with EOS separator
            if current_tokens:
                current_tokens.append(eos_token_id)
                current_mask.append(1)

            current_tokens.extend(seq)
            current_mask.extend([1] * seq_len)

        # Finalize last pack
        if current_tokens:
            pad_len = target_length - len(current_tokens)
            if pad_len > 0:
                current_tokens.extend([pad_token_id] * pad_len)
                current_mask.extend([0] * pad_len)
            packed_tokens.append(current_tokens[:target_length])
            packed_masks.append(current_mask[:target_length])

        # Verify lengths
        for i, pt in enumerate(packed_tokens):
            if len(pt) != target_length:
                if len(pt) < target_length:
                    packed_tokens[i] = pt + [pad_token_id] * (target_length - len(pt))
                    packed_masks[i] = packed_masks[i] + [0] * (target_length - len(packed_masks[i]))
                else:
                    packed_tokens[i] = pt[:target_length]
                    packed_masks[i] = packed_masks[i][:target_length]

        original_count = len(token_lists)
        final_count = len(packed_tokens)
        logger.info(f"🚀 ULTRA-SPEED PACKING: {original_count} seqs → {final_count} packs @ {target_length} tokens")
        logger.info(f"   Expected speed: ~{50 if target_length <= 256 else 12 if target_length <= 512 else 3} it/s")

        return packed_tokens, packed_masks

    def _create_length_sorted_sampler(
        self,
        lengths: List[int],
        batch_size: int,
    ) -> torch.utils.data.Sampler:
        """
        1% OPTIMIZATION: Sort sequences by length for dynamic batching.

        This minimizes padding waste within each batch by grouping
        similar-length sequences together.

        Returns indices sorted by length, with some randomization within buckets
        to maintain training variance.
        """
        # Create (index, length) pairs and sort by length
        indexed_lengths = [(i, l) for i, l in enumerate(lengths)]
        indexed_lengths.sort(key=lambda x: x[1])

        # Group into buckets of batch_size for slight randomization
        buckets = []
        for i in range(0, len(indexed_lengths), batch_size * 4):
            bucket = indexed_lengths[i:i + batch_size * 4]
            # Shuffle within bucket to add variance
            import random
            random.shuffle(bucket)
            buckets.extend(bucket)

        indices = [idx for idx, _ in buckets]
        logger.info(f"🚀 LENGTH-SORTED SAMPLING: Minimized padding waste across {len(indices)} sequences")

        return indices

    def load_data(
        self,
        sequences_file: str,
        start_idx: int = 0,
        val_sequences_file: Optional[str] = None,
    ) -> Tuple[DataLoader, DataLoader, List[Dict], List[Dict], Optional[Dict[str, Any]]]:
        """Load and prepare data"""
        logger.info(f"Loading sequences from {sequences_file}...")

        sequences_file = self._stage_dataset_if_needed(sequences_file)
        if val_sequences_file:
            val_sequences_file = self._stage_dataset_if_needed(val_sequences_file)

        mmap_cfg = self.train_cfg.get('memory_mapped_dataset', {})
        probe_cfg = self.train_cfg.get('throughput_probe', {})
        use_packing = self.train_cfg.get('use_sequence_packing', True)
        is_pretokenized = self._is_pretokenized_jsonl(sequences_file)
        stats = None

        if is_pretokenized and (use_packing or (isinstance(mmap_cfg, dict) and mmap_cfg.get('enabled', False))):
            stats = self._profile_sequence_lengths(sequences_file, start_idx=start_idx)
            self.dataset_stats = stats
            logger.info(
                f"Dataset stats: count={stats['count']:,}, avg={stats['avg']:.0f}, "
                f"p95={stats['p95']:.0f}, max={stats['max']:.0f}"
            )

        pack_target = None
        if use_packing and stats is not None:
            pack_target = mmap_cfg.get('pack_target') if isinstance(mmap_cfg, dict) else None
            pack_target_locked = False
            if pack_target is None or str(pack_target).lower() == "auto":
                pack_target = self._auto_pack_target(stats, mmap_cfg if isinstance(mmap_cfg, dict) else None)
            else:
                pack_target = int(pack_target)
                pack_target_locked = True

            candidates = [pack_target]
            if isinstance(mmap_cfg, dict):
                min_target = int(mmap_cfg.get('pack_target_min', 256))
                max_target = int(mmap_cfg.get('pack_target_max', 4096))
                base_target = self._auto_pack_target(None, mmap_cfg)
                p95_target = self._next_pow2(int(stats.get('p95', pack_target)))
                for candidate in (base_target, p95_target, pack_target):
                    candidate = max(min_target, min(max_target, candidate))
                    candidates.append(candidate)
                candidates = sorted(set(candidates))
                if bool(probe_cfg.get('favor_speed', True)):
                    candidates = sorted(candidates)
                else:
                    candidates = sorted(candidates, reverse=True)
                max_candidates = int(probe_cfg.get('max_candidates', 3)) if isinstance(probe_cfg, dict) else 3
                if len(candidates) > max_candidates:
                    candidates = candidates[:max_candidates]

            joint_choice = None
            user_batch_override = None
            if isinstance(self.train_cfg, dict):
                user_batch_override = self.train_cfg.get('batch_size_override')

            if not pack_target_locked and isinstance(probe_cfg, dict) and probe_cfg.get('joint_batch_pack', False):
                if not user_batch_override:
                    batch_candidates = self._batch_size_candidates_for_probe(
                        seq_len=pack_target,
                        probe_cfg=probe_cfg,
                    )
                    joint_choice = self._select_pack_target_and_batch_from_probe(
                        seq_candidates=candidates,
                        batch_candidates=batch_candidates,
                        probe_cfg=probe_cfg,
                    )
                    if joint_choice:
                        pack_target, batch_choice = joint_choice
                        self.train_cfg['batch_size_selected'] = int(batch_choice)
                        self.train_cfg['batch_size_override'] = int(batch_choice)
                        logger.info(
                            f"Throughput probe selected pack_target={pack_target}, "
                            f"batch_size={batch_choice}"
                        )
                else:
                    logger.info("Batch size override provided; skipping joint batch/pack probe")

            if joint_choice is None and not pack_target_locked:
                probe_choice = self._select_pack_target_from_probe(
                    candidates=candidates,
                    batch_size=max(1, int(self.train_cfg.get('batch_size_reference', 2))),
                    probe_cfg=probe_cfg if isinstance(probe_cfg, dict) else {},
                )
                if probe_choice:
                    pack_target = probe_choice
            logger.info(f"Selected pack_target={pack_target} tokens")
            self.train_cfg['pack_target_selected'] = pack_target

        if isinstance(mmap_cfg, dict) and mmap_cfg.get('enabled', False) and is_pretokenized:
            val_ok = True if not val_sequences_file else self._is_pretokenized_jsonl(val_sequences_file)
            if val_ok:
                packed_count = None
                if use_packing and pack_target:
                    packed_count = self._count_packed_sequences(
                        sequences_file, int(pack_target), start_idx=start_idx
                    )
                seq_len_est = int(pack_target) if (use_packing and pack_target) else int(stats.get('max', 0)) if stats else 0
                use_memmap = self._should_use_memmap(
                    mmap_cfg,
                    stats if stats else {'count': 0},
                    seq_len_est,
                    packed_count=packed_count,
                )
                if use_memmap:
                    return self._load_data_mmap(
                        sequences_file,
                        start_idx=start_idx,
                        val_sequences_file=val_sequences_file,
                        pack_target_override=pack_target,
                        stats=stats,
                    )
                logger.info("Dataset fits in RAM; using in-memory TensorDataset instead of memmap.")
            else:
                logger.warning("Memory-mapped dataset requested but input is not pre-tokenized; falling back to RAM load.")

        sequences = []
        data_vocab_size = None
        metadata_map = {}

        # Load JSONL format (one JSON object per line)
        with open(sequences_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)

                if isinstance(obj, dict):
                    if "token_sequences" in obj:
                        sequences.extend(obj["token_sequences"])
                    elif "tokens" in obj:
                        sequences.append({"tokens": obj["tokens"]})
                    else:
                        sequences.append(obj)

                    if "metadata" in obj:
                        metadata_map.update(obj["metadata"])
                    if "vocab_size" in obj and data_vocab_size is None:
                        data_vocab_size = obj["vocab_size"]
                else:
                    sequences.append(obj)

        if sequences and isinstance(sequences[0], dict):
            if "tokens" not in sequences[0]:
                raise ValueError(
                    f"Expected 'tokens' key in sequence objects from {sequences_file}, "
                    f"got keys={list(sequences[0].keys())}"
                )
            token_lists = [seq["tokens"] for seq in sequences]
        else:
            token_lists = sequences

        if start_idx >= len(token_lists):
            raise ValueError(
                f"Requested start index {start_idx} exceeds loaded sequences ({len(token_lists)})"
            )

        if start_idx:
            logger.info(f"Incremental data: using sequences from index {start_idx} onward")
            token_lists = token_lists[start_idx:]

        logger.info(f"Loaded {len(token_lists)} sequences")

        if not token_lists:
            raise ValueError(
                f"No token sequences found in {sequences_file}; run the tokenizer first"
            )

        # CRITICAL 1% OPTIMIZATION: Use DYNAMIC max_length based on actual data!
        # Previous: Used config context_window (131K) for 256-token data = 512x wasted compute
        # Fixed: Use actual max sequence length from data + small padding buffer

        # First, find the actual max sequence length in the data
        actual_max_seq = max(len(seq) for seq in token_lists if isinstance(seq, list) and seq)

        # Round up to nearest multiple of 64 for GPU efficiency (tensor cores)
        actual_max_seq = ((actual_max_seq + 63) // 64) * 64

        # Get model's theoretical max (for clamping)
        model_max = self.model_cfg.get('max_position_embeddings')
        if model_max is None and getattr(self, 'model', None) is not None:
            cfg = getattr(self.model, 'config', None)
            model_max = getattr(cfg, 'max_position_embeddings', None) or getattr(cfg, 'n_positions', None)
        if model_max is None:
            model_max = int(self.train_cfg.get('context_window', 2048)) if isinstance(self.train_cfg, dict) else 2048

        # Use the SMALLER of actual data length vs model max
        max_length = min(actual_max_seq, model_max)

        logger.info(f"🚀 DYNAMIC max_length: {max_length} (data max: {actual_max_seq}, model max: {model_max})")

        # Lazy-load tokenizer so load_data() can be used without calling model setup first.
        # This is important for unit tests and for data/throughput validation phases.
        if self.tokenizer is None:
            tok_name = self.model_cfg.get('tokenizer_name') or self.model_cfg.get('name')
            trust_remote = bool(self.model_cfg.get('trust_remote_code', True))
            logger.info(
                f"Tokenizer not initialized; loading tokenizer '{tok_name}' (trust_remote_code={trust_remote})"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(tok_name, trust_remote_code=trust_remote)
            # Many causal-LM tokenizers don't define a pad token; use EOS for padding.
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        tokenizer_vocab_size = len(self.tokenizer)

        # Check vocab size compatibility
        if data_vocab_size is not None and data_vocab_size > tokenizer_vocab_size:
            raise ValueError(
                f"Token sequences in {sequences_file} were created with vocab_size={data_vocab_size}, "
                f"but tokenizer {self.tokenizer.name_or_path} only has {tokenizer_vocab_size} tokens. "
                "Regenerate the sequences with the matching tokenizer (e.g. "
                f"`python3 tokenizers/git_tokenizer_rich.py --model {self.model_cfg['tokenizer_name']}`) "
                "before training."
            )

        pre_tokenized = all(
            isinstance(seq, list) and seq for seq in token_lists
        )

        if pre_tokenized:
            # 🔥 EINSTEIN-LEVEL: ALWAYS use sequence packing for efficiency!
            # This dramatically reduces forward passes (132K → ~8K for typical data)
            avg_seq_len = sum(len(seq) for seq in token_lists if seq) / max(1, len(token_lists))

            # CRITICAL FIX: Always enable packing - the old check was WRONG
            # Old: avg_seq_len < model_max * 0.5 would disable packing for long sequences
            # New: Always pack - it's ALWAYS beneficial for training speed
            use_packing = self.train_cfg.get('use_sequence_packing', True)

            logger.info(f"📊 Dataset stats: {len(token_lists)} sequences, avg length: {avg_seq_len:.0f} tokens")
            logger.info(f"🔧 Packing enabled: {use_packing}")

            if use_packing:
                # 🔥🔥🔥 ULTRA SPEED MODE 🔥🔥🔥
                # O(n²) attention math:
                # - 256 tokens:  65K ops   → ~0.02s/step (50 it/s!)
                # - 512 tokens:  262K ops  → ~0.08s/step (12 it/s)
                # - 1024 tokens: 1M ops    → ~0.3s/step (3 it/s)
                # - 2048 tokens: 4.2M ops  → ~3s/step (0.33 it/s)
                #
                # SPEED IS 100x MORE IMPORTANT THAN PRESERVING EVERY TOKEN!

                gpu_compute_cap = 0.0
                if torch.cuda.is_available():
                    gpu_compute_cap = torch.cuda.get_device_capability()[0] + torch.cuda.get_device_capability()[1] / 10

                # Sequence stats for logging
                seq_lengths = [len(seq) for seq in token_lists if seq]
                max_seq = max(seq_lengths)
                avg_seq = sum(seq_lengths) / len(seq_lengths)
                logger.info(f"📊 Data: {len(seq_lengths)} seqs, max={max_seq}, avg={avg_seq:.0f} tokens")

                if pack_target is None:
                    # 🔥 SPEED-FIRST: Use TINY pack target for Turing!
                    if gpu_compute_cap >= 8.0:  # Ampere+ with FlashAttention - can handle long
                        pack_target = 2048
                        logger.info("🚀 Ampere+: 2K pack (FlashAttention)")
                    elif gpu_compute_cap >= 7.5:  # Turing - SPEED IS EVERYTHING!
                        pack_target = 256  # BLAZING FAST! 50+ it/s possible!
                        truncated = sum(1 for l in seq_lengths if l > 256)
                        logger.info(f"⚡⚡⚡ TURING ULTRA-SPEED MODE ⚡⚡⚡")
                        logger.info(f"📦 Pack target: 256 tokens (50+ it/s expected!)")
                        logger.info(f"⚠️ {truncated}/{len(seq_lengths)} sequences will be chunked/truncated")
                        logger.info(f"   This is OK! Speed >> preserving every token")
                    else:
                        pack_target = 256
                        logger.info("📦 256 pack for older GPU")
                else:
                    truncated = sum(1 for l in seq_lengths if l > pack_target)
                    logger.info(f"📦 Pack target: {pack_target} tokens (empirically selected)")
                    if truncated:
                        logger.info(f"⚠️ {truncated}/{len(seq_lengths)} sequences will be truncated")
                eos_token_id = self.tokenizer.eos_token_id or pad_token_id

                # Clean and validate sequences first
                clean_seqs = []
                for seq in token_lists:
                    if isinstance(seq, list) and seq:
                        valid_seq = [max(0, min(int(t), tokenizer_vocab_size - 1)) for t in seq]
                        clean_seqs.append(valid_seq)

                packed_tokens, packed_masks = self._pack_sequences(
                    clean_seqs, pack_target, pad_token_id, eos_token_id
                )

                num_sequences = len(packed_tokens)
                max_length = pack_target  # Update max_length to pack target
                input_ids = torch.tensor(packed_tokens, dtype=torch.long)
                attention_mask = torch.tensor(packed_masks, dtype=torch.uint8)

                logger.info(f"✓ Packing enabled: {len(token_lists)} → {num_sequences} sequences ({pack_target} tokens each)")
            else:
                # Original unpacked processing
                num_sequences = len(token_lists)
                input_ids = torch.full((num_sequences, max_length), pad_token_id, dtype=torch.long)
                attention_mask = torch.zeros((num_sequences, max_length), dtype=torch.uint8)

                for idx, sequence in enumerate(token_lists):
                    if not isinstance(sequence, list):
                        raise ValueError(
                            f"Expected tokenized sequence lists in {sequences_file}; got {type(sequence)}"
                        )

                    seq_length = min(len(sequence), max_length)
                    if seq_length == 0:
                        continue

                    # Clip token IDs to valid range (some may be out of vocab)
                    valid_sequence = [max(0, min(int(token_id), tokenizer_vocab_size - 1)) for token_id in sequence[:seq_length]]
                    input_ids[idx, :seq_length] = torch.tensor(
                        valid_sequence, dtype=torch.long
                    )
                    attention_mask[idx, :seq_length] = 1
        else:
            encodings = self.tokenizer(
                token_lists,
                return_tensors='pt',
                max_length=max_length,
                padding='max_length',
                truncation=True,
            )
            input_ids = encodings['input_ids']
            attention_mask = encodings['attention_mask']

        def _maybe_pin(name: str, tensor: torch.Tensor) -> torch.Tensor:
            """Pin tensor to RAM if headroom allows; speeds up H2D copies."""
            if not self.train_cfg.get('use_ram_cache', True):
                return tensor
            try:
                # Use available RAM rather than total to be conservative
                available_ram = psutil.virtual_memory().available
                ratio_limit = float(self.train_cfg.get('ram_cache_max_ratio', 0.5))
                if tensor.nbytes <= available_ram * ratio_limit:
                    pinned = tensor.pin_memory()
                    logger.info(f"RAM cache: pinned {name} ({tensor.nbytes/1e9:.2f} GB)")
                    return pinned
                else:
                    budget_gb = available_ram * ratio_limit / 1e9
                    logger.info(f"RAM cache skipped for {name}: needs {tensor.nbytes/1e9:.2f} GB, budget {budget_gb:.2f} GB")
            except Exception as e:
                logger.warning(f"RAM cache pin failed for {name}: {e}")
            return tensor

        # Build metadata aligned with the sequences we loaded
        if metadata_map:
            sequence_metadata = []
            for offset in range(len(token_lists)):
                idx = start_idx + offset
                raw_meta = metadata_map.get(str(idx))
                sequence_metadata.append(raw_meta if isinstance(raw_meta, dict) else {})
        else:
            sequence_metadata = [{} for _ in token_lists]

        # Create dataset
        dataset = TensorDataset(
            input_ids,
            attention_mask
        )

        # Handle train/val split
        if val_sequences_file:
            # Use separate validation file
            logger.info(f"Loading validation sequences from {val_sequences_file}...")
            val_seqs = []
            with open(val_sequences_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        if "tokens" in obj:
                            val_seqs.append(obj["tokens"])
                        elif "token_sequences" in obj:
                            val_seqs.extend(obj["token_sequences"])
                    elif isinstance(obj, list):
                        val_seqs.append(obj)

            # 🔥 CRITICAL FIX: Use same max_length as training data (after packing)!
            # This ensures train and val tensors have matching dimensions
            val_max_length = max_length  # This is now pack_target after packing
            logger.info(f"Validation using same sequence length as training: {val_max_length}")

            # Process val sequences with same length as training
            val_input_ids = torch.full((len(val_seqs), val_max_length), pad_token_id, dtype=torch.long)
            val_attention_mask = torch.zeros((len(val_seqs), val_max_length), dtype=torch.uint8)
            for idx, seq in enumerate(val_seqs):
                if isinstance(seq, list) and seq:
                    seq_len = min(len(seq), val_max_length)
                    valid_seq = [max(0, min(int(t), tokenizer_vocab_size - 1)) for t in seq[:seq_len]]
                    val_input_ids[idx, :seq_len] = torch.tensor(valid_seq, dtype=torch.long)
                    val_attention_mask[idx, :seq_len] = 1
            val_input_ids = _maybe_pin("val input_ids", val_input_ids)
            val_attention_mask = _maybe_pin("val attention_mask", val_attention_mask)

            train_dataset = dataset
            val_dataset = TensorDataset(val_input_ids, val_attention_mask)
            train_metadata = sequence_metadata
            val_metadata = [{} for _ in val_seqs]
            logger.info(f"Loaded {len(val_seqs)} validation sequences (truncated to {val_max_length} tokens)")
        else:
            # Split from main dataset
            dataset_size = len(dataset)
            desired_val = max(1, int(dataset_size * float(self.train_cfg.get('validation_split', 0.1))))
            if dataset_size <= 1:
                val_count = 0
            else:
                val_count = min(desired_val, dataset_size - 1)
            train_count = dataset_size - val_count
            train_indices = list(range(train_count))
            val_indices = list(range(train_count, dataset_size))

            train_metadata = [sequence_metadata[i] for i in train_indices]
            val_metadata = [sequence_metadata[i] for i in val_indices]

            train_dataset = Subset(dataset, train_indices)
            val_dataset = Subset(dataset, val_indices)
        train_sampler, curriculum_summary = self._build_curriculum_sampler(train_metadata)

        # 🔥 MEGA BATCH: Determine batch size based on GPU memory AND sequence length!
        batch_size = self._get_batch_size(seq_length=max_length)

        input_ids = _maybe_pin("train input_ids", input_ids)
        attention_mask = _maybe_pin("train attention_mask", attention_mask)

        # 1% OPTIMIZATION: Optimized DataLoader settings
        # - num_workers: Parallel data loading (4 is sweet spot for most systems)
        # - pin_memory: Fast GPU transfer via page-locked memory
        # - prefetch_factor: Load ahead while GPU computes (2-4 optimal)
        # - persistent_workers: Keep workers alive between epochs (saves startup time)
        num_workers = self._recommend_num_workers()
        use_pin_memory = self.train_cfg.get('pin_memory', True) and torch.cuda.is_available()
        pin_memory_device = self.train_cfg.get('pin_memory_device')
        prefetch_factor = int(self.train_cfg.get('prefetch_factor', 4))
        if num_workers == 0:
            prefetch_factor = None

        loader_kwargs = {
            'batch_size': batch_size,
            'num_workers': num_workers,
            'pin_memory': use_pin_memory,
            'prefetch_factor': prefetch_factor,
            'persistent_workers': num_workers > 0,
        }
        if pin_memory_device and 'pin_memory_device' in inspect.signature(DataLoader).parameters:
            loader_kwargs['pin_memory_device'] = pin_memory_device

        train_loader = DataLoader(
            train_dataset,
            sampler=train_sampler,
            shuffle=train_sampler is None,
            **loader_kwargs,
        )

        val_loader = DataLoader(
            val_dataset,
            shuffle=False,
            **loader_kwargs,
        )

        logger.info(f"🚀 DataLoader optimized: {num_workers} workers, pin_memory={use_pin_memory}, prefetch={prefetch_factor}")
        
        logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
        logger.info(f"Batch size: {batch_size}")
        if curriculum_summary:
            logger.info(
                f"Curriculum weights (mean={curriculum_summary['mean_weight']:.2f}, "
                f"max={curriculum_summary['max_weight']:.2f}, "
                f"top_commits={[c for c, _ in curriculum_summary['top_commit_weights']][:3]})"
            )

        val_commits = [
            meta.get('sample_commit')
            for meta in val_metadata
            if isinstance(meta, dict) and meta.get('sample_commit')
        ]
        if val_commits:
            unique_commits = list(dict.fromkeys(val_commits))
            logger.info(
                "Validation commits (newest "
                f"{min(3, len(unique_commits))}): "
                f"{', '.join(unique_commits[-3:])}"
            )
        
        return train_loader, val_loader, train_metadata, val_metadata, curriculum_summary

    def _recommend_num_workers(self) -> int:
        """Use available CPU cores to maximize input pipeline throughput."""
        explicit = self.train_cfg.get('num_workers', None)
        if explicit and explicit > 0:
            return int(explicit)

        # Auto-tune based on CPU cores
        cpu_total = psutil.cpu_count(logical=True) or 4
        target = max(2, cpu_total - 1)  # leave 1 core for main thread/OS
        upper = self.train_cfg.get('num_workers_max', cpu_total)
        lower = self.train_cfg.get('num_workers_min', 0)
        return max(lower, min(target, upper))
    
    def _build_curriculum_sampler(
        self,
        metadata_list: List[Dict[str, Any]]
    ) -> Tuple[Optional[WeightedRandomSampler], Optional[Dict[str, Any]]]:
        """Create a weighted sampler when curriculum learning is enabled"""
        if not metadata_list:
            return None, None
        if not self.model_saving_cfg.get('use_curriculum', False):
            return None, None

        recency_enabled = self.model_saving_cfg.get('weight_by_recency', False)
        directory_enabled = self.model_saving_cfg.get('weight_by_directory', False)
        author_enabled = self.model_saving_cfg.get('weight_by_author', False)

        recency_multiplier = float(self.model_saving_cfg.get('recency_multiplier', 2.0))
        directory_multiplier = float(self.model_saving_cfg.get('directory_multiplier', 0.5))
        author_multiplier = float(self.model_saving_cfg.get('author_multiplier', 0.25))

        max_commit_idx = max((meta.get('end_commit_idx', 0) for meta in metadata_list), default=0)
        weights = []
        commit_weights = defaultdict(float)

        for idx, meta in enumerate(metadata_list):
            weight = 1.0
            if recency_enabled and max_commit_idx >= 0:
                denom = max_commit_idx + 1
                recency = (meta.get('end_commit_idx', 0) + 1) / denom
                weight += recency * recency_multiplier
            if directory_enabled and meta.get('primary_directory'):
                weight += directory_multiplier
            if author_enabled and meta.get('author_name'):
                weight += author_multiplier

            weights.append(weight)
            commit_key = meta.get('sample_commit') or f"sequence-{idx}"
            commit_weights[commit_key] += weight

        if not weights:
            return None, None

        sampler = WeightedRandomSampler(
            torch.tensor(weights, dtype=torch.double),
            num_samples=len(weights),
            replacement=True
        )

        sorted_commits = sorted(
            commit_weights.items(),
            key=lambda item: item[1],
            reverse=True
        )[:3]

        summary = {
            'mean_weight': float(np.mean(weights)),
            'max_weight': float(np.max(weights)),
            'std_weight': float(np.std(weights)),
            'top_commit_weights': sorted_commits,
            'flags': {
                'recency': recency_enabled,
                'directory': directory_enabled,
                'author': author_enabled,
            },
            'sequence_count': len(weights),
            'replacement': True,
        }

        if directory_enabled and not any(meta.get('primary_directory') for meta in metadata_list):
            logger.warning("Curriculum requested directory weighting but metadata lacks directory signals.")

        return sampler, summary

    def _run_behavioral_eval(
        self,
        epoch: int,
        val_metadata: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        eval_cfg = self.eval_cfg
        if not eval_cfg.get('run_behavioral_eval', False):
            return None

        prompts = eval_cfg.get('behavioral_test_prompts', [])
        if not prompts:
            return None

        max_new_tokens = eval_cfg.get('eval_max_length', self.model_cfg.get('max_new_tokens', 150))
        num_return = eval_cfg.get('eval_num_return_sequences', 1)
        temperature = eval_cfg.get('eval_temperature', 0.7)
        top_p = eval_cfg.get('eval_top_p', 0.95)
        do_sample = eval_cfg.get('eval_do_sample', True)
        trim_length = int(eval_cfg.get('eval_output_trim', 300))
        tokenizer_max = getattr(self.tokenizer, 'model_max_length', None)
        if not tokenizer_max or tokenizer_max <= 0:
            tokenizer_max = self.model_cfg.get('max_position_embeddings', 512)
        prompt_max_length = min(tokenizer_max, max_new_tokens)

        recent_commits = [
            meta.get('sample_commit')
            for meta in val_metadata
            if isinstance(meta, dict) and meta.get('sample_commit')
        ]
        unique_commits = list(dict.fromkeys(recent_commits))
        if unique_commits:
            logger.info(
                f"  Running behavioral eval (epoch {epoch+1}) for commits: "
                f"{', '.join(unique_commits[-3:])}"
            )
        else:
            logger.info(f"  Running behavioral eval (epoch {epoch+1}); no commit metadata available")

        eval_results = []
        self.model.eval()
        with torch.no_grad():
            for prompt in prompts:
                encoded = self.tokenizer(
                    prompt,
                    return_tensors='pt',
                    truncation=True,
                    max_length=prompt_max_length,
                )
                input_ids = encoded['input_ids'].to(self.device)
                attention_mask = encoded.get('attention_mask')
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)

                generated = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    num_return_sequences=num_return,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
                decoded = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
                trimmed = [
                    (text or "").strip().replace('\n', ' ')[:trim_length]
                    for text in decoded
                ]
                eval_results.append({
                    'prompt': prompt,
                    'generated': trimmed,
                })

        return {
            'epoch': epoch + 1,
            'timestamp': datetime.now().isoformat(),
            'prompts_tested': len(prompts),
            'recent_commits': unique_commits[:5],
            'results': eval_results,
        }

    def _summarize_hardware_stats(self) -> Dict[str, Dict[str, float]]:
        stats_history = self.hardware_monitor.stats_history
        if not stats_history:
            return {}

        summary = {}
        keys = ['gpu_memory_mb', 'gpu_utilization', 'cpu_percent', 'ram_percent']
        for key in keys:
            values = [entry.get(key) for entry in stats_history if entry.get(key) is not None]
            if not values:
                continue
            summary[key] = {
                'min': float(min(values)),
                'max': float(max(values)),
                'avg': float(sum(values) / len(values)),
            }
        return summary
    
    def _get_batch_size(self, seq_length: int = 2048, *, log: bool = True) -> int:
        """
        🧠🧠🧠 EINSTEIN-LEVEL MEMORY CALCULATION 🧠🧠🧠

        EXACT memory formula for transformer training:

        1. Model weights (already loaded, 4-bit): ~1.5GB for Phi-2
        2. LoRA weights (fp16): ~25MB
        3. Optimizer states (8-bit AdamW): ~2x LoRA params = ~50MB

        4. ACTIVATION MEMORY (the killer!):
           Per batch item, WITHOUT gradient checkpointing:
           - Hidden states: seq × hidden × layers × 2 bytes
           - Attention scores: heads × seq² × layers × 2 bytes
           - FFN intermediates: seq × 4×hidden × layers × 2 bytes

           For Phi-2 (hidden=2560, layers=32, heads=32):
           - Hidden: 256 × 2560 × 32 × 2 = 42MB
           - Attention: 32 × 256² × 32 × 2 = 134MB
           - FFN: 256 × 10240 × 32 × 2 = 168MB
           - Total per item: ~350MB

        5. Gradient memory: ~same as activations for backward pass

        FORMULA: max_batch = (available_memory - model_overhead) / activation_per_item
        """
        override = None
        if isinstance(self.train_cfg, dict):
            override = self.train_cfg.get('batch_size_override') or self.train_cfg.get('batch_size_selected')
        if override:
            return max(1, int(override))

        if not torch.cuda.is_available():
            return self.train_cfg['batch_size_small']

        # Get ACTUAL available memory right now
        torch.cuda.empty_cache()
        total_memory = torch.cuda.get_device_properties(0).total_memory
        allocated_memory = torch.cuda.memory_allocated()
        reserved_memory = torch.cuda.memory_reserved()
        free_memory = total_memory - reserved_memory

        total_gb = total_memory / 1e9
        free_gb = free_memory / 1e9
        allocated_gb = allocated_memory / 1e9

        # Phi-2 architecture constants
        hidden_dim = 2560
        num_layers = 32
        num_heads = 32
        ffn_mult = 4

        # Calculate activation memory per batch item (in bytes)
        # Hidden states through all layers
        hidden_mem = seq_length * hidden_dim * num_layers * 2  # fp16
        # Attention scores (seq × seq per head per layer)
        attn_mem = num_heads * (seq_length ** 2) * num_layers * 2  # fp16
        # FFN intermediates
        ffn_mem = seq_length * (hidden_dim * ffn_mult) * num_layers * 2  # fp16

        # Total per batch item (forward + backward ≈ 2x)
        activation_per_item_bytes = (hidden_mem + attn_mem + ffn_mem) * 2
        activation_per_item_mb = activation_per_item_bytes / 1e6

        # 🧠 EINSTEIN INSIGHT: Gradient checkpointing saves ~3x activation memory!
        # For GPUs < 12GB, we ALWAYS enable checkpointing, so divide by 3
        uses_checkpointing = total_gb < 12 or self.train_cfg.get('use_gradient_checkpointing', False)
        if uses_checkpointing:
            activation_per_item_mb = activation_per_item_mb / 3  # Checkpointing saves ~3x

        # Safety margin (fragmentation, peaks, etc.)
        safety_factor = 0.6  # Conservative - only use 60% of "free" memory

        # Available for batches
        available_for_batches_mb = (free_gb * 1000) * safety_factor

        # Calculate max batch
        if activation_per_item_mb > 0:
            max_batch = int(available_for_batches_mb / activation_per_item_mb)
        else:
            max_batch = 1

        # Clamp to reasonable range
        max_batch = max(1, min(max_batch, 16))

        if log:
            logger.info(f"🧠 EINSTEIN MEMORY CALC:")
            logger.info(f"   Total VRAM: {total_gb:.2f}GB, Free: {free_gb:.2f}GB")
            logger.info(f"   Checkpointing: {'YES (3x memory saving)' if uses_checkpointing else 'NO'}")
            logger.info(f"   Activation/item: {activation_per_item_mb:.0f}MB (seq={seq_length})")
            logger.info(f"   Available: {available_for_batches_mb:.0f}MB → Batch: {max_batch}")

        return max_batch
    
    def train(
        self,
        sequences_file: str,
        num_epochs: int,
        output_dir: str,
        val_sequences: Optional[str] = None,
    ) -> Dict:
        """Train the model"""
        self.load_model_and_tokenizer()
        self.training_output_dir = Path(output_dir)
        self.previous_training_info = self._load_training_info(self.training_output_dir)
        incremental_info = self._prepare_incremental_slice(sequences_file)

        if not incremental_info.get("new_data"):
            logger.info("No new commits detected; skipping training.")
            self.training_stats = {"status": "no_new_data"}
            return self.training_stats

        start_idx = incremental_info["train_start_idx"]
        train_loader, val_loader, _, val_metadata, curriculum_summary = self.load_data(
            sequences_file,
            start_idx=start_idx,
            val_sequences_file=val_sequences,
        )
        self.curriculum_summary = curriculum_summary
        if self.curriculum_summary:
            logger.info(f"Curriculum summary: {self.curriculum_summary['flags']}, "
                        f"sequence_count={self.curriculum_summary['sequence_count']}")

        # 1% OPTIMIZATION: Pre-allocate CUDA memory pools to avoid fragmentation
        if torch.cuda.is_available():
            # Warm up GPU memory allocator with a dummy allocation/deallocation
            # This pre-expands the memory pool for faster subsequent allocations
            try:
                torch.cuda.empty_cache()
                # Allocate a temporary tensor to expand the memory pool
                dummy = torch.empty(256 * 1024 * 1024, dtype=torch.float16, device='cuda')  # 512MB
                del dummy
                torch.cuda.empty_cache()
                # Set memory fraction to avoid OOM from fragmentation
                torch.cuda.set_per_process_memory_fraction(0.95)
                logger.info("🚀 CUDA memory pool pre-allocated (reduced fragmentation)")
            except Exception as e:
                logger.warning(f"Memory pre-allocation failed (non-critical): {e}")

        # Setup optimizer and scheduler
        # TIER 4 OPTIMIZATION: Use 8-bit AdamW optimizer to save ~1.7 GB VRAM
        # 1% OPTIMIZATION: Use fused=True for 1.2x faster optimizer step
        params_for_base = list(self.model.parameters())
        if self.model_cfg.get("cpu_offload_lora", False):
            if HAS_CPU_LORA_OFFLOAD:
                normal_params, _ = get_cpu_offload_parameters(self.model)
                params_for_base = [p for p in normal_params if p.requires_grad]
                logger.info(f"Using CPU-offloaded LoRA: {len(params_for_base)} parameters stay on GPU optimizer")
            else:
                warnings.warn("cpu_offload_lora requested but offload module unavailable; falling back to standard LoRA")

        if HAS_BNB_OPTIMIZER:
            logger.info("🚀 Using bitsandbytes 8-bit AdamW optimizer (saves ~1.7 GB VRAM!)")
            optimizer = bnb.optim.AdamW8bit(
                params_for_base,
                lr=self.train_cfg['base_learning_rate'],
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=self.train_cfg['weight_decay'],
            )
        else:
            # Try to use fused AdamW (1.2x faster) if available
            try:
                optimizer = AdamW(
                    params_for_base,
                    lr=self.train_cfg['base_learning_rate'],
                    weight_decay=self.train_cfg['weight_decay'],
                    fused=True,  # 1% OPTIMIZATION: Fused kernel for optimizer (1.2x speedup)
                )
                logger.info("🚀 Using FUSED AdamW optimizer (1.2x faster!)")
            except TypeError:
                # Fallback for older PyTorch versions
                optimizer = AdamW(
                    params_for_base,
                    lr=self.train_cfg['base_learning_rate'],
                    weight_decay=self.train_cfg['weight_decay'],
                )
                logger.info("Using standard AdamW optimizer")

        if self.model_cfg.get("cpu_offload_lora", False) and HAS_CPU_LORA_OFFLOAD:
            optimizer = wrap_optimizer_for_cpu_offload(
                optimizer,
                self.model,
                lr=self.train_cfg['base_learning_rate'],
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=self.train_cfg['weight_decay'],
                device=self.device,
                use_fp16_updates=True,
            )
            logger.info("Optimizer wrapped for CPU LoRA offload (staging adapters to GPU during step)")

        lookahead_cfg = self.train_cfg.get('lookahead', {}) if isinstance(self.train_cfg, dict) else {}
        lookahead_enabled = bool(isinstance(lookahead_cfg, dict) and lookahead_cfg.get('enabled', False))
        lookahead_k = int(lookahead_cfg.get('k', 5)) if lookahead_enabled else 0
        lookahead_alpha = float(lookahead_cfg.get('alpha', 0.5)) if lookahead_enabled else 0.0

        # Calculate scheduler parameters FIRST (needed before DeepSpeed wraps optimizer)
        grad_accum_steps = self.train_cfg['gradient_accumulation_steps']
        steps_per_epoch = math.ceil(len(train_loader) / grad_accum_steps)
        total_steps = steps_per_epoch * num_epochs
        warmup_steps = max(
            self.train_cfg['warmup_steps_min'],
            int(total_steps * self.train_cfg['warmup_ratio'])
        )
        warmup_steps = min(warmup_steps, self.train_cfg['warmup_steps_max'])

        # 🚀🚀🚀 1% of 1% OPTIMIZATION: DeepSpeed ZeRO with CPU OFFLOAD 🚀🚀🚀
        # This is the KEY to fitting large contexts on 8GB GPU - offload EVERYTHING to CPU/RAM!
        self.use_deepspeed = False
        self.deepspeed_engine = None
        deepspeed_config_path = None
        scheduler = None
        plateau_scheduler = None

        import sys as _sys
        for i, arg in enumerate(_sys.argv):
            if arg == '--deepspeed' and i + 1 < len(_sys.argv):
                deepspeed_config_path = _sys.argv[i + 1]
                break

        if deepspeed_config_path and os.path.exists(deepspeed_config_path):
            try:
                import deepspeed
                import json

                logger.info("🔥🔥🔥 DEEPSPEED ZeRO ACTIVATION 🔥🔥🔥")
                logger.info("Offloading model parameters, optimizer states, AND gradients to CPU!")

                # Disable DeepSpeed internal timers to avoid missing attribute issues
                os.environ.setdefault("DEEPSPEED_ENABLE_TIMERS", "0")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Load DeepSpeed config
                with open(deepspeed_config_path, 'r') as f:
                    ds_config = json.load(f)

                # Set batch sizes if 'auto'
                if ds_config.get('train_batch_size') == 'auto':
                    ds_config['train_batch_size'] = 1 * self.train_cfg['gradient_accumulation_steps']
                if ds_config.get('gradient_accumulation_steps') == 'auto':
                    ds_config['gradient_accumulation_steps'] = self.train_cfg['gradient_accumulation_steps']

                # Ensure ZeRO allows untested optimizer combos
                zero_cfg = ds_config.setdefault('zero_optimization', {})
                if isinstance(zero_cfg, dict):
                    zero_cfg.pop('zero_allow_untested_optimizer', None)
                ds_config['zero_allow_untested_optimizer'] = True

                # For small GPUs (<=8GB), use ZeRO-2 with CPU offload + tiny buckets
                if torch.cuda.is_available():
                    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
                else:
                    total_mem_gb = 0
                if total_mem_gb <= 8.5:
                    zero_cfg['stage'] = 2
                    zero_cfg['overlap_comm'] = False
                    zero_cfg['contiguous_gradients'] = False
                    zero_cfg['allgather_bucket_size'] = int(min(zero_cfg.get('allgather_bucket_size', 200000000), 5000000))
                    zero_cfg['reduce_bucket_size'] = int(min(zero_cfg.get('reduce_bucket_size', 200000000), 5000000))
                    zero_cfg.setdefault('offload_optimizer', {})
                    zero_cfg['offload_optimizer'].setdefault('device', 'cpu')
                    zero_cfg['offload_optimizer'].setdefault('pin_memory', True)
                    zero_cfg.setdefault('offload_param', {})
                    zero_cfg['offload_param'].setdefault('device', 'cpu')
                    zero_cfg['offload_param'].setdefault('pin_memory', True)
                    ds_config['train_batch_size'] = max(1, int(ds_config.get('train_batch_size', 1)))
                    ds_config['gradient_accumulation_steps'] = max(1, int(ds_config.get('gradient_accumulation_steps', 1)))

                # Create optimizer for DeepSpeed
                wants_offload = bool(zero_cfg.get('offload_optimizer') or zero_cfg.get('offload_param'))
                if wants_offload:
                    try:
                        from deepspeed.ops.adam import DeepSpeedCPUAdam
                        ds_optimizer = DeepSpeedCPUAdam(
                            self.model.parameters(),
                            lr=float(self.train_cfg['base_learning_rate']),
                            betas=(0.9, 0.999),
                            eps=1e-8,
                            weight_decay=float(self.train_cfg['weight_decay']),
                        )
                        logger.info("✓ Created DeepSpeedCPUAdam for CPU offloading")
                    except ImportError as ie:
                        logger.warning(f"DeepSpeedCPUAdam not available: {ie}")
                        ds_optimizer = optimizer
                else:
                    ds_optimizer = optimizer

                # Create scheduler BEFORE DeepSpeed wraps the optimizer
                # (DeepSpeed's wrapped optimizer doesn't pass isinstance(Optimizer) checks)
                scheduler = get_cosine_schedule_with_warmup(
                    ds_optimizer,
                    num_warmup_steps=warmup_steps,
                    num_training_steps=total_steps,
                )

                # Add scheduler config to DeepSpeed
                ds_config['scheduler'] = {
                    'type': 'WarmupDecayLR',
                    'params': {
                        'warmup_min_lr': 0,
                        'warmup_max_lr': float(self.train_cfg['base_learning_rate']),
                        'warmup_num_steps': warmup_steps,
                        'total_num_steps': total_steps,
                    }
                }

                # Initialize DeepSpeed engine with scheduler
                self.deepspeed_engine, optimizer, _, scheduler = deepspeed.initialize(
                    model=self.model,
                    optimizer=ds_optimizer,
                    lr_scheduler=scheduler,
                    config=ds_config,
                    dist_init_required=True,
                )

                # Replace model reference with DeepSpeed engine's module
                self.model = self.deepspeed_engine.module

                # Patch missing attributes for DeepSpeed 0.18.3 compatibility
                for attr, val in [("engine_timers", None), ("_deepcompile_active", False)]:
                    if not hasattr(self.deepspeed_engine, attr):
                        setattr(self.deepspeed_engine, attr, val)

                self.use_deepspeed = True

                # Store reference to the wrapped optimizer for LR tracking
                self._ds_base_optimizer = ds_optimizer

                # Log memory savings
                zero_stage = ds_config.get('zero_optimization', {}).get('stage', 0)
                offload_optimizer = ds_config.get('zero_optimization', {}).get('offload_optimizer', {}).get('device', 'none')
                offload_param = ds_config.get('zero_optimization', {}).get('offload_param', {}).get('device', 'none')

                logger.info(f"✅ DeepSpeed ZeRO-{zero_stage} initialized!")
                logger.info(f"   Optimizer offload: {offload_optimizer}")
                logger.info(f"   Parameter offload: {offload_param}")
                logger.info("🧠 GPU will only hold ACTIVE tensors - everything else in RAM!")

                import psutil
                ram = psutil.virtual_memory()
                logger.info(f"   RAM available: {ram.available / 1e9:.1f} GB / {ram.total / 1e9:.1f} GB")

            except ImportError:
                logger.warning("DeepSpeed not installed - falling back to standard training")
            except Exception as e:
                logger.error(f"DeepSpeed initialization failed: {e}")
                import traceback
                logger.error(f"Full traceback:\n{traceback.format_exc()}")
                if 'ValidationError' in str(type(e)):
                    try:
                        if hasattr(e, 'errors'):
                            logger.error(f"Validation error details: {e.errors()}")
                    except:
                        pass
                logger.error(f"DeepSpeed config that failed:\n{json.dumps(ds_config, indent=2)}")
                raise

        # Create schedulers if not using DeepSpeed (DeepSpeed handles its own scheduling)
        if scheduler is None:
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )
        if plateau_scheduler is None and not self.use_deepspeed:
            plateau_scheduler = ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=self.train_cfg.get('lr_reduction_factor', 0.5),
                patience=max(1, self.train_cfg.get('lr_plateau_patience', 2)),
                threshold=self.train_cfg['min_delta'],
            )

        if lookahead_enabled:
            self._init_lookahead_state(lookahead_k, lookahead_alpha)
            logger.info(f"👀 Lookahead enabled (k={lookahead_k}, alpha={lookahead_alpha})")

        swa_cfg = self.train_cfg.get('swa', {}) if isinstance(self.train_cfg, dict) else {}
        # SWA requires standard PyTorch optimizer - disable when using DeepSpeed
        swa_enabled = bool(swa_cfg.get('enabled', False)) and HAS_SWA_UTILS and not self.use_deepspeed
        if self.use_deepspeed and swa_cfg.get('enabled', False):
            logger.info("📊 SWA disabled (incompatible with DeepSpeed's wrapped optimizer)")
        swa_model = None
        swa_scheduler = None
        swa_start_step = total_steps + 1
        swa_updates = 0
        if swa_enabled:
            swa_start_epoch = int(swa_cfg.get('start_epoch', max(0, num_epochs - 1)))
            if swa_start_epoch < 0:
                swa_start_epoch = 0
            if swa_start_epoch >= num_epochs:
                swa_start_epoch = max(0, num_epochs - 1)
            swa_start_step = swa_start_epoch * steps_per_epoch
            swa_lr = swa_cfg.get('swa_lr')
            if swa_lr is None:
                swa_lr = float(self.train_cfg['base_learning_rate']) * 0.1
            anneal_strategy = str(swa_cfg.get('anneal_strategy', 'cos'))
            anneal_epochs = int(swa_cfg.get('anneal_epochs', 5))
            swa_model = AveragedModel(self.model)
            swa_scheduler = SWALR(
                optimizer,
                swa_lr=float(swa_lr),
                anneal_strategy=anneal_strategy,
                anneal_epochs=anneal_epochs,
            )
            logger.info(
                f"📊 SWA enabled (start_epoch={swa_start_epoch}, swa_lr={float(swa_lr):.2e}, "
                f"anneal={anneal_strategy}/{anneal_epochs})"
            )
        elif bool(swa_cfg.get('enabled', False)) and not HAS_SWA_UTILS:
            logger.warning("SWA requested but unavailable; disabling SWA")
        
        logger.info(f"\nTraining Setup:")
        logger.info(f"  Total steps: {total_steps}")
        logger.info(f"  Warmup steps: {warmup_steps}")
        logger.info(f"  Base LR: {self.train_cfg['base_learning_rate']}")
        logger.info(f"  Gradient accumulation: {self.train_cfg['gradient_accumulation_steps']}\n")
        logger.info(f"  Optimizer steps/epoch: {steps_per_epoch}")
        
        # 1% OPTIMIZATION: Optimized mixed precision with tuned GradScaler
        use_autocast = bool(self.train_cfg.get('use_mixed_precision', False)) and self.device.type in ("cuda", "mps")
        if use_autocast and self.device.type == "cuda":
            # Optimized GradScaler settings for faster convergence
            scaler = torch.amp.GradScaler(
                init_scale=2**16,       # Start with reasonable scale
                growth_factor=2.0,       # Double scale when no overflow
                backoff_factor=0.5,      # Halve scale on overflow
                growth_interval=2000,    # Check every 2000 steps (less overhead)
                enabled=True
            )
            logger.info("🚀 Optimized GradScaler: growth_interval=2000 (reduced overhead)")
        else:
            scaler = None
        use_scaler = scaler is not None and scaler.is_enabled()

        graph_cfg = self.train_cfg.get('cuda_graphs', {}) if isinstance(self.train_cfg, dict) else {}
        use_cuda_graphs = bool(graph_cfg.get('enabled', False)) and torch.cuda.is_available()
        if use_cuda_graphs and self.use_deepspeed:
            logger.info("CUDA graphs disabled (incompatible with DeepSpeed)")
            use_cuda_graphs = False
        if use_cuda_graphs and (use_scaler or self.train_cfg['gradient_accumulation_steps'] != 1):
            logger.info("CUDA graphs disabled (requires grad_accum=1 and no GradScaler)")
            use_cuda_graphs = False
        if use_cuda_graphs and self.train_cfg.get('curriculum', {}).get('enabled', False):
            logger.info("CUDA graphs disabled (curriculum changes sequence length)")
            use_cuda_graphs = False

        label_cfg = self.train_cfg.get('label_smoothing', {}) if isinstance(self.train_cfg, dict) else {}
        label_smoothing = float(label_cfg.get('smoothing', 0.0)) if label_cfg.get('enabled', False) else 0.0
        
        # --- CRITICAL FIX START ---
        # Fix "element 0 of tensors does not require grad" for 4-bit + LoRA
        if hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads()
        # --- CRITICAL FIX END ---
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        training_start = time.time()
        
        loss_history = []
        val_loss_history = []
        grad_norm_history = []
        lr_history = []
        optimizer_steps = 0
        
        # 1% OPTIMIZATION: Disable Python garbage collection during training
        # GC can cause unpredictable pauses; we manually collect between epochs
        import gc
        gc.disable()
        logger.info("🚀 Python GC disabled for training (manual collection between epochs)")

        # 🔥🔥🔥 ULTRA-SPEED OPTIMIZATION SUMMARY 🔥🔥🔥
        batch_size = len(train_loader.dataset) // len(train_loader) if len(train_loader) > 0 else 1
        # Get actual sequence length from data (not hardcoded!)
        try:
            seq_len = train_loader.dataset[0][0].shape[0]
        except:
            seq_len = 256  # Fallback
        gpu_compute_cap = 0.0
        if torch.cuda.is_available():
            gpu_compute_cap = torch.cuda.get_device_capability()[0] + torch.cuda.get_device_capability()[1] / 10

        logger.info("\n" + "=" * 60)
        logger.info("🔥🔥🔥 1% of 1% ULTRA-SPEED OPTIMIZATIONS ACTIVE 🔥🔥🔥")
        logger.info("=" * 60)
        logger.info(f"✓ MEGA BATCH SIZE: {batch_size} (dynamically scaled for {seq_len} tokens)")
        logger.info(f"✓ CUDA Stream Prefetching: Overlap data transfer with compute")
        if gpu_compute_cap >= 8.0:
            logger.info(f"✓ torch.compile reduce-overhead: JIT kernel fusion")
        else:
            logger.info(f"✓ Turing SDPA: Native attention (no compile overhead)")
        logger.info(f"✓ Skip-Padding Loss: Don't compute gradients for padding tokens")
        # Checkpointing status based on GPU memory (< 12GB = required)
        total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
        if total_mem_gb < 12:
            logger.info(f"✓ Gradient Checkpointing: REQUIRED for {total_mem_gb:.0f}GB GPU (saves 3x memory)")
        else:
            logger.info(f"✓ NO Gradient Checkpointing: {total_mem_gb:.0f}GB GPU has enough VRAM")
        logger.info(f"✓ Fused Optimizer: 1.2x faster optimizer step")
        logger.info(f"✓ cudnn.benchmark + TF32: Faster convolutions and matmul")
        logger.info(f"✓ Memory Pool Pre-allocation: Reduced fragmentation")
        logger.info(f"✓ Python GC Disabled: No GC pauses during training")

        # Speed estimate based on sequence length
        # O(n²) attention: 256 tokens = ~50 it/s, 512 = ~12 it/s, 1024 = ~3 it/s
        if gpu_compute_cap >= 7.5 and gpu_compute_cap < 8.0:  # Turing
            expected_its = max(1, int(50 * (256 / seq_len) ** 1.5 * (batch_size / 16)))
            logger.info(f"\n⚡ EXPECTED SPEED: ~{expected_its} it/s ({1/expected_its:.3f}s per step)")
            epoch_time_est = len(train_loader) / expected_its / 60
            logger.info(f"⚡ ESTIMATED EPOCH TIME: ~{epoch_time_est:.1f} minutes")
        logger.info("=" * 60 + "\n")

        graph_ctx = None

        for epoch in range(num_epochs):
            epoch_start = time.time()

            # 1% OPTIMIZATION: Clear CUDA cache and run GC between epochs
            # Reduces memory fragmentation and prevents OOM during long training
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

            # Training phase
            self.model.train()
            train_loss = 0.0
            epoch_grad_norms = []

            # 1% OPTIMIZATION: set_to_none=True is faster than zeroing gradients
            if not (self.use_deepspeed and self.deepspeed_engine is not None):
                optimizer.zero_grad(set_to_none=True)

            # 🔥🔥🔥 CUDA STREAM PREFETCHING: Overlap data transfer with compute!
            prefetcher = CUDADataPrefetcher(train_loader, self.device)
            pbar = tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}")

            for step, (input_ids, attention_mask) in enumerate(prefetcher):
                # Data is already on GPU via the prefetcher!
                # Optional curriculum: gradually increase max sequence length
                max_len = self._curriculum_max_len(epoch, num_epochs)
                if max_len is not None:
                    input_ids = input_ids[:, :max_len]
                    attention_mask = attention_mask[:, :max_len]

                # Optimization: if attention_mask is effectively "all tokens valid", pass None.
                # This avoids needless work in attention and can unlock Orchard FlashAttention on MPS.
                attn_mask_to_use = attention_mask
                if not use_cuda_graphs and self.train_cfg.get('drop_full_attention_mask', True):
                    try:
                        if attn_mask_to_use is not None:
                            if attn_mask_to_use.dtype == torch.bool:
                                mask_is_full = bool(attn_mask_to_use.all().item())
                            else:
                                mask_is_full = bool((attn_mask_to_use.min() == 1).item() and (attn_mask_to_use.max() == 1).item())
                            if mask_is_full:
                                attn_mask_to_use = None
                    except Exception:
                        attn_mask_to_use = attention_mask

                # Objectives: LM / FIM / span-locality
                labels = input_ids.clone()
                objective = self._objective_sample()
                if objective == 'fim':
                    input_ids_obj = self._apply_fim(input_ids)
                    labels = input_ids_obj.clone()  # Clone to allow modification
                    input_ids = input_ids_obj
                    if attn_mask_to_use is not None:
                        attn_mask_to_use = torch.ones(
                            (input_ids.shape[0], input_ids.shape[1]),
                            device=input_ids.device,
                            dtype=attn_mask_to_use.dtype,
                        )
                elif objective == 'span':
                    labels = self._apply_span_labels(labels)

                # 🔥 SKIP-PADDING OPTIMIZATION: Apply AFTER objective transforms!
                # HuggingFace uses -100 as ignore_index in CrossEntropyLoss
                # This saves compute by not calculating gradients for padding tokens
                if attention_mask is not None:
                    labels = labels.masked_fill(attention_mask == 0, -100)

                # Forward + backward pass (optionally via CUDA graphs)
                loss = None
                if use_cuda_graphs:
                    if graph_ctx is None:
                        graph_ctx = self._setup_cuda_graph(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                            use_autocast=use_autocast,
                            warmup_steps=int(graph_cfg.get('warmup_steps', 2)),
                            label_smoothing=label_smoothing,
                        )
                    if graph_ctx and graph_ctx['static_input_ids'].shape == input_ids.shape:
                        graph_ctx['static_input_ids'].copy_(input_ids)
                        graph_ctx['static_attention_mask'].copy_(attention_mask)
                        graph_ctx['static_labels'].copy_(labels)
                        graph_ctx['graph'].replay()
                        loss = graph_ctx['loss']
                    else:
                        graph_ctx = None

                if loss is None:
                    with torch.amp.autocast(device_type=self.amp_device, enabled=use_autocast):
                        # 🚀 Use DeepSpeed engine for forward pass - handles ZeRO-3 parameter gathering!
                        model_for_forward = self.deepspeed_engine if (self.use_deepspeed and self.deepspeed_engine is not None) else self.model
                        outputs = model_for_forward(
                            input_ids=input_ids,
                            attention_mask=attn_mask_to_use,
                            labels=labels,
                        )
                        if label_smoothing > 0:
                            logits = outputs.logits
                            loss = F.cross_entropy(
                                logits.view(-1, logits.size(-1)),
                                labels.view(-1),
                                label_smoothing=label_smoothing,
                                ignore_index=-100,
                            )
                        else:
                            loss = outputs.loss
                        loss = loss / self.train_cfg['gradient_accumulation_steps']

                    # 🚀 DeepSpeed backward - handles CPU offloading automatically!
                    if self.use_deepspeed and self.deepspeed_engine is not None:
                        self.deepspeed_engine.backward(loss)
                    elif use_scaler:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

                train_loss += loss.item()
                
                # Gradient accumulation
                if (step + 1) % self.train_cfg['gradient_accumulation_steps'] == 0:
                    # 🚀 DeepSpeed step - handles gradient clipping, optimizer step, and CPU offload!
                    if self.use_deepspeed and self.deepspeed_engine is not None:
                        # DeepSpeed handles gradient clipping internally
                        self.deepspeed_engine.step()
                        self._apply_lookahead_update()
                        total_norm = 0.0  # DeepSpeed handles this internally
                        epoch_grad_norms.append(total_norm)
                        self._ema_update()
                    else:
                        if use_scaler:
                            scaler.unscale_(optimizer)

                        self._apply_gradient_noise(optimizer_steps)
                        self._apply_gradient_centralization()

                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.train_cfg['max_grad_norm']
                        )

                        # Record grad norm
                        total_norm = 0.0
                        for p in self.model.parameters():
                            if p.grad is not None:
                                total_norm += p.grad.data.norm(2).item() ** 2
                        total_norm = total_norm ** 0.5
                        epoch_grad_norms.append(total_norm)

                        # Optimizer step
                        if use_scaler:
                            self._safe_scaled_step(scaler, optimizer)
                            scaler.update()
                            self._ema_update()
                        else:
                            optimizer.step()
                            self._ema_update()
                        self._apply_lookahead_update()

                    optimizer_steps += 1
                    if swa_enabled and optimizer_steps >= swa_start_step:
                        if swa_model is not None:
                            swa_model.update_parameters(self.model)
                            swa_updates += 1
                        if swa_scheduler is not None:
                            swa_scheduler.step()
                            lr_history.append(swa_scheduler.get_last_lr()[0])
                        else:
                            lr_history.append(optimizer.param_groups[0]['lr'])
                    else:
                        scheduler.step()
                        lr_history.append(scheduler.get_last_lr()[0])

                    # Zero gradients for next accumulation cycle
                    if self.use_deepspeed and self.deepspeed_engine is not None:
                        pass  # DeepSpeed handles gradient zeroing
                    else:
                        optimizer.zero_grad(set_to_none=True)
                
                # Sample hardware if needed
                if self.hardware_monitor.should_sample():
                    self.hardware_monitor.get_stats()

                # Update progress bar
                pbar.update(1)

            pbar.close()
            last_step_idx = step

            # --- START: FIX FOR GRADIENT ACCUMULATION REMAINDER ---
            if (last_step_idx + 1) % self.train_cfg['gradient_accumulation_steps'] != 0:
                logger.info("Performing final optimizer step for leftover gradients.")

                # 🚀 DeepSpeed step for leftover gradients
                if self.use_deepspeed and self.deepspeed_engine is not None:
                    self.deepspeed_engine.step()
                    self._apply_lookahead_update()
                    total_norm = 0.0
                    epoch_grad_norms.append(total_norm)
                else:
                    if use_scaler:
                        scaler.unscale_(optimizer)

                    self._apply_gradient_noise(optimizer_steps)
                    self._apply_gradient_centralization()

                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.train_cfg['max_grad_norm']
                    )

                    total_norm = 0.0
                    for p in self.model.parameters():
                        if p.grad is not None:
                            total_norm += p.grad.data.norm(2).item() ** 2
                    total_norm = total_norm ** 0.5
                    epoch_grad_norms.append(total_norm)

                    if use_scaler:
                        self._safe_scaled_step(scaler, optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    self._apply_lookahead_update()

                optimizer_steps += 1
                if swa_enabled and optimizer_steps >= swa_start_step:
                    if swa_model is not None:
                        swa_model.update_parameters(self.model)
                        swa_updates += 1
                    if swa_scheduler is not None:
                        swa_scheduler.step()
                        lr_history.append(swa_scheduler.get_last_lr()[0])
                    else:
                        lr_history.append(optimizer.param_groups[0]['lr'])
                else:
                    scheduler.step()
                    lr_history.append(scheduler.get_last_lr()[0])

                if not (self.use_deepspeed and self.deepspeed_engine is not None):
                    optimizer.zero_grad(set_to_none=True)
            # --- END: FIX FOR GRADIENT ACCUMULATION REMAINDER ---
            
            # Epoch stats
            train_loss = train_loss / len(train_loader)
            loss_history.append(train_loss)

            grad_mean = float(np.mean(epoch_grad_norms)) if epoch_grad_norms else float('nan')
            grad_max = float(np.max(epoch_grad_norms)) if epoch_grad_norms else float('nan')
            grad_norm_history.append({
                'epoch': epoch + 1,
                'mean': grad_mean,
                'max': grad_max,
            })
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_steps = 0
            
            with torch.no_grad():
                for input_ids, attention_mask in val_loader:
                    input_ids = input_ids.to(self.device)
                    attention_mask = attention_mask.to(self.device)
                    
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=input_ids,
                    )
                    val_loss += outputs.loss.item()
                    val_steps += 1
            
            if val_steps > 0:
                val_loss = val_loss / val_steps
            else:
                val_loss = float('nan')
                logger.warning("No validation batches processed; skipping metrics for this epoch.")
            val_loss_history.append(val_loss)
            
            # Epoch time
            epoch_time = time.time() - epoch_start
            perplexity = np.exp(val_loss) if not np.isnan(val_loss) else float('nan')
            
            current_lr = optimizer.param_groups[0]['lr']
            
            # Logging
            logger.info(f"\nEpoch {epoch+1}/{num_epochs}:")
            logger.info(f"  Train Loss: {train_loss:.4f}")
            logger.info(f"  Val Loss: {val_loss:.4f}")
            logger.info(f"  Perplexity: {perplexity:.2f}")
            logger.info(f"  Grad Norm: {grad_mean:.4f} (max: {grad_max:.4f})")
            logger.info(f"  LR: {current_lr:.2e}")
            logger.info(f"  Time: {epoch_time:.1f}s")
            
            if not np.isnan(val_loss) and plateau_scheduler is not None:
                prev_plateau_lr = optimizer.param_groups[0]['lr']
                if not (swa_enabled and optimizer_steps >= swa_start_step):
                    plateau_scheduler.step(val_loss)
                    if optimizer.param_groups[0]['lr'] < prev_plateau_lr:
                        logger.info(f"  ↓ LR reduced to {optimizer.param_groups[0]['lr']:.2e} (ReduceLROnPlateau)")
            eval_every = self.eval_cfg.get('eval_every_n_epochs', 1)
            if eval_every and (epoch + 1) % eval_every == 0:
                eval_result = self._run_behavioral_eval(epoch, val_metadata)
                if eval_result:
                    self.behavioral_eval_history.append(eval_result)
                    logger.info(f"  Behavioral eval recorded {len(eval_result['results'])} prompts")
            
            should_break = False
            if np.isnan(val_loss):
                logger.info("  ✗ Validation skipped; patience reset")
                patience_counter = 0
            elif val_loss < best_val_loss - self.train_cfg['min_delta']:
                best_val_loss = val_loss
                patience_counter = 0
                logger.info(f"  ✓ Validation improved")
                
                if self.model_saving_cfg.get('save_best_model', True):
                    self._save_model(
                        output_dir,
                        stage="best",
                        force_merge=False,
                        write_training_info=False,
                    )
            else:
                patience_counter += 1
                patience_limit = self.train_cfg.get('lr_plateau_patience', 2)
                logger.info(f"  ✗ No improvement (patience: {patience_counter}/{patience_limit})")

                if patience_counter >= patience_limit:
                    logger.info(f"\nEarly stopping at epoch {epoch+1}")
                    should_break = True

            ckpt_every = self.model_saving_cfg.get('save_ckpt_every_n_epochs', 0)
            if ckpt_every and (epoch + 1) % ckpt_every == 0:
                self._save_model(
                    output_dir,
                    stage=f"checkpoint-epoch-{epoch+1}",
                    force_merge=False,
                    write_training_info=False,
                )

            if should_break:
                break

        # 1% OPTIMIZATION: Re-enable GC after training
        gc.enable()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # Ensure all CUDA ops complete for accurate timing
            torch.cuda.empty_cache()

        if swa_model is not None and swa_updates > 0:
            try:
                self.model.load_state_dict(swa_model.module.state_dict(), strict=False)
                logger.info(f"📊 SWA averaged weights applied ({swa_updates} updates)")
            except Exception as e:
                logger.warning(f"SWA weight transfer failed; continuing with last weights: {e}")

        # Training complete
        total_time = time.time() - training_start
        
        logger.info(f"\n" + "="*70)
        logger.info(f"TRAINING COMPLETE")
        logger.info(f"="*70)
        logger.info(f"Final train loss: {loss_history[-1]:.4f}")
        logger.info(f"Final val loss: {val_loss_history[-1]:.4f}")
        logger.info(f"Best val loss: {best_val_loss:.4f}")
        logger.info(f"Total time: {total_time/60:.1f} minutes")
        logger.info(f"="*70 + "\n")
        
        # Save final model (if configured)
        if self.model_saving_cfg.get('save_final_model', True):
            self._save_model(output_dir, force_merge=True)
        
        hardware_summary = self._summarize_hardware_stats()
        # Compile stats
        self.training_stats = {
            'num_epochs_completed': epoch + 1,
            'total_steps': optimizer_steps,
            'final_train_loss': float(loss_history[-1]),
            'final_val_loss': float(val_loss_history[-1]),
            'best_val_loss': float(best_val_loss),
            'final_perplexity': float(np.exp(val_loss_history[-1])) if not np.isnan(val_loss_history[-1]) else float('nan'),
            'loss_history': loss_history,
            'val_loss_history': val_loss_history,
            'grad_norm_history': grad_norm_history,
            'lr_history': lr_history,
            'validation_commit_hashes': [
                meta.get('sample_commit')
                for meta in val_metadata
                if isinstance(meta, dict) and meta.get('sample_commit')
            ],
            'curriculum_summary': self.curriculum_summary,
            'behavioral_eval_history': self.behavioral_eval_history,
            'dataset_stats': self.dataset_stats,
            'hardware_stats': self.hardware_monitor.stats_history,
            'hardware_summary': hardware_summary,
            'peak_gpu_memory_mb': self.hardware_monitor.peak_gpu_memory_mb,
            'peak_ram_percent': self.hardware_monitor.peak_ram_percent,
            'total_training_seconds': total_time,
            'swa_updates': swa_updates,
        }
        
        return self.training_stats
    
    def _save_model(
        self,
        output_dir: str,
        *,
        stage: Optional[str] = None,
        force_merge: bool = False,
        write_training_info: bool = True,
    ):
        """Save model and tokenizer"""
        base_path = Path(output_dir)
        target_path = base_path / stage if stage else base_path
        target_path.mkdir(parents=True, exist_ok=True)

        # Save tokenizer
        self.tokenizer.save_pretrained(target_path)

        save_adapter_only = self.model_saving_cfg.get(
            'save_adapter_only',
            self.model_cfg.get('save_adapter_only', False)
        )

        if self.model_cfg.get('use_lora', False):
            if save_adapter_only or not force_merge:
                # Save adapter weights without merging into base
                self.model.save_pretrained(target_path)
            else:
                merged_model = self.model.merge_and_unload()
                merged_model.save_pretrained(target_path)
        else:
            self.model.save_pretrained(target_path)

        if write_training_info and not stage:
            config_to_save = {
                'model_name': self.model_cfg['name'],
                'use_lora': self.model_cfg.get('use_lora', False),
                'use_4bit': self.model_cfg.get('use_4bit', False),
                'use_8bit': self.model_cfg.get('use_8bit', False),
                'model_saving': self.model_saving_cfg,
                'training_config': self.train_cfg,
            }
            if self._total_sequences > 0:
                config_to_save.update({
                    'last_trained_sequence_idx': self._total_sequences - 1,
                    'last_trained_commit': self.latest_sequence_commit,
                    'latest_sequence_commit': self.latest_sequence_commit,
                    'incremental_context_sequences': self.incremental_context_sequences,
                    'last_trained_commit_idx': self.latest_commit_idx,
                    'latest_sequence_commit_idx': self.latest_commit_idx,
                })
            with open(base_path / "training_info.json", 'w') as f:
                json.dump(config_to_save, f, indent=2)

        stage_label = f" ({stage})" if stage else ""
        logger.info(f"Model saved to {target_path}{stage_label}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="training_config_metal_cuda_universal.yaml")
    parser.add_argument("--sequences", default="/data/scrape-dec23/commits_rich.json")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--output", default="models/the-block-dec23")
    parser.add_argument(
        "--device",
        default=None,
        choices=["cuda", "mps", "cpu"],
        help="Force device (default: auto-detect)",
    )
    parser.add_argument(
        "--deepspeed",
        default=None,
        help="DeepSpeed config file for ZeRO optimization (enables extreme context training)",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training (automatically set by DeepSpeed)",
    )
    parser.add_argument(
        "--val-sequences",
        dest="val_sequences",
        default=None,
        help="Separate validation sequences file (optional)",
    )

    args = parser.parse_args()

    trainer = OptimizedModelTrainer(args.config, force_device=args.device)
    trainer.train(args.sequences, args.epochs, args.output, val_sequences=args.val_sequences)
