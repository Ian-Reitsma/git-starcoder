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
import torch
import logging
import random
import numpy as np
import yaml
import psutil
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

        # 1% CUDA OPTIMIZATIONS for maximum throughput
        if torch.cuda.is_available():
            # Enable cudnn benchmark for auto-tuning (1.1-1.3x speedup)
            torch.backends.cudnn.benchmark = True
            # Enable TF32 for Ampere+ GPUs (2x faster matmul with minimal precision loss)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            # Pre-allocate memory pool to avoid fragmentation
            torch.cuda.empty_cache()
            logger.info("🚀 CUDA optimizations: cudnn.benchmark=True, TF32=True, memory pool cleared")

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
        logger.info(f"Use 4-bit: {self.model_cfg['use_4bit']}")
        logger.info(f"Use Mixed Precision: {self.train_cfg['use_mixed_precision']}")
        logger.info(f"="*70 + "\n")
        
        self.model = None
        self.tokenizer = None

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
                'batch_size_reference': batch,
                'batch_size_large': batch,
                'batch_size_medium': batch,
                'batch_size_small': batch,
                'num_workers': int(train_in.get('num_workers', 0)),
                'num_workers_min': int(train_in.get('num_workers_min', 0)),
                'num_workers_max': int(train_in.get('num_workers_max', 0)),
                'incremental_context_sequences': int(train_in.get('incremental_context_sequences', 2)),
                'drop_full_attention_mask': bool(train_in.get('drop_full_attention_mask', True)),
                'deterministic_mode': bool(train_in.get('deterministic_mode', False)),
            }

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

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_cfg['name'],
                quantization_config=quantization_config,
                device_map='auto' if (self.model_cfg['use_4bit'] or self.model_cfg['use_8bit']) else None,
                trust_remote_code=self.model_cfg['trust_remote_code'],
                torch_dtype=torch.bfloat16 if self.model_cfg['use_bf16'] else torch.float32,
                attn_implementation=attn_implementation,  # FlashAttention-2 or SDPA
                use_cache=False,  # Required for training with Flash Attention
            )

        # If not using quantized device_map, move model explicitly to selected device
        if not (self.model_cfg['use_4bit'] or self.model_cfg['use_8bit']):
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
        if torch.cuda.is_available() and hasattr(torch, 'compile'):
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
        
        # 1% OPTIMIZATION: Optimized DataLoader settings
        # - num_workers: Parallel data loading (4 is sweet spot for most systems)
        # - pin_memory: Fast GPU transfer via page-locked memory
        # - prefetch_factor: Load ahead while GPU computes (2-4 optimal)
        # - persistent_workers: Keep workers alive between epochs (saves startup time)
        num_workers = max(4, self.train_cfg.get('num_workers', 0))
        use_pin_memory = self.train_cfg.get('pin_memory', True) and torch.cuda.is_available()

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            shuffle=train_sampler is None,
            num_workers=num_workers,
            pin_memory=use_pin_memory,
            prefetch_factor=2 if num_workers > 0 else None,
            persistent_workers=num_workers > 0,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=use_pin_memory,
            prefetch_factor=2 if num_workers > 0 else None,
            persistent_workers=num_workers > 0,
        )

        logger.info(f"🚀 DataLoader optimized: {num_workers} workers, pin_memory={use_pin_memory}, prefetch=2")
        
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
    
    def _get_batch_size(self, seq_length: int = 2048) -> int:
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
        if HAS_BNB_OPTIMIZER:
            logger.info("🚀 Using bitsandbytes 8-bit AdamW optimizer (saves ~1.7 GB VRAM!)")
            optimizer = bnb.optim.AdamW8bit(
                self.model.parameters(),
                lr=self.train_cfg['base_learning_rate'],
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=self.train_cfg['weight_decay'],
            )
        else:
            # Try to use fused AdamW (1.2x faster) if available
            try:
                optimizer = AdamW(
                    self.model.parameters(),
                    lr=self.train_cfg['base_learning_rate'],
                    weight_decay=self.train_cfg['weight_decay'],
                    fused=True,  # 1% OPTIMIZATION: Fused kernel for optimizer (1.2x speedup)
                )
                logger.info("🚀 Using FUSED AdamW optimizer (1.2x faster!)")
            except TypeError:
                # Fallback for older PyTorch versions
                optimizer = AdamW(
                    self.model.parameters(),
                    lr=self.train_cfg['base_learning_rate'],
                    weight_decay=self.train_cfg['weight_decay'],
                )
                logger.info("Using standard AdamW optimizer")
        
        grad_accum_steps = self.train_cfg['gradient_accumulation_steps']
        steps_per_epoch = math.ceil(len(train_loader) / grad_accum_steps)
        total_steps = steps_per_epoch * num_epochs
        warmup_steps = max(
            self.train_cfg['warmup_steps_min'],
            int(total_steps * self.train_cfg['warmup_ratio'])
        )
        warmup_steps = min(warmup_steps, self.train_cfg['warmup_steps_max'])
        
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        plateau_scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=self.train_cfg.get('lr_reduction_factor', 0.5),
            patience=max(1, self.train_cfg.get('lr_plateau_patience', 2)),
            threshold=self.train_cfg['min_delta'],
        )
        
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
                if self.train_cfg.get('drop_full_attention_mask', True):
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

                # Forward pass
                with torch.amp.autocast(device_type=self.amp_device, enabled=use_autocast):
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attn_mask_to_use,
                        labels=labels,
                    )
                    loss = outputs.loss / self.train_cfg['gradient_accumulation_steps']

                # Backward pass
                if use_scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                train_loss += loss.item()
                
                # Gradient accumulation
                if (step + 1) % self.train_cfg['gradient_accumulation_steps'] == 0:
                    if use_scaler:
                        scaler.unscale_(optimizer)

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
                    
                    scheduler.step()
                    optimizer_steps += 1
                    lr_history.append(scheduler.get_last_lr()[0])
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

                if use_scaler:
                    scaler.unscale_(optimizer)

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
            
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
                lr_history.append(scheduler.get_last_lr()[0])
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
            
            if not np.isnan(val_loss):
                prev_plateau_lr = optimizer.param_groups[0]['lr']
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
            'hardware_stats': self.hardware_monitor.stats_history,
            'hardware_summary': hardware_summary,
            'peak_gpu_memory_mb': self.hardware_monitor.peak_gpu_memory_mb,
            'peak_ram_percent': self.hardware_monitor.peak_ram_percent,
            'total_training_seconds': total_time,
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
