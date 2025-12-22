#!/usr/bin/env python3
"""training/mps_quant_backend.py

MPS-native quantization backend (no bitsandbytes) intended to make StarCoder2-3B
fine-tuning viable on Apple Silicon.

What this actually provides (today):
- A **memory-oriented** path: freeze base weights, store selected Linear weights
  as int8 + per-group symmetric scales, and dequantize to fp16/bf16 on the fly.
- A LoRA adapter per quantized Linear so training updates stay lightweight.

Important constraints:
- This is not a fused int8 GEMM kernel; matmul is still fp16/bf16 after
  dequantization. The win is reduced *storage* (VRAM/unified memory), not compute.
- It targets StarCoder2 / GPT-style modules (c_attn/c_proj/mlp.*) by name.

This backend is designed to be called from model_trainer_unified.py when:
- device is MPS/Metal, AND
- the config requests 4-bit/8-bit (CUDA bitsandbytes) quantization.

Then we treat that request as: "use the MPS-native int8+LoRA path".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class MPSQuantConfig:
    """Config for MPS-native weight-only quantization."""

    # Weight storage dtype
    quant_dtype: str = "int8"  # only int8 implemented

    # Group-wise symmetric per-output-channel quantization.
    # Each row is quantized in chunks of group_size along in_features.
    group_size: int = 128

    # Compute dtype after dequant (for linear op)
    compute_dtype: str = "float16"  # "float16" or "bfloat16"

    # LoRA params
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05

    # Which module names to replace.
    # StarCoder2 uses GPT-2 style naming: c_attn, c_proj, mlp.c_fc, mlp.c_proj.
    target_name_substrings: Tuple[str, ...] = (
        "c_attn",
        "c_proj",
        "mlp.c_fc",
        "mlp.c_proj",
    )

    @classmethod
    def from_trainer_cfg(cls, model_cfg: Dict[str, Any], quant_cfg: Optional[Dict[str, Any]] = None) -> "MPSQuantConfig":
        quant_cfg = quant_cfg or {}
        # allow overriding via YAML keys under either model or quantization
        return cls(
            quant_dtype=str(quant_cfg.get("mps_quant_dtype", model_cfg.get("mps_quant_dtype", "int8"))),
            group_size=int(quant_cfg.get("mps_group_size", model_cfg.get("mps_group_size", 128))),
            compute_dtype=str(quant_cfg.get("mps_compute_dtype", model_cfg.get("mps_compute_dtype", "float16"))),
            lora_rank=int(quant_cfg.get("lora_rank", model_cfg.get("lora_rank", 32))),
            lora_alpha=int(quant_cfg.get("lora_alpha", model_cfg.get("lora_alpha", 64))),
            lora_dropout=float(quant_cfg.get("lora_dropout", model_cfg.get("lora_dropout", 0.05))),
        )


def _compute_dtype(cfg: MPSQuantConfig) -> torch.dtype:
    if cfg.compute_dtype.lower() in ("fp16", "float16"):
        return torch.float16
    if cfg.compute_dtype.lower() in ("bf16", "bfloat16"):
        return torch.bfloat16
    return torch.float16


class QuantizedLinear(nn.Module):
    """Weight-only int8 quantized linear with group-wise symmetric scales."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool,
        cfg: MPSQuantConfig,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.cfg = cfg

        self.group_size = int(cfg.group_size)
        self.num_groups = (self.in_features + self.group_size - 1) // self.group_size
        self.padded_in_features = self.num_groups * self.group_size

        # Buffers (frozen)
        self.register_buffer("weight_q", torch.zeros((self.out_features, self.padded_in_features), dtype=torch.int8))
        # per-(out, group) scales
        self.register_buffer("scales", torch.ones((self.out_features, self.num_groups), dtype=torch.float16))
        self.register_buffer("bias", torch.zeros((self.out_features,), dtype=torch.float16) if bias else None)

    @torch.no_grad()
    def load_from_float_weight(self, w: torch.Tensor, b: Optional[torch.Tensor] = None) -> None:
        """Quantize from float weight. Uses symmetric quantization per (row, group)."""
        if w.shape != (self.out_features, self.in_features):
            raise ValueError(f"Weight shape mismatch: expected {(self.out_features, self.in_features)}, got {tuple(w.shape)}")

        # Move to CPU for quantization to avoid slow Python loops on MPS
        w_cpu = w.detach().to("cpu", dtype=torch.float32)

        q = torch.zeros((self.out_features, self.padded_in_features), dtype=torch.int8)
        s = torch.empty((self.out_features, self.num_groups), dtype=torch.float16)

        for g in range(self.num_groups):
            start = g * self.group_size
            end = min((g + 1) * self.group_size, self.in_features)
            block = w_cpu[:, start:end]

            # symmetric per-row max
            row_absmax = block.abs().amax(dim=1)  # (out,)
            # avoid div-by-zero
            scale = (row_absmax / 127.0).clamp(min=1e-8)  # (out,)

            # quant: round(w/scale)
            q_block = torch.round(block / scale.unsqueeze(1)).clamp(-128, 127).to(torch.int8)

            q[:, start:end] = q_block
            s[:, g] = scale.to(torch.float16)

        # store
        self.weight_q.copy_(q)
        self.scales.copy_(s)

        if self.bias is not None and b is not None:
            self.bias.copy_(b.detach().to("cpu", dtype=torch.float16))

    def _dequant_weight(self, device: torch.device) -> torch.Tensor:
        # Vectorized dequantization with in-place ops to minimize memory allocations.
        compute_dtype = _compute_dtype(self.cfg)

        q = self.weight_q
        s = self.scales

        if q.device != device:
            q = q.to(device=device)
        if s.device != device or s.dtype != compute_dtype:
            s = s.to(device=device, dtype=compute_dtype)

        # (out, padded) -> (out, groups, group_size)
        # Use contiguous + view to avoid copy when possible
        q3 = q.to(dtype=compute_dtype).view(self.out_features, self.num_groups, self.group_size)
        # Broadcast multiply with minimal temp allocation
        w3 = q3.mul(s.view(self.out_features, self.num_groups, 1))
        # Reshape and slice in one op
        return w3.reshape(self.out_features, self.padded_in_features)[:, : self.in_features].contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self._dequant_weight(x.device)
        b = None
        if self.bias is not None:
            b = self.bias.to(device=x.device, dtype=w.dtype)
        return F.linear(x, w, b)


class LoRALinear(nn.Module):
    """Plain LoRA adapter for a linear projection."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        r: int,
        alpha: int,
        dropout: float,
        dtype: torch.dtype,
    ):
        super().__init__()
        self.r = int(r)
        self.alpha = int(alpha)
        self.scale = float(alpha) / float(r) if r > 0 else 1.0
        self.dropout = nn.Dropout(dropout)

        self.A = nn.Linear(in_features, r, bias=False, dtype=dtype)
        self.B = nn.Linear(r, out_features, bias=False, dtype=dtype)

        # init: A ~ Kaiming, B = 0
        nn.init.kaiming_uniform_(self.A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.B(self.A(self.dropout(x))) * self.scale


class QuantizedLoRALinear(nn.Module):
    """Quantized base + trainable LoRA."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool,
        cfg: MPSQuantConfig,
    ):
        super().__init__()
        self.cfg = cfg
        self.base = QuantizedLinear(in_features, out_features, bias=bias, cfg=cfg)
        self.lora = LoRALinear(
            in_features,
            out_features,
            r=cfg.lora_rank,
            alpha=cfg.lora_alpha,
            dropout=cfg.lora_dropout,
            dtype=_compute_dtype(cfg),
        )

    @torch.no_grad()
    def load_base(self, w: torch.Tensor, b: Optional[torch.Tensor]) -> None:
        self.base.load_from_float_weight(w, b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora(x)


def _should_replace(name: str, cfg: MPSQuantConfig) -> bool:
    return any(sub in name for sub in cfg.target_name_substrings)


def _get_parent_module(model: nn.Module, module_name: str) -> Tuple[nn.Module, str]:
    # module_name like "transformer.h.0.attn.c_attn"
    parts = module_name.split(".")
    if len(parts) == 1:
        raise ValueError("Cannot replace top-level module")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


@torch.no_grad()
def load_quantized_starcoder2_mps(
    *,
    pretrained_model: str,
    device: torch.device,
    cfg: MPSQuantConfig,
    trust_remote_code: bool = True,
) -> nn.Module:
    """Load StarCoder2-3B with MPS-native quantization+LoRA replacements."""

    from transformers import AutoModelForCausalLM

    # Load base model on CPU in fp32 (so we can quantize deterministically)
    logger.info(f"[mps-quant] Loading base model on CPU: {pretrained_model}")
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model,
        torch_dtype=torch.float32,
        device_map={"": "cpu"},
        trust_remote_code=bool(trust_remote_code),
    )

    # Freeze everything by default
    for p in model.parameters():
        p.requires_grad = False

    # Identify target linears first (avoid mutating while iterating generator)
    targets: List[Tuple[str, nn.Linear]] = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and _should_replace(name, cfg):
            targets.append((name, mod))

    logger.info(f"[mps-quant] Found {len(targets)} target Linear layers to replace")

    replaced = 0
    for name, lin in targets:
        parent, attr = _get_parent_module(model, name)

        qlora = QuantizedLoRALinear(
            lin.in_features,
            lin.out_features,
            bias=lin.bias is not None,
            cfg=cfg,
        )
        qlora.load_base(lin.weight, lin.bias)

        setattr(parent, attr, qlora)
        replaced += 1
        if replaced % 25 == 0:
            logger.info(f"[mps-quant] Replaced {replaced}/{len(targets)}")

    # Move to device
    logger.info(f"[mps-quant] Moving model to {device}")
    model.to(device)

    # Basic stats
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"[mps-quant] Params: total={total/1e6:.1f}M trainable={trainable/1e6:.1f}M")

    return model
