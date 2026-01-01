"""
CPU LoRA Offload Toolkit
------------------------
Stores LoRA weights in pinned CPU memory and stages them to the GPU only when
needed. Designed for 8GB-class GPUs where RAM is abundant.

Components
- CPUOffloadedLoRALinear: Lightweight drop-in that wraps target Linear/Conv1D
  layers with CPU-resident LoRA matrices.
- CPUOffloadOptimizer: Wraps an existing optimizer to update CPU LoRA weights
  without keeping optimizer state on the GPU.
- apply_cpu_offloaded_lora / wrap_optimizer_for_cpu_offload: Helper functions
  that make this a drop-in for the unified model trainer.
"""

from __future__ import annotations

import logging
import math
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _is_conv1d(module: nn.Module) -> bool:
    """Detect the GPT-style Conv1D used in transformers GPT2 blocks."""
    return module.__class__.__name__ == "Conv1D"


class CPUOffloadedLoRALinear(nn.Module):
    """LoRA adapter whose trainable weights live in pinned CPU memory."""

    def __init__(
        self,
        base_module: nn.Module,
        r: int,
        lora_alpha: int,
        lora_dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if not hasattr(base_module, "weight"):
            raise ValueError("Base module must expose a weight attribute")

        self.base = base_module
        self.r = int(max(r, 0))
        self.scaling = float(lora_alpha) / max(self.r, 1)
        self.dropout = nn.Dropout(lora_dropout) if lora_dropout > 0 else nn.Identity()
        self.is_conv1d = _is_conv1d(base_module)

        in_features, out_features = self._infer_features()

        # Freeze base weights; only LoRA trains.
        for p in self.base.parameters():
            p.requires_grad = False

        if self.r > 0:
            a = torch.empty((self.r, in_features), device="cpu", dtype=self.base.weight.dtype)
            b = torch.empty((out_features, self.r), device="cpu", dtype=self.base.weight.dtype)
            nn.init.kaiming_uniform_(a, a=math.sqrt(5))
            nn.init.zeros_(b)

            # Pin for fast PCIe transfer; tag for optimizer split later.
            self.lora_A = nn.Parameter(a.pin_memory())
            self.lora_B = nn.Parameter(b.pin_memory())
            self.lora_A._cpu_offload_lora = True  # type: ignore[attr-defined]
            self.lora_B._cpu_offload_lora = True  # type: ignore[attr-defined]
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

    def _infer_features(self) -> Tuple[int, int]:
        weight = self.base.weight

        if hasattr(self.base, "in_features") and hasattr(self.base, "out_features"):
            return int(self.base.in_features), int(self.base.out_features)

        # Conv1D stores weight as (in_features, out_features)
        if self.is_conv1d:
            return int(weight.shape[0]), int(weight.shape[1])

        # Linear weight uses (out_features, in_features)
        return int(weight.shape[1]), int(weight.shape[0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.base(x)

        if self.r == 0 or self.lora_A is None or self.lora_B is None:
            return result

        # Stage LoRA weights to the compute device just-in-time.
        target = x.device
        a = self.lora_A.to(target, non_blocking=True)
        b = self.lora_B.to(target, non_blocking=True)

        # Standard LoRA low-rank update: dropout(x) @ A^T @ B^T
        lora_in = self.dropout(x)
        lora_intermediate = torch.matmul(lora_in, a.transpose(0, 1))
        lora_update = torch.matmul(lora_intermediate, b.transpose(0, 1)) * self.scaling

        return result + lora_update

    def extra_repr(self) -> str:
        return f"r={self.r}, scaling={self.scaling:.4f}, pinned_cpu=True"


def _should_wrap_module(module_name: str, targets: Sequence[str]) -> bool:
    return any(t in module_name for t in targets)


def _repin_lora_parameters(module: nn.Module) -> None:
    """Force all LoRA parameters tagged for offload back to pinned CPU."""
    for _, param in module.named_parameters():
        if getattr(param, "_cpu_offload_lora", False):
            with torch.no_grad():
                param.data = param.data.detach().cpu().pin_memory()
            if param.grad is not None and param.grad.device != torch.device("cpu"):
                param.grad = param.grad.detach().cpu()


def apply_cpu_offloaded_lora(model: nn.Module, lora_cfg: dict) -> nn.Module:
    """Patch target modules with CPU-offloaded LoRA adapters."""
    targets: Sequence[str] = lora_cfg.get("target_modules", []) or []
    r = int(lora_cfg.get("r", 8))
    alpha = int(lora_cfg.get("lora_alpha", max(r, 1)))
    dropout = float(lora_cfg.get("lora_dropout", 0.0))

    if not targets:
        raise ValueError("target_modules must be provided for CPU LoRA offload")

    # Freeze full model; only the injected adapters train.
    for p in model.parameters():
        p.requires_grad = False

    wrapped = 0
    for module_name, module in list(model.named_modules()):
        if not _should_wrap_module(module_name, targets):
            continue

        if not isinstance(module, nn.Linear) and not _is_conv1d(module):
            continue

        parts = module_name.split(".")
        parent = model
        for key in parts[:-1]:
            parent = getattr(parent, key)

        new_module = CPUOffloadedLoRALinear(module, r=r, lora_alpha=alpha, lora_dropout=dropout)
        setattr(parent, parts[-1], new_module)
        wrapped += 1

    _repin_lora_parameters(model)

    model.cpu_lora_offload_active = True  # type: ignore[attr-defined]
    model.cpu_lora_offload_targets = list(targets)  # type: ignore[attr-defined]
    logger.info(f"Applied CPU-offloaded LoRA to {wrapped} modules (r={r}, alpha={alpha}, dropout={dropout})")
    if wrapped == 0:
        logger.warning("No modules matched for CPU LoRA offload; verify target_modules names.")

    return model


def get_cpu_offload_parameters(model: nn.Module) -> Tuple[List[torch.nn.Parameter], List[torch.nn.Parameter]]:
    """Split parameters into (non-offloaded, offloaded-LoRA)."""
    offloaded: List[torch.nn.Parameter] = []
    normal: List[torch.nn.Parameter] = []

    for _, param in model.named_parameters():
        if getattr(param, "_cpu_offload_lora", False):
            offloaded.append(param)
        else:
            normal.append(param)

    return normal, offloaded


class CPUOffloadOptimizer:
    """Wraps a base optimizer and updates LoRA CPU parameters on GPU staging."""

    def __init__(
        self,
        base_optimizer: torch.optim.Optimizer,
        lora_params: Iterable[torch.nn.Parameter],
        *,
        lr: float,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        offload_device: Optional[torch.device] = None,
        use_fp16_updates: Optional[bool] = None,
    ) -> None:
        self.base_optimizer = base_optimizer
        self.lora_params = [p for p in lora_params if p.requires_grad]
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.offload_device = offload_device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Use fp16 math on GPU to reduce optimizer FLOPs/latency; keep master weights on CPU.
        if use_fp16_updates is None:
            self.use_fp16_updates = self.offload_device.type == "cuda"
        else:
            self.use_fp16_updates = bool(use_fp16_updates)

        self.state = {}
        for param in self.lora_params:
            state = {
                "step": 0,
                "exp_avg": torch.zeros_like(param, device="cpu").pin_memory(),
                "exp_avg_sq": torch.zeros_like(param, device="cpu").pin_memory(),
            }
            self.state[param] = state

        # Mirror param_groups for logging convenience
        self.param_groups = self.base_optimizer.param_groups
        self.defaults = getattr(self.base_optimizer, "defaults", {})

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.base_optimizer.zero_grad(set_to_none=set_to_none)
        for param in self.lora_params:
            if param.grad is None:
                continue
            if set_to_none:
                param.grad = None
            else:
                param.grad.detach_()
                param.grad.zero_()

    def state_dict(self) -> dict:
        return {
            "base_optimizer": self.base_optimizer.state_dict(),
            "lora_state": {
                "hyperparams": {
                    "lr": self.lr,
                    "betas": self.betas,
                    "eps": self.eps,
                    "weight_decay": self.weight_decay,
                },
                "state": self.state,
            },
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.base_optimizer.load_state_dict(state_dict.get("base_optimizer", {}))
        lora_state = state_dict.get("lora_state", {})
        saved_state = lora_state.get("state", {})
        for param in self.lora_params:
            if param in saved_state:
                self.state[param] = saved_state[param]

    @property
    def device(self) -> torch.device:
        return self.offload_device

    def _current_lr(self) -> float:
        if not self.base_optimizer.param_groups:
            return self.lr
        return float(self.base_optimizer.param_groups[0].get("lr", self.lr))

    def _step_lora(self) -> None:
        if not self.lora_params:
            return

        lr = self._current_lr()
        beta1, beta2 = self.betas

        for param in self.lora_params:
            grad = param.grad
            if grad is None:
                continue

            state = self.state[param]
            state["step"] += 1
            step = state["step"]

            # Stage tensors to the chosen device for the math, then copy back.
            grad_dev = grad.detach().to(self.offload_device, non_blocking=True)
            param_dev = param.data.to(self.offload_device, non_blocking=True)
            exp_avg = state["exp_avg"].to(self.offload_device, non_blocking=True)
            exp_avg_sq = state["exp_avg_sq"].to(self.offload_device, non_blocking=True)

            target_dtype = param_dev.dtype
            if self.use_fp16_updates and self.offload_device.type == "cuda":
                target_dtype = torch.float16
                grad_dev = grad_dev.to(target_dtype)
                param_dev = param_dev.to(target_dtype)
                exp_avg = exp_avg.to(target_dtype)
                exp_avg_sq = exp_avg_sq.to(target_dtype)

            exp_avg.mul_(beta1).add_(grad_dev, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad_dev, grad_dev, value=1 - beta2)

            denom = exp_avg_sq.sqrt().add_(self.eps)
            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step
            step_size = lr * (bias_correction2**0.5) / bias_correction1

            if self.weight_decay:
                param_dev.add_(param_dev, alpha=-self.weight_decay * lr)

            param_dev.addcdiv_(exp_avg, denom, value=-step_size)

            # Copy updates back to pinned CPU.
            with torch.no_grad():
                param.data.copy_(param_dev.to("cpu").to(param.data.dtype))
                state["exp_avg"] = exp_avg.to("cpu").pin_memory().to(param.data.dtype)
                state["exp_avg_sq"] = exp_avg_sq.to("cpu").pin_memory().to(param.data.dtype)

            param.grad = None

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.base_optimizer.step()
        self._step_lora()
        return loss

    def add_param_group(self, *args, **kwargs):
        return self.base_optimizer.add_param_group(*args, **kwargs)


def wrap_optimizer_for_cpu_offload(
    optimizer: torch.optim.Optimizer,
    model: nn.Module,
    *,
    lr: float,
    betas: Tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    device: Optional[torch.device] = None,
    use_fp16_updates: Optional[bool] = None,
) -> CPUOffloadOptimizer:
    """Return an optimizer wrapper that keeps LoRA weights on CPU."""

    _, offloaded = get_cpu_offload_parameters(model)
    _repin_lora_parameters(model)

    return CPUOffloadOptimizer(
        optimizer,
        offloaded,
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
        offload_device=device,
        use_fp16_updates=use_fp16_updates,
    )
