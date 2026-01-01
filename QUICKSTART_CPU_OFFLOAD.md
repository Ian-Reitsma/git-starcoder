# Quickstart: CPU LoRA Offload

This makes LoRA adapters live on CPU pinned memory and stage through the GPU only during `optimizer.step()`. Net effect on 8GB GPU: rank 64 + batch 4 with lower VRAM.

## 3 Code Changes (training/model_trainer_unified.py)
1) Import helper:
```python
from cpu_offload_implementation import apply_cpu_offloaded_lora, get_cpu_offload_parameters, wrap_optimizer_for_cpu_offload
```
2) Replace `get_peft_model(...)` with `apply_cpu_offloaded_lora(...)` inside `_apply_lora` when `cpu_offload_lora` is true.
3) Build optimizer on non-offloaded params and wrap it:
```python
normal_params, _ = get_cpu_offload_parameters(self.model)
optimizer = AdamW(normal_params, ...)
optimizer = wrap_optimizer_for_cpu_offload(optimizer, self.model, lr=..., weight_decay=..., device=self.device)
```

Already implemented in this repo—just enable via config.

## Config Toggle
Set `cpu_offload_lora: true` anywhere in the optimization/quantization section (alternate schema) or under `model` for canonical configs.

Example (universal config):
```yaml
optimization:
  cpu_offload_lora: true
  lora_rank: 64
  batch_size: 4
```

## Verification Checklist
- Log line: `LoRA CPU offload: True` and `CPU-offloaded LoRA to N modules`
- VRAM during train: <= ~6.8 GB at bs=4, r=64
- Step time overhead: ~5–6 ms vs baseline, amortized by larger batch
- LoRA optimizer path uses fp16 math on GPU for the staged update to cut latency; master weights stay on CPU in original dtype
- If import missing: install dependencies (`torch`, `transformers`, `peft`); module is local, no extra install needed

## Rollback
Set `cpu_offload_lora: false` to revert to standard PEFT LoRA without touching code.
