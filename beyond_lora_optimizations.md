# Beyond LoRA: Next Optimizations (priority order)

1) Smart memory defragmentation (+5% effective VRAM)
- Run `torch.cuda.empty_cache()` + GC between epochs (already wired)
- Add optional allocator warmups per N steps to smooth peaks

2) Selective gradient checkpointing (-8% speed, +~300 MB)
- Checkpoint only middle transformer blocks where activation volume peaks
- Keep embeddings + final blocks live to preserve throughput

3) Gradient quantization (-25% grad/opt memory)
- Quantize stored gradients to fp16/bf16 before optimizer step
- Pair with CPU LoRA offload to shrink host bandwidth needs

4) Local attention for long contexts (+~30% throughput on 4K+ tokens)
- Swap full attention with windowed/block local attention on lower layers
- Keep top layers global for quality

5) Mixed-rank LoRA (expressive where it matters)
- Attention/QKV: r=64–96, FFN: r=8–16
- Keeps quality gains while containing CPU bandwidth

6) Operator fusion (already applied in SDPA/torch.compile paths)
- Verify fused MLP/attention kernels remain enabled post offload
- If FlashAttention v2 available, prefer it for >2K context
