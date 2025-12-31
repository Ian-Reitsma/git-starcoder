#!/usr/bin/env python3
"""
Simple test for custom Turing FlashAttention kernel integration.

Tests:
1. Model loads with original Phi-2 architecture (32×80)
2. Custom kernel patches all attention layers
3. Forward/backward pass works
"""

import torch
import logging
import sys
import os
from transformers import AutoModelForCausalLM, AutoConfig
import math

# Add training directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'training'))

from flash_attn_turing_ext import FlashAttentionTuringFunction
from training.model_trainer_unified import (
    PhiCustomTuringAttention,
    patch_model_with_custom_fa1_turing,
    PhiAttention
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_simple():
    """Simple integration test"""

    logger.info("="*80)
    logger.info("SIMPLE CUSTOM TURING KERNEL TEST")
    logger.info("="*80)

    # Load model with 4-bit quantization
    logger.info("\n1. Loading Phi-2 model with 4-bit quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-2",
        load_in_4bit=True,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    logger.info("✓ Model loaded")

    # Verify original architecture
    logger.info("\n2. Verifying original Phi-2 architecture...")
    config = model.config
    hidden_size = config.hidden_size
    num_heads = config.num_attention_heads
    head_dim = hidden_size // num_heads

    logger.info(f"  Hidden size: {hidden_size}")
    logger.info(f"  Num heads: {num_heads}")
    logger.info(f"  Head dim: {head_dim}")

    assert hidden_size == 2560, f"Expected 2560, got {hidden_size}"
    assert num_heads == 32, f"Expected 32, got {num_heads}"
    assert head_dim == 80, f"Expected 80, got {head_dim}"
    logger.info("✓ Original architecture confirmed: 32 heads × 80 head_dim")

    # Patch with custom kernel
    logger.info("\n3. Patching model with custom Turing kernel...")
    model = patch_model_with_custom_fa1_turing(model, FlashAttentionTuringFunction)

    # Verify patching
    logger.info("\n4. Verifying all attention layers are patched...")
    patched = 0
    total = 0
    for name, module in model.named_modules():
        if 'attn' in name.lower() and hasattr(module, 'q_proj'):
            total += 1
            if isinstance(module, PhiCustomTuringAttention):
                patched += 1

    logger.info(f"  Patched: {patched}/{total} layers")
    assert patched == total, f"Not all layers patched: {patched}/{total}"
    logger.info(f"✓ All {patched} attention layers patched")

    # Test forward/backward pass
    logger.info("\n5. Testing forward/backward pass...")
    model.train()

    # Create small test input
    input_ids = torch.randint(0, config.vocab_size, (1, 64), device=model.device)

    # Forward
    outputs = model(input_ids, labels=input_ids)
    loss = outputs.loss
    logger.info(f"  ✓ Forward pass successful! Loss: {loss.item():.4f}")

    # Backward
    loss.backward()
    logger.info(f"  ✓ Backward pass successful!")

    # Check gradients
    grads = sum(1 for p in model.parameters() if p.requires_grad and p.grad is not None)
    logger.info(f"  ✓ Gradients computed for {grads} parameters")

    logger.info("\n" + "="*80)
    logger.info("✓✓✓ CUSTOM TURING KERNEL INTEGRATION WORKS! ✓✓✓")
    logger.info("="*80)
    logger.info("Summary:")
    logger.info(f"  ✓ Original Phi-2 architecture: {num_heads}×{head_dim}")
    logger.info(f"  ✓ All {patched} attention layers patched")
    logger.info(f"  ✓ Forward/backward pass working")
    logger.info(f"  ✓ Ready to train with head_dim=80 on Turing!")
    logger.info("="*80)

    return True

if __name__ == "__main__":
    try:
        success = test_simple()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
