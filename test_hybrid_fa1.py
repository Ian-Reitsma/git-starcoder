#!/usr/bin/env python3
"""Test Hybrid FlashAttention-1 (FA1 forward + standard backward)"""

import torch
import logging
from training.model_trainer_unified import HybridFlashAttentionFunction

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_hybrid_fa1():
    """Test hybrid FA1 with head_dim=80 (Phi-2's dimension)"""

    device = torch.device("cuda:0")
    batch = 2
    seqlen = 128
    nheads = 32
    headdim = 80  # Phi-2's head dimension - this is what failed with pure FA1!

    logger.info(f"Testing Hybrid FA1 with head_dim={headdim} on Turing GPU")
    logger.info(f"Shape: batch={batch}, seq={seqlen}, heads={nheads}, head_dim={headdim}")

    # Create random QKV
    qkv = torch.randn(
        batch, seqlen, 3, nheads, headdim,
        device=device, dtype=torch.float16, requires_grad=True
    )

    logger.info("✓ Created QKV tensor")

    # Forward pass using hybrid function
    try:
        softmax_scale = 1.0 / (headdim ** 0.5)
        output = HybridFlashAttentionFunction.apply(
            qkv, 0.0, softmax_scale, True
        )
        logger.info(f"✓ Forward pass successful! Output shape: {output.shape}")
    except Exception as e:
        logger.error(f"✗ Forward pass failed: {e}")
        return False

    # Backward pass (this is where pure FA1 failed!)
    try:
        loss = output.sum()
        loss.backward()
        logger.info(f"✓ Backward pass successful!")
        logger.info(f"✓ QKV gradients shape: {qkv.grad.shape}")
    except Exception as e:
        logger.error(f"✗ Backward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    logger.info("\n" + "=" * 80)
    logger.info("✓✓✓ HYBRID FA1 WORKS ON TURING WITH HEAD_DIM=80! ✓✓✓")
    logger.info("=" * 80)
    return True

if __name__ == "__main__":
    success = test_hybrid_fa1()
    exit(0 if success else 1)
