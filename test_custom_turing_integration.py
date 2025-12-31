#!/usr/bin/env python3
"""
Test Custom Turing FlashAttention Integration

Verifies that:
1. Model loads with original Phi-2 architecture (32×80)
2. Custom kernel patches all attention layers
3. Forward/backward pass works through the entire model
4. Preserves 4-bit quantization
"""

import torch
import logging
import sys
import os

# Add training directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'training'))

from training.model_trainer_unified import OptimizedModelTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_custom_turing_integration():
    """Test full integration of custom Turing kernel with model trainer"""

    logger.info("=" * 80)
    logger.info("TESTING CUSTOM TURING KERNEL INTEGRATION")
    logger.info("=" * 80)

    # Create minimal config for testing
    config = {
        'model': {
            'name': 'microsoft/phi-2',  # Phi-2 with 32×80 architecture
            'use_4bit': True,  # Test with quantization
            'use_8bit': False,
            'use_bf16': False,
            'use_lora': False,
            'load_in_4bit': True,
            'bnb_4bit_quant_type': 'nf4',
            'bnb_4bit_compute_dtype': 'float16',
            'bnb_4bit_use_double_quant': True,
        },
        'training': {
            'output_dir': '/tmp/test_custom_turing',
            'base_learning_rate': 1e-4,
            'batch_size': 1,
            'gradient_accumulation_steps': 1,
            'num_epochs': 1,
            'warmup_steps': 0,
            'save_every_n_steps': 1000,
            'max_sequence_length': 512,
            'use_mixed_precision': True,
            'use_gradient_checkpointing': False,
            'seed': 42,
        },
        'data': {
            'sequences_file': '/tmp/dummy_sequences.jsonl',
            'validation_split': 0.0,
        }
    }

    # Create dummy sequences file and config file
    import json
    import yaml
    os.makedirs('/tmp', exist_ok=True)
    with open('/tmp/dummy_sequences.jsonl', 'w') as f:
        json.dump({'input_ids': [1, 2, 3, 4, 5]}, f)

    config_path = '/tmp/test_custom_turing_config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f)

    logger.info("Step 1: Creating OptimizedModelTrainer instance...")
    try:
        trainer = OptimizedModelTrainer(config_path)
        logger.info("✓ ModelTrainer created successfully")
    except Exception as e:
        logger.error(f"✗ Failed to create ModelTrainer: {e}")
        import traceback
        traceback.print_exc()
        return False

    logger.info("\nStep 2: Verifying model architecture...")
    try:
        # Check that model has the original Phi-2 architecture
        config_obj = trainer.model.config
        hidden_size = config_obj.hidden_size
        num_heads = config_obj.num_attention_heads
        head_dim = hidden_size // num_heads

        logger.info(f"  Hidden size: {hidden_size}")
        logger.info(f"  Num heads: {num_heads}")
        logger.info(f"  Head dim: {head_dim}")

        assert hidden_size == 2560, f"Expected hidden_size=2560, got {hidden_size}"
        assert num_heads == 32, f"Expected num_heads=32, got {num_heads}"
        assert head_dim == 80, f"Expected head_dim=80, got {head_dim}"

        logger.info("✓ Model has original Phi-2 architecture (32×80)")
    except Exception as e:
        logger.error(f"✗ Architecture verification failed: {e}")
        return False

    logger.info("\nStep 3: Verifying attention layers are patched...")
    try:
        from training.model_trainer_unified import PhiCustomTuringAttention

        patched_count = 0
        total_attn_count = 0

        for name, module in trainer.model.named_modules():
            if 'attn' in name.lower() and hasattr(module, 'q_proj'):
                total_attn_count += 1
                if isinstance(module, PhiCustomTuringAttention):
                    patched_count += 1
                    logger.debug(f"  ✓ {name} is PhiCustomTuringAttention")
                else:
                    logger.warning(f"  ⚠ {name} is {type(module).__name__}")

        logger.info(f"  Patched: {patched_count}/{total_attn_count} attention layers")

        assert patched_count > 0, "No attention layers were patched!"
        assert patched_count == total_attn_count, \
            f"Not all attention layers patched: {patched_count}/{total_attn_count}"

        logger.info("✓ All attention layers patched with custom kernel")
    except Exception as e:
        logger.error(f"✗ Patching verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    logger.info("\nStep 4: Testing forward/backward pass...")
    try:
        # Create dummy input
        batch_size = 1
        seq_len = 128
        vocab_size = config_obj.vocab_size

        input_ids = torch.randint(
            0, vocab_size, (batch_size, seq_len),
            device=trainer.model.device
        )

        logger.info(f"  Input shape: {input_ids.shape}")

        # Forward pass
        trainer.model.train()
        outputs = trainer.model(input_ids, labels=input_ids)
        loss = outputs.loss

        logger.info(f"  ✓ Forward pass successful! Loss: {loss.item():.4f}")

        # Backward pass
        loss.backward()

        logger.info(f"  ✓ Backward pass successful!")

        # Check gradients exist
        grad_count = 0
        for name, param in trainer.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_count += 1

        logger.info(f"  ✓ Gradients computed for {grad_count} parameters")

        assert grad_count > 0, "No gradients computed!"

    except Exception as e:
        logger.error(f"✗ Forward/backward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    logger.info("\nStep 5: Verifying 4-bit quantization preserved...")
    try:
        from bitsandbytes.nn import Linear4bit

        quant_count = 0
        for name, module in trainer.model.named_modules():
            if isinstance(module, Linear4bit):
                quant_count += 1

        logger.info(f"  Found {quant_count} 4-bit quantized layers")
        assert quant_count > 0, "No 4-bit quantized layers found!"

        logger.info("✓ 4-bit quantization preserved")
    except Exception as e:
        logger.error(f"✗ Quantization check failed: {e}")
        return False

    logger.info("\n" + "=" * 80)
    logger.info("✓✓✓ CUSTOM TURING KERNEL INTEGRATION SUCCESSFUL! ✓✓✓")
    logger.info("=" * 80)
    logger.info("Summary:")
    logger.info("  ✓ Original Phi-2 architecture (32 heads × 80 head_dim)")
    logger.info(f"  ✓ All {patched_count} attention layers patched")
    logger.info("  ✓ Forward/backward pass working")
    logger.info(f"  ✓ {quant_count} layers still 4-bit quantized")
    logger.info("  ✓ Ready for training with head_dim=80 on Turing GPU!")
    logger.info("=" * 80)

    return True


if __name__ == "__main__":
    success = test_custom_turing_integration()
    sys.exit(0 if success else 1)
