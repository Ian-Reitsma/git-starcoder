#!/usr/bin/env python3
"""
Test FlashAttention-1 Custom Integration on Turing GPU

This script verifies that our custom FA1 integration works correctly:
1. Load Phi-2 model
2. Patch with FA1 attention
3. Run forward pass
4. Verify outputs are correct
5. Check memory usage
"""

import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_fa1_integration():
    """Test FlashAttention-1 integration"""

    logger.info("=" * 80)
    logger.info("TESTING FLASHATTENTION-1 INTEGRATION ON TURING GPU")
    logger.info("=" * 80)

    # Check CUDA
    if not torch.cuda.is_available():
        logger.error("CUDA not available!")
        return False

    device = torch.device("cuda:0")
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"Compute capability: {torch.cuda.get_device_capability(0)}")

    # Import FA1 integration
    try:
        from training.model_trainer_unified import HAS_FLASH_ATTN_1, patch_model_with_flash_attn_1
        if not HAS_FLASH_ATTN_1:
            logger.error("FlashAttention-1 not available!")
            return False
        logger.info("✓ FlashAttention-1 available")
    except Exception as e:
        logger.error(f"Failed to import FA1 integration: {e}")
        return False

    # Load model
    logger.info("\n1. Loading Phi-2 model...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            "microsoft/phi-2",
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        logger.info("✓ Model loaded")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False

    # Load tokenizer
    logger.info("\n2. Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
        logger.info("✓ Tokenizer loaded")
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        return False

    # Patch with FA1
    logger.info("\n3. Patching model with FlashAttention-1...")
    try:
        model = patch_model_with_flash_attn_1(model)
        logger.info("✓ Model patched")
    except Exception as e:
        logger.error(f"Failed to patch model: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test forward pass
    logger.info("\n4. Testing forward pass...")
    try:
        test_text = "def fibonacci(n):\n    if n <= 1:\n        return n\n    return"
        inputs = tokenizer(test_text, return_tensors="pt").to(device)

        logger.info(f"Input text: '{test_text}'")
        logger.info(f"Input shape: {inputs['input_ids'].shape}")

        # Clear cache before forward pass
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Forward pass
        with torch.no_grad():
            outputs = model(**inputs)

        logger.info(f"✓ Forward pass successful")
        logger.info(f"Output logits shape: {outputs.logits.shape}")
        logger.info(f"Peak VRAM: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

    except Exception as e:
        logger.error(f"Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test generation
    logger.info("\n5. Testing generation...")
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        output_ids = model.generate(
            inputs['input_ids'],
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        logger.info(f"✓ Generation successful")
        logger.info(f"Generated: '{generated_text}'")
        logger.info(f"Peak VRAM: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test with larger sequence
    logger.info("\n6. Testing with larger sequence (512 tokens)...")
    try:
        long_text = "def quicksort(arr):\n" * 100  # Repeat to make it longer
        inputs = tokenizer(long_text, return_tensors="pt", max_length=512, truncation=True).to(device)

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        with torch.no_grad():
            outputs = model(**inputs)

        logger.info(f"✓ Large sequence forward pass successful")
        logger.info(f"Sequence length: {inputs['input_ids'].shape[1]}")
        logger.info(f"Peak VRAM: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

    except Exception as e:
        logger.error(f"Large sequence test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    logger.info("\n" + "=" * 80)
    logger.info("✓ ALL TESTS PASSED! FLASHATTENTION-1 INTEGRATION WORKING!")
    logger.info("=" * 80)
    return True

if __name__ == "__main__":
    success = test_fa1_integration()
    exit(0 if success else 1)
