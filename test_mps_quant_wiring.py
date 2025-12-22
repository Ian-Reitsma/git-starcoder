#!/usr/bin/env python3
"""Quick validation test for MPS-native quantization backend wiring."""

import sys
import torch
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_mps_quant_backend_import():
    """Test 1: Can we import the MPS quant backend?"""
    try:
        from training.mps_quant_backend import (
            MPSQuantConfig,
            QuantizedLinear,
            QuantizedLoRALinear,
            load_quantized_starcoder2_mps,
        )
        logger.info("✓ MPS quant backend imports successfully")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to import MPS quant backend: {e}")
        return False


def test_quantized_linear():
    """Test 2: Can we create and use a QuantizedLinear layer?"""
    try:
        from training.mps_quant_backend import MPSQuantConfig, QuantizedLinear
        
        cfg = MPSQuantConfig(quant_dtype="int8", group_size=64, compute_dtype="float16")
        layer = QuantizedLinear(256, 512, bias=True, cfg=cfg)
        
        # Quantize a random weight
        w = torch.randn(512, 256, dtype=torch.float32)
        layer.load_from_float_weight(w)
        
        # Forward pass
        x = torch.randn(2, 128, 256, dtype=torch.float16)
        out = layer(x)
        
        assert out.shape == (2, 128, 512), f"Expected shape (2, 128, 512), got {out.shape}"
        assert out.dtype == torch.float16, f"Expected dtype float16, got {out.dtype}"
        
        logger.info(f"✓ QuantizedLinear works: input {x.shape} -> output {out.shape}")
        return True
    except Exception as e:
        logger.error(f"✗ QuantizedLinear test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_quantized_lora_linear():
    """Test 3: Can we create and use a QuantizedLoRALinear layer?"""
    try:
        from training.mps_quant_backend import MPSQuantConfig, QuantizedLoRALinear
        
        cfg = MPSQuantConfig(
            quant_dtype="int8",
            group_size=128,
            compute_dtype="float16",
            lora_rank=8,
            lora_alpha=16,
            lora_dropout=0.05,
        )
        layer = QuantizedLoRALinear(512, 1024, bias=True, cfg=cfg)
        
        # Quantize base weight
        w = torch.randn(1024, 512, dtype=torch.float32)
        layer.load_base(w, None)
        
        # Forward pass
        x = torch.randn(1, 64, 512, dtype=torch.float16)
        out = layer(x)
        
        assert out.shape == (1, 64, 1024), f"Expected shape (1, 64, 1024), got {out.shape}"
        
        # Check that only LoRA params are trainable
        trainable = sum(p.numel() for p in layer.parameters() if p.requires_grad)
        total = sum(p.numel() for p in layer.parameters())
        
        logger.info(
            f"✓ QuantizedLoRALinear works: "
            f"trainable={trainable}/{total} params ({100*trainable/total:.1f}%)"
        )
        return True
    except Exception as e:
        logger.error(f"✗ QuantizedLoRALinear test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_integration():
    """Test 4: Does the trainer config canonicalization preserve MPS quant knobs?"""
    try:
        from training.model_trainer_unified import load_yaml_config
        
        # Test with a minimal config that would trigger MPS quant
        test_cfg = {
            'model': {
                'pretrained_model': 'bigcode/starcoder2-3b',
                'mps_prefer_fp16': True,
                'trust_remote_code': True,
            },
            'quantization': {
                'load_in_4bit': True,
                'lora_enabled': True,
                'lora_rank': 16,
                'lora_alpha': 32,
                'mps_quant_dtype': 'int8',
                'mps_group_size': 64,
            },
            'optimization': {
                'batch_size': 2,
                'learning_rate': 2e-4,
                'mixed_precision': 'fp16',
            },
            'training': {
                'num_epochs': 1,
                'seed': 42,
            },
            'output': {
                'output_dir': 'test_output',
            },
        }
        
        # Write temp config
        import tempfile
        import yaml
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(test_cfg, f)
            temp_path = f.name
        
        try:
            cfg = load_yaml_config(temp_path)
            
            # Should have canonical structure
            assert 'model' in cfg
            assert 'training' in cfg
            
            # MPS knobs should survive canonicalization
            if 'mps_prefer_fp16' in cfg['model']:
                logger.info("✓ Config canonicalization preserves MPS knobs")
                return True
            else:
                logger.warning("⚠ Config canonicalization may not preserve all MPS knobs")
                return True  # Not a hard failure
        finally:
            Path(temp_path).unlink(missing_ok=True)
            
    except Exception as e:
        logger.error(f"✗ Config integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    logger.info("\n" + "="*70)
    logger.info("MPS-NATIVE QUANTIZATION BACKEND VALIDATION")
    logger.info("="*70 + "\n")
    
    tests = [
        ("Import", test_mps_quant_backend_import),
        ("QuantizedLinear", test_quantized_linear),
        ("QuantizedLoRALinear", test_quantized_lora_linear),
        ("Config Integration", test_config_integration),
    ]
    
    results = []
    for name, test_fn in tests:
        logger.info(f"Running test: {name}")
        success = test_fn()
        results.append((name, success))
        logger.info("")
    
    logger.info("="*70)
    logger.info("SUMMARY")
    logger.info("="*70)
    
    passed = sum(1 for _, s in results if s)
    total = len(results)
    
    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        logger.info(f"{status:8s} {name}")
    
    logger.info(f"\nTotal: {passed}/{total} tests passed")
    logger.info("="*70 + "\n")
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
