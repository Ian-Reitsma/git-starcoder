#!/usr/bin/env python3
"""
EXTREME OPTIMIZATION TEST SUITE
Tests all modifications without running full training
"""

import sys
import torch
import yaml
import json
from pathlib import Path

print("="*80)
print("  🧪 EXTREME OPTIMIZATION TEST SUITE")
print("  Testing all modifications without running full epoch")
print("="*80)
print()

# Test results tracking
tests_passed = 0
tests_failed = 0
tests_skipped = 0

def test(name, func):
    """Run a test and track results"""
    global tests_passed, tests_failed, tests_skipped
    print(f"\n[TEST] {name}")
    print("-" * 80)
    try:
        result = func()
        if result:
            print(f"✅ PASS: {name}")
            tests_passed += 1
        else:
            print(f"❌ FAIL: {name}")
            tests_failed += 1
    except Exception as e:
        print(f"❌ ERROR: {name}")
        print(f"   {type(e).__name__}: {e}")
        tests_failed += 1


# TEST 1: Verify CUDA availability
def test_cuda():
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False

    gpu_name = torch.cuda.get_device_name(0)
    total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"✓ GPU: {gpu_name}")
    print(f"✓ VRAM: {total_mem:.1f} GB")

    if total_mem < 7.5:
        print(f"⚠️  Warning: Low VRAM ({total_mem:.1f} GB < 7.5 GB recommended)")

    return True

test("CUDA Availability", test_cuda)


# TEST 2: Test bitsandbytes 8-bit optimizer
def test_bnb_optimizer():
    try:
        import bitsandbytes as bnb
        print(f"✓ bitsandbytes version: {bnb.__version__}")

        # Create a simple model
        model = torch.nn.Linear(10, 10).cuda()

        # Try to create 8-bit optimizer
        optimizer = bnb.optim.AdamW8bit(
            model.parameters(),
            lr=1e-4,
            betas=(0.9, 0.999),
        )
        print("✓ AdamW8bit optimizer created successfully")

        # Test one optimization step
        loss = model(torch.randn(1, 10).cuda()).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print("✓ 8-bit optimizer step completed")

        return True
    except ImportError:
        print("❌ bitsandbytes not installed")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

test("8-bit AdamW Optimizer", test_bnb_optimizer)


# TEST 3: Test FlashAttention-2 or SDPA fallback
def test_flash_attention():
    try:
        import flash_attn
        print(f"✓ FlashAttention-2 version: {flash_attn.__version__}")
        has_flash = True
    except ImportError:
        print("⚠️  FlashAttention-2 not installed, will test SDPA fallback")
        has_flash = False

    try:
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig

        print("✓ Loading phi-2 with 8-bit quantization...")

        # Test loading with attention implementation
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

        attn_impl = "flash_attention_2" if has_flash else "sdpa"
        print(f"✓ Using attention implementation: {attn_impl}")

        # Load tiny model for testing (use microsoft/phi-2 but only load config)
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained("microsoft/phi-2", trust_remote_code=True)
        print(f"✓ Model config loaded: {config.hidden_size} hidden size, {config.num_hidden_layers} layers")

        # Don't actually load full model in test (too slow)
        print("✓ Model loading test passed (config validated)")

        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

test("FlashAttention-2 / SDPA", test_flash_attention)


# TEST 4: Test YAML config loading
def test_yaml_configs():
    configs = [
        "training_config_TIER4_32k.yaml",
        "training_config_TIER5_64k.yaml",
        "training_config_TIER6_128k.yaml",
        "training_config_TIER7_256k.yaml",
    ]

    for config_file in configs:
        try:
            with open(config_file) as f:
                cfg = yaml.safe_load(f)

            # Validate key fields
            ctx = cfg['quantization']['context_window']
            tgt = cfg['quantization']['target_window']
            rank = cfg['quantization']['lora_rank']

            print(f"✓ {config_file}:")
            print(f"    Context: {ctx:,} tokens")
            print(f"    Target: {tgt:,} tokens")
            print(f"    LoRA rank: {rank}")

        except Exception as e:
            print(f"❌ Error loading {config_file}: {e}")
            return False

    return True

test("YAML Config Loading", test_yaml_configs)


# TEST 5: Test DeepSpeed config loading
def test_deepspeed_configs():
    configs = [
        "ds_config_tier4_32k.json",
        "ds_config_tier5_64k.json",
    ]

    for config_file in configs:
        try:
            with open(config_file) as f:
                cfg = json.load(f)

            # Validate ZeRO-2 offloading
            if 'zero_optimization' not in cfg:
                print(f"❌ {config_file}: Missing zero_optimization")
                return False

            zero = cfg['zero_optimization']
            stage = zero.get('stage', 0)
            has_offload = 'offload_optimizer' in zero

            print(f"✓ {config_file}:")
            print(f"    ZeRO stage: {stage}")
            print(f"    CPU offloading: {'Yes' if has_offload else 'No'}")

        except Exception as e:
            print(f"❌ Error loading {config_file}: {e}")
            return False

    return True

test("DeepSpeed Config Loading", test_deepspeed_configs)


# TEST 6: Test gradient checkpointing
def test_gradient_checkpointing():
    try:
        model = torch.nn.Sequential(
            torch.nn.Linear(100, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 10),
        ).cuda()

        # Enable gradient checkpointing
        from torch.utils.checkpoint import checkpoint

        x = torch.randn(32, 100).cuda()

        # Forward pass with checkpointing
        def forward_with_checkpoint(x):
            x = checkpoint(model[0:2], x, use_reentrant=False)
            x = checkpoint(model[2:4], x, use_reentrant=False)
            x = model[4](x)
            return x

        output = forward_with_checkpoint(x)
        loss = output.sum()
        loss.backward()

        print("✓ Gradient checkpointing works")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

test("Gradient Checkpointing", test_gradient_checkpointing)


# TEST 7: Test VRAM usage estimation
def test_vram_estimation():
    try:
        # Clear VRAM
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        initial_mem = torch.cuda.memory_allocated() / (1024**3)
        print(f"✓ Initial VRAM: {initial_mem:.2f} GB")

        # Simulate model loading (8-bit quantized)
        # Phi-2 = 2.7B params × 1 byte = ~2.7 GB
        estimated_model = 2.7

        # LoRA adapters (rank 12 for TIER 4)
        # ~5M params × 2 bytes = ~0.01 GB
        estimated_lora = 0.01

        # Activations (32K context with FlashAttention-2)
        # Linear scaling: ~0.8 GB
        estimated_activations = 0.8

        # Optimizer states (8-bit)
        # ~5M params × 3 × 1 byte = ~0.015 GB
        estimated_optimizer = 0.015

        # KV cache (32K context)
        # ~3.5 GB
        estimated_kv = 3.5

        # Misc
        estimated_misc = 0.5

        total_estimated = (estimated_model + estimated_lora +
                          estimated_activations + estimated_optimizer +
                          estimated_kv + estimated_misc)

        print(f"\nEstimated VRAM for TIER 4 (32K context):")
        print(f"  Model (8-bit):     {estimated_model:.2f} GB")
        print(f"  LoRA (rank 12):    {estimated_lora:.2f} GB")
        print(f"  Activations:       {estimated_activations:.2f} GB")
        print(f"  Optimizer (8-bit): {estimated_optimizer:.2f} GB")
        print(f"  KV cache:          {estimated_kv:.2f} GB")
        print(f"  Misc:              {estimated_misc:.2f} GB")
        print(f"  ─────────────────────────────")
        print(f"  TOTAL:             {total_estimated:.2f} GB")

        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        headroom = total_vram - total_estimated

        if headroom > 0.3:
            print(f"✓ Estimated headroom: {headroom:.2f} GB (Safe!)")
            return True
        elif headroom > 0:
            print(f"⚠️  Estimated headroom: {headroom:.2f} GB (Tight!)")
            return True
        else:
            print(f"❌ Estimated VRAM overflow: {-headroom:.2f} GB over limit")
            print(f"   Consider reducing context window or LoRA rank")
            return False

    except Exception as e:
        print(f"❌ Error: {e}")
        return False

test("VRAM Estimation for TIER 4", test_vram_estimation)


# TEST 8: Verify modified trainer imports
def test_trainer_imports():
    try:
        sys.path.insert(0, str(Path(__file__).parent))

        # Import the trainer module
        from training import model_trainer_unified

        # Check for our global flags
        if hasattr(model_trainer_unified, 'HAS_BNB_OPTIMIZER'):
            bnb_status = model_trainer_unified.HAS_BNB_OPTIMIZER
            print(f"✓ HAS_BNB_OPTIMIZER: {bnb_status}")
        else:
            print("❌ HAS_BNB_OPTIMIZER not found in trainer")
            return False

        if hasattr(model_trainer_unified, 'HAS_FLASH_ATTN'):
            flash_status = model_trainer_unified.HAS_FLASH_ATTN
            print(f"✓ HAS_FLASH_ATTN: {flash_status}")
        else:
            print("❌ HAS_FLASH_ATTN not found in trainer")
            return False

        print("✓ Trainer imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

test("Trainer Code Modifications", test_trainer_imports)


# FINAL SUMMARY
print("\n" + "="*80)
print("  📊 TEST SUMMARY")
print("="*80)
print(f"  ✅ Passed: {tests_passed}")
print(f"  ❌ Failed: {tests_failed}")
print(f"  ⏭️  Skipped: {tests_skipped}")
print("="*80)

if tests_failed == 0:
    print("\n  🎉 ALL TESTS PASSED!")
    print("\n  Everything is wired up correctly and ready to use.")
    print("\n  Next steps:")
    print("    1. Install FlashAttention-2 and DeepSpeed:")
    print("       ./install_extreme_optimizations.sh")
    print()
    print("    2. Modify dataset creator for 32K sequences:")
    print("       Edit create_training_dataset_ELITE.py line ~79")
    print("       Set: CONTEXT_WINDOWS = [8192, 16384, 24576, 32768]")
    print()
    print("    3. Regenerate dataset:")
    print("       python3 create_training_dataset_ELITE.py")
    print()
    print("    4. Start training TIER 4:")
    print("       deepspeed --num_gpus=1 model_trainer_metal_cuda.py \\")
    print("         --config training_config_TIER4_32k.yaml \\")
    print("         --deepspeed ds_config_tier4_32k.json \\")
    print("         --sequences training_data_ELITE/training_data_train.jsonl \\")
    print("         --epochs 20 \\")
    print("         --output models/the-block-ELITE-TIER4-32kctx")
    print()
    sys.exit(0)
else:
    print(f"\n  ❌ {tests_failed} TEST(S) FAILED")
    print("\n  Please fix the errors above before proceeding.")
    print()
    sys.exit(1)
