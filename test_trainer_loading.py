#!/usr/bin/env python3
"""
Test that simulates the exact model loading process the trainer uses.
This ensures the custom kernel will load correctly during actual training.
"""

import torch
import sys
import os
from transformers import AutoModelForCausalLM, AutoConfig
from transformers.models.phi.modeling_phi import PhiAttention

# Add training directory to path (same as trainer does)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'training'))

print("="*80)
print("SIMULATING TRAINER MODEL LOADING")
print("="*80)

# Step 1: Load model exactly as trainer does
print("\n1. Loading Phi-2 model with 4-bit quantization...")
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-2",
    load_in_4bit=True,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True,
    use_cache=False
)
print("✓ Model loaded")

# Step 2: Check HAS_FLASH_ATTN_1 flag
try:
    from flash_attn.flash_attention import FlashAttention
    HAS_FLASH_ATTN_1 = True
    print("✓ FlashAttention-1 available")
except:
    HAS_FLASH_ATTN_1 = False
    print("✗ FlashAttention-1 NOT available")
    sys.exit(1)

# Step 3: Check if model is Phi
model_name = "microsoft/phi-2"
is_phi = 'phi' in model_name.lower()
print(f"✓ Is Phi model: {is_phi}")

# Step 4: Simulate trainer's custom kernel loading logic
if HAS_FLASH_ATTN_1 and is_phi:
    print("\n2. Loading custom Turing kernel (as trainer does)...")
    print("="*80)
    print("🔧 LOADING CUSTOM FLASHATTENTION TURING KERNEL")
    print("="*80)

    try:
        # Import exactly as trainer does
        training_dir = os.path.dirname(os.path.abspath(__file__))
        training_dir = os.path.join(training_dir, 'training')
        sys.path.insert(0, training_dir)

        from flash_attn_turing_ext import FlashAttentionTuringFunction
        print("✓ Custom Turing kernel loaded successfully!")

        # Patch model exactly as trainer does
        from training.model_trainer_unified import patch_model_with_custom_fa1_turing
        model = patch_model_with_custom_fa1_turing(model, FlashAttentionTuringFunction)
        print("✓ Model patched with custom kernel")

    except Exception as e:
        print(f"✗ Failed to load custom kernel: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# Step 5: Verify patching
print("\n3. Verifying all attention layers are patched...")
from training.model_trainer_unified import PhiCustomTuringAttention

patched = 0
total = 0
for name, module in model.named_modules():
    if isinstance(module, PhiAttention):
        total += 1
    elif isinstance(module, PhiCustomTuringAttention):
        patched += 1

print(f"  Patched: {patched}/32 expected")
if patched != 32:
    print(f"✗ ERROR: Expected 32 patched layers, got {patched}")
    sys.exit(1)

print("✓ All 32 attention layers patched correctly")

# Step 6: Test forward/backward
print("\n4. Testing forward/backward pass...")
model.train()

config = model.config
input_ids = torch.randint(0, config.vocab_size, (1, 64), device=model.device)

outputs = model(input_ids, labels=input_ids)
loss = outputs.loss
print(f"  ✓ Forward pass! Loss: {loss.item():.4f}")

loss.backward()
print(f"  ✓ Backward pass!")

print("\n" + "="*80)
print("✓✓✓ TRAINER LOADING SIMULATION SUCCESSFUL! ✓✓✓")
print("="*80)
print("The custom kernel will load correctly during actual training!")
print("="*80)
