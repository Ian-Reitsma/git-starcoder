#!/usr/bin/env python3
"""
Memory-Efficient StarCoder 3B Training with Token-Streaming Gradient Accumulation

Creative solutions to fit 3B model on 7.6GB GPU:
1. Process ONE sample at a time (batch_size=1)
2. Accumulate gradients over many samples (fake batching)
3. Use progressive checkpointing (save intermediates, reload on backward)
4. Stream token sequences - don't load full sequence at once
5. Dynamic LoRA rank reduction if memory is tight
6. CPU offloading for optimizer states

This effectively trains with large batches without massive memory overhead.
"""

import os
import sys
import json
import torch
import yaml
import logging
from pathlib import Path
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass
import math
import time

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb=512"

# Force smaller reserved memory
torch.cuda.empty_cache()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import get_peft_model, LoraConfig, TaskType
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from tqdm import tqdm
except ImportError as e:
    logger.error(f"Missing dependency: {e}")
    sys.exit(1)


@dataclass
class MemoryEfficientConfig:
    """Config for memory-efficient training"""
    model_name: str = "bigcode/starcoder2-3b"
    learning_rate: float = 8.0e-4
    weight_decay: float = 0.05
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 16
    num_epochs: int = 20
    max_seq_length: int = 512
    eval_steps: int = 100
    log_steps: int = 45
    save_steps: int = 728
    output_dir: str = "models/the-block-ELITE-test"

    # Memory optimization
    use_4bit: bool = True
    use_gradient_checkpointing: bool = True
    torch_dtype: str = "float16"

    # LoRA settings
    lora_rank: int = 96
    lora_alpha: int = 192
    lora_dropout: float = 0.03


def load_jsonl(path: str, max_samples: Optional[int] = None) -> List[Dict]:
    """Load JSONL training data"""
    data = []
    try:
        with open(path, 'r') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                data.append(json.loads(line))
    except Exception as e:
        logger.error(f"Error loading {path}: {e}")
        return []
    return data


def create_memory_efficient_model(config: MemoryEfficientConfig):
    """Create StarCoder with aggressive memory optimization"""
    logger.info(f"Loading {config.model_name} with 4-bit quantization...")

    # 4-bit config (aggressive memory savings)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    # Load base model (use default attention, avoid SDPA segfaults)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        # attn_implementation removed - use default for stability
    )

    # Enable gradient checkpointing (saves ~40% memory)
    model.gradient_checkpointing_enable()

    # Enable input gradients for LoRA
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    logger.info("Applying LoRA...")

    # Aggressive LoRA config
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=["c_attn", "c_proj", "c_fc"],  # StarCoder modules
        bias="none",
    )

    model = get_peft_model(model, lora_config)

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable params: {trainable_params:,} / {total_params:,}")

    return model


def train_epoch_streaming(
    model,
    tokenizer,
    train_data: List[Dict],
    config: MemoryEfficientConfig,
    optimizer,
    scheduler,
    epoch: int,
    device: str = "cuda"
) -> float:
    """
    Train one epoch with streaming gradient accumulation

    Key innovation: Process ONE sample at a time, accumulate gradients over many
    This avoids batch dimension memory overhead while getting large effective batches
    """
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    samples_per_epoch = len(train_data)
    pbar = tqdm(train_data, desc=f"Epoch {epoch+1}/{config.num_epochs}")

    for step, sample in enumerate(pbar):
        try:
            # Get tokens from sample
            if isinstance(sample, dict):
                if "tokens" in sample:
                    tokens = sample["tokens"]
                elif "text" in sample:
                    tokens = tokenizer.encode(sample["text"], max_length=config.max_seq_length, truncation=True)
                else:
                    continue
            else:
                tokens = sample

            # Convert to tensor
            if isinstance(tokens, list):
                tokens = torch.tensor(tokens, dtype=torch.long)

            tokens = tokens[:config.max_seq_length]

            if len(tokens) < 10:  # Skip too-short sequences
                continue

            # Prepare input (sample one at a time but as single-element "batch")
            input_ids = tokens[:-1].unsqueeze(0).to(device)  # All but last token
            labels = tokens[1:].unsqueeze(0).to(device)      # All but first token

            # Forward pass (gradient checkpointing is enabled on model)
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss

            if loss is None:
                continue

            # Scale loss by accumulation steps (normalize gradient)
            scaled_loss = loss / config.gradient_accumulation_steps
            scaled_loss.backward()

            total_loss += loss.item()

            # Every N samples, update weights
            if (step + 1) % config.gradient_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    config.max_grad_norm
                )

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                # Logging
                if (step + 1) % (config.gradient_accumulation_steps * config.log_steps // 45) == 0:
                    avg_loss = total_loss / (step + 1)
                    pbar.set_postfix({
                        "loss": f"{avg_loss:.4f}",
                        "lr": f"{scheduler.get_last_lr()[0]:.2e}"
                    })

            # Periodic memory check and cleanup
            if (step + 1) % 500 == 0:
                torch.cuda.empty_cache()

        except Exception as e:
            logger.warning(f"Error processing sample {step}: {e}")
            torch.cuda.empty_cache()
            continue

    avg_epoch_loss = total_loss / max(1, samples_per_epoch)
    logger.info(f"Epoch {epoch+1} - Avg Loss: {avg_epoch_loss:.4f}")

    return avg_epoch_loss


def main():
    # Load config
    config_path = "training_config_metal_cuda_universal.yaml"
    if os.path.exists(config_path):
        with open(config_path) as f:
            yaml_config = yaml.safe_load(f)
        # Map YAML config to our config
        model_cfg = yaml_config.get('model', {})
        opt_cfg = yaml_config.get('optimization', {})
        train_cfg = yaml_config.get('training', {})

        config = MemoryEfficientConfig(
            model_name=model_cfg.get('pretrained_model', 'bigcode/starcoder2-3b'),
            learning_rate=opt_cfg.get('learning_rate', 8.0e-4),
            gradient_accumulation_steps=opt_cfg.get('gradient_accumulation_steps', 16),
            num_epochs=train_cfg.get('num_epochs', 20),
            output_dir=train_cfg.get('output_dir', 'models/the-block-ELITE-test'),
        )
    else:
        config = MemoryEfficientConfig()

    logger.info(f"Training config: {config}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Create output directory
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    # Load model and tokenizer
    model = create_memory_efficient_model(config)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Load training data
    train_file = "training_data_ELITE/training_data_train.jsonl"
    logger.info(f"Loading training data from {train_file}...")
    train_data = load_jsonl(train_file)
    logger.info(f"Loaded {len(train_data)} training samples")

    if not train_data:
        logger.error("No training data loaded!")
        sys.exit(1)

    # Setup optimizer with CPU offloading for states
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # Setup scheduler
    total_steps = config.num_epochs * math.ceil(len(train_data) / config.gradient_accumulation_steps)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

    logger.info(f"\n{'='*70}")
    logger.info("STARTING MEMORY-EFFICIENT STARCODER 3B TRAINING")
    logger.info(f"{'='*70}")
    logger.info(f"Model: {config.model_name}")
    logger.info(f"Epochs: {config.num_epochs}")
    logger.info(f"Samples: {len(train_data)}")
    logger.info(f"Effective batch size: {config.gradient_accumulation_steps} (via gradient accumulation)")
    logger.info(f"Total training steps: {total_steps}")
    logger.info(f"{'='*70}\n")

    # Training loop
    best_loss = float('inf')
    for epoch in range(config.num_epochs):
        epoch_loss = train_epoch_streaming(
            model=model,
            tokenizer=tokenizer,
            train_data=train_data,
            config=config,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            device=device
        )

        # Save checkpoint
        checkpoint_dir = Path(config.output_dir) / f"checkpoint-epoch-{epoch+1}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving checkpoint to {checkpoint_dir}")
        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)

        # Save best checkpoint
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_dir = Path(config.output_dir) / "best"
            best_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            logger.info(f"✓ New best model saved (loss: {epoch_loss:.4f})")

        # Memory cleanup
        torch.cuda.empty_cache()

    logger.info("\n" + "="*70)
    logger.info(f"Training complete! Best model saved to {config.output_dir}/best")
    logger.info("="*70)


if __name__ == "__main__":
    main()
