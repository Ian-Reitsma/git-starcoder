#!/usr/bin/env python3
"""
Resume training from a saved checkpoint
This script loads a previously saved LoRA checkpoint and continues fine-tuning
"""

import os
import sys
import torch
import yaml
import json
import logging
from pathlib import Path
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig, get_peft_model

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_checkpoint_for_training(
    checkpoint_path: str,
    base_model_name: str = "gpt2",
    device: str = "cuda"
) -> tuple:
    """
    Load a checkpoint (LoRA adapter) and prepare it for continued training

    Args:
        checkpoint_path: Path to the LoRA checkpoint (e.g., models/the-block-ELITE-test/best)
        base_model_name: Base model identifier
        device: Device to load on (cuda/cpu/mps)

    Returns:
        (model, tokenizer) tuple ready for training
    """
    logger.info(f"Loading checkpoint from {checkpoint_path}")

    # Check if checkpoint exists
    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    # Load tokenizer from checkpoint
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

    # Load base model
    logger.info(f"Loading base model: {base_model_name}")
    try:
        # Try loading with 4-bit quantization (if available)
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="auto" if device == "cuda" else device,
            trust_remote_code=True,
        )
        logger.info("✓ Loaded with 4-bit quantization")
    except Exception as e:
        logger.warning(f"Could not load with 4-bit quantization: {e}")
        logger.info("Fallback: Loading without quantization")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="auto" if device == "cuda" else device,
            trust_remote_code=True,
        )

    # Load LoRA adapter onto the base model
    logger.info(f"Loading LoRA adapter from {checkpoint_path}")
    model = PeftModel.from_pretrained(
        base_model,
        checkpoint_path,
        is_trainable=True,  # Important: allows continued training
    )

    # Enable gradient computation
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    logger.info("✓ Checkpoint loaded and ready for training")

    return model, tokenizer


def resume_training_from_checkpoint(
    checkpoint_path: str,
    sequences_file: str,
    config_path: str,
    output_dir: str,
    resume_epoch: int = 2,
    total_epochs: int = 20,
    device: str = "cuda",
    base_model_name: str = "gpt2"
):
    """
    Resume training from a checkpoint

    Note: This is a simplified version. For full compatibility, use the main trainer.
    """
    logger.info(f"\n{'='*60}")
    logger.info("RESUME TRAINING FROM CHECKPOINT")
    logger.info(f"{'='*60}")
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Resume from epoch: {resume_epoch}")
    logger.info(f"Total epochs: {total_epochs}")
    logger.info(f"Data: {sequences_file}")
    logger.info(f"{'='*60}\n")

    # Load the checkpoint
    model, tokenizer = load_checkpoint_for_training(
        checkpoint_path,
        base_model_name=base_model_name,
        device=device
    )

    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    logger.info("\n✓ Successfully loaded:")
    model_name = config.get('model', {}).get('model_name') or config.get('bigcode/starcoder2-3b', 'unknown')
    logger.info(f"  - Model: {model_name}")
    logger.info(f"  - Device: {device}")
    logger.info(f"  - LoRA weights: {checkpoint_path}")
    logger.info(f"\nNext step: Use the loaded model and continue training")
    logger.info(f"\nTo continue, you need to:")
    logger.info(f"  1. Import the model and tokenizer from this script")
    logger.info(f"  2. Load your training data from {sequences_file}")
    logger.info(f"  3. Continue training from epoch {resume_epoch}")

    return model, tokenizer, config


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Resume training from checkpoint")
    parser.add_argument("--checkpoint", default="models/the-block-ELITE-test/best",
                       help="Path to checkpoint")
    parser.add_argument("--base-model", default="gpt2",
                       help="Base model name (default: gpt2)")
    parser.add_argument("--config", default="training_config_metal_cuda_universal.yaml",
                       help="Training config")
    parser.add_argument("--sequences", default="training_data_ELITE/training_data_train.jsonl",
                       help="Training sequences file")
    parser.add_argument("--output", default="models/the-block-ELITE-test",
                       help="Output directory")
    parser.add_argument("--resume-epoch", type=int, default=2,
                       help="Which epoch to resume from (default: 2)")
    parser.add_argument("--total-epochs", type=int, default=20,
                       help="Total epochs to train (default: 20)")
    parser.add_argument("--device", choices=["cuda", "cpu", "mps"], default="cuda",
                       help="Device to use")

    args = parser.parse_args()

    # Run resume
    model, tokenizer, config = resume_training_from_checkpoint(
        checkpoint_path=args.checkpoint,
        sequences_file=args.sequences,
        config_path=args.config,
        output_dir=args.output,
        resume_epoch=args.resume_epoch,
        total_epochs=args.total_epochs,
        device=args.device,
        base_model_name=args.base_model,
    )

    logger.info("\n✓ Checkpoint loaded successfully!")
    logger.info(f"\nModel info:")
    logger.info(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    logger.info(f"\nYou can now use the model for continued training or inference.")
