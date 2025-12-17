#!/usr/bin/env python3
"""
Unified Model Trainer - Metal (macOS) + CUDA (Linux) Support

This extends the existing OptimizedModelTrainer to:
- Auto-detect device (Metal/CUDA/CPU)
- Adapt training config based on device
- Patch models for device-specific optimizations
- Use appropriate attention backend (Metal FlashAttention, SDPA, xFormers, native)

Usage:
    python model_trainer_metal_cuda.py \
      --config training_config_metal_cuda_universal.yaml \
      --sequences data/token_sequences.json \
      --epochs 10 \
      --output models/the-block-metal-cuda
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional

try:
    from training.model_trainer_unified import OptimizedModelTrainer
except ImportError:
    OptimizedModelTrainer = None

try:
    from device_backend import get_device_backend
except ImportError:
    get_device_backend = None

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MetalCudaUnifiedTrainer:
    """Wrapper around OptimizedModelTrainer with device backend integration."""

    def __init__(
        self,
        config_path: str,
        force_device: Optional[str] = None,
        verbose: bool = False,
    ):
        """Initialize trainer with device backend.
        
        Args:
            config_path: Path to YAML config
            force_device: Force specific device ("cuda", "mps", "cpu", or None for auto)
            verbose: Enable verbose logging
        """
        if OptimizedModelTrainer is None:
            raise ImportError("model_trainer_unified not available")
        if get_device_backend is None:
            raise ImportError("device_backend not available")

        self.config_path = Path(config_path)
        self.verbose = verbose

        # Initialize the base trainer (trainer is now device-backend aware)
        self.trainer = OptimizedModelTrainer(
            str(self.config_path),
            force_device=force_device,
            verbose_device_backend=verbose,
        )

        # Reuse trainer-selected backend for logging/summary
        self.device_backend = getattr(self.trainer, "device_backend", None)
        if verbose and self.device_backend is not None:
            self.device_backend.log_summary()

    def _load_and_adapt_config(self) -> Dict[str, Any]:
        """Load config and adapt for device backend."""
        import yaml
        
        logger.info(f"Loading config: {self.config_path}")
        with open(self.config_path) as f:
            config = yaml.safe_load(f)
        
        # Get device-specific overrides
        overrides = self.device_backend.get_model_config_overrides()
        
        # Apply overrides to model config
        if "model" in config:
            config["model"].update(overrides)
        
        # Log adapted config
        if self.verbose:
            logger.info(f"Config adapted for {self.device_backend.config.device_type}")
            logger.info(f"  torch_dtype: {config['model'].get('torch_dtype')}")
            logger.info(f"  gradient_checkpointing: {config['model'].get('gradient_checkpointing')}")
        
        return config

    def train(self, sequences_path: str, epochs: int, output_dir: str) -> None:
        """Run training.
        
        Args:
            sequences_path: Path to token sequences
            epochs: Number of training epochs
            output_dir: Output directory for checkpoints
        """
        if self.device_backend is not None:
            logger.info(f"Starting training on {self.device_backend.config.device_type}")
            logger.info(f"  Attention backend: {self.device_backend.config.attention_backend}")
            logger.info(f"  Max VRAM: {self.device_backend.config.max_vram_gb:.1f} GB")
        else:
            logger.info(f"Starting training on device={getattr(self.trainer, 'device', 'unknown')}")
        
        # Call base trainer
        self.trainer.train(sequences_path, epochs, output_dir)
        
        logger.info(f"Training complete. Model saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Unified Metal/CUDA Model Trainer"
    )
    parser.add_argument(
        "--config",
        default="training_config_metal_cuda_universal.yaml",
        help="Path to training config",
    )
    parser.add_argument(
        "--sequences",
        required=True,
        help="Path to token sequences",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--output",
        default="models/the-block-metal-cuda",
        help="Output directory",
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=["cuda", "mps", "cpu"],
        help="Force specific device (default: auto-detect)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    
    args = parser.parse_args()
    
    try:
        trainer = MetalCudaUnifiedTrainer(
            config_path=args.config,
            force_device=args.device,
            verbose=args.verbose,
        )
        
        trainer.train(
            sequences_path=args.sequences,
            epochs=args.epochs,
            output_dir=args.output,
        )
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
