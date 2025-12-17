#!/usr/bin/env python3
"""
Git Repository Model Training Pipeline

Fine-tunes a language model on Git commit history to understand
the structure and evolution of your codebase.

Supports multiple base models:
- CodeLlama-7B (best for code)
- Llama-2-7B (good general purpose)
- GPT-2 (lightweight, fast)

Usage:
    python model_trainer.py --input data/token_sequences.json --model-type llama2 --output-dir models/the-block
"""

import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple
import argparse
import logging

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        get_linear_schedule_with_warmup,
        PreTrainedTokenizer,
    )
    import numpy as np
except ImportError:
    print("Please install required packages:")
    print("pip install torch pytorch-lightning transformers")
    sys.exit(1)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GitCommitDataset(Dataset):
    """Dataset for Git commit token sequences"""
    
    def __init__(
        self,
        token_sequences: List[List[int]],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 1024,
    ):
        self.token_sequences = token_sequences
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.token_sequences)
    
    def __getitem__(self, idx):
        tokens = self.token_sequences[idx]
        
        # Pad or truncate to max_length
        if len(tokens) < self.max_length:
            tokens = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        else:
            tokens = tokens[:self.max_length]
        
        attention_mask = [1 if t != self.tokenizer.pad_token_id else 0 for t in tokens]
        
        return {
            "input_ids": torch.tensor(tokens, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(tokens, dtype=torch.long),
        }


class GitModelLightning(pl.LightningModule):
    """PyTorch Lightning wrapper for model training"""
    
    def __init__(
        self,
        model_name: str,
        learning_rate: float = 5e-5,
        warmup_steps: int = 1000,
        total_steps: int = 10000,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        logger.info(f"Loading model: {model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
    
    def forward(self, **kwargs):
        return self.model(**kwargs)
    
    def training_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs.loss
        
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True
        )
        return loss
    
    def validation_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs.loss
        
        self.log(
            "val_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True
        )
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=0.1,
        )
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.total_steps,
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            }
        }


class ModelTrainer:
    """Main training orchestrator"""
    
    def __init__(self, output_dir: str = "models", use_gpu: bool = True):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_gpu = use_gpu and torch.cuda.is_available()
        
        if self.use_gpu:
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"Available VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            logger.info("Using CPU")
    
    def load_token_sequences(self, path: str) -> Tuple[List[List[int]], int]:
        """Load token sequences from JSON"""
        logger.info(f"Loading token sequences from {path}")
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        token_sequences = data["token_sequences"]
        vocab_size = data["vocab_size"]
        
        logger.info(f"Loaded {len(token_sequences)} sequences with vocab size {vocab_size}")
        return token_sequences, vocab_size
    
    def train(
        self,
        token_sequences: List[List[int]],
        model_name: str,
        batch_size: int = 4,
        num_epochs: int = 3,
        learning_rate: float = 5e-5,
        max_length: int = 1024,
        validation_split: float = 0.1,
    ) -> str:
        """Train model on token sequences"""
        
        logger.info(f"Preparing training data (batch_size={batch_size}, max_length={max_length})...")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Create dataset
        dataset = GitCommitDataset(
            token_sequences,
            tokenizer,
            max_length=max_length,
        )
        
        # Split into train/val
        train_size = int((1 - validation_split) * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset,
            [train_size, val_size],
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0 if self.use_gpu else 2,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0 if self.use_gpu else 2,
        )
        
        logger.info(f"Train set: {len(train_dataset)}, Val set: {len(val_dataset)}")
        
        # Calculate total steps
        num_batches = len(train_loader)
        total_steps = num_batches * num_epochs
        warmup_steps = total_steps // 10
        
        logger.info(f"Total steps: {total_steps}, Warmup steps: {warmup_steps}")
        
        # Initialize model
        logger.info(f"Initializing model: {model_name}")
        lit_model = GitModelLightning(
            model_name=model_name,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
        )
        
        # Setup callbacks
        checkpoint_callback = ModelCheckpoint(
            dirpath=str(self.output_dir),
            filename="the-block-git-model-{epoch:02d}-{val_loss:.2f}",
            save_top_k=3,
            monitor="val_loss",
            mode="min",
            save_last=True,
        )
        
        early_stop_callback = EarlyStopping(
            monitor="val_loss",
            patience=2,
            verbose=True,
            mode="min",
        )
        
        # Initialize trainer
        trainer = pl.Trainer(
            max_epochs=num_epochs,
            accelerator="gpu" if self.use_gpu else "cpu",
            devices=1 if self.use_gpu else None,
            callbacks=[checkpoint_callback, early_stop_callback],
            log_every_n_steps=50,
            precision="16-mixed" if self.use_gpu else "32",
            gradient_accumulation_steps=1,
        )
        
        # Train
        logger.info("Starting training...")
        trainer.fit(lit_model, train_loader, val_loader)
        
        # Save final model
        final_model_path = self.output_dir / "the-block-git-model-final"
        logger.info(f"Saving model to {final_model_path}")
        lit_model.model.save_pretrained(str(final_model_path))
        tokenizer.save_pretrained(str(final_model_path))
        
        return str(final_model_path)
    
    def evaluate(self, model_path: str, token_sequences: List[List[int]]) -> Dict:
        """Evaluate trained model"""
        logger.info(f"Evaluating model: {model_path}")
        
        model = AutoModelForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Move to GPU if available
        if self.use_gpu:
            model = model.cuda()
        
        model.eval()
        
        # Evaluate perplexity
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for tokens in token_sequences[:100]:  # Evaluate on first 100 sequences
                if len(tokens) < 2:
                    continue
                
                input_ids = torch.tensor([tokens[:-1]], dtype=torch.long)
                labels = torch.tensor([tokens[1:]], dtype=torch.long)
                
                if self.use_gpu:
                    input_ids = input_ids.cuda()
                    labels = labels.cuda()
                
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss
                
                total_loss += loss.item() * (len(tokens) - 1)
                total_tokens += len(tokens) - 1
        
        perplexity = torch.exp(torch.tensor(total_loss / total_tokens))
        
        return {
            "model_path": model_path,
            "perplexity": perplexity.item(),
            "total_loss": total_loss,
            "total_tokens": total_tokens,
        }


def main():
    parser = argparse.ArgumentParser(description="Train model on Git commit history")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input token sequences JSON file"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="gpt2",
        help="HuggingFace model name to fine-tune"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Output directory for trained models"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for training"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=1024,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate model after training"
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU usage"
    )
    
    args = parser.parse_args()
    
    trainer = ModelTrainer(output_dir=args.output_dir, use_gpu=not args.no_gpu)
    
    # Load data
    token_sequences, vocab_size = trainer.load_token_sequences(args.input)
    
    # Train
    model_path = trainer.train(
        token_sequences=token_sequences,
        model_name=args.model_name,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
    )
    
    logger.info(f"Model trained and saved to {model_path}")
    
    # Optionally evaluate
    if args.evaluate:
        eval_results = trainer.evaluate(model_path, token_sequences)
        print("\n" + "="*60)
        print("Evaluation Results")
        print("="*60)
        for key, value in eval_results.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")


if __name__ == "__main__":
    main()
