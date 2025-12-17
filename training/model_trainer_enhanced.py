#!/usr/bin/env python3
"""
Enhanced Model Trainer with Detailed Statistics

Training features:
- Dynamic epoch determination
- Early stopping based on validation metrics
- Comprehensive per-epoch statistics
- GPU/CPU/RAM monitoring
- Detailed loss tracking
- Gradient analysis
- Learning rate schedule tracking
- Model checkpoint comparison
"""

import os
import sys
import json
import time
import torch
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict, field
import numpy as np

try:
    from torch.utils.data import DataLoader, TensorDataset
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import get_linear_schedule_with_warmup
    from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling
    from tqdm import tqdm
except ImportError:
    print("Install: pip install torch transformers tqdm")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingStats:
    """Comprehensive training statistics"""
    epoch: int = 0
    total_epochs: int = 0
    steps_completed: int = 0
    total_steps: int = 0
    
    # Loss tracking
    train_loss: float = 0.0
    train_loss_history: List[float] = field(default_factory=list)
    val_loss: float = 0.0
    val_loss_history: List[float] = field(default_factory=list)
    perplexity: float = 0.0
    
    # Hardware monitoring
    gpu_memory_mb: float = 0.0
    gpu_memory_peak_mb: float = 0.0
    cpu_percent: float = 0.0
    ram_mb: float = 0.0
    ram_percent: float = 0.0
    temp_celsius: float = 0.0
    
    # Gradients
    avg_grad_norm: float = 0.0
    max_grad_norm: float = 0.0
    
    # Learning rate
    learning_rate: float = 0.0
    
    # Timing
    epoch_time_seconds: float = 0.0
    total_time_seconds: float = 0.0
    estimated_remaining_seconds: float = 0.0
    
    # Data
    train_samples_processed: int = 0
    total_train_samples: int = 0
    
    # Stopping criteria
    patience_remaining: int = 0
    best_val_loss: float = float('inf')
    epochs_without_improvement: int = 0


class HardwareMonitor:
    """Monitor GPU/CPU/RAM/Thermal performance"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.has_gpu = torch.cuda.is_available()
    
    def get_stats(self) -> Dict:
        stats = {}
        
        # GPU stats
        if self.has_gpu:
            try:
                gpu_props = torch.cuda.get_device_properties(0)
                stats['gpu_name'] = gpu_props.name
                stats['gpu_total_memory_gb'] = gpu_props.total_memory / 1e9
                
                gpu_allocated = torch.cuda.memory_allocated() / 1e6
                gpu_reserved = torch.cuda.memory_reserved() / 1e6
                stats['gpu_memory_allocated_mb'] = gpu_allocated
                stats['gpu_memory_reserved_mb'] = gpu_reserved
                stats['gpu_memory_allocated_pct'] = (gpu_allocated / (gpu_props.total_memory / 1e6)) * 100
                
                # Try to get utilization
                try:
                    import subprocess
                    result = subprocess.run(
                        ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits', '-i', '0'],
                        capture_output=True, text=True, timeout=5
                    )
                    stats['gpu_utilization_pct'] = float(result.stdout.strip())
                except:
                    stats['gpu_utilization_pct'] = None
            except Exception as e:
                logger.warning(f"Could not get GPU stats: {e}")
        
        # CPU stats
        try:
            import psutil
            cpu_percent = psutil.cpu_percent(interval=0.1)
            ram_info = psutil.virtual_memory()
            
            stats['cpu_percent'] = cpu_percent
            stats['cpu_count'] = psutil.cpu_count()
            stats['ram_available_gb'] = ram_info.available / 1e9
            stats['ram_total_gb'] = ram_info.total / 1e9
            stats['ram_percent'] = ram_info.percent
            
            # Temperature (if available)
            try:
                temps = psutil.sensors_temperatures()
                if temps:
                    first_temp = list(temps.values())[0][0].current
                    stats['temp_celsius'] = first_temp
            except:
                pass
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"Could not get CPU stats: {e}")
        
        return stats
    
    def format_stats(self, stats: Dict) -> str:
        """Format stats for display"""
        lines = []
        
        if 'gpu_name' in stats:
            pct = stats.get('gpu_memory_allocated_pct', 0)
            mb = stats.get('gpu_memory_allocated_mb', 0)
            total = stats.get('gpu_total_memory_gb', 0) * 1000
            lines.append(f"  GPU ({stats['gpu_name']}): {pct:.1f}% | {mb:.0f}MB / {total:.0f}MB")
        
        if 'cpu_percent' in stats:
            lines.append(f"  CPU: {stats['cpu_percent']:.1f}% | {stats['cpu_count']} cores")
        
        if 'ram_percent' in stats:
            lines.append(f"  RAM: {stats['ram_percent']:.1f}% | {stats['ram_total_gb']:.0f}GB available")
        
        if 'temp_celsius' in stats:
            lines.append(f"  Thermal: {stats['temp_celsius']:.1f}°C")
        
        return "\n".join(lines)


class EnhancedModelTrainer:
    """Enhanced trainer with comprehensive statistics"""
    
    def __init__(
        self,
        model_name: str = 'gpt2-medium',
        output_dir: str = 'models/the-block-git-model-final',
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.device = torch.device(device)
        self.monitor = HardwareMonitor()
        
        # Training parameters
        self.learning_rate = 5e-5
        self.batch_size = 8
        self.num_workers = 8
        self.warmup_ratio = 0.1
        self.weight_decay = 0.01
        
        # Early stopping
        self.patience = 3
        self.min_delta = 1e-4
        
        # Statistics
        self.stats = TrainingStats()
        self.all_train_losses = []
        self.all_val_losses = []
        self.gradients_history = []
        self.lr_history = []
        
        # Load model and tokenizer
        logger.info(f"Loading {model_name} model...")
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()) / 1e6:.0f}M")
    
    def calculate_gradient_stats(self) -> Tuple[float, float]:
        """Calculate gradient norm statistics"""
        total_norm = 0
        max_norm = 0
        
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                max_norm = max(max_norm, param_norm.item())
        
        total_norm = total_norm ** 0.5
        return total_norm, max_norm
    
    def format_time(self, seconds: float) -> str:
        """Format seconds to human readable time"""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"
    
    def train(
        self,
        train_dataset,
        num_epochs: int = 5,
        eval_dataset=None,
    ) -> Dict:
        """
        Enhanced training loop with detailed statistics.
        """
        logger.info("\n" + "="*70)
        logger.info("STARTING TRAINING")
        logger.info("="*70)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        
        num_training_steps = len(train_loader) * num_epochs
        num_warmup_steps = int(num_training_steps * self.warmup_ratio)
        
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        
        self.stats.total_epochs = num_epochs
        self.stats.total_steps = num_training_steps
        self.stats.total_train_samples = len(train_dataset)
        
        logger.info(f"\nTraining Configuration:")
        logger.info(f"  Model: {self.model_name}")
        logger.info(f"  Epochs: {num_epochs}")
        logger.info(f"  Batch size: {self.batch_size}")
        logger.info(f"  Learning rate: {self.learning_rate}")
        logger.info(f"  Warmup steps: {num_warmup_steps} ({self.warmup_ratio*100:.0f}%)")
        logger.info(f"  Total training steps: {num_training_steps}")
        logger.info(f"  Training samples: {len(train_dataset):,}")
        logger.info(f"  Device: {self.device}")
        
        training_start = time.time()
        
        for epoch in range(num_epochs):
            self.stats.epoch = epoch + 1
            epoch_start = time.time()
            
            logger.info(f"\n" + "-"*70)
            logger.info(f"EPOCH {epoch + 1}/{num_epochs}")
            logger.info("-"*70)
            
            # Training phase
            self.model.train()
            epoch_train_loss = 0
            progress_bar = tqdm(train_loader, desc=f"Training epoch {epoch + 1}")
            
            for step, batch in enumerate(progress_bar):
                self.stats.steps_completed += 1
                self.stats.train_samples_processed = (step + 1) * self.batch_size
                
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device) if len(batch) > 1 else None
                labels = input_ids.clone()
                labels[labels == self.tokenizer.pad_token_id] = -100
                
                outputs = self.model(
                    input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss
                
                epoch_train_loss += loss.item()
                avg_train_loss = epoch_train_loss / (step + 1)
                
                loss.backward()
                avg_grad_norm, max_grad_norm = self.calculate_gradient_stats()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                current_lr = scheduler.get_last_lr()[0]
                self.lr_history.append(current_lr)
                
                # Update progress
                progress_bar.set_postfix({
                    'loss': f"{avg_train_loss:.4f}",
                    'grad_norm': f"{avg_grad_norm:.2f}",
                    'lr': f"{current_lr:.2e}",
                })
                
                # Periodic stats
                if (step + 1) % max(1, len(train_loader) // 5) == 0:
                    hw_stats = self.monitor.get_stats()
                    logger.info(f"  Step {step + 1}/{len(train_loader)} | Loss: {avg_train_loss:.4f}")
                    logger.info("  Hardware:")
                    for line in self.monitor.format_stats(hw_stats).split("\n"):
                        logger.info(line)
            
            self.stats.train_loss = avg_train_loss
            self.all_train_losses.append(avg_train_loss)
            self.stats.train_loss_history.append(avg_train_loss)
            
            epoch_time = time.time() - epoch_start
            self.stats.epoch_time_seconds = epoch_time
            
            # Validation phase (if provided)
            if eval_dataset is not None:
                val_loss = self._evaluate(eval_dataset)
                self.stats.val_loss = val_loss
                self.stats.val_loss_history.append(val_loss)
                self.all_val_losses.append(val_loss)
                self.stats.perplexity = np.exp(val_loss)
                
                # Early stopping check
                if val_loss < self.stats.best_val_loss - self.min_delta:
                    self.stats.best_val_loss = val_loss
                    self.stats.epochs_without_improvement = 0
                    logger.info(f"  ✓ Validation loss improved: {val_loss:.4f}")
                    self._save_checkpoint(epoch)
                else:
                    self.stats.epochs_without_improvement += 1
                    logger.info(f"  • No improvement. Patience: {self.patience - self.stats.epochs_without_improvement}/{self.patience}")
                    
                    if self.stats.epochs_without_improvement >= self.patience:
                        logger.info(f"  Early stopping triggered after {epoch + 1} epochs")
                        break
            
            # Summary
            total_time = time.time() - training_start
            self.stats.total_time_seconds = total_time
            
            if num_epochs > 0:
                avg_epoch_time = total_time / (epoch + 1)
                remaining_epochs = num_epochs - epoch - 1
                self.stats.estimated_remaining_seconds = avg_epoch_time * remaining_epochs
            
            logger.info(f"\n  Epoch Summary:")
            logger.info(f"    Training loss: {self.stats.train_loss:.4f}")
            if eval_dataset is not None:
                logger.info(f"    Validation loss: {self.stats.val_loss:.4f}")
                logger.info(f"    Perplexity: {self.stats.perplexity:.2f}")
            logger.info(f"    Time: {self.format_time(epoch_time)}")
            logger.info(f"    Total time: {self.format_time(total_time)}")
            logger.info(f"    Remaining: {self.format_time(self.stats.estimated_remaining_seconds)}")
        
        return self._generate_final_report()
    
    def _evaluate(self, eval_dataset) -> float:
        """Evaluate on validation set"""
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
        
        self.model.eval()
        total_eval_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Evaluating", disable=True):
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device) if len(batch) > 1 else None
                labels = input_ids.clone()
                labels[labels == self.tokenizer.pad_token_id] = -100
                
                outputs = self.model(
                    input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                total_eval_loss += outputs.loss.item()
        
        avg_eval_loss = total_eval_loss / len(eval_loader)
        return avg_eval_loss
    
    def _save_checkpoint(self, epoch: int):
        """Save model checkpoint"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(self.output_dir))
        self.tokenizer.save_pretrained(str(self.output_dir))
        logger.info(f"  Checkpoint saved to {self.output_dir}")
    
    def _generate_final_report(self) -> Dict:
        """Generate comprehensive training report"""
        logger.info("\n" + "="*70)
        logger.info("TRAINING COMPLETE - FINAL STATISTICS")
        logger.info("="*70)
        
        report = {
            'model_name': self.model_name,
            'epochs_completed': self.stats.epoch,
            'total_steps': self.stats.steps_completed,
            'training': {
                'final_train_loss': self.stats.train_loss,
                'final_val_loss': self.stats.val_loss,
                'final_perplexity': self.stats.perplexity,
                'best_val_loss': self.stats.best_val_loss,
                'min_train_loss': min(self.all_train_losses) if self.all_train_losses else 0,
                'max_train_loss': max(self.all_train_losses) if self.all_train_losses else 0,
            },
            'timing': {
                'total_seconds': self.stats.total_time_seconds,
                'total_minutes': self.stats.total_time_seconds / 60,
                'total_hours': self.stats.total_time_seconds / 3600,
                'seconds_per_epoch': self.stats.total_time_seconds / self.stats.epoch if self.stats.epoch > 0 else 0,
            },
            'hardware': {
                'device': str(self.device),
                'has_gpu': torch.cuda.is_available(),
            },
            'configuration': {
                'batch_size': self.batch_size,
                'learning_rate': self.learning_rate,
                'warmup_ratio': self.warmup_ratio,
                'weight_decay': self.weight_decay,
                'patience': self.patience,
            },
        }
        
        # Log report
        logger.info(f"\nEpochs completed: {report['epochs_completed']}")
        logger.info(f"Total steps: {report['total_steps']}")
        logger.info(f"\nLoss progression:")
        logger.info(f"  Final training loss: {report['training']['final_train_loss']:.4f}")
        logger.info(f"  Final validation loss: {report['training']['final_val_loss']:.4f}")
        logger.info(f"  Final perplexity: {report['training']['final_perplexity']:.2f}")
        logger.info(f"  Best validation loss: {report['training']['best_val_loss']:.4f}")
        logger.info(f"\nTiming:")
        logger.info(f"  Total time: {self.format_time(report['timing']['total_seconds'])}")
        logger.info(f"  Average per epoch: {self.format_time(report['timing']['seconds_per_epoch'])}")
        logger.info(f"\nModel saved to: {self.output_dir}")
        logger.info("="*70 + "\n")
        
        return report


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced model trainer")
    parser.add_argument("--data-path", type=str, help="Path to training data")
    parser.add_argument("--model", type=str, default="gpt2-medium", help="Model name")
    parser.add_argument("--epochs", type=int, help="Number of epochs (auto-determined if not specified)")
    parser.add_argument("--output", type=str, default="models/the-block-git-model-final", help="Output directory")
    
    args = parser.parse_args()
    
    trainer = EnhancedModelTrainer(model_name=args.model, output_dir=args.output)
    
    # Note: This would need actual data loading logic
    logger.info("Trainer initialized (data loading would go here)")


if __name__ == "__main__":
    main()
