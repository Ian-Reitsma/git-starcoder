#!/usr/bin/env python3
"""
Fixed & Optimized Model Trainer

Key improvements:
- Actual data loading and dataset creation
- Train/validation split (90/10)
- Hardware-aware batch size selection
- Deterministic training (seed management)
- Config-based hyperparameters
- Per-epoch and final reporting with min/max statistics
- Hardware stats every fixed interval (not per-step)
"""

import os
import sys
import json
import time
import torch
import logging
import random
import numpy as np
import yaml
import psutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict, field

try:
    from torch.utils.data import DataLoader, TensorDataset, random_split
    from torch.optim import AdamW
    # Scheduler helper lives in transformers, not torch
    try:
        from transformers.optimization import get_linear_schedule_with_warmup
    except ImportError:
        from transformers import get_linear_schedule_with_warmup
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    from tqdm import tqdm
except ImportError as e:
    raise ImportError(
        "Missing dependency for training/model_trainer_fixed.py: "
        f"{e}. Install: pip install torch transformers pyyaml tqdm"
    )

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_yaml_config(config_path: str) -> Dict:
    """Load training configuration from YAML"""
    with open(config_path) as f:
        return yaml.safe_load(f)


def set_seeds(seed: int = 42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seeds set to {seed}")


class HardwareMonitor:
    """Monitor GPU/CPU/RAM/Thermal with time-based sampling"""
    
    def __init__(self, interval_seconds: float = 10.0):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.has_gpu = torch.cuda.is_available()
        self.interval = interval_seconds
        self.last_sample_time = time.time()
        self.stats_history = []
        self.peak_gpu_memory_mb = 0
        self.peak_ram_percent = 0
    
    def should_sample(self) -> bool:
        """Check if it's time to sample"""
        elapsed = time.time() - self.last_sample_time
        return elapsed >= self.interval
    
    def get_stats(self) -> Dict:
        """Get current hardware statistics"""
        stats = {'timestamp': time.time()}
        
        # GPU stats
        if self.has_gpu:
            try:
                gpu_props = torch.cuda.get_device_properties(0)
                stats['gpu_name'] = gpu_props.name
                stats['gpu_total_memory_gb'] = gpu_props.total_memory / 1e9
                
                gpu_allocated = torch.cuda.memory_allocated() / 1e6
                stats['gpu_memory_allocated_mb'] = gpu_allocated
                stats['gpu_memory_allocated_pct'] = (gpu_allocated / (gpu_props.total_memory / 1e6)) * 100
                
                self.peak_gpu_memory_mb = max(self.peak_gpu_memory_mb, gpu_allocated)
                
                # Try to get utilization
                try:
                    import subprocess
                    result = subprocess.run(
                        ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits', '-i', '0'],
                        capture_output=True, text=True, timeout=5
                    )
                    stats['gpu_utilization_pct'] = float(result.stdout.strip())
                except:
                    pass
            except Exception as e:
                logger.warning(f"Could not get GPU stats: {e}")
        
        # CPU and RAM stats
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            ram_info = psutil.virtual_memory()
            
            stats['cpu_percent'] = cpu_percent
            stats['cpu_count'] = psutil.cpu_count()
            stats['ram_percent'] = ram_info.percent
            stats['ram_available_gb'] = ram_info.available / 1e9
            stats['ram_total_gb'] = ram_info.total / 1e9
            
            self.peak_ram_percent = max(self.peak_ram_percent, ram_info.percent)
            
            # Temperature
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
            logger.warning(f"Could not get CPU/RAM stats: {e}")
        
        self.last_sample_time = time.time()
        self.stats_history.append(stats)
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
            lines.append(f"  RAM: {stats['ram_percent']:.1f}% | {stats['ram_total_gb']:.0f}GB total")
        
        if 'temp_celsius' in stats:
            lines.append(f"  Thermal: {stats['temp_celsius']:.1f}°C")
        
        return "\n".join(lines)


class OptimizedModelTrainer:
    """Trainer with data loading, splitting, and config-based parameters"""
    
    def __init__(
        self,
        model_name: str = 'gpt2-medium',
        output_dir: str = 'models/the-block-git-model-final',
        config_path: str = 'training_config.yaml',
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.device = torch.device(device)
        
        # Load configuration
        self.config = load_yaml_config(config_path)
        training_cfg = self.config['training']
        
        logger.info(f"Training config loaded from {config_path}")
        
        # Set seeds first
        set_seeds(training_cfg['seed'])
        
        # Hardware-aware batch size
        self.batch_size = self._get_batch_size()
        self.num_workers = self._get_num_workers()
        
        # Learning rate (scaled by batch size)
        self.learning_rate = training_cfg['base_learning_rate']
        if self.batch_size != training_cfg['batch_size_reference']:
            scale = self.batch_size / training_cfg['batch_size_reference']
            self.learning_rate *= scale
            logger.info(f"Learning rate scaled by {scale:.2f}x (batch size {self.batch_size})")
        
        # Other hyperparameters
        self.warmup_ratio = training_cfg['warmup_ratio']
        self.weight_decay = training_cfg['weight_decay']
        self.patience = training_cfg['patience']
        self.min_delta = training_cfg['min_delta']
        self.val_split = training_cfg['validation_split']
        
        # Hardware monitor
        self.monitor = HardwareMonitor(
            interval_seconds=self.config['hardware_monitoring']['collection_interval_seconds']
        )
        
        # Statistics
        self.train_losses = []
        self.val_losses = []
        self.lr_history = []
        self.grad_norm_history = []
        
        # Load model and tokenizer
        logger.info(f"Loading {model_name} model...")
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        
        num_params = sum(p.numel() for p in self.model.parameters()) / 1e6
        logger.info(f"Model loaded: {num_params:.0f}M parameters")
    
    def _get_batch_size(self) -> int:
        """Get batch size based on GPU memory"""
        training_cfg = self.config['training']
        hw_cfg = self.config['hardware_monitoring']
        
        if not torch.cuda.is_available():
            return training_cfg['batch_size_small']
        
        try:
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            if gpu_memory_gb >= hw_cfg['gpu_memory_threshold_large_gb']:
                batch_size = training_cfg['batch_size_large']
                logger.info(f"GPU memory: {gpu_memory_gb:.1f}GB → batch_size={batch_size} (large)")
            elif gpu_memory_gb >= hw_cfg['gpu_memory_threshold_medium_gb']:
                batch_size = training_cfg['batch_size_medium']
                logger.info(f"GPU memory: {gpu_memory_gb:.1f}GB → batch_size={batch_size} (medium)")
            else:
                batch_size = training_cfg['batch_size_small']
                logger.info(f"GPU memory: {gpu_memory_gb:.1f}GB → batch_size={batch_size} (small)")
            
            return batch_size
        except:
            return training_cfg['batch_size_reference']
    
    def _get_num_workers(self) -> int:
        """Get num_workers based on CPU count"""
        training_cfg = self.config['training']
        hw_cfg = self.config['hardware_monitoring']
        
        cpu_count = psutil.cpu_count() or 1
        num_workers = min(
            training_cfg['num_workers'],
            max(training_cfg['num_workers_min'], cpu_count // 2)
        )
        logger.info(f"CPU cores: {cpu_count} → num_workers={num_workers}")
        return num_workers
    
    def load_data(self, data_path: str) -> Tuple:
        """Load token sequences and create datasets"""
        logger.info(f"\nLoading token sequences from {data_path}...")
        
        with open(data_path) as f:
            sequences = json.load(f)
        
        num_sequences = len(sequences)
        logger.info(f"Loaded {num_sequences} sequences")
        
        # Convert to tensor dataset
        # Each sequence is a list of token IDs (length 2048)
        token_ids = [torch.tensor(seq['tokens']) for seq in sequences]
        dataset = TensorDataset(*torch.stack(token_ids))
        
        # Create train/val split
        n = len(dataset)
        n_val = max(1, int(self.val_split * n))
        n_train = n - n_val
        
        train_dataset, val_dataset = random_split(dataset, [n_train, n_val])
        
        logger.info(f"Train/val split: {n_train} / {n_val} ({self.val_split*100:.0f}% validation)")
        
        return train_dataset, val_dataset
    
    def calculate_training_params(self, num_sequences: int) -> Dict:
        """Calculate optimal epochs based on actual sequence count"""
        epoch_cfg = self.config['epoch_calculation']
        
        total_tokens = num_sequences * 2048
        target_tokens = epoch_cfg['target_tokens']
        
        # Formula: epochs = clamp(floor(target_tokens / total_tokens), min, max)
        ideal_epochs = target_tokens / total_tokens
        epochs = int(np.floor(ideal_epochs))
        epochs = np.clip(epochs, epoch_cfg['min_epochs'], epoch_cfg['max_epochs'])
        
        logger.info(f"\nTraining parameter calculation:")
        logger.info(f"  Total tokens: {total_tokens:,} ({total_tokens/1e6:.1f}M)")
        logger.info(f"  Target tokens: {target_tokens:,} ({target_tokens/1e6:.1f}M)")
        logger.info(f"  Ideal epochs: {ideal_epochs:.2f} → Clamped: {epochs}")
        
        steps_per_epoch = (num_sequences * (1 - self.val_split)) // self.batch_size
        if (num_sequences * (1 - self.val_split)) % self.batch_size != 0:
            steps_per_epoch += 1
        
        total_steps = steps_per_epoch * epochs
        warmup_steps = min(
            max(self.config['training']['warmup_steps_min'], int(0.1 * total_steps)),
            self.config['training']['warmup_steps_max']
        )
        
        # Estimate training time (rough: 1.5 seconds per step on RTX 2060)
        estimated_time_minutes = (total_steps * 1.5) / 60
        
        params = {
            'epochs': epochs,
            'steps_per_epoch': steps_per_epoch,
            'total_steps': total_steps,
            'warmup_steps': warmup_steps,
            'estimated_time_minutes': estimated_time_minutes,
        }
        
        logger.info(f"  Epochs: {epochs}")
        logger.info(f"  Steps/epoch: {steps_per_epoch}")
        logger.info(f"  Total steps: {total_steps}")
        logger.info(f"  Warmup steps: {warmup_steps}")
        logger.info(f"  Estimated time: {estimated_time_minutes:.1f}m")
        
        return params
    
    def format_time(self, seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            return f"{seconds/3600:.2f}h"
    
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
    
    def train(
        self,
        train_dataset,
        val_dataset,
        num_epochs: int = 5,
    ) -> Dict:
        """Complete training loop"""
        logger.info("\n" + "="*70)
        logger.info("STARTING TRAINING")
        logger.info("="*70)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.config['training']['pin_memory'],
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.config['training']['pin_memory'],
        )
        
        num_training_steps = len(train_loader) * num_epochs
        num_warmup_steps = int(num_training_steps * self.warmup_ratio)
        
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        
        logger.info(f"\nTraining Configuration:")
        logger.info(f"  Model: {self.model_name}")
        logger.info(f"  Epochs: {num_epochs}")
        logger.info(f"  Batch size: {self.batch_size}")
        logger.info(f"  Learning rate: {self.learning_rate:.2e}")
        logger.info(f"  Total training steps: {num_training_steps}")
        logger.info(f"  Warmup steps: {num_warmup_steps}")
        logger.info(f"  Device: {self.device}")
        
        training_start = time.time()
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            logger.info(f"\n" + "-"*70)
            logger.info(f"EPOCH {epoch + 1}/{num_epochs}")
            logger.info("-"*70)
            
            # Training phase
            self.model.train()
            epoch_train_loss = 0
            progress_bar = tqdm(train_loader, desc=f"Training")
            
            for batch_idx, batch in enumerate(progress_bar):
                input_ids = batch[0].to(self.device)
                labels = input_ids.clone()
                labels[labels == self.tokenizer.pad_token_id] = -100
                
                outputs = self.model(input_ids, labels=labels)
                loss = outputs.loss
                
                epoch_train_loss += loss.item()
                avg_train_loss = epoch_train_loss / (batch_idx + 1)
                
                loss.backward()
                avg_grad_norm, max_grad_norm = self.calculate_gradient_stats()
                self.grad_norm_history.append((avg_grad_norm, max_grad_norm))
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                current_lr = scheduler.get_last_lr()[0]
                self.lr_history.append(current_lr)
                
                progress_bar.set_postfix({
                    'loss': f"{avg_train_loss:.4f}",
                    'grad': f"{avg_grad_norm:.2f}",
                    'lr': f"{current_lr:.2e}",
                })
            
            self.train_losses.append(epoch_train_loss / len(train_loader))
            train_loss = self.train_losses[-1]
            
            # Validation phase
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validating", disable=True):
                    input_ids = batch[0].to(self.device)
                    labels = input_ids.clone()
                    labels[labels == self.tokenizer.pad_token_id] = -100
                    
                    outputs = self.model(input_ids, labels=labels)
                    val_loss += outputs.loss.item()
            
            val_loss = val_loss / len(val_loader)
            self.val_losses.append(val_loss)
            perplexity = np.exp(val_loss)
            
            # Early stopping check
            if val_loss < best_val_loss - self.min_delta:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                logger.info(f"  ✓ Validation loss improved: {val_loss:.4f}")
                self._save_checkpoint()
            else:
                epochs_without_improvement += 1
                logger.info(f"  • No improvement. Patience: {self.patience - epochs_without_improvement}/{self.patience}")
                
                if epochs_without_improvement >= self.patience:
                    logger.info(f"  Early stopping triggered after {epoch + 1} epochs")
                    break
            
            # Epoch summary
            epoch_time = time.time() - epoch_start
            total_time = time.time() - training_start
            avg_epoch_time = total_time / (epoch + 1)
            remaining_epochs = num_epochs - epoch - 1
            remaining_time = avg_epoch_time * remaining_epochs
            
            logger.info(f"\n  Epoch Summary:")
            logger.info(f"    Train loss: {train_loss:.4f}")
            logger.info(f"    Val loss: {val_loss:.4f}")
            logger.info(f"    Perplexity: {perplexity:.2f}")
            logger.info(f"    Time: {self.format_time(epoch_time)}")
            logger.info(f"    Total: {self.format_time(total_time)}")
            logger.info(f"    Remaining: {self.format_time(remaining_time)}")
            
            # Hardware stats if available
            if self.monitor.should_sample():
                hw_stats = self.monitor.get_stats()
                logger.info(f"  Hardware:")
                for line in self.monitor.format_stats(hw_stats).split("\n"):
                    logger.info(line)
        
        return self._generate_final_report(time.time() - training_start)
    
    def _save_checkpoint(self):
        """Save best model checkpoint"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(self.output_dir))
        self.tokenizer.save_pretrained(str(self.output_dir))
    
    def _generate_final_report(self, total_time: float) -> Dict:
        """Generate comprehensive training report"""
        logger.info("\n" + "="*70)
        logger.info("TRAINING COMPLETE - FINAL STATISTICS")
        logger.info("="*70)
        
        # Min/max statistics
        min_train_loss = min(self.train_losses) if self.train_losses else 0
        max_train_loss = max(self.train_losses) if self.train_losses else 0
        min_val_loss = min(self.val_losses) if self.val_losses else 0
        max_val_loss = max(self.val_losses) if self.val_losses else 0
        
        if self.grad_norm_history:
            avg_norms, max_norms = zip(*self.grad_norm_history)
            min_grad_norm = min(avg_norms)
            max_grad_norm = max(max_norms)
        else:
            min_grad_norm = 0
            max_grad_norm = 0
        
        if self.lr_history:
            min_lr = min(self.lr_history)
            max_lr = max(self.lr_history)
        else:
            min_lr = 0
            max_lr = 0
        
        report = {
            'model_name': self.model_name,
            'config': self.config,  # Include full config for reproducibility
            'epochs_completed': len(self.train_losses),
            'training': {
                'final_train_loss': self.train_losses[-1] if self.train_losses else 0,
                'final_val_loss': self.val_losses[-1] if self.val_losses else 0,
                'final_perplexity': float(np.exp(self.val_losses[-1])) if self.val_losses else 0,
                'best_val_loss': min(self.val_losses) if self.val_losses else 0,
                'min_train_loss': min_train_loss,
                'max_train_loss': max_train_loss,
                'min_val_loss': min_val_loss,
                'max_val_loss': max_val_loss,
                'loss_history': {
                    'train': [float(x) for x in self.train_losses],
                    'val': [float(x) for x in self.val_losses],
                },
            },
            'gradients': {
                'min_norm': min_grad_norm,
                'max_norm': max_grad_norm,
                'history': [(float(avg), float(max_)) for avg, max_ in self.grad_norm_history],
            },
            'learning_rate': {
                'min': min_lr,
                'max': max_lr,
                'history': [float(x) for x in self.lr_history],
            },
            'hardware': {
                'device': str(self.device),
                'has_gpu': torch.cuda.is_available(),
                'batch_size': self.batch_size,
                'num_workers': self.num_workers,
                'peak_gpu_memory_mb': self.monitor.peak_gpu_memory_mb,
                'peak_ram_percent': self.monitor.peak_ram_percent,
            },
            'timing': {
                'total_seconds': total_time,
                'total_minutes': total_time / 60,
                'seconds_per_epoch': total_time / len(self.train_losses) if self.train_losses else 0,
            },
        }
        
        # Log summary
        logger.info(f"\nEpochs completed: {report['epochs_completed']}")
        logger.info(f"\nLoss progression:")
        logger.info(f"  Train: {min_train_loss:.4f} → {self.train_losses[-1]:.4f}")
        logger.info(f"  Val: {min_val_loss:.4f} → {self.val_losses[-1]:.4f}")
        logger.info(f"  Final perplexity: {report['training']['final_perplexity']:.2f}")
        logger.info(f"\nGradient norms:")
        logger.info(f"  Min: {min_grad_norm:.4f}")
        logger.info(f"  Max: {max_grad_norm:.4f}")
        logger.info(f"\nLearning rate:")
        logger.info(f"  Min: {min_lr:.2e}")
        logger.info(f"  Max: {max_lr:.2e}")
        logger.info(f"\nHardware peaks:")
        logger.info(f"  GPU memory: {self.monitor.peak_gpu_memory_mb:.0f}MB")
        logger.info(f"  RAM: {self.monitor.peak_ram_percent:.1f}%")
        logger.info(f"\nTiming:")
        logger.info(f"  Total: {self.format_time(total_time)}")
        logger.info(f"  Per-epoch: {self.format_time(report['timing']['seconds_per_epoch'])}")
        logger.info(f"\nModel saved to: {self.output_dir}")
        logger.info("="*70 + "\n")
        
        return report


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Optimized model trainer")
    parser.add_argument("--data-path", type=str, required=True, help="Path to token sequences JSON")
    parser.add_argument("--model", type=str, default="gpt2-medium", help="Model name")
    parser.add_argument("--epochs", type=int, help="Number of epochs (auto-calculated if not specified)")
    parser.add_argument("--output", type=str, default="models/the-block-git-model-final", help="Output directory")
    parser.add_argument("--config", type=str, default="training_config.yaml", help="Config file path")
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = OptimizedModelTrainer(
        model_name=args.model,
        output_dir=args.output,
        config_path=args.config,
    )
    
    # Load data
    train_dataset, val_dataset = trainer.load_data(args.data_path)
    
    # Calculate epochs if not provided
    if args.epochs is None:
        training_params = trainer.calculate_training_params(len(train_dataset) + len(val_dataset))
        epochs = training_params['epochs']
    else:
        epochs = args.epochs
    
    # Train
    report = trainer.train(train_dataset, val_dataset, num_epochs=epochs)
    
    # Save report
    report_path = Path(args.output) / "training_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    logger.info(f"Training report saved to {report_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
