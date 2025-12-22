#!/usr/bin/env python3
"""Pre-training validation script for StarCoder2-3B fine-tuning.

Validates:
- Repository structure and git history
- Dataset files and format
- Training configuration
- Hardware and memory requirements
- Model accessibility

Run this before starting a long training run to catch issues early.
"""

import sys
import json
import yaml
import torch
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class PreTrainingValidator:
    def __init__(self, config_path: str, target_repo: Optional[str] = None):
        self.config_path = Path(config_path)
        self.target_repo = Path(target_repo) if target_repo else None
        self.config = None
        self.errors = []
        self.warnings = []
        
    def validate_all(self) -> bool:
        """Run all validation checks."""
        logger.info("\n" + "="*70)
        logger.info("PRE-TRAINING VALIDATION")
        logger.info("="*70 + "\n")
        
        checks = [
            ("Configuration", self.validate_config),
            ("Hardware", self.validate_hardware),
            ("Target Repository", self.validate_target_repo),
            ("Dataset", self.validate_dataset),
            ("Model Access", self.validate_model_access),
            ("Memory Requirements", self.validate_memory),
        ]
        
        for name, check_fn in checks:
            logger.info(f"Checking: {name}")
            try:
                check_fn()
                logger.info(f"✓ {name}: PASS\n")
            except Exception as e:
                self.errors.append(f"{name}: {e}")
                logger.error(f"✗ {name}: FAIL - {e}\n")
        
        self._print_summary()
        return len(self.errors) == 0
    
    def validate_config(self):
        """Validate training configuration file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found: {self.config_path}")
        
        with open(self.config_path) as f:
            self.config = yaml.safe_load(f)
        
        # Check required sections
        required = ['model', 'training']
        for section in required:
            if section not in self.config:
                raise ValueError(f"Config missing required section: {section}")
        
        # Validate model config
        model_cfg = self.config['model']
        if 'pretrained_model' not in model_cfg and 'name' not in model_cfg:
            raise ValueError("Config must specify model.pretrained_model or model.name")
        
        # Check for dataset paths if using alternate schema
        if 'quantization' in self.config:
            quant = self.config['quantization']
            if 'train_path' in quant or 'val_path' in quant:
                logger.info("  Found dataset paths in config")
        
        logger.info(f"  Config structure: OK")
        logger.info(f"  Model: {model_cfg.get('pretrained_model') or model_cfg.get('name')}")
    
    def validate_hardware(self):
        """Validate hardware availability."""
        device_available = {
            'cuda': torch.cuda.is_available(),
            'mps': torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False,
            'cpu': True,
        }
        
        available = [k for k, v in device_available.items() if v]
        if not any(device_available.values()):
            raise RuntimeError("No training device available (CUDA/MPS/CPU)")
        
        logger.info(f"  Available devices: {', '.join(available)}")
        
        # Check MPS on Mac
        if device_available['mps']:
            import platform
            mac_ver = platform.mac_ver()[0]
            logger.info(f"  macOS version: {mac_ver}")
            logger.info(f"  PyTorch version: {torch.__version__}")
            
            # Warn if PyTorch is too old for good MPS support
            torch_version = tuple(int(x) for x in torch.__version__.split('.')[:2])
            if torch_version < (2, 0):
                self.warnings.append("PyTorch < 2.0; MPS support may be limited")
        
        # Check CUDA compute capability if available
        if device_available['cuda']:
            props = torch.cuda.get_device_properties(0)
            logger.info(f"  GPU: {props.name}")
            logger.info(f"  CUDA compute: {props.major}.{props.minor}")
            logger.info(f"  GPU memory: {props.total_memory / 1e9:.1f} GB")
    
    def validate_target_repo(self):
        """Validate target repository for training."""
        if self.target_repo is None:
            self.warnings.append("No target repository specified; skipping repo checks")
            return
        
        if not self.target_repo.exists():
            raise FileNotFoundError(f"Target repo not found: {self.target_repo}")
        
        # Check if it's a git repo
        git_dir = self.target_repo / ".git"
        if not git_dir.exists():
            raise ValueError(f"Target is not a git repository: {self.target_repo}")
        
        # Count commits
        try:
            result = subprocess.run(
                ['git', '-C', str(self.target_repo), 'rev-list', '--count', 'HEAD'],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                commit_count = int(result.stdout.strip())
                logger.info(f"  Commits: {commit_count}")
                if commit_count < 50:
                    self.warnings.append(f"Low commit count ({commit_count}); may have insufficient training data")
        except Exception as e:
            self.warnings.append(f"Could not count commits: {e}")
        
        # Check for Rust files if training on Rust codebase
        rust_files = list(self.target_repo.rglob("*.rs"))
        if rust_files:
            logger.info(f"  Rust files: {len(rust_files)}")
        
        # Estimate LOC
        try:
            loc_count = 0
            for rs_file in rust_files[:100]:  # Sample
                try:
                    loc_count += len(rs_file.read_text().splitlines())
                except:
                    pass
            if rust_files:
                estimated_total = (loc_count * len(rust_files)) // min(100, len(rust_files))
                logger.info(f"  Estimated LOC: {estimated_total:,}")
        except:
            pass
    
    def validate_dataset(self):
        """Validate dataset files exist and are properly formatted."""
        # Check if using alternate schema with explicit paths
        dataset_paths = []
        
        if 'quantization' in self.config:
            quant = self.config['quantization']
            for key in ['train_path', 'val_path', 'test_path']:
                if key in quant:
                    dataset_paths.append(Path(quant[key]))
        
        # Check if using data.train_path style
        if 'data' in self.config:
            data = self.config['data']
            for key in ['train_path', 'val_path', 'test_path']:
                if key in 
                    dataset_paths.append(Path(data[key]))
        
        if not dataset_paths:
            self.warnings.append("No dataset paths found in config; ensure they're provided at runtime")
            return
        
        missing = []
        for path in dataset_paths:
            if not path.exists():
                missing.append(str(path))
        
        if missing:
            raise FileNotFoundError(f"Dataset files missing: {', '.join(missing)}")
        
        logger.info(f"  Dataset files: {len(dataset_paths)} found")
        
        # Validate format of first file
        first_file = dataset_paths[0]
        try:
            if first_file.suffix == '.json':
                with open(first_file) as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        logger.info(f"  Training examples: {len(data)}")
                        if len(data) < 100:
                            self.warnings.append(f"Low training example count: {len(data)}")
            elif first_file.suffix == '.jsonl':
                with open(first_file) as f:
                    lines = f.readlines()
                    logger.info(f"  Training examples: {len(lines)}")
        except Exception as e:
            self.warnings.append(f"Could not parse dataset file: {e}")
    
    def validate_model_access(self):
        """Validate access to pretrained model."""
        model_name = self.config['model'].get('pretrained_model') or self.config['model'].get('name')
        
        # For HuggingFace models, check if accessible
        if '/' in model_name:  # Likely a HF model ID
            try:
                from transformers import AutoConfig
                config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
                logger.info(f"  Model config accessible: {model_name}")
                if hasattr(config, 'vocab_size'):
                    logger.info(f"  Vocab size: {config.vocab_size:,}")
            except Exception as e:
                raise RuntimeError(f"Cannot access model '{model_name}': {e}")
    
    def validate_memory(self):
        """Estimate memory requirements and compare to available."""
        model_name = self.config['model'].get('pretrained_model') or self.config['model'].get('name')
        
        # Rough estimates for common models
        param_estimates = {
            'starcoder2-3b': 3.0,  # billion params
            'gpt2': 0.124,
            'phi-2': 2.7,
        }
        
        params_b = 3.0  # default
        for key, val in param_estimates.items():
            if key in model_name.lower():
                params_b = val
                break
        
        # Estimate memory (very rough)
        # int8 quant: ~1 byte/param base + ~0.1 byte/param for LoRA + activations
        # Full precision: ~2 bytes/param (fp16) + LoRA + activations
        
        use_quant = self.config.get('quantization', {}).get('load_in_4bit') or \
                    self.config.get('quantization', {}).get('load_in_8bit') or \
                    self.config.get('model', {}).get('use_4bit') or \
                    self.config.get('model', {}).get('use_8bit')
        
        if use_quant:
            base_mem_gb = params_b * 1.0  # ~1 GB per billion params (int8)
            lora_mem_gb = params_b * 0.1
            activation_mem_gb = 2.0  # rough estimate for batch_size=2, seq=2048
        else:
            base_mem_gb = params_b * 2.0  # ~2 GB per billion params (fp16)
            lora_mem_gb = params_b * 0.2
            activation_mem_gb = 3.0
        
        total_est = base_mem_gb + lora_mem_gb + activation_mem_gb
        
        logger.info(f"  Model size: ~{params_b:.1f}B parameters")
        logger.info(f"  Estimated VRAM: {total_est:.1f} GB (with quant={use_quant})")
        
        # Check available memory
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            available_gb = props.total_memory / 1e9
            logger.info(f"  Available GPU memory: {available_gb:.1f} GB")
            
            if total_est > available_gb * 0.9:
                self.warnings.append(
                    f"Estimated memory ({total_est:.1f} GB) may exceed available ({available_gb:.1f} GB); "
                    "consider reducing batch_size or sequence length"
                )
        elif torch.backends.mps.is_available():
            # MPS uses unified memory; harder to estimate
            import psutil
            total_ram_gb = psutil.virtual_memory().total / 1e9
            logger.info(f"  System RAM (unified): {total_ram_gb:.1f} GB")
            
            if total_est > total_ram_gb * 0.3:  # Use at most 30% of RAM for model
                self.warnings.append(
                    f"Estimated memory ({total_est:.1f} GB) may be high for unified memory; "
                    "monitor memory usage during training"
                )
    
    def _print_summary(self):
        """Print validation summary."""
        logger.info("\n" + "="*70)
        logger.info("VALIDATION SUMMARY")
        logger.info("="*70)
        
        if self.errors:
            logger.error(f"\n✗ {len(self.errors)} ERROR(S):")
            for err in self.errors:
                logger.error(f"  - {err}")
        
        if self.warnings:
            logger.warning(f"\n⚠ {len(self.warnings)} WARNING(S):")
            for warn in self.warnings:
                logger.warning(f"  - {warn}")
        
        if not self.errors and not self.warnings:
            logger.info("\n✓ ALL CHECKS PASSED - READY TO TRAIN")
        elif not self.errors:
            logger.info("\n✓ NO CRITICAL ERRORS - READY TO TRAIN (with warnings)")
        else:
            logger.error("\n✗ VALIDATION FAILED - FIX ERRORS BEFORE TRAINING")
        
        logger.info("="*70 + "\n")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Validate setup before training")
    parser.add_argument(
        '--config',
        default='training_config_metal_cuda_universal.yaml',
        help='Path to training config'
    )
    parser.add_argument(
        '--repo',
        default=None,
        help='Path to target repository (e.g., ~/projects/the-block)'
    )
    args = parser.parse_args()
    
    validator = PreTrainingValidator(args.config, args.repo)
    success = validator.validate_all()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
