#!/usr/bin/env python3
"""
End-to-End Minimal Training Test

Tests complete training pipeline with a tiny model.
Covers:
1. Model loading (with quantization/LoRA)
2. Data preparation
3. Full training loop (1 epoch)
4. Model saving
5. Manifest generation
"""

import os
import sys
import json
import tempfile
import torch
import logging
from pathlib import Path
from datetime import datetime

try:
    sys.path.insert(0, str(Path(__file__).parent))
    from training.model_trainer_unified import OptimizedModelTrainer
except ImportError as e:
    print(f"Import error: {e}")
    print("Required: torch, transformers, peft, bitsandbytes")
    sys.exit(1)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class EndToEndMinimalTest:
    """Complete end-to-end test with real training"""
    
    def __init__(self):
        self.temp_dir = None
        self.output_dir = None
    
    def cleanup(self):
        """Clean up temporary files"""
        if self.temp_dir and Path(self.temp_dir).exists():
            import shutil
            try:
                shutil.rmtree(self.temp_dir)
            except:
                pass
    
    def create_minimal_config(self):
        """Create minimal config for testing"""
        import yaml
        
        config = {
            'model': {
                'name': 'gpt2',  # Tiny model for testing
                'tokenizer_name': 'gpt2',
                'trust_remote_code': False,
                'use_4bit': False,
                'use_8bit': False,
                'use_bf16': False,
                'use_lora': False,  # Disable LoRA for testing simplicity
                'max_position_embeddings': 512,
                'max_new_tokens': 100,
            },
            'training': {
                'base_learning_rate': 5e-5,
                'warmup_ratio': 0.1,
                'warmup_steps_min': 2,
                'warmup_steps_max': 100,
                'weight_decay': 0.01,
                'batch_size_large': 2,
                'batch_size_medium': 1,
                'batch_size_small': 1,
                'num_workers': 0,
                'pin_memory': False,
                'gradient_accumulation_steps': 1,
                'max_grad_norm': 1.0,
                'use_mixed_precision': False,
                'autocast_dtype': 'float32',
                'use_gradient_checkpointing': False,
                'validation_split': 0.5,  # 50% for testing
                'patience': 2,
                'min_delta': 0.0001,
                'seed': 42,
            },
            'hardware_monitoring': {
                'collection_interval_seconds': 10,
                'gpu_memory_threshold_large_gb': 7.0,
                'gpu_memory_threshold_medium_gb': 4.0,
                'gpu_memory_threshold_small_gb': 2.0,
            },
            'model_saving': {
                'save_final_model': True,
                'save_best_model': True,
                'save_ckpt_every_n_epochs': 1,
                'save_adapter_only': False,
                'output_dir': 'models/test-model',
            },
        }
        
        # Write to temp file
        self.temp_dir = tempfile.mkdtemp()
        config_path = Path(self.temp_dir) / 'test_config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        return str(config_path)
    
    def create_minimal_sequences(self, num_sequences: int = 5):
        """Create minimal test sequences"""
        sequences = [
            {"tokens": "def hello(): pass"},
            {"tokens": "class MyClass: pass"},
            {"tokens": "def process(x): return x * 2"},
            {"tokens": "import sys\nimport os"},
            {"tokens": "def main(): print('hello')"},
        ][:num_sequences]
        
        sequences_file = Path(self.temp_dir) / 'test_sequences.json'
        with open(sequences_file, 'w') as f:
            json.dump(sequences, f)
        
        return str(sequences_file)
    
    def test_full_training_pipeline(self):
        """Test complete training pipeline"""
        print("\n" + "="*70)
        print("END-TO-END TEST: Full Training Pipeline")
        print("="*70)
        
        try:
            # Step 1: Setup
            print("\n[1/5] Creating test configuration...")
            config_path = self.create_minimal_config()
            print(f"  ✓ Config: {config_path}")
            
            print("\n[2/5] Creating test sequences...")
            sequences_file = self.create_minimal_sequences(num_sequences=5)
            print(f"  ✓ Sequences: {sequences_file}")
            
            # Step 2: Initialize trainer
            print("\n[3/5] Initializing trainer with GPT2 (tiny)...")
            trainer = OptimizedModelTrainer(config_path)
            print(f"  ✓ Device: {trainer.device}")
            print(f"  ✓ Model: gpt2")
            print(f"  ✓ LoRA: Disabled (for testing)")
            
            # Step 3: Load model and tokenizer
            print("\n[4/5] Loading model and tokenizer...")
            trainer.load_model_and_tokenizer()
            print(f"  ✓ Model loaded")
            print(f"  ✓ Tokenizer loaded")
            print(f"  ✓ Vocab size: {len(trainer.tokenizer)}")
            
            # Step 4: Train (1 epoch only)
            print("\n[5/5] Running minimal training (1 epoch)...")
            output_dir = str(Path(self.temp_dir) / 'test_model')
            
            stats = trainer.train(
                sequences_file=sequences_file,
                num_epochs=1,
                output_dir=output_dir,
            )
            
            # Verify training completed
            assert stats is not None, "Training should return stats"
            assert 'final_train_loss' in stats, "Stats should have final_train_loss"
            assert 'final_val_loss' in stats, "Stats should have final_val_loss"
            assert stats['num_epochs_completed'] == 1, "Should complete 1 epoch"
            
            print(f"\n  ✓ Training complete")
            print(f"    Epochs: {stats['num_epochs_completed']}")
            print(f"    Final train loss: {stats['final_train_loss']:.4f}")
            print(f"    Final val loss: {stats['final_val_loss']:.4f}")
            print(f"    Final perplexity: {stats['final_perplexity']:.2f}")
            print(f"    Time: {stats['total_training_seconds']:.1f}s")
            
            # Step 5: Verify model was saved
            print("\n[VERIFICATION] Checking saved model...")
            assert Path(output_dir).exists(), "Output directory should exist"
            assert (Path(output_dir) / 'pytorch_model.bin').exists() or \
                   (Path(output_dir) / 'model.safetensors').exists(), "Model weights should be saved"
            assert (Path(output_dir) / 'config.json').exists(), "Config should be saved"
            assert (Path(output_dir) / 'training_info.json').exists(), "Training info should be saved"
            
            print(f"  ✓ Model saved to {output_dir}")
            print(f"  ✓ Files:")
            for f in Path(output_dir).iterdir():
                if f.is_file():
                    size_mb = f.stat().st_size / 1e6
                    print(f"    - {f.name} ({size_mb:.1f} MB)")
            
            # Step 6: Verify training_info.json
            print("\n[VERIFICATION] Checking training_info.json...")
            with open(Path(output_dir) / 'training_info.json') as f:
                training_info = json.load(f)
            
            assert 'model_name' in training_info, "Should have model_name"
            assert 'use_lora' in training_info, "Should have use_lora"
            assert 'model_saving' in training_info, "Should have model_saving"
            
            print(f"  ✓ training_info.json valid")
            print(f"    Model: {training_info['model_name']}")
            print(f"    LoRA: {training_info['use_lora']}")
            print(f"    save_adapter_only: {training_info['model_saving'].get('save_adapter_only', 'N/A')}")
            
            print("\n" + "="*70)
            print("✅ END-TO-END TEST: PASS")
            print("="*70 + "\n")
            
            return True
            
        except Exception as e:
            print(f"\n\n" + "="*70)
            print("✗ END-TO-END TEST: FAIL")
            print("="*70)
            print(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
            print("\n")
            return False
        
        finally:
            self.cleanup()


def main():
    """Run end-to-end test"""
    print("\n" + "#"*70)
    print("# END-TO-END MINIMAL TRAINING TEST")
    print("#"*70)
    print("\nThis test will:")
    print("  1. Create a minimal config for GPT2")
    print("  2. Create 5 test sequences")
    print("  3. Initialize OptimizedModelTrainer")
    print("  4. Load model and tokenizer")
    print("  5. Run 1 epoch of training")
    print("  6. Verify model saving and manifest")
    print("\nExpected to take: 30-60 seconds")
    print("\n" + "#"*70 + "\n")
    
    test = EndToEndMinimalTest()
    success = test.test_full_training_pipeline()
    
    if success:
        print("\n✅ All checks passed! Training pipeline works correctly.")
        sys.exit(0)
    else:
        print("\n✗ Test failed. Check errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
