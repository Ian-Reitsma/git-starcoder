#!/usr/bin/env python3
"""End-to-end trainer integration test (no downloads).

This test protects the pipeline against refactors that break:
- OptimizedModelTrainer init
- load_model_and_tokenizer
- one forward/backward/optimizer step

It uses the trainer's `synthetic_model` mode to avoid network/model downloads.

Run:
  python3 -m unittest -v test_trainer_one_step.py
"""

import json
import tempfile
import unittest
from pathlib import Path


class TestTrainerOneStep(unittest.TestCase):
    def test_trainer_one_step_synthetic(self):
        from training.model_trainer_unified import OptimizedModelTrainer

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)

            # Minimal config matching the trainer schema
            cfg_path = td / "tiny_config.yaml"
            cfg_path.write_text(
                """
model:
  synthetic_model: true
  synthetic_vocab_size: 256
  synthetic_n_positions: 128
  # Values below are unused by synthetic mode, but kept for schema clarity.
  name: "__synthetic__"
  tokenizer_name: "__synthetic__"
  trust_remote_code: false
  use_lora: false
  use_4bit: false
  use_8bit: false
  use_bf16: false

training:
  seed: 123
  use_mixed_precision: false
  use_gradient_checkpointing: false
  gradient_accumulation_steps: 1
  max_grad_norm: 1.0
  base_learning_rate: 0.001
  weight_decay: 0.0
  warmup_ratio: 0.0
  warmup_steps_min: 0
  warmup_steps_max: 0
  min_delta: 0.0
  lr_reduction_factor: 0.5
  lr_plateau_patience: 1
  pin_memory: false
  batch_size_reference: 2
  batch_size_large: 2
  batch_size_medium: 2
  batch_size_small: 2
  num_workers: 0
  num_workers_min: 0
  num_workers_max: 0
  incremental_context_sequences: 0
  drop_full_attention_mask: true

  objectives:
    fim_rate: 0.2
    span_rate: 0.2
    span_frac: 0.25

  curriculum:
    enabled: true
    start_max_len: 32
    end_max_len: 64
    warmup_epochs: 2

  ema:
    enabled: true
    decay: 0.9

evaluation: {}
model_saving:
  save_final_model: false
hardware_monitoring:
  collection_interval_seconds: 999
""".lstrip()
            )

            # Synthetic sequences file in the format trainer accepts
            seq_path = td / "seq.json"
            # 8 sequences of length 64
            token_sequences = [[i % 256 for i in range(j, j + 64)] for j in range(8)]
            seq_path.write_text(json.dumps({"token_sequences": token_sequences, "metadata": {}}))

            # Instantiate and run only the minimal pieces needed
            trainer = OptimizedModelTrainer(str(cfg_path), force_device="cpu")
            trainer.load_model_and_tokenizer()

            # Run a tiny training invocation (1 epoch, output dir inside temp)
            out_dir = td / "out"
            out_dir.mkdir(parents=True, exist_ok=True)

            # The trainer's train() will do much more; keep epochs=1.
            trainer.train(str(seq_path), 1, str(out_dir))

            # Assert training_stats gets populated
            self.assertTrue(isinstance(trainer.training_stats, dict))
            self.assertIn('final_train_loss', trainer.training_stats)


if __name__ == "__main__":
    unittest.main(verbosity=2)
