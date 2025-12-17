#!/usr/bin/env python3
"""MPS smoke test: one forward/backward/step.

Goal:
- Validate that the default trainer stack can execute a minimal training step on macOS MPS.
- Does not require model downloads.
- Does not require Orchard to be present; if Orchard is present, DeviceBackend will enable it.

Run:
  python3 test_mps_smoke_training.py
"""

import unittest


class TestMPSSmokeTraining(unittest.TestCase):
    def test_one_step_on_mps(self):
        try:
            import torch
        except Exception as e:
            self.skipTest(f"torch not available: {e}")

        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and torch.backends.mps.is_built()):
            self.skipTest("MPS not available on this machine")

        try:
            from device_backend import get_device_backend
        except Exception as e:
            self.skipTest(f"device_backend not available: {e}")

        # Build a tiny model with random weights (no downloads)
        from transformers import GPT2Config, GPT2LMHeadModel

        cfg = GPT2Config(
            vocab_size=128,
            n_positions=64,
            n_ctx=64,
            n_embd=64,
            n_layer=2,
            n_head=4,
        )
        model = GPT2LMHeadModel(cfg)

        backend = get_device_backend(force_device="mps", verbose=False)
        backend.setup()
        backend.patch_model(model)

        device = torch.device("mps")
        model.to(device)
        model.train()

        # One tiny batch
        input_ids = torch.randint(0, cfg.vocab_size, (2, 32), device=device)
        labels = input_ids.clone()

        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        out = model(input_ids=input_ids, labels=labels)
        loss = out.loss
        self.assertTrue(torch.isfinite(loss).item())

        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
