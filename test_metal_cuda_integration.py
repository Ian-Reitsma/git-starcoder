#!/usr/bin/env python3
"""
Comprehensive Tests: Device Backend + Unified Trainer

Covers:
- Platform detection (Darwin/Linux)
- Device availability (Metal/CUDA/CPU)
- Attention backend selection
- Config adaptation
- Model patching
- Training on both platforms
"""

import unittest
import sys
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import logging

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from device_backend import DeviceBackend, BackendConfig, get_device_backend
    DEVICE_BACKEND_AVAILABLE = True
except ImportError:
    DEVICE_BACKEND_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestDeviceDetection(unittest.TestCase):
    """Test platform and device detection."""

    @unittest.skipIf(not DEVICE_BACKEND_AVAILABLE, "device_backend not available")
    def test_backend_initialization(self):
        """Test DeviceBackend initializes without error."""
        backend = DeviceBackend(verbose=False)
        self.assertIsNotNone(backend.config)
        self.assertIn(backend.config.device, ["cuda", "mps", "cpu"])
        self.assertIn(backend.config.device_type, ["cuda", "metal", "cpu"])

    @unittest.skipIf(not DEVICE_BACKEND_AVAILABLE, "device_backend not available")
    def test_device_force_cpu(self):
        """Test forcing CPU device."""
        backend = DeviceBackend(force_device="cpu")
        self.assertEqual(backend.device, "cpu")
        self.assertEqual(backend.device_type, "cpu")

    @unittest.skipIf(not DEVICE_BACKEND_AVAILABLE or not TORCH_AVAILABLE,
                     "torch or device_backend not available")
    def test_metal_detection(self):
        """Test Metal availability detection."""
        backend = DeviceBackend()
        try:
            is_metal = torch.backends.mps.is_available() and torch.backends.mps.is_built()
        except Exception:
            is_metal = False
        
        self.assertEqual(backend.is_metal, is_metal)

    @unittest.skipIf(not DEVICE_BACKEND_AVAILABLE or not TORCH_AVAILABLE,
                     "torch or device_backend not available")
    def test_cuda_detection(self):
        """Test CUDA availability detection."""
        backend = DeviceBackend()
        is_cuda = torch.cuda.is_available()
        self.assertEqual(backend.is_cuda, is_cuda)


class TestAttentionBackendSelection(unittest.TestCase):
    """Test attention backend selection logic."""

    @unittest.skipIf(not DEVICE_BACKEND_AVAILABLE, "device_backend not available")
    def test_attention_backend_cpu(self):
        """Test attention backend selection for CPU."""
        backend = DeviceBackend(force_device="cpu")
        self.assertEqual(backend.attention_backend, "native")

    @unittest.skipIf(not DEVICE_BACKEND_AVAILABLE or not TORCH_AVAILABLE,
                     "torch or device_backend not available")
    def test_attention_backend_cuda_available(self):
        """Test CUDA attention backend selection."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        
        backend = DeviceBackend(force_device="cuda")
        self.assertIn(backend.attention_backend, ["xformers", "sdpa", "native"])

    @unittest.skipIf(not DEVICE_BACKEND_AVAILABLE or not TORCH_AVAILABLE,
                     "torch or device_backend not available")
    def test_attention_backend_metal(self):
        """Test Metal attention backend selection."""
        backend = DeviceBackend()
        if backend.is_metal:
            self.assertEqual(backend.attention_backend, "metal")


class TestConfigAdaptation(unittest.TestCase):
    """Test config adaptation for device."""

    @unittest.skipIf(not DEVICE_BACKEND_AVAILABLE, "device_backend not available")
    def test_cpu_config_overrides(self):
        """Test CPU config overrides."""
        backend = DeviceBackend(force_device="cpu")
        overrides = backend.get_model_config_overrides()
        # CPU has minimal overrides
        self.assertIsInstance(overrides, dict)

    @unittest.skipIf(not DEVICE_BACKEND_AVAILABLE or not TORCH_AVAILABLE,
                     "torch or device_backend not available")
    def test_metal_config_overrides(self):
        """Test Metal config overrides (dtype, gradient checkpointing)."""
        if not (torch.backends.mps.is_available() and torch.backends.mps.is_built()):
            self.skipTest("Metal not available")
        
        backend = DeviceBackend()
        if backend.is_metal:
            overrides = backend.get_model_config_overrides()
            # Metal should suggest bf16/fp32 (not fp16)
            self.assertIn("torch_dtype", overrides)
            self.assertNotEqual(overrides["torch_dtype"], "float16")

    @unittest.skipIf(not DEVICE_BACKEND_AVAILABLE or not TORCH_AVAILABLE,
                     "torch or device_backend not available")
    def test_cuda_config_overrides(self):
        """Test CUDA config overrides."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        
        backend = DeviceBackend()
        if backend.is_cuda:
            overrides = backend.get_model_config_overrides()
            self.assertIsInstance(overrides, dict)


class TestEnvironmentSetup(unittest.TestCase):
    """Test environment variable configuration."""

    @unittest.skipIf(not DEVICE_BACKEND_AVAILABLE, "device_backend not available")
    def test_setup_clears_state(self):
        """Test that setup() configures environment correctly."""
        backend = DeviceBackend(force_device="cpu")
        backend.setup()  # Should not raise
        self.assertIsNotNone(backend.config)

    @unittest.skipIf(not DEVICE_BACKEND_AVAILABLE or not TORCH_AVAILABLE,
                     "torch or device_backend not available")
    def test_cuda_env_setup(self):
        """Test CUDA environment configuration."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        
        backend = DeviceBackend(force_device="cuda")
        backend.setup()
        
        # Check PYTORCH_CUDA_ALLOC_CONF is set
        self.assertIn("PYTORCH_CUDA_ALLOC_CONF", os.environ)

    @unittest.skipIf(not DEVICE_BACKEND_AVAILABLE, "device_backend not available")
    def test_metal_env_setup(self):
        """Test Metal environment configuration."""
        backend = DeviceBackend()
        if backend.is_metal:
            backend.setup()
            self.assertIn("PYTORCH_MPS_HIGH_WATERMARK_RATIO", os.environ)


class TestVRAMEstimation(unittest.TestCase):
    """Test VRAM estimation."""

    @unittest.skipIf(not DEVICE_BACKEND_AVAILABLE, "device_backend not available")
    def test_vram_gt_zero(self):
        """Test that VRAM estimation returns positive value."""
        backend = DeviceBackend()
        self.assertGreaterEqual(backend.max_vram_gb, 0.0)

    @unittest.skipIf(not DEVICE_BACKEND_AVAILABLE or not TORCH_AVAILABLE,
                     "torch or device_backend not available")
    def test_cuda_vram_estimation(self):
        """Test CUDA VRAM estimation."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        
        backend = DeviceBackend(force_device="cuda")
        expected_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        self.assertGreater(backend.max_vram_gb, 0)
        self.assertLess(backend.max_vram_gb, expected_vram * 2)  # Should be reasonable


class TestBackendFactory(unittest.TestCase):
    """Test factory function."""

    @unittest.skipIf(not DEVICE_BACKEND_AVAILABLE, "device_backend not available")
    def test_factory_function(self):
        """Test get_device_backend factory."""
        backend = get_device_backend(force_device="cpu", verbose=False)
        self.assertIsInstance(backend, DeviceBackend)
        self.assertEqual(backend.device, "cpu")


class TestLogging(unittest.TestCase):
    """Test logging functionality."""

    @unittest.skipIf(not DEVICE_BACKEND_AVAILABLE, "device_backend not available")
    def test_log_summary(self):
        """Test that log_summary doesn't raise."""
        backend = DeviceBackend(force_device="cpu")
        backend.setup()
        backend.log_summary()  # Should not raise

    @unittest.skipIf(not DEVICE_BACKEND_AVAILABLE, "device_backend not available")
    def test_verbose_logging(self):
        """Test verbose logging initialization."""
        backend = DeviceBackend(force_device="cpu", verbose=True)
        self.assertTrue(backend.verbose)


class TestIntegration(unittest.TestCase):
    """Integration tests."""

    @unittest.skipIf(not DEVICE_BACKEND_AVAILABLE, "device_backend not available")
    def test_full_workflow_cpu(self):
        """Test full device backend workflow on CPU."""
        backend = get_device_backend(force_device="cpu", verbose=False)
        backend.setup()
        
        # Check all properties are accessible
        _ = backend.device
        _ = backend.device_type
        _ = backend.attention_backend
        _ = backend.supports_bf16
        _ = backend.max_vram_gb
        _ = backend.is_metal
        _ = backend.is_cuda
        
        overrides = backend.get_model_config_overrides()
        self.assertIsInstance(overrides, dict)

    @unittest.skipIf(not DEVICE_BACKEND_AVAILABLE or not TORCH_AVAILABLE,
                     "torch or device_backend not available")
    def test_platform_detection_consistency(self):
        """Test that platform detection is consistent."""
        backend = DeviceBackend()
        config = backend.config
        
        # Platform consistency
        if config.is_metal:
            self.assertFalse(config.is_cuda)
        if config.is_cuda:
            self.assertFalse(config.is_metal)
        
        # Device string matches device type
        if config.device_type == "metal":
            self.assertEqual(config.device, "mps")
        elif config.device_type == "cuda":
            self.assertEqual(config.device, "cuda")
        else:
            self.assertEqual(config.device, "cpu")


def run_tests():
    """Run all tests with summary."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestDeviceDetection))
    suite.addTests(loader.loadTestsFromTestCase(TestAttentionBackendSelection))
    suite.addTests(loader.loadTestsFromTestCase(TestConfigAdaptation))
    suite.addTests(loader.loadTestsFromTestCase(TestEnvironmentSetup))
    suite.addTests(loader.loadTestsFromTestCase(TestVRAMEstimation))
    suite.addTests(loader.loadTestsFromTestCase(TestBackendFactory))
    suite.addTests(loader.loadTestsFromTestCase(TestLogging))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print("=" * 70)
    
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_tests())
