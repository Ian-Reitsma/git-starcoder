"""pytest configuration for integration tests"""
import warnings
import pytest

# Suppress the PyTorch MPS pin_memory warning
# This warning occurs on Apple Silicon when pin_memory=True is set,
# which is not supported on MPS backend
warnings.filterwarnings(
    "ignore",
    message=".*pin_memory.*MPS.*",
    category=UserWarning,
    module="torch.utils.data.dataloader"
)

# Additional PyTorch warnings to suppress
warnings.filterwarnings(
    "ignore",
    message=".*pin_memory argument is set as true but not supported on MPS.*",
    category=UserWarning
)
