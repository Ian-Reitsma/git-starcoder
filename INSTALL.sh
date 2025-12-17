#!/bin/bash

# Installation script for The Block Git Scraping & Model Training Pipeline
# Tested on: macOS (M1), Linux (Ubuntu 22.04)

set -e

echo "========================================"
echo "Git Scraping Pipeline Installer"
echo "========================================"

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

if ! python3 -c 'import sys; exit(0 if sys.version_info >= (3, 9) else 1)'; then
    echo "ERROR: Python 3.9+ required"
    exit 1
fi

echo ""
echo "Creating virtual environment..."
python3 -m venv venv

echo "Activating virtual environment..."
source venv/bin/activate

echo ""
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

echo ""
echo "Installing dependencies..."
echo "This may take 5-10 minutes..."
echo "Note: on macOS, bitsandbytes/pygit2 are skipped automatically (CUDA-only / build toolchain)."

# Install deps (requirements.txt contains platform markers).
if ! pip install -r requirements.txt; then
    echo ""
    echo "ERROR: pip install -r requirements.txt failed."
    echo "Most common causes on macOS:" 
    echo "  - Using an old pip (upgrade pip/setuptools/wheel)"
    echo "  - Building native deps without Xcode CLT"
    echo ""
    echo "Try:"
    echo "  xcode-select --install"
    echo "  pip install --upgrade pip setuptools wheel"
    echo "  pip install -r requirements.txt"
    exit 1
fi

echo ""
echo "Verifying installation..."

# Test imports
python3 << 'PYEOF'
import sys
print("Checking imports...")

missing = []

try:
    import torch
    print(f"  ✓ torch {torch.__version__}")
except Exception as e:
    print(f"  ✗ torch failed: {e}")
    missing.append("torch")

try:
    import numpy as np
    print(f"  ✓ numpy {np.__version__}")
except Exception as e:
    print(f"  ✗ numpy failed: {e}")
    missing.append("numpy")

try:
    import yaml
    print(f"  ✓ pyyaml {yaml.__version__}")
except Exception as e:
    print(f"  ✗ pyyaml failed: {e}")
    missing.append("pyyaml")

try:
    import transformers
    print(f"  ✓ transformers {transformers.__version__}")
except Exception as e:
    print(f"  ✗ transformers failed: {e}")
    missing.append("transformers")

try:
    from git import Repo
    print(f"  ✓ GitPython")
except Exception as e:
    print(f"  ✗ GitPython failed: {e}")
    missing.append("GitPython")

if missing:
    print("\nERROR: Missing required dependencies:", ", ".join(missing))
    print("\nFix (inside venv):")
    print("  pip install --upgrade pip setuptools wheel")
    if "torch" in missing:
        print("  pip install torch")
    if "numpy" in missing:
        print("  pip install numpy")
    if "pyyaml" in missing:
        print("  pip install pyyaml")
    if "transformers" in missing:
        print("  pip install transformers")
    if "GitPython" in missing:
        print("  pip install GitPython")
    sys.exit(1)

print("\n✓ All dependencies installed successfully!")
PYEOF

if [ $? -ne 0 ]; then
    echo "ERROR: Installation verification failed"
    exit 1
fi

echo ""
echo "========================================"
echo "Installation Complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "  1. Activate environment: source venv/bin/activate"
echo "  2. Run pipeline: python3 run_pipeline.py --repo /path/to/repo --run all"
echo "  3. Or run individual steps (see README.md)"
echo ""
echo "Quick test:"
echo "  python3 run_pipeline.py --help"
echo ""
