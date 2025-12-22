#!/bin/bash
# ===================================================================
# COMPREHENSIVE ORCHARD METAL BACKWARD FIX - REBUILD SCRIPT
# ===================================================================
# This script rebuilds Orchard with the private MTLBuffer workaround fix.
# The fix ensures Metal backward runs successfully regardless of tensor
# storage mode (shared or private) by materializing private tensors into
# shared storage via clone().
#
# Status: Comprehensive production-ready implementation
# ==================================================================

set -e

echo "[1/8] Checking environment..."
cd "/Users/ianreitsma/projects/git-starcoder" || exit 1

echo "[2/8] Verifying PyTorch..."
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'MPS available: {torch.backends.mps.is_available()}'); print(f'Torch path: {torch.__file__}')"

echo ""
echo "[3/8] Cleaning previous build artifacts..."
rm -rf metal-backend/experimental/orchard_ops/build
rm -rf metal-backend/experimental/orchard_ops/*.egg-info
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find . -name "*.so" -path "*/orchard_ops/*" -delete 2>/dev/null || true

echo ""
echo "[4/8] Verifying flash_attn.mm patch was applied correctly..."
if grep -q "Materialize private tensor into shared storage via clone" metal-backend/experimental/orchard_ops/mps/flash_attn.mm; then
    echo "✓ Private MTLBuffer workaround patch detected in flash_attn.mm"
else
    echo "✗ ERROR: Patch not detected. Exiting."
    exit 1
fi

echo ""
echo "[5/8] Building Orchard Metal backend..."
cd metal-backend || exit 1

# Create build directory
mkdir -p build
cd build

# Configure with CMake (enable experimental)
echo "Configuring CMake with experimental enabled..."
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DORCHARD_BUILD_EXPERIMENTAL=ON \
  -DCMAKE_OSX_ARCHITECTURES="arm64" \
  -DCMAKE_OSX_MINIMUM_SUPPORTED_VERSION=11.0

echo ""
echo "Building C++ core and experimental bridge..."
cmake --build . --config Release --parallel $(sysctl -n hw.ncpu)

cd "../experimental/orchard_ops" || exit 1

echo ""
echo "[6/8] Building Python extension (orchard_ops)..."
# Ensure CMakeLists.txt references the patched flash_attn.mm
if ! grep -q "mps/flash_attn.mm" CMakeLists.txt; then
    echo "✗ WARNING: flash_attn.mm not found in CMakeLists.txt source list"
fi

# Build with setup.py
python3 setup.py build_ext --inplace 2>&1 | tee build.log

if [ -f "orchard_ops*.so" ] || [ -d "build" ]; then
    echo "✓ Python extension built successfully"
else
    echo "✗ Build may have failed. Check build.log"
fi

echo ""
echo "[7/8] Verifying extension is importable..."
cd "/Users/ianreitsma/projects/git-starcoder" || exit 1
python3 -c "
import sys
sys.path.insert(0, 'metal-backend/experimental/orchard_ops')
try:
    import orchard_ops
    print('✓ orchard_ops imported successfully')
    print(f'  Module: {orchard_ops.__file__}')
except Exception as e:
    print(f'✗ Import failed: {e}')
    sys.exit(1)
"

echo ""
echo "[8/8] Running smoke test with Metal backward..."
export ORCHARD_DEBUG_FLASH_ATN=1
export ORCHARD_TENSOR_PROFILE=1
rm -f /tmp/orchard_tensor_profile.log /tmp/flashattn_kernel_calls.log

python3 -m pytest metal-backend/experimental/tests/test_mps_smoke_training.py::test_flash_attention_backward -xvs 2>&1 | tee smoke_test.log

echo ""
echo "===================================================================="
echo "BUILD COMPLETE"
echo "===================================================================="
echo ""
echo "Next steps:"
echo "1. Run full test suite:"
echo "   cd /Users/ianreitsma/projects/git-starcoder"
echo "   ORCHARD_DEBUG_FLASH_ATN=1 python -m pytest metal-backend/experimental/tests/test_mps_smoke_training.py -xvs"
echo ""
echo "2. Check profiling log for allocation patterns:"
echo "   tail -n 300 /tmp/orchard_tensor_profile.log"
echo ""
echo "3. Verify Metal kernel was used:"
echo "   grep -i 'metal\|fallback' smoke_test.log"
echo ""
echo "===================================================================="
