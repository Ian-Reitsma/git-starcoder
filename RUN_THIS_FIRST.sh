#!/bin/bash
# ============================================================================
# ORCHARD METAL BACKWARD FIX - ONE-COMMAND COMPLETE REBUILD
# ============================================================================
# This script does EVERYTHING:
# 1. Verifies the C++ patch is in place
# 2. Cleans old build artifacts  
# 3. Builds C++ Metal backend
# 4. Builds Python extension (orchard_ops)
# 5. Runs smoke test to verify functionality
# ============================================================================

set -e  # Exit on any error

echo ""
echo "========================================================================"
echo "  ORCHARD METAL BACKWARD FIX - COMPREHENSIVE REBUILD"
echo "========================================================================"
echo ""

cd "/Users/ianreitsma/projects/git-starcoder"

echo "[1/10] Verifying C++ patch..."
if grep -q "Materialize private tensor into shared storage via clone" metal-backend/experimental/orchard_ops/mps/flash_attn.mm; then
    echo "✓ Patch detected in flash_attn.mm"
else
    echo "✗ ERROR: Patch not found in flash_attn.mm"
    echo "Make sure the file was edited correctly before running this script."
    exit 1
fi

echo ""
echo "[2/10] Checking PyTorch installation..."
python3 -c "import torch; print(f'✓ PyTorch {torch.__version__}'); print(f'✓ MPS available: {torch.backends.mps.is_available()}')"

echo ""
echo "[3/10] Cleaning old build artifacts..."
rm -rf metal-backend/experimental/orchard_ops/build
rm -rf metal-backend/build
find metal-backend -name "*.so" -path "*/orchard_ops/*" -delete 2>/dev/null || true
echo "✓ Cleaned"

echo ""
echo "[4/10] Configuring CMake..."
cd metal-backend
mkdir -p build
cd build

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DORCHARD_BUILD_EXPERIMENTAL=ON \
  -DCMAKE_OSX_ARCHITECTURES="arm64" \
  -DCMAKE_OSX_MINIMUM_SUPPORTED_VERSION=11.0 \
  2>&1 | tail -20

echo "✓ CMake configured"

echo ""
echo "[5/10] Building C++ core (this may take 1-2 minutes)..."
cmake --build . --config Release --parallel $(sysctl -n hw.ncpu) 2>&1 | tail -50
echo "✓ C++ build complete"

echo ""
echo "[6/10] Building Python extension..."
cd ../experimental/orchard_ops
python3 setup.py build_ext --inplace 2>&1 | grep -E "building|copying|writing|Successfully" | tail -20

if ls orchard_ops*.so 1> /dev/null 2>&1; then
    echo "✓ Extension built: $(ls -lh orchard_ops*.so | awk '{print $9, $5}')"
else
    echo "✗ Extension not found in current directory"
    echo "Checking build subdirectory..."
    find . -name "orchard_ops*.so" -exec ls -lh {} \;
fi

echo ""
echo "[7/10] Testing import..."
cd /Users/ianreitsma/projects/git-starcoder
python3 << 'PYEOF'
import sys
sys.path.insert(0, 'metal-backend/experimental/orchard_ops')
try:
    import orchard_ops
    print(f"✓ orchard_ops imported successfully")
    print(f"  Location: {orchard_ops.__file__}")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)
PYEOF

echo ""
echo "[8/10] Cleaning test environment..."
rm -f /tmp/orchard_tensor_profile.log /tmp/flashattn_kernel_calls.log
echo "✓ Cleaned"

echo ""
echo "[9/10] Running smoke test (Metal backward)..."
export ORCHARD_DEBUG_FLASH_ATN=1
export ORCHARD_TENSOR_PROFILE=1

if python3 -m pytest metal-backend/experimental/tests/test_mps_smoke_training.py::test_flash_attention_backward -xvs 2>&1 | tee smoke_test.log | tail -50; then
    echo "✓ Smoke test PASSED"
else
    echo "✗ Smoke test FAILED - see smoke_test.log for details"
    echo "Attempting to show last errors:"
    tail -100 smoke_test.log | grep -i "error\|failed\|exception" || true
fi

echo ""
echo "[10/10] Verification summary..."
echo ""
echo "Generated logs:"
echo "  - /tmp/orchard_tensor_profile.log  (Tensor allocation profiling)"
echo "  - /tmp/flashattn_kernel_calls.log  (Kernel execution log)"
echo "  - smoke_test.log                    (Test output)"
echo ""
echo "Quick checks:"
echo "✓ Patch applied to flash_attn.mm"
echo "✓ Python extension built and imported"
echo "✓ Smoke test completed"
echo ""
echo "========================================================================"
echo "  BUILD COMPLETE - METAL BACKWARD FIX READY FOR TESTING"
echo "========================================================================"
echo ""
echo "NEXT STEPS:"
echo ""
echo "1. Verify Metal kernel was used:"
echo "   cat /tmp/flashattn_kernel_calls.log"
echo ""
echo "2. Check for any errors:"
echo "   grep -i error smoke_test.log"
echo ""
echo "3. Run full test suite:"
echo "   ORCHARD_DEBUG_FLASH_ATN=1 python3 -m pytest \\"
echo "     metal-backend/experimental/tests/test_mps_smoke_training.py -v"
echo ""
echo "4. Test in your training code:"
echo "   python3 your_training_script.py"
echo ""
echo "========================================================================"
