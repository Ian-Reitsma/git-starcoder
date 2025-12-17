#!/bin/bash

# Test runner script for The Block pipeline
# Comprehensive testing with detailed output

set -e

echo ""
echo "#######################################################################"
echo "#                   COMPREHENSIVE TEST SUITE                        #"
echo "#              The Block Git Scraping & Training Pipeline           #"
echo "#######################################################################"
echo ""

# Check Python
echo "[1/4] Checking Python..."
if ! command -v python3 &> /dev/null; then
    echo "✗ Python 3 not found"
    exit 1
fi
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo "✓ Python $PYTHON_VERSION"
echo ""

# Check virtual environment
echo "[2/4] Checking virtual environment..."
if [ ! -d "venv" ]; then
    echo "✗ Virtual environment not found"
    echo "  Run: bash INSTALL.sh"
    exit 1
fi
echo "✓ Virtual environment exists"
echo ""

# Activate virtual environment
echo "[3/4] Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Run tests
echo "[4/4] Running comprehensive tests..."
echo ""
python3 tests/test_all.py
TEST_EXIT=$?

echo ""
if [ $TEST_EXIT -eq 0 ]; then
    echo "#######################################################################"
    echo "#✓ ALL TESTS PASSED - System is ready for use"
    echo "#######################################################################"
    echo ""
    echo "Next steps:"
    echo "  1. Run: python3 run_pipeline_optimized.py --repo /path/to/repo --verbose"
    echo "  2. Wait ~10 minutes for model training"
    echo "  3. Find model at: models/the-block-git-model-final/"
    echo ""
else
    echo "#######################################################################"
    echo "#✗ SOME TESTS FAILED - Please review output above"
    echo "#######################################################################"
    echo ""
    exit 1
fi
