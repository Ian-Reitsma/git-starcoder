#!/bin/bash

# Quick-start script for Metal/CUDA unified training
#
# Usage:
#   ./run_metal_cuda.sh [OPTIONS]
#
# Options:
#   --config CONFIG       Training config (default: training_config_metal_cuda_universal.yaml)
#   --sequences SEQ       Path to sequences (required if not set in config)
#   --epochs N            Number of epochs (default: 3)
#   --output DIR          Output directory (default: models/the-block-metal-cuda)
#   --device DEVICE       Force device: cuda, mps, cpu (default: auto)
#   --test                Run tests instead of training
#   --verbose             Enable verbose logging
#   --help                Show this message

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
CONFIG="training_config_metal_cuda_universal.yaml"
SEQUENCES=""
EPOCHS=3
OUTPUT="models/the-block-metal-cuda"
DEVICE=""
RUN_TESTS=false
VERBOSE=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --sequences)
            SEQUENCES="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --output)
            OUTPUT="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --test)
            RUN_TESTS=true
            shift
            ;;
        --verbose)
            VERBOSE="--verbose"
            shift
            ;;
        --help)
            grep "^#" "$0" | sed 's/^# //' | head -20
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Function to print colored output
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

# Check Python
if ! command -v python3 &> /dev/null; then
    log_error "Python 3 not found"
    exit 1
fi

log_info "Python: $(python3 --version)"

# Run tests if requested
if [ "$RUN_TESTS" = true ]; then
    log_info "Running integration tests..."
    python3 test_metal_cuda_integration.py
    exit 0
fi

# Validate inputs
if [ -z "$SEQUENCES" ]; then
    log_error "--sequences is required"
    exit 1
fi

if [ ! -f "$CONFIG" ]; then
    log_error "Config not found: $CONFIG"
    exit 1
fi

if [ ! -f "$SEQUENCES" ]; then
    log_error "Sequences not found: $SEQUENCES"
    exit 1
fi

# Check for required modules
log_info "Checking dependencies..."
python3 -c "import torch; print(f'  PyTorch: {torch.__version__}')" || {
    log_error "PyTorch not installed"
    exit 1
}

python3 -c "import yaml; print(f'  PyYAML: OK')" || {
    log_error "PyYAML not installed"
    exit 1
}

# Create output directory
mkdir -p "$OUTPUT"
log_info "Output directory: $OUTPUT"

# Build command
CMD="python3 model_trainer_metal_cuda.py"
CMD="$CMD --config $CONFIG"
CMD="$CMD --sequences $SEQUENCES"
CMD="$CMD --epochs $EPOCHS"
CMD="$CMD --output $OUTPUT"

if [ -n "$DEVICE" ]; then
    CMD="$CMD --device $DEVICE"
fi

if [ -n "$VERBOSE" ]; then
    CMD="$CMD $VERBOSE"
fi

# Print summary
echo ""
log_info "Configuration Summary"
echo "  Config: $CONFIG"
echo "  Sequences: $SEQUENCES"
echo "  Epochs: $EPOCHS"
echo "  Output: $OUTPUT"
if [ -n "$DEVICE" ]; then
    echo "  Device: $DEVICE (forced)"
else
    echo "  Device: auto-detect"
fi
echo ""

# Run training
log_info "Starting training..."
echo ""

if [ -n "$VERBOSE" ]; then
    # With verbose, show output directly
    eval "$CMD"
else
    # Without verbose, show progress
    eval "$CMD" 2>&1 | tee "$OUTPUT/training.log"
fi

TRAINING_STATUS=$?

if [ $TRAINING_STATUS -eq 0 ]; then
    echo ""
    log_info "Training completed successfully"
    log_info "Model saved to: $OUTPUT"
    echo ""
else
    echo ""
    log_error "Training failed with status $TRAINING_STATUS"
    echo "Check $OUTPUT/training.log for details"
    exit 1
fi
