#!/bin/bash
# Ultimate training launcher with pre-flight checks and optimizations
# Usage: ./train_starcoder_optimized.sh [--repo PATH] [--config PATH] [--epochs N]

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step() { echo -e "\n${BLUE}==> $1${NC}"; }

# Defaults
CONFIG="training_config_metal_cuda_universal.yaml"
REPO=""
EPOCHS=3
OUTPUT="models/starcoder2-optimized"
SKIP_VALIDATION=false
VERBOSE=false

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --config) CONFIG="$2"; shift 2 ;;
        --repo) REPO="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --output) OUTPUT="$2"; shift 2 ;;
        --skip-validation) SKIP_VALIDATION=true; shift ;;
        --verbose) VERBOSE=true; shift ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --config PATH         Training config (default: $CONFIG)"
            echo "  --repo PATH           Target repository to train on"
            echo "  --epochs N            Number of epochs (default: $EPOCHS)"
            echo "  --output DIR          Output directory (default: $OUTPUT)"
            echo "  --skip-validation     Skip pre-training validation"
            echo "  --verbose             Verbose output"
            echo "  --help                Show this message"
            exit 0
            ;;
        *) log_error "Unknown option: $1"; exit 1 ;;
    esac
done

log_step "StarCoder2-3B Optimized Training Launcher"

# Check Python/venv
if [[ ! -f ".venv/bin/python" ]]; then
    log_error "Virtual environment not found at .venv/"
    log_info "Run: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
    exit 1
fi

PYTHON=".venv/bin/python"
log_info "Using Python: $PYTHON"

# Check config exists
if [[ ! -f "$CONFIG" ]]; then
    log_error "Config not found: $CONFIG"
    exit 1
fi
log_info "Config: $CONFIG"

# Validation
if [[ "$SKIP_VALIDATION" == "false" ]]; then
    log_step "Running pre-training validation"
    
    if [[ -n "$REPO" ]]; then
        $PYTHON validate_training_setup.py --config "$CONFIG" --repo "$REPO" || {
            log_error "Validation failed"
            exit 1
        }
    else
        $PYTHON validate_training_setup.py --config "$CONFIG" || {
            log_error "Validation failed"
            exit 1
        }
    fi
    
    log_info "Validation passed"
else
    log_warn "Skipping validation (--skip-validation)"
fi

# Check if dataset needs to be generated
log_step "Checking dataset"

# Parse dataset path from config
DATASET_DIR=$(grep -A5 'train_path:' "$CONFIG" | grep 'train_path:' | awk '{print $2}' | tr -d '"' | xargs dirname 2>/dev/null || echo "")

if [[ -n "$DATASET_DIR" ]] && [[ ! -d "$DATASET_DIR" ]]; then
    log_warn "Dataset directory not found: $DATASET_DIR"
    
    if [[ -n "$REPO" ]]; then
        log_info "Generating enhanced dataset from $REPO"
        
        $PYTHON run_pipeline_enhanced.py \
            --repo "$REPO" \
            --base-dir "./data_enhanced" \
            --config "$CONFIG" || {
            log_error "Dataset generation failed"
            exit 1
        }
        
        log_info "Dataset generated"
    else
        log_error "Dataset not found and no --repo specified"
        exit 1
    fi
else
    log_info "Dataset exists: $DATASET_DIR"
fi

# Display training parameters
log_step "Training Configuration"
log_info "Config file: $CONFIG"
log_info "Epochs: $EPOCHS"
log_info "Output: $OUTPUT"

if [[ -n "$REPO" ]]; then
    log_info "Training on: $REPO"
fi

# Detect device
DEVICE="auto"
if command -v nvidia-smi &> /dev/null; then
    log_info "CUDA detected"
    DEVICE="cuda"
elif [[ "$(uname)" == "Darwin" ]]; then
    log_info "macOS detected (will use MPS)"
    DEVICE="mps"
else
    log_info "CPU training"
    DEVICE="cpu"
fi

# Final confirmation
log_step "Ready to start training"
log_warn "This will start a multi-hour training run."
echo -n "Continue? [y/N] "
read -r response
if [[ ! "$response" =~ ^[Yy]$ ]]; then
    log_info "Aborted by user"
    exit 0
fi

# Launch training
log_step "Starting training"

VERBOSE_FLAG=""
if [[ "$VERBOSE" == "true" ]]; then
    VERBOSE_FLAG="--verbose"
fi

# Use run_metal_cuda.sh if it exists, otherwise direct trainer invocation
if [[ -f "run_metal_cuda.sh" ]]; then
    ./run_metal_cuda.sh \
        --config "$CONFIG" \
        --epochs "$EPOCHS" \
        --output "$OUTPUT" \
        $VERBOSE_FLAG
else
    $PYTHON -m training.model_trainer_unified \
        --config "$CONFIG" \
        --epochs "$EPOCHS" \
        --output "$OUTPUT" \
        --device "$DEVICE"
fi

TRAIN_EXIT=$?

if [[ $TRAIN_EXIT -eq 0 ]]; then
    log_step "Training completed successfully"
    log_info "Model saved to: $OUTPUT"
    
    # Show model info
    if [[ -f "$OUTPUT/training_info.json" ]]; then
        log_info "Training stats:"
        cat "$OUTPUT/training_info.json" | grep -E '(final_|best_|total_)' || true
    fi
else
    log_error "Training failed with exit code $TRAIN_EXIT"
    exit $TRAIN_EXIT
fi

log_step "Done"
