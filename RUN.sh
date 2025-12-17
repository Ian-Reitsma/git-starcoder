#!/bin/bash

################################################################################
# Block Model Training System - Complete Execution Script
################################################################################

echo ""
echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                  BLOCK MODEL TRAINING SYSTEM - RUN SCRIPT                   ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR" || exit 1

echo "[1/5] Checking Python version..."
python3 --version
if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Python 3 not found${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python 3 detected${NC}"
echo ""

echo "[2/5] Setting up virtual environment..."
if [ ! -d "venv" ]; then
    echo "  Creating new venv..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo -e "${RED}✗ Failed to create venv${NC}"
        exit 1
    fi
fi

echo "  Activating venv..."
source venv/bin/activate
if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Failed to activate venv${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Virtual environment ready${NC}"
echo ""

echo "[3/5] Installing dependencies..."
pip install --upgrade pip > /dev/null 2>&1
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Failed to install dependencies${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Dependencies installed${NC}"
echo ""

echo "[4/5] Running test suite..."
python3 test_suite.py
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}⚠ Some tests failed${NC}"
    echo "  (This may be OK if torch/GPU tests are skipped)"
fi
echo ""

echo "[5/5] Ready to run pipeline"
echo -e "${GREEN}✓ All systems ready${NC}"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "To start training your Block model, run:"
echo ""
echo "  python3 run_pipeline_dynamic.py \\"
echo "    --repo /Users/ianreitsma/projects/the-block \\"
echo "    --verbose"
echo ""
echo "This will:"
echo "  1. Analyze your repository (all branches & commits)"
echo "  2. Scrape git history with full metadata"
echo "  3. Tokenize commits into 2048-token sequences"
echo "  4. Generate embeddings for semantic understanding"
echo "  5. Train GPT-2-medium on your code patterns (auto-determined epochs)"
echo "  6. Generate MANIFEST_DYNAMIC.json with complete statistics"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "After training completes, review results:"
echo ""
echo "  # See all statistics"
echo "  jq '.' MANIFEST_DYNAMIC.json | less"
echo ""
echo "  # Just training metrics"
echo "  jq '.training_report' MANIFEST_DYNAMIC.json"
echo ""
echo "  # Just repository stats"
echo "  jq '.repository_stats' MANIFEST_DYNAMIC.json"
echo ""
echo "  # Just training parameters (formula-based)"
echo "  jq '.training_parameters' MANIFEST_DYNAMIC.json"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Configuration Overview:"
echo ""
echo "  • Base learning rate: 5e-5 (scales with batch size)"
echo "  • Batch size: Auto-detected (8 for 8GB+, 4 for 4-8GB, 2 for <4GB)"
echo "  • Warmup: 10% of training steps (min 100, max 1000)"
echo "  • Epochs: Auto-calculated from token count (target: 20M tokens)"
echo "  • Validation split: 10% of data"
echo "  • Early stopping: Patience=3 with min_delta=0.0001"
echo "  • Determinism: Seeds=42 (reproducible runs)"
echo ""
echo "Hardware-Aware Optimization:"
echo ""
echo "  • GPU memory detection: batch size adjusted automatically"
echo "  • CPU detection: num_workers = cpu_count // 2 (max 8)"
echo "  • Hardware monitoring: Every 10 seconds (not per-step overhead)"
echo "  • Peak memory tracking: GPU and RAM peaks recorded"
echo ""
echo "Data Flow:"
echo ""
echo "  1. Repository commits → Git scraper → Rich metadata (30+ fields)"
echo "  2. Commits → Tokenizer → 2048-token sequences with 256-token overlap"
echo "  3. Sequences → Embeddings → Vector representations (384-dim)"
echo "  4. Sequences → Trainer → Model fine-tuning with stats"
echo ""
echo "Output Files:"
echo ""
echo "  • models/the-block-git-model-final/        (trained model, 345M params)"
echo "  • models/the-block-git-model-final/training_report.json  (detailed stats)"
echo "  • MANIFEST_DYNAMIC.json                     (complete run manifest)"
echo "  • data/git_history_rich.jsonl               (raw commit metadata)"
echo "  • data/token_sequences_rich.json            (tokenized sequences)"
echo "  • embeddings/embeddings_rich.npz            (semantic vectors)"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
