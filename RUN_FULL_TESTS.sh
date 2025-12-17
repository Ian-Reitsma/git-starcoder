#!/bin/bash

# Full Coverage Test Suite Runner
# Run all three comprehensive test suites with verbose output
#
# Usage:
#   bash RUN_FULL_TESTS.sh
#   bash RUN_FULL_TESTS.sh /path/to/repo
#   REPO_PATH=/path/to/repo bash RUN_FULL_TESTS.sh

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get repository path
REPO_PATH="${1:-${REPO_PATH}}"

if [ -z "$REPO_PATH" ]; then
    echo -e "${RED}Error: Repository path not specified${NC}"
    echo "Usage: $0 /path/to/repo"
    echo "  or: REPO_PATH=/path/to/repo $0"
    exit 1
fi

if [ ! -d "$REPO_PATH" ]; then
    echo -e "${RED}Error: Repository path does not exist: $REPO_PATH${NC}"
    exit 1
fi

echo -e "${BLUE}"
echo "################################################################################"
echo "# FULL COVERAGE TEST SUITE - RUNNER"
echo "################################################################################"
echo -e "${NC}"

echo -e "\nTest Environment:"
echo "  Repository: $REPO_PATH"
echo "  Working directory: $(pwd)"
echo "  Python version: $(python3 --version)"
echo "  Start time: $(date)"

echo -e "\n${BLUE}Tests to run:${NC}"
echo "  1. Behavioral Evaluation Test (~10 min)"
echo "  2. Pipeline Orchestration Test (~10 min)"
echo "  3. StarCoder2-3B + 4-bit + LoRA Test (~20 min)"
echo "  Total: ~35-40 minutes"

echo -e "\n${YELLOW}Press ENTER to begin...${NC}"
read -r

# Track results
TOTAL_TESTS=3
PASSED_TESTS=0
FAILED_TESTS=0
start_time=$(date +%s)

# Test 1: Behavioral Evaluation
echo -e "\n${BLUE}================================================================================"
echo "[1/3] Running Behavioral Evaluation Test"
echo "================================================================================\n${NC}"

if python3 test_behavioral_evaluation.py; then
    echo -e "${GREEN}✓ Behavioral Evaluation Test PASSED${NC}"
    ((PASSED_TESTS++))
else
    echo -e "${RED}✗ Behavioral Evaluation Test FAILED${NC}"
    ((FAILED_TESTS++))
fi

# Test 2: Pipeline Orchestration
echo -e "\n${BLUE}================================================================================"
echo "[2/3] Running Pipeline Orchestration Test"
echo "================================================================================\n${NC}"

if python3 test_pipeline_orchestration.py "$REPO_PATH"; then
    echo -e "${GREEN}✓ Pipeline Orchestration Test PASSED${NC}"
    ((PASSED_TESTS++))
else
    echo -e "${RED}✗ Pipeline Orchestration Test FAILED${NC}"
    ((FAILED_TESTS++))
fi

# Test 3: StarCoder2-3B + 4-bit + LoRA
echo -e "\n${BLUE}================================================================================"
echo "[3/3] Running StarCoder2-3B + 4-bit + LoRA Test"
echo "================================================================================\n${NC}"

if python3 test_starcoder_lora_quantization.py; then
    echo -e "${GREEN}✓ StarCoder2-3B + 4-bit + LoRA Test PASSED${NC}"
    ((PASSED_TESTS++))
else
    echo -e "${RED}✗ StarCoder2-3B + 4-bit + LoRA Test FAILED${NC}"
    ((FAILED_TESTS++))
fi

# Calculate timing
end_time=$(date +%s)
duration=$((end_time - start_time))
minutes=$((duration / 60))
seconds=$((duration % 60))

# Print final report
echo -e "\n${BLUE}================================================================================"
echo "# FINAL REPORT"
echo "================================================================================\n${NC}"

echo "Test Results:"
echo "  Total tests: $TOTAL_TESTS"
echo "  Passed: $(echo -e "${GREEN}${PASSED_TESTS}${NC}")"
echo "  Failed: $(echo -e "${RED}${FAILED_TESTS}${NC}")"

if [ "$FAILED_TESTS" -eq 0 ]; then
    success_rate=100
else
    success_rate=$((PASSED_TESTS * 100 / TOTAL_TESTS))
fi

echo "  Success rate: ${success_rate}%"
echo ""
echo "Timing:"
echo "  Total duration: ${minutes}m ${seconds}s"
echo "  Start time: $(date -r $start_time '+%Y-%m-%d %H:%M:%S')"
echo "  End time: $(date '+%Y-%m-%d %H:%M:%S')"

echo ""
if [ "$FAILED_TESTS" -eq 0 ]; then
    echo -e "${GREEN}✅✅✅ ALL TESTS PASSED! ✅✅✅${NC}"
    echo -e "${GREEN}Your system is production-ready with full test coverage!${NC}"
    exit 0
else
    echo -e "${RED}✗✗✗ ${FAILED_TESTS} TEST(S) FAILED ✗✗✗${NC}"
    echo -e "${RED}Review output above for details${NC}"
    exit 1
fi
