#!/bin/bash
# DriftBench Quick Start Script
#
# This script demonstrates basic DriftBench usage with example configurations.

set -e

echo "=========================================="
echo "DriftBench Quick Start"
echo "=========================================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if driftbench is installed
if ! command -v driftbench &> /dev/null; then
    echo -e "${RED}Error: driftbench not installed${NC}"
    echo "Install with: pip install -e ."
    exit 1
fi

echo -e "${GREEN}✓ DriftBench installed${NC}"
echo ""

# Example 1: Predict drift (no inference needed)
echo "=========================================="
echo "Example 1: Predict Drift Risk (Quick)"
echo "=========================================="
echo ""
echo "Command:"
echo "  driftbench predict \\"
echo "    --model llama-3.1-8b \\"
echo "    --hardware b200 \\"
echo "    --precision fp8 \\"
echo "    --framework tensorrt-llm \\"
echo "    --workload safety"
echo ""

read -p "Press Enter to run..."

driftbench predict \
    --model llama-3.1-8b \
    --hardware b200 \
    --precision fp8 \
    --framework tensorrt-llm \
    --workload safety

echo ""
echo -e "${GREEN}✓ Example 1 complete${NC}"
echo ""

# Example 2: Predict with baseline comparison
echo "=========================================="
echo "Example 2: Predict with Baseline"
echo "=========================================="
echo ""
echo "Command:"
echo "  driftbench predict \\"
echo "    --model llama-3.1-8b \\"
echo "    --hardware h200 \\"
echo "    --precision fp16 \\"
echo "    --framework vllm \\"
echo "    --workload code \\"
echo "    --baseline 'h100/fp16/vllm'"
echo ""

read -p "Press Enter to run..."

driftbench predict \
    --model llama-3.1-8b \
    --hardware h200 \
    --precision fp16 \
    --framework vllm \
    --workload code \
    --baseline "h100/fp16/vllm"

echo ""
echo -e "${GREEN}✓ Example 2 complete${NC}"
echo ""

# Example 3: Compare configurations (requires model and prompts)
echo "=========================================="
echo "Example 3: Compare Configurations"
echo "=========================================="
echo ""
echo "This example requires:"
echo "  - Model at /workspace/mnt/exp/models/llama-3.1-8b"
echo "  - Prompts at /workspace/mnt/exp/datasets/safety/advbench_prompts.jsonl"
echo ""

# Check if files exist
MODEL_PATH="/workspace/mnt/exp/models/llama-3.1-8b"
PROMPTS_PATH="/workspace/mnt/exp/datasets/safety/advbench_prompts.jsonl"

if [ ! -d "$MODEL_PATH" ]; then
    echo -e "${YELLOW}⚠ Model not found: $MODEL_PATH${NC}"
    echo "Skipping Example 3 (requires model for inference)"
elif [ ! -f "$PROMPTS_PATH" ]; then
    echo -e "${YELLOW}⚠ Prompts not found: $PROMPTS_PATH${NC}"
    echo "Skipping Example 3 (requires prompts)"
else
    echo "Command:"
    echo "  driftbench compare \\"
    echo "    --baseline 'llama-3.1-8b/h100/fp16/vllm' \\"
    echo "    --test 'llama-3.1-8b/h100/fp8/vllm' \\"
    echo "    --prompts $PROMPTS_PATH \\"
    echo "    --workload safety \\"
    echo "    --max-prompts 10 \\"
    echo "    --output /tmp/drift_report.json"
    echo ""
    
    read -p "Press Enter to run (this will take a few minutes)..."
    
    driftbench compare \
        --baseline "llama-3.1-8b/h100/fp16/vllm" \
        --test "llama-3.1-8b/h100/fp8/vllm" \
        --prompts "$PROMPTS_PATH" \
        --workload safety \
        --max-prompts 10 \
        --output /tmp/drift_report.json
    
    echo ""
    echo "Report saved to: /tmp/drift_report.json"
    echo ""
    echo "View report:"
    echo "  cat /tmp/drift_report.json | jq"
    echo ""
    echo -e "${GREEN}✓ Example 3 complete${NC}"
fi

echo ""
echo "=========================================="
echo "Quick Start Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Read documentation: cat driftbench/README.md"
echo "  2. Run driftbench --help for all options"
echo "  3. Try with your own models and prompts"
echo ""
echo "For more examples, see:"
echo "  - driftbench/README.md"
echo "  - examples/ directory"
echo ""
