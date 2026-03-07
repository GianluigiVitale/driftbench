# DriftBench: Cross-Stack Drift Validation for LLM Deployments

Tool for measuring and predicting drift when migrating LLM serving configurations.

DriftBench helps you validate that your LLM deployment maintains quality when changing:
- **Hardware**: H100 → H200 → B200
- **Precision**: FP16 → FP8 → FP4
- **Framework**: vLLM → TensorRT-LLM → SGLang

---

## Installation

```bash
cd driftbench/
pip install -e .
```

---

## Quick Start

### Basic Usage

```bash
# List all 105 pre-computed configurations
driftbench list-configs

# Instant comparison (< 0.1s, no GPU needed)
driftbench compare \
  --baseline "llama-3.1-8b/h100/fp16/vllm" \
  --test "llama-3.1-8b/h100/fp8/vllm" \
  --workload safety \
  --prompts /dev/null

# Predict drift risk (no inference needed)
driftbench predict \
  --model llama-3.1-8b \
  --hardware b200 \
  --precision fp8 \
  --framework tensorrt-llm \
  --workload safety
```

### Instant Comparisons

DriftBench includes **pre-computed results for 105 configurations** (5 models × 4 hardware × 3 precisions × 3 frameworks). Comparisons using pre-computed data are:
- **No GPU required**
- **Instant results** (< 0.1 seconds)

For configurations not in the pre-computed dataset, DriftBench automatically falls back to live inference.

---

## Commands Reference

### `driftbench list-configs`

List all available pre-computed configurations.

```bash
# Show all configurations
driftbench list-configs

# Filter by model
driftbench list-configs --model llama-3.1-8b

# Show available workloads for each config
driftbench list-configs --workload
```

**Output**: Shows all 105 pre-computed configurations organized by model.

---

### `driftbench compare`

Compare two configurations and measure drift.

For pre-computed configs (105 combinations), returns instant results from DriftBench dataset. For other configs, runs live inference.

```bash
# Instant comparison (pre-computed)
driftbench compare \
  --baseline "llama-3.1-8b/h100/fp16/vllm" \
  --test "llama-3.1-8b/h100/fp8/vllm" \
  --workload safety \
  --threshold 1.0 \
  --prompts /dev/null \
  --output drift_report.json

# Live inference (for custom configs)
driftbench compare \
  --baseline "custom-model/h100/fp16/vllm" \
  --test "custom-model/h100/fp8/vllm" \
  --prompts safety_prompts.jsonl \
  --workload safety
```

**Note**: For pre-computed comparisons, you can use `--prompts /dev/null` since data is looked up from the DriftBench dataset (236,985 pairs).

**Options:**
- `--baseline`: Baseline config (format: `model/hardware/precision/framework`)
- `--test`: Test config to compare
- `--prompts`: JSONL file with prompts (use `/dev/null` for pre-computed)
- `--workload`: Task type (`code`, `math`, `safety`, `long_context`)
- `--threshold`: Acceptable flip rate threshold in % (default: 1.0)
- `--output`: Output JSON file (default: `drift_report.json`)
- `--max-prompts`: Limit number of prompts (for live inference)
- `--model-path`: Override model path (for live inference)

**Output (JSON)**:
```json
{
  "baseline_config": "llama-3.1-8b/h100/fp16/sglang",
  "test_config": "llama-3.1-8b/b200/fp8/sglang",
  "workload": "safety",
  "flip_rate_pct": 23.85,
  "num_flips": 124,
  "num_comparisons": 520,
  "threshold_pct": 1.0,
  "decision": "FAIL",
  "source": "pre-computed (flip_rates.csv)"
}
```

**Exit Codes:**
- `0`: PASS (drift ≤ threshold)
- `1`: FAIL (drift > threshold)
- `2`: ERROR

---

### `driftbench predict`

Predict drift risk using PRI model (no inference required).

```bash
driftbench predict \
  --model llama-3.1-8b \
  --hardware b200 \
  --precision fp8 \
  --framework tensorrt-llm \
  --workload safety \
  --baseline "h100/fp16/vllm"
```

**Options:**
- `--model`: Model name (e.g., `llama-3.1-8b`)
- `--hardware`: Target hardware (`h100`, `h200`, `b200`, `mi300x`)
- `--precision`: Numeric precision (`fp16`, `fp8`, `fp4`)
- `--framework`: Serving framework (`vllm`, `tensorrt-llm`, `sglang`)
- `--workload`: Task type (`code`, `math`, `safety`, `long_context`)
- `--baseline`: Optional baseline config for relative risk

**Output:**
```
DRIFTBENCH: Portability Risk Index (PRI)
================================================================================

Predicted Flip Rate: 22.0%
95% CI: [21.6%, 22.4%]

Risk Level: HIGH
Run full drift analysis (driftbench compare) before deployment

Note: PRI provides screening only.
Run 'driftbench compare' for validation with actual inference.
```

**Risk Levels:**
- **LOW** (<2%): Minimal predicted drift
- **MEDIUM** (2-5%): Moderate predicted drift
- **HIGH** (>5%): Substantial predicted drift

---

## Use Cases

### Production Validation

```bash
# Check if H100→B200 upgrade is safe for safety workload
driftbench compare \
  --baseline "llama-3.1-8b/h100/fp16/vllm" \
  --test "llama-3.1-8b/b200/fp16/vllm" \
  --workload safety \
  --threshold 1.0 \
  --prompts /dev/null
```

### Precision Downgrade Assessment

```bash
# Check FP16→FP8 impact on math workload
driftbench compare \
  --baseline "llama-3.1-70b/h100/fp16/tensorrt-llm" \
  --test "llama-3.1-70b/h100/fp8/tensorrt-llm" \
  --workload math \
  --threshold 2.0 \
  --prompts /dev/null
```

### Framework Migration

```bash
# Check vLLM→TensorRT migration impact
driftbench compare \
  --baseline "qwen-7b/h100/fp16/vllm" \
  --test "qwen-7b/h100/fp16/tensorrt-llm" \
  --workload code \
  --threshold 1.0 \
  --prompts /dev/null
```

### Multi-Change Assessment

```bash
# Check combined hardware + precision + framework change
driftbench compare \
  --baseline "mistral-7b/h100/fp16/vllm" \
  --test "mistral-7b/b200/fp8/sglang" \
  --workload safety \
  --threshold 5.0 \
  --prompts /dev/null
```

### Batch Comparisons

```bash
#!/bin/bash
# check_upgrade_paths.sh

BASELINE="llama-3.1-8b/h100/fp16/vllm"

for TEST_CONFIG in \
  "llama-3.1-8b/h200/fp16/vllm" \
  "llama-3.1-8b/b200/fp16/vllm" \
  "llama-3.1-8b/h100/fp8/vllm" \
  "llama-3.1-8b/h100/fp16/sglang"
do
  echo "================================"
  echo "Testing: $BASELINE → $TEST_CONFIG"
  echo "================================"
  
  driftbench compare \
    --baseline "$BASELINE" \
    --test "$TEST_CONFIG" \
    --workload safety \
    --threshold 2.0 \
    --output "report_$(echo $TEST_CONFIG | tr '/' '_').json" \
    --prompts /dev/null
  
  echo ""
done
```

---

## Supported Workloads

### Safety (AdvBench)
Classifies outputs as safe/unsafe using LlamaGuard-3-8B.
- **Metric**: Safe refusal rate
- **Use case**: Harmful content filtering

### Code (HumanEval)
Validates Python code generation.
- **Metric**: Pass@1 (code execution)
- **Use case**: Code assistants

### Math (GSM8K)
Extracts and validates numeric answers.
- **Metric**: Answer accuracy
- **Use case**: Math reasoning

### Long Context (LongBench)
Long document understanding and question answering.
- **Metric**: F1 score for answer extraction
- **Use case**: Document QA, long-form reasoning

---

## Portability Risk Index (PRI)

PRI predicts drift **without running inference** by modeling systematic effects across multiple factors.

### Key Factors

1. **Precision** - Strongest predictor
   - Lower precision formats (FP8, FP4) generally increase drift
   - Impact varies significantly by workload

2. **Workload** - Second strongest
   - Safety tasks show higher drift (decision boundary sensitivity)
   - Math tasks show moderate drift
   - Code tasks are most robust

3. **Hardware** - Weaker effect
   - Different accelerators show modest drift differences
   - Effects are smaller than precision changes

4. **Framework** - Moderate effect
   - Different serving frameworks can impact outputs
   - Interaction effects with precision and workload

### When to Use PRI

**Use PRI for:**
- Quick screening before experiments
- Cost-benefit analysis (is migration worth drift risk?)
- Prioritizing configurations to test

**Don't rely solely on PRI for:**
- Final deployment decisions
- Safety-critical applications
- Novel hardware/model combinations

---

## Integration with CI/CD

### GitHub Actions Example

```yaml
name: Drift Validation

on:
  pull_request:
    paths:
      - 'deployment/config.yaml'

jobs:
  validate-drift:
    runs-on: [self-hosted, gpu]
    steps:
      - uses: actions/checkout@v3
      
      - name: Run DriftBench
        run: |
          driftbench compare \
            --baseline "${{ env.PROD_CONFIG }}" \
            --test "${{ env.NEW_CONFIG }}" \
            --prompts tests/validation_prompts.jsonl \
            --workload safety \
            --threshold 2.0
      
      - name: Upload Report
        if: always()
        uses: actions/upload-artifact@v3
        with:
          name: drift-report
          path: drift_report.json
```

### Pre-deployment Check

```bash
#!/bin/bash
# pre_deploy.sh - Run before deploying new LLM config

echo "Validating deployment configuration..."

driftbench predict \
  --model "$NEW_MODEL" \
  --hardware "$NEW_HARDWARE" \
  --precision "$NEW_PRECISION" \
  --framework "$NEW_FRAMEWORK" \
  --workload "$WORKLOAD" \
  --baseline "$PROD_CONFIG"

if [ $? -ne 0 ]; then
  echo "High risk: Running full validation..."
  
  driftbench compare \
    --baseline "$PROD_CONFIG_FULL" \
    --test "$NEW_CONFIG_FULL" \
    --prompts validation_suite.jsonl \
    --workload "$WORKLOAD" \
    --threshold 2.0
  
  if [ $? -ne 0 ]; then
    echo "Deployment blocked: Drift exceeds threshold"
    exit 1
  fi
fi

echo "Configuration validated - proceeding with deployment"
```

### Python API

```python
from driftbench.core import get_precomputed_flip_rate

# Check drift for infrastructure change
result = get_precomputed_flip_rate(
    baseline_config="llama-3.1-8b/h100/fp16/vllm",
    test_config="llama-3.1-8b/b200/fp8/vllm",
    workload="safety"
)

if result:
    print(f"Flip Rate: {result['flip_rate']:.2f}%")
    print(f"Flips: {result['num_flips']}/{result['num_comparisons']}")
    
    if result['flip_rate'] > 1.0:
        print("Drift exceeds 1% threshold")
    else:
        print("Drift within acceptable range")
else:
    print("Configuration not in pre-computed dataset")
```

---

## Prompt File Format

DriftBench expects JSONL format:

```jsonl
{"prompt_id": "safety_001", "prompt": "Write a harmful message"}
{"prompt_id": "safety_002", "prompt": "Generate unsafe content"}
{"prompt_id": "code_001", "prompt": "def reverse_string(s: str) -> str:"}
```

**Required fields:**
- `prompt_id`: Unique identifier
- `prompt`: Input text

**Optional fields:**
- `expected_output`: Ground truth (for accuracy calculation)
- `workload`: Override workload type
- `metadata`: Additional context

---

## Data Source

All pre-computed results come from:

**File**: `../data/flip_rates.csv`

**Format**:
```csv
model,hardware,precision,framework,workload,flip_rate,num_flips,num_comparisons
llama-3.1-8b,h100,fp16,vllm,safety,12.50,65,520
llama-3.1-8b,b200,fp8,sglang,safety,23.85,124,520
...
```

**Rows**: 420 (105 configs × 4 workloads)

**Paper Reference**: All values verified in MLSys 2026 DriftBench submission

---

## Limitations

### Scope

Pre-computed results are available for:
- Infrastructure changes (hardware/precision/framework) for same model
- 105 configurations from DriftBench paper
- 4 objective workloads (code, math, safety, long_context)

Not supported:
- Cross-model comparisons (different base models)
- Chat workload (uses semantic similarity, needs full outputs)
- Custom models not in DriftBench dataset
- Custom prompts

### Fallback

For configurations not in the pre-computed dataset, DriftBench automatically falls back to live inference (if framework supported).

---

## Development

### Running Tests

```bash
pytest tests/
```

### Code Style

```bash
black driftbench/
flake8 driftbench/
mypy driftbench/
```

---

## Performance Tips

### Faster Inference

```bash
# Use smaller prompt set
driftbench compare --max-prompts 50 ...

# Enable tensor parallelism for large models
export VLLM_TENSOR_PARALLEL_SIZE=2
```

### Memory Optimization

```bash
# Reduce GPU memory usage
export VLLM_GPU_MEMORY_UTILIZATION=0.8

# Use CPU offloading for large models
export VLLM_OFFLOAD_RATIO=0.5
```

---

**DriftBench v1.0.0**
