# DriftBench CLI

Tool for instant drift comparison and risk prediction in LLM serving configurations.

## Overview

DriftBench CLI enables instant comparison of LLM serving configurations without re-running inference:

- **Instant results** (< 0.1 seconds)
- **No GPU required** (uses pre-computed data)
- **Full coverage**: 105 configurations from DriftBench
- Exit codes, JSON output, CI/CD integration
- Pre-computed results for 236,985 prompt-response pairs
- Portability Risk Index (PRI) model for prediction
- Command-line interface and Python API

---

## Quick Start

### Installation

```bash
cd driftbench
pip install -e .
```

### Run Production Case (23.85% Drift)

Reproduce the exact production validation case from the paper:

```bash
python3 -m driftbench.cli compare \
  --baseline "llama-3.1-8b/h100/fp16/sglang" \
  --test "llama-3.1-8b/b200/fp8/sglang" \
  --workload safety
```

**Expected Output:**
```
Flip Rate: 23.85%
124/520 outputs changed
FAIL: Drift (23.85%) exceeds threshold (1.0%)
```

### Automated Test

```bash
python3 test_installation.py
```

This will install the package, test all commands, and verify the production case.

---

## Commands

### List Configurations

```bash
python3 -m driftbench.cli list-configs
```

Shows all 105 pre-computed configurations (models, hardware, precision, frameworks).

### Compare Configurations

```bash
python3 -m driftbench.cli compare \
  --baseline "llama-3.1-8b/h100/fp16/vllm" \
  --test "llama-3.1-8b/h100/fp8/vllm" \
  --workload safety \
  --threshold 1.0 \
  --output drift_report.json
```

**Exit Codes:**
- `0` = PASS (drift ≤ threshold)
- `1` = FAIL (drift > threshold)
- `2` = ERROR

### Predict Drift Risk (PRI)

```bash
python3 -m driftbench.cli predict \
  --model llama-3.1-8b \
  --hardware b200 \
  --precision fp8 \
  --framework sglang \
  --workload safety
```

Returns predicted flip rate with 95% confidence intervals.

---

## Verification

All commands tested with 40+ test cases. All numerical results match the paper.

| Test | Result | Details |
|------|--------|---------|
| Production case | PASS | 23.85%, 124/520 (exact match) |
| PRI prediction | PASS | 24.00% (within CI of actual 23.85%) |
| All configurations | PASS | 105 configurations found |
| Data files | PASS | flip_rates.csv (421 rows), pri_model.pkl (1.7 MB) |

### Performance

| Operation | Time | GPU Required |
|-----------|------|--------------|
| list-configs | < 0.1s | No |
| compare (pre-computed) | < 0.1s | No |
| compare (live inference) | 5-30 min | Yes (1-4 GPUs) |
| predict (PRI) | < 0.5s | No |

---

## Supported Configurations

**Models** (6):
- llama-3.1-8b, llama-3.1-70b
- llama-3.1-8b-instruct-fp4 (quantized variant)
- mistral-7b, mixtral-8x7b, qwen-7b

*Note: All models are instruct-tuned versions*

**Hardware**:
- NVIDIA H100
- NVIDIA H200
- NVIDIA B200
- AMD MI300X

**Precision**:
- FP16
- FP8
- FP4

**Frameworks**:
- vLLM 
- SGLang 
- TensorRT-LLM

**Workloads** (4):
- code (HumanEval, 164 problems)
- math (GSM8K, 500 problems)
- safety (AdvBench, 520 prompts)
- long_context (LongBench, 100 documents)

**Total**: 105 configurations × 4 workloads = 420 pre-computed comparisons

---

## Pre-Computed Data

**Source**: `data/flip_rates.csv` (420 rows)

**Coverage:**
- 236,985 prompt-response pairs from DriftBench
- All numerical results from the paper

**Performance:**
- **Instant**: < 0.1 seconds (vs 5-30 minutes for live inference)
- **No GPU required**

---

## Repository Structure

```
driftbench/
├── driftbench/           # Main package
│   ├── __init__.py      # Package initialization
│   ├── cli.py           # Command-line interface
│   ├── core.py          # Comparison logic
│   ├── pri.py           # PRI model
│   ├── evaluators.py    # Task evaluators
│   ├── setup.py         # Installation
│   └── README.md        # Technical documentation
├── models/
│   └── pri_model.pkl    # Pre-trained PRI model
├── data/
│   └── flip_rates.csv   # Pre-computed flip rates (420 rows)
├── test_installation.py # Automated test script
├── driftbench_cli.sh    # CLI wrapper script
└── README.md            # This file
```

---

## Documentation

- **`driftbench/README.md`** - Complete technical documentation, API reference, usage examples
- **Paper**: Section 4 (DriftBench) and Appendix

## License

- **Code:** MIT License
- **Data:** CC BY 4.0

**Archived:** [10.5281/zenodo.19361066](https://doi.org/10.5281/zenodo.19361066)

---

**DriftBench CLI v1.0** | MLSys 2026
