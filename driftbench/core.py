"""
Core inference and drift computation logic for DriftBench.

Integrates with existing experiment infrastructure from run_experiment.py.
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from driftbench.evaluators import get_evaluator


def run_comparison(config: str, prompt_file: str, workload: str,
                   model_path_override: Optional[str] = None,
                   max_prompts: Optional[int] = None) -> List[Dict]:
    """
    Run inference for a given configuration.
    
    Args:
        config: String like "llama-3.1-8b/h100/fp16/vllm"
        prompt_file: Path to JSONL with prompts
        workload: Workload type
        model_path_override: Optional override for model path
        max_prompts: Optional limit on number of prompts
    
    Returns:
        List of dicts with prompt_id, prompt, and generated_text
    """
    # Parse config
    parts = config.split('/')
    if len(parts) != 4:
        raise ValueError(f"Config must be model/hardware/precision/framework, got: {config}")
    
    model, hardware, precision, framework = parts
    
    # Load prompts
    with open(prompt_file) as f:
        prompts_data = [json.loads(line) for line in f if line.strip()]
    
    if max_prompts:
        prompts_data = prompts_data[:max_prompts]
    
    prompts = [p['prompt'] for p in prompts_data]
    
    # Determine model path
    if model_path_override:
        model_path = model_path_override
    else:
        # Default paths
        model_path = f'/workspace/mnt/exp/models/{model}'
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
    
    # Initialize model based on framework
    if framework == 'vllm':
        return _run_vllm_inference(model_path, prompts_data, precision)
    
    elif framework == 'tensorrt-llm':
        return _run_tensorrt_inference(model_path, prompts_data, precision)
    
    elif framework == 'sglang':
        return _run_sglang_inference(model_path, prompts_data, precision)
    
    else:
        raise NotImplementedError(f"Framework {framework} not yet supported")


def _run_vllm_inference(model_path: str, prompts_data: List[Dict], 
                        precision: str) -> List[Dict]:
    """Run inference using vLLM framework."""
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        raise ImportError("vLLM not installed. Install with: pip install vllm")
    
    # Map precision to dtype
    dtype_map = {
        'fp32': 'float32',
        'fp16': 'float16',
        'bf16': 'bfloat16',
        'fp8': 'float8_e4m3fn',
    }
    
    dtype = dtype_map.get(precision, 'float16')
    
    # Initialize vLLM
    llm = LLM(
        model=model_path,
        dtype=dtype,
        enforce_eager=True,
        gpu_memory_utilization=0.9
    )
    
    # Sampling parameters (deterministic)
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=512,
        seed=42,
        top_p=1.0
    )
    
    # Extract prompts
    prompts = [p['prompt'] for p in prompts_data]
    
    # Generate
    outputs = llm.generate(prompts, sampling_params)
    
    # Format results
    results = []
    for prompt_data, output in zip(prompts_data, outputs):
        results.append({
            'prompt_id': prompt_data['prompt_id'],
            'prompt': prompt_data['prompt'],
            'generated_text': output.outputs[0].text
        })
    
    return results


def _run_tensorrt_inference(model_path: str, prompts_data: List[Dict],
                            precision: str) -> List[Dict]:
    """Run inference using TensorRT-LLM framework."""
    # Placeholder - would integrate with TensorRT-LLM
    raise NotImplementedError(
        "TensorRT-LLM support coming soon. "
        "For now, generate outputs using run_experiment.py and analyze with existing tools."
    )


def _run_sglang_inference(model_path: str, prompts_data: List[Dict],
                         precision: str) -> List[Dict]:
    """Run inference using SGLang framework."""
    # Placeholder - would integrate with SGLang
    raise NotImplementedError(
        "SGLang support coming soon. "
        "For now, generate outputs using run_experiment.py and analyze with existing tools."
    )


def compute_flip_rate(baseline_results: List[Dict], 
                     test_results: List[Dict],
                     workload: str) -> Tuple[float, List[Dict], Dict]:
    """
    Compute flip rate by comparing baseline vs test outputs.
    
    Args:
        baseline_results: List of baseline outputs
        test_results: List of test outputs
        workload: Workload type
    
    Returns:
        Tuple of (flip_rate, flips_list, metrics_dict)
    """
    if len(baseline_results) != len(test_results):
        raise ValueError(
            f"Result count mismatch: baseline={len(baseline_results)}, "
            f"test={len(test_results)}"
        )
    
    # Get evaluator for workload
    evaluator = get_evaluator(workload)
    
    flips = []
    baseline_correct_count = 0
    test_correct_count = 0
    
    for baseline, test in zip(baseline_results, test_results):
        if baseline['prompt_id'] != test['prompt_id']:
            raise ValueError(
                f"Prompt ID mismatch: {baseline['prompt_id']} vs {test['prompt_id']}"
            )
        
        # Evaluate both outputs
        baseline_label, baseline_confidence = evaluator.evaluate(
            baseline['generated_text'],
            baseline['prompt']
        )
        
        test_label, test_confidence = evaluator.evaluate(
            test['generated_text'],
            test['prompt']
        )
        
        # Track correctness
        if baseline_label:
            baseline_correct_count += 1
        if test_label:
            test_correct_count += 1
        
        # Check for flip
        if baseline_label != test_label:
            flips.append({
                'prompt_id': baseline['prompt_id'],
                'baseline_label': 'correct' if baseline_label else 'incorrect',
                'test_label': 'correct' if test_label else 'incorrect',
                'baseline_confidence': baseline_confidence,
                'test_confidence': test_confidence,
                'baseline_output': baseline['generated_text'][:200] + '...',
                'test_output': test['generated_text'][:200] + '...'
            })
    
    # Compute metrics
    total = len(baseline_results)
    flip_rate = (len(flips) / total) * 100
    
    baseline_accuracy = (baseline_correct_count / total) * 100
    test_accuracy = (test_correct_count / total) * 100
    accuracy_delta = test_accuracy - baseline_accuracy
    
    metrics = {
        'baseline_accuracy_pct': round(baseline_accuracy, 2),
        'test_accuracy_pct': round(test_accuracy, 2),
        'accuracy_delta_pp': round(accuracy_delta, 2),
        'agreement_rate_pct': round(100 - flip_rate, 2)
    }
    
    return flip_rate, flips, metrics


def load_existing_outputs(output_dir: Path, workload: str) -> List[Dict]:
    """
    Load existing experiment outputs from run_experiment.py.
    
    Useful for analyzing already-generated results without re-running inference.
    
    Args:
        output_dir: Directory containing JSON output files
        workload: Workload filter
    
    Returns:
        List of output dicts
    """
    results = []
    
    for json_file in sorted(output_dir.glob(f"*--{workload}--*.json")):
        try:
            with open(json_file) as f:
                data = json.load(f)
                results.append({
                    'prompt_id': data['prompt_id'],
                    'prompt': data['prompt'],
                    'generated_text': data.get('generated_text', '')
                })
        except Exception as e:
            print(f"Warning: Could not load {json_file}: {e}")
    
    return results


def get_precomputed_flip_rate(baseline_config: str, test_config: str, workload: str,
                               flip_rates_path: str = None) -> Optional[Dict]:
    """
    Get pre-computed flip rate from flip_rates.csv for instant comparison.
    
    All 420 comparisons (105 configs × 4 workloads) are pre-computed in flip_rates.csv.
    This provides instant results without re-running inference or re-analyzing outputs.
    
    Args:
        baseline_config: Baseline config like "llama-3.1-8b/h100/fp16/vllm" 
        test_config: Test config like "llama-3.1-8b/h100/fp8/vllm"
        workload: Workload type (code, math, safety, long_context)
        flip_rates_path: Path to flip_rates.csv
    
    Returns:
        Dict with flip_rate, num_flips, num_comparisons, or None if not found
    """
    try:
        import pandas as pd
    except ImportError:
        return None
    
    # Default path logic - try multiple locations
    if flip_rates_path is None:
        # Try relative path (for installed package)
        package_dir = Path(__file__).parent
        relative_path = package_dir.parent / 'data' / 'flip_rates.csv'
        
        # Try original workspace path (for development)
        workspace_path = Path('/workspace/mnt/exp/analysis/metrics/flip_rates.csv')
        
        if relative_path.exists():
            flip_rates_path = str(relative_path)
        elif workspace_path.exists():
            flip_rates_path = str(workspace_path)
        else:
            return None
    
    # Parse configs
    baseline_parts = baseline_config.split('/')
    test_parts = test_config.split('/')
    
    if len(baseline_parts) != 4 or len(test_parts) != 4:
        return None
    
    baseline_model, baseline_hw, baseline_prec, baseline_fw = baseline_parts
    test_model, test_hw, test_prec, test_fw = test_parts
    
    # For now, we only support same-model comparisons (infrastructure changes)
    # This matches the paper's focus: hardware/precision/framework changes
    if baseline_model != test_model:
        return None
    
    # Load flip_rates.csv
    flip_rates_file = Path(flip_rates_path)
    if not flip_rates_file.exists():
        return None
    
    df = pd.read_csv(flip_rates_file)
    
    # Find the test config in the data
    # Note: flip_rates.csv shows drift relative to a baseline (typically H100/FP16/vLLM)
    # For arbitrary baseline vs test, we query the test config's row
    match = df[
        (df['model'] == test_model) &
        (df['hardware'] == test_hw) &
        (df['precision'] == test_prec) &
        (df['framework'] == test_fw) &
        (df['workload'] == workload)
    ]
    
    if len(match) == 0:
        return None
    
    row = match.iloc[0]
    
    return {
        'flip_rate': float(row['flip_rate']),
        'num_flips': int(row['num_flips']),
        'num_comparisons': int(row['num_comparisons']),
        'baseline_config': baseline_config,
        'test_config': test_config,
        'workload': workload,
        'source': 'pre-computed (flip_rates.csv)'
    }


def list_available_configs(flip_rates_path: str = None) -> List[str]:
    """
    List all available pre-computed configurations from flip_rates.csv.
    
    Returns:
        List of config strings in format "model/hardware/precision/framework"
    """
    try:
        import pandas as pd
    except ImportError:
        return []
    
    # Default path logic
    if flip_rates_path is None:
        package_dir = Path(__file__).parent
        relative_path = package_dir.parent / 'data' / 'flip_rates.csv'
        workspace_path = Path('/workspace/mnt/exp/analysis/metrics/flip_rates.csv')
        
        if relative_path.exists():
            flip_rates_path = str(relative_path)
        elif workspace_path.exists():
            flip_rates_path = str(workspace_path)
        else:
            return []
    
    flip_rates_file = Path(flip_rates_path)
    if not flip_rates_file.exists():
        return []
    
    df = pd.read_csv(flip_rates_file)
    
    # Extract unique configs
    configs = df[['model', 'hardware', 'precision', 'framework']].drop_duplicates()
    config_list = [f"{row['model']}/{row['hardware']}/{row['precision']}/{row['framework']}" 
                   for _, row in configs.iterrows()]
    
    return sorted(config_list)


def get_available_workloads_for_config(config: str,
                                       flip_rates_path: str = None) -> List[str]:
    """
    Get available workloads for a specific configuration.
    
    Args:
        config: Config string like "llama-3.1-8b/h100/fp16/vllm"
        flip_rates_path: Path to flip_rates.csv
    
    Returns:
        List of available workload names
    """
    try:
        import pandas as pd
    except ImportError:
        return []
    
    parts = config.split('/')
    if len(parts) != 4:
        return []
    
    model, hardware, precision, framework = parts
    
    # Default path logic
    if flip_rates_path is None:
        package_dir = Path(__file__).parent
        relative_path = package_dir.parent / 'data' / 'flip_rates.csv'
        workspace_path = Path('/workspace/mnt/exp/analysis/metrics/flip_rates.csv')
        
        if relative_path.exists():
            flip_rates_path = str(relative_path)
        elif workspace_path.exists():
            flip_rates_path = str(workspace_path)
        else:
            return []
    
    flip_rates_file = Path(flip_rates_path)
    if not flip_rates_file.exists():
        return []
    
    df = pd.read_csv(flip_rates_file)
    
    # Find workloads for this config
    match = df[
        (df['model'] == model) &
        (df['hardware'] == hardware) &
        (df['precision'] == precision) &
        (df['framework'] == framework)
    ]
    
    return sorted(match['workload'].unique().tolist())
