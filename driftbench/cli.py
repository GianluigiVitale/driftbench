#!/usr/bin/env python3
"""
DriftBench CLI - Cross-Stack Drift Validation for LLM Deployments

Usage:
    driftbench compare --baseline CONFIG1 --test CONFIG2 --prompts FILE
    driftbench predict --model MODEL --hardware HW --precision PREC --framework FW --workload TYPE
"""

import click
import json
import sys
from pathlib import Path
from typing import Optional

@click.group()
@click.version_option(version="1.0.0")
def cli():
    """
    DriftBench - Measure and predict cross-stack drift in LLM serving.
    
    Examples:
        # Compare two configurations
        driftbench compare \\
            --baseline "llama-3.1-8b/h100/fp16/vllm" \\
            --test "llama-3.1-8b/h100/fp8/vllm" \\
            --prompts safety_prompts.jsonl \\
            --workload safety
        
        # Predict drift risk (no inference needed)
        driftbench predict \\
            --model llama-3.1-8b \\
            --hardware b200 \\
            --precision fp8 \\
            --framework tensorrt-llm \\
            --workload safety
    """
    pass


@cli.command()
@click.option('--baseline', 'baseline_config', required=True,
              help='Baseline config: model/hardware/precision/framework (e.g., llama-3.1-8b/h100/fp16/vllm)')
@click.option('--test', 'test_config', required=True,
              help='Test config to compare against baseline')
@click.option('--prompts', 'prompt_file', type=click.Path(exists=True),
              help='JSONL file with prompts (only needed for live inference, not for pre-computed comparisons)')
@click.option('--workload', required=True, 
              type=click.Choice(['code', 'math', 'safety', 'long_context']),
              help='Workload type for task-specific evaluation')
@click.option('--output', default='drift_report.json',
              help='Output file for results (default: drift_report.json)')
@click.option('--threshold', default=1.0, type=float,
              help='Acceptable flip rate threshold in %% (default: 1.0)')
@click.option('--model-path', type=click.Path(),
              help='Override model path (default: /workspace/mnt/exp/models/MODEL)')
@click.option('--max-prompts', type=int,
              help='Limit number of prompts to process')
def compare(baseline_config, test_config, prompt_file, workload, output, 
            threshold, model_path, max_prompts):
    """
    Compare two LLM serving configurations for drift.
    
    This command runs inference on both configurations and measures how often
    they produce different task outcomes (flip rate).
    
    Example:
        driftbench compare \\
            --baseline "llama-3.1-8b/h100/fp16/vllm" \\
            --test "llama-3.1-8b/h100/fp8/vllm" \\
            --prompts /workspace/mnt/exp/datasets/safety/advbench_prompts.jsonl \\
            --workload safety \\
            --threshold 1.0 \\
            --max-prompts 100
    """
    from driftbench.core import run_comparison, compute_flip_rate
    
    click.echo("=" * 80)
    click.echo("DRIFTBENCH: Cross-Stack Drift Validation")
    click.echo("=" * 80)
    click.echo(f"\nConfiguration:")
    click.echo(f"   Baseline: {baseline_config}")
    click.echo(f"   Test:     {test_config}")
    click.echo(f"   Workload: {workload}")
    if prompt_file:
        click.echo(f"   Prompts:  {prompt_file}")
        if max_prompts:
            click.echo(f"   Limit:    {max_prompts} prompts")
    
    try:
        # Try to get pre-computed flip rate from flip_rates.csv (instant!)
        from driftbench.core import get_precomputed_flip_rate
        
        click.echo("\nChecking for pre-computed results in flip_rates.csv...")
        precomputed = get_precomputed_flip_rate(baseline_config, test_config, workload)
        
        if precomputed:
            # Instant results!
            click.echo(f"   Found pre-computed comparison!")
            click.echo(f"   Using instant lookup (no inference or re-analysis needed)\n")
            
            flip_rate = precomputed['flip_rate']
            num_flips = precomputed['num_flips']
            num_comparisons = precomputed['num_comparisons']
            
            # Create summary report
            click.echo("=" * 80)
            click.echo("DRIFT COMPARISON RESULTS (PRE-COMPUTED)")
            click.echo("=" * 80)
            click.echo(f"\nFlip Rate: {flip_rate:.2f}%")
            click.echo(f"{num_flips}/{num_comparisons} outputs changed")
            click.echo(f"\nThreshold: {threshold}%")
            
            # Decision
            passes = flip_rate <= threshold
            if passes:
                click.echo(f"\nPASS: Drift ({flip_rate:.2f}%) is within threshold ({threshold}%)")
                decision = "PASS"
                exit_code = 0
            else:
                click.echo(f"\nFAIL: Drift ({flip_rate:.2f}%) exceeds threshold ({threshold}%)")
                decision = "FAIL"
                exit_code = 1
            
            # Save report
            report = {
                "baseline_config": baseline_config,
                "test_config": test_config,
                "workload": workload,
                "flip_rate_pct": flip_rate,
                "num_flips": num_flips,
                "num_comparisons": num_comparisons,
                "threshold_pct": threshold,
                "decision": decision,
                "source": precomputed['source']
            }
            
            with open(output, 'w') as f:
                json.dump(report, f, indent=2)
            
            click.echo(f"\nSaved report to: {output}")
            click.echo("=" * 80)
            sys.exit(exit_code)
        
        # If not in pre-computed data, fall back to running inference
        click.echo(f"   No pre-computed results for {baseline_config} vs {test_config}")
        click.echo("   Falling back to live inference...\n")
        
        # Check if prompts file is provided for live inference
        if not prompt_file:
            click.echo("Error: --prompts is required for live inference (not in pre-computed data)", err=True)
            click.echo("Tip: Use --prompts <your_prompts.jsonl> or check if configs are correct", err=True)
            sys.exit(2)
        
        # Run inference on baseline config
        click.echo("Running inference on baseline config...")
        baseline_results = run_comparison(
            baseline_config, 
            prompt_file, 
            workload,
            model_path_override=model_path,
            max_prompts=max_prompts
        )
        click.echo(f"   Generated {len(baseline_results)} outputs")
        
        # Run inference on test config
        click.echo("\nRunning inference on test config...")
        test_results = run_comparison(
            test_config, 
            prompt_file, 
            workload,
            model_path_override=model_path,
            max_prompts=max_prompts
        )
        click.echo(f"   Generated {len(test_results)} outputs")
        
        # Compute drift metrics
        click.echo("\nComputing drift metrics...")
        flip_rate, flips, metrics = compute_flip_rate(
            baseline_results, 
            test_results, 
            workload
        )
        
        # Generate report
        report = {
            'driftbench_version': '1.0.0',
            'baseline_config': baseline_config,
            'test_config': test_config,
            'workload': workload,
            'num_prompts': len(baseline_results),
            'flip_rate_pct': round(flip_rate, 2),
            'num_flips': len(flips),
            'threshold_pct': threshold,
            'decision': 'PASS' if flip_rate <= threshold else 'FAIL',
            'metrics': metrics,
            'flip_examples': flips[:10]  # First 10 flip examples
        }
        
        # Save report
        output_path = Path(output)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Display results
        click.echo(f"\n{'=' * 80}")
        click.echo(f"DRIFT ANALYSIS RESULTS")
        click.echo(f"{'=' * 80}")
        click.echo(f"\nFlip Rate: {click.style(f'{flip_rate:.2f}%', bold=True, fg='red' if flip_rate > threshold else 'green')}")
        click.echo(f"   Flips Detected: {len(flips)}/{len(baseline_results)}")
        click.echo(f"   Threshold: {threshold:.2f}%")
        
        # Decision
        if flip_rate <= threshold:
            click.echo(f"\n{click.style('PASS', bold=True, fg='green')} - Drift within acceptable threshold")
        else:
            click.echo(f"\n{click.style('FAIL', bold=True, fg='red')} - Drift exceeds threshold")
            click.echo(f"   Drift is {flip_rate/threshold:.1f}x higher than acceptable")
        
        # Show metrics if available
        if metrics:
            click.echo(f"\nAdditional Metrics:")
            for key, value in metrics.items():
                click.echo(f"   {key}: {value}")
        
        # Show example flips
        if flips and len(flips) > 0:
            click.echo(f"\nExample Flips (first 3):")
            for i, flip in enumerate(flips[:3], 1):
                click.echo(f"\n   Flip {i}: {flip['prompt_id']}")
                click.echo(f"   Baseline: {flip['baseline_label']}")
                click.echo(f"   Test:     {flip['test_label']}")
        
        click.echo(f"\nFull report saved to: {output_path.absolute()}")
        
        # Exit with appropriate code
        sys.exit(0 if flip_rate <= threshold else 1)
        
    except Exception as e:
        click.echo(f"\nError: {e}", err=True)
        import traceback
        traceback.print_exc()
        sys.exit(2)


@cli.command()
@click.option('--model', required=True,
              help='Model name (e.g., llama-3.1-8b, llama-3.1-70b)')
@click.option('--hardware', required=True,
              type=click.Choice(['h100', 'h200', 'b200', 'mi300x']),
              help='Target hardware platform')
@click.option('--precision', required=True,
              type=click.Choice(['fp16', 'fp8', 'fp4']),
              help='Numerical precision format (pre-computed data available for: fp16, fp8, fp4)')
@click.option('--framework', required=True,
              type=click.Choice(['vllm', 'tensorrt-llm', 'sglang']),
              help='Serving framework')
@click.option('--workload', required=True,
              type=click.Choice(['code', 'math', 'safety', 'long_context']),
              help='Workload type')
@click.option('--baseline', 
              help='Baseline config for comparison (e.g., h100/fp16/vllm)')
def predict(model, hardware, precision, framework, workload, baseline):
    """
    Predict drift risk using PRI model (no inference required).
    
    This command uses the Portability Risk Index (PRI) to estimate drift
    without running inference. Useful for quick screening before deployment.
    
    Example:
        driftbench predict \\
            --model llama-3.1-8b \\
            --hardware b200 \\
            --precision fp8 \\
            --framework tensorrt-llm \\
            --workload safety \\
            --baseline "h100/fp16/vllm"
    """
    from driftbench.pri import load_pri_model, predict_drift
    
    click.echo("=" * 80)
    click.echo("DRIFTBENCH: Portability Risk Index (PRI)")
    click.echo("=" * 80)
    click.echo(f"\nTarget Configuration:")
    click.echo(f"   Model:     {model}")
    click.echo(f"   Hardware:  {hardware}")
    click.echo(f"   Precision: {precision}")
    click.echo(f"   Framework: {framework}")
    click.echo(f"   Workload:  {workload}")
    
    if baseline:
        click.echo(f"\nBaseline Configuration:")
        click.echo(f"   {baseline}")
    
    try:
        # Load PRI model
        click.echo("\nLoading PRI model...")
        pri_model = load_pri_model()
        click.echo("   Model loaded")
        
        # Predict drift
        click.echo("\nPredicting drift risk...")
        prediction = predict_drift(
            pri_model,
            model=model,
            hardware=hardware,
            precision=precision,
            framework=framework,
            workload=workload,
            baseline=baseline
        )
        
        predicted_flip_rate = prediction['predicted_flip_rate']
        confidence_interval = prediction.get('confidence_interval', (None, None))
        risk_level = prediction['risk_level']
        recommendation = prediction['recommendation']
        
        # Display results
        click.echo(f"\n{'=' * 80}")
        click.echo(f"PORTABILITY RISK INDEX (PRI)")
        click.echo(f"{'=' * 80}")
        
        # Color based on risk
        color_map = {'LOW': 'green', 'MEDIUM': 'yellow', 'HIGH': 'red'}
        color = color_map.get(risk_level, 'white')
        
        click.echo(f"\nPredicted Flip Rate: {click.style(f'{predicted_flip_rate:.2f}%', bold=True, fg=color)}")
        
        if confidence_interval[0] is not None:
            click.echo(f"   95% CI: [{confidence_interval[0]:.2f}%, {confidence_interval[1]:.2f}%]")
        
        click.echo(f"\nRisk Level: {click.style(risk_level, bold=True, fg=color)}")
        click.echo(f"   {recommendation}")
        
        # Show factors
        if 'factors' in prediction:
            click.echo(f"\nRisk Factors:")
            for factor, impact in prediction['factors'].items():
                click.echo(f"   {factor}: {impact}")
        
        click.echo(f"\nNote: PRI provides screening only.")
        click.echo(f"   Run 'driftbench compare' for validation with actual inference.")
        
        # Exit code based on risk
        exit_code = {'LOW': 0, 'MEDIUM': 0, 'HIGH': 1}.get(risk_level, 0)
        sys.exit(exit_code)
        
    except Exception as e:
        click.echo(f"\nError: {e}", err=True)
        import traceback
        traceback.print_exc()
        sys.exit(2)


@cli.command(name='list-configs')
@click.option('--model', help='Filter by model (optional)')
@click.option('--workload', help='Show workloads for each config', is_flag=True)
def list_configs(model, workload):
    """
    List all available pre-computed configurations.
    
    Shows all 105 configurations from the DriftBench dataset that can be
    instantly compared without re-running inference.
    """
    from driftbench.core import list_available_configs, get_available_workloads_for_config
    from collections import defaultdict
    
    click.echo("=" * 80)
    click.echo("AVAILABLE PRE-COMPUTED CONFIGURATIONS")
    click.echo("=" * 80)
    click.echo("\nThese configurations can be instantly compared using 'driftbench compare'\n")
    
    configs = list_available_configs()
    
    if not configs:
        click.echo("No pre-computed configurations found.")
        click.echo("   Check that flip_rates.csv exists at: /workspace/mnt/exp/analysis/metrics/")
        return
    
    # Group by model
    by_model = defaultdict(list)
    for config in configs:
        config_model = config.split('/')[0]
        if model and model not in config_model:
            continue
        by_model[config_model].append(config)
    
    # Display
    total_count = 0
    for config_model in sorted(by_model.keys()):
        click.echo(f"Model: {click.style(config_model, bold=True)}")
        click.echo(f"   {len(by_model[config_model])} configurations:\n")
        
        for cfg in sorted(by_model[config_model]):
            total_count += 1
            click.echo(f"   {total_count:3d}. {cfg}")
            
            if workload:
                workloads = get_available_workloads_for_config(cfg)
                click.echo(f"        Workloads: {', '.join(workloads)}")
        
        click.echo()
    
    click.echo("=" * 80)
    click.echo(f"Total: {total_count} pre-computed configurations available")
    click.echo("\nUsage:")
    click.echo("   driftbench compare \\")
    click.echo("       --baseline <config> \\")
    click.echo("       --test <config> \\")
    click.echo("       --workload <workload>")
    click.echo("\n   Note: --prompts flag not needed for pre-computed comparisons")
    click.echo("=" * 80)


@cli.command()
def version():
    """Show DriftBench version information."""
    click.echo("DriftBench v1.0.0")
    click.echo("Cross-Stack Drift Validation for LLM Deployments")
    click.echo("\nFor more information: https://anonymous.4open.science/r/driftbench")


if __name__ == '__main__':
    cli()
