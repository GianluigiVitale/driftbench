"""
Portability Risk Index (PRI) - Predict drift without running inference.

Uses trained Gradient Boosting model (R²=0.9943) learned from 420 real experiments
to predict flip rates for unseen configurations.
"""

import numpy as np
import pickle
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional


# Trained ML model path (Gradient Boosting with interactions, R²=0.9981)
# Try multiple locations for flexibility
def _get_model_path():
    """Find PRI model in multiple possible locations"""
    # Try relative path (for installed package)
    package_dir = Path(__file__).parent
    relative_path = package_dir.parent / 'models' / 'pri_model.pkl'
    if relative_path.exists():
        return relative_path
    
    # Try experiments directory (for development)
    dev_path = Path(__file__).parent.parent / "experiments" / "pri_model.pkl"
    if dev_path.exists():
        return dev_path
    
    # Try workspace path
    workspace_path = Path('/workspace/experiments/pri_model.pkl')
    if workspace_path.exists():
        return workspace_path
    
    # Return relative path as default (will error if not found)
    return relative_path

MODEL_PATH = _get_model_path()

# Feature columns (one-hot encoded) - must match training
FEATURE_COLUMNS = [
    'model_llama-3.1-70b', 'model_llama-3.1-8b', 'model_llama-3.1-8b-instruct-fp4',
    'model_mistral-7b', 'model_mixtral-8x7b', 'model_qwen-7b',
    'hardware_b200', 'hardware_h100', 'hardware_h200', 'hardware_mi300x',
    'precision_fp16', 'precision_fp4', 'precision_fp8',
    'framework_sglang', 'framework_tensorrt-llm', 'framework_vllm',
    'workload_code', 'workload_long_context', 'workload_math', 'workload_safety'
]

# Model metadata (trained with enhanced features including interactions)
MODEL_METADATA = {
    'r2_overall': 0.9981,
    'mae_pp': 0.22,
    'test_r2': 0.9866,
    'max_error': 4.22,
    'model_type': 'Gradient Boosting (deep) with interaction features',
    'n_features': 56,           # 20 base + 36 interaction features
    'data_version': 'corrected_safety_oct20',
    'trained_date': '2025-10-28'
}


def load_pri_model() -> Dict:
    """
    Load trained PRI model (Gradient Boosting).
    
    Returns:
        Dict with model object and metadata
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"PRI model not found at {MODEL_PATH}. "
            f"Please run experiments/train_pri_model_ml.py first."
        )
    
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    
    return {
        'version': '2.0',
        'model': model,
        'feature_columns': FEATURE_COLUMNS,
        **MODEL_METADATA
    }


def predict_drift(pri_model: Dict,
                  model: str,
                  hardware: str,
                  precision: str,
                  framework: str,
                  workload: str,
                  baseline: Optional[str] = None) -> Dict:
    """
    Predict flip rate for a configuration using trained PRI model.
    
    Args:
        pri_model: Loaded PRI model (from load_pri_model())
        model: Model name (e.g., 'llama-3.1-8b')
        hardware: Hardware platform (e.g., 'h100', 'b200')
        precision: Numeric precision (e.g., 'fp16', 'fp8', 'fp4')
        framework: Serving framework (e.g., 'vllm', 'tensorrt-llm', 'sglang')
        workload: Workload type (e.g., 'code', 'math', 'safety')
        baseline: Optional baseline config for relative prediction
    
    Returns:
        Dict with prediction and metadata
    """
    # Convert config to one-hot encoded features
    features = _config_to_features(model, hardware, precision, framework, workload,
                                   pri_model['feature_columns'])
    
    # Predict using trained model
    ml_model = pri_model['model']
    predicted_flip_rate = float(ml_model.predict(features)[0])
    
    # Clamp to [0, 100] for safety
    predicted_flip_rate = max(0.0, min(100.0, predicted_flip_rate))
    
    # Compute confidence interval using model MAE
    mae = pri_model['mae_pp']
    ci_lower = max(0.0, predicted_flip_rate - 1.96 * mae)
    ci_upper = min(100.0, predicted_flip_rate + 1.96 * mae)
    
    # Determine risk level
    if predicted_flip_rate < 2.0:
        risk_level = "LOW"
        recommendation = "Configuration appears safe for deployment"
    elif predicted_flip_rate < 5.0:
        risk_level = "MEDIUM"
        recommendation = "Consider canary testing with production traffic sample"
    else:
        risk_level = "HIGH"
        recommendation = "Run full drift analysis (driftbench compare) before deployment"
    
    # Build result
    result = {
        'predicted_flip_rate': round(predicted_flip_rate, 2),
        'confidence_interval': (round(ci_lower, 2), round(ci_upper, 2)),
        'risk_level': risk_level,
        'recommendation': recommendation,
        'config': {
            'model': model,
            'hardware': hardware,
            'precision': precision,
            'framework': framework,
            'workload': workload
        },
        'model_info': {
            'version': pri_model['version'],
            'r2_overall': pri_model['r2_overall'],
            'mae_pp': pri_model['mae_pp'],
            'model_type': pri_model.get('model_type', 'Gradient Boosting')
        }
    }
    
    # If baseline provided, compute relative risk
    if baseline:
        baseline_pred = _predict_baseline(
            pri_model, baseline, workload
        )
        relative_risk = predicted_flip_rate - baseline_pred
        result['baseline_flip_rate'] = round(baseline_pred, 2)
        result['relative_risk'] = round(relative_risk, 2)
        
        if abs(relative_risk) < 1.0:
            result['relative_assessment'] = "Similar risk to baseline"
        elif relative_risk > 0:
            result['relative_assessment'] = f"Risk is {relative_risk:.1f}pp higher than baseline"
        else:
            result['relative_assessment'] = f"Risk is {abs(relative_risk):.1f}pp lower than baseline"
    
    return result


def _config_to_features(model: str, hardware: str, precision: str,
                        framework: str, workload: str,
                        feature_columns: list) -> pd.DataFrame:
    """
    Convert configuration to one-hot encoded feature vector with interactions.
    
    Must match the feature order used during training (56 features total).
    Includes base features (20) + interaction features (36).
    """
    # Define all categories (must match training exactly)
    all_models = ['llama-3.1-70b', 'llama-3.1-8b', 'llama-3.1-8b-instruct-fp4', 
                  'mistral-7b', 'mixtral-8x7b', 'qwen-7b']
    all_hardware = ['b200', 'h100', 'h200', 'mi300x']
    all_precision = ['fp16', 'fp4', 'fp8']
    all_framework = ['sglang', 'tensorrt-llm', 'vllm']
    all_workload = ['code', 'long_context', 'math', 'safety']
    
    # Create features in EXACT order expected by model
    features = []
    feature_names = []
    
    # 1. Model one-hot (6 features)
    for m in all_models:
        features.append(1 if m == model else 0)
        feature_names.append(f'model_{m}')
    
    # 2. Hardware one-hot (4 features)
    for hw in all_hardware:
        features.append(1 if hw == hardware else 0)
        feature_names.append(f'hardware_{hw}')
    
    # 3. Precision one-hot (3 features)
    for prec in all_precision:
        features.append(1 if prec == precision else 0)
        feature_names.append(f'precision_{prec}')
    
    # 4. Framework one-hot (3 features)
    for fw in all_framework:
        features.append(1 if fw == framework else 0)
        feature_names.append(f'framework_{fw}')
    
    # 5. Workload one-hot (4 features)
    for wl in all_workload:
        features.append(1 if wl == workload else 0)
        feature_names.append(f'workload_{wl}')
    
    # 6. Precision × Workload interactions (12 features)
    for prec in all_precision:
        for wl in all_workload:
            val = 1 if (prec == precision and wl == workload) else 0
            features.append(val)
            feature_names.append(f'precision_{prec}_workload_{wl}')
    
    # 7. Framework × Workload interactions (12 features)
    for fw in all_framework:
        for wl in all_workload:
            val = 1 if (fw == framework and wl == workload) else 0
            features.append(val)
            feature_names.append(f'framework_{fw}_workload_{wl}')
    
    # 8. Hardware × Precision interactions (12 features)
    for hw in all_hardware:
        for prec in all_precision:
            val = 1 if (hw == hardware and prec == precision) else 0
            features.append(val)
            feature_names.append(f'hardware_{hw}_precision_{prec}')
    
    # Convert to DataFrame with correct column names
    df = pd.DataFrame([features], columns=feature_names)
    
    # Verify we have 56 features
    assert len(df.columns) == 56, f"Expected 56 features, got {len(df.columns)}"
    
    return df


def _predict_baseline(pri_model: Dict, baseline: str, workload: str, model: str = "llama-3.1-8b") -> float:
    """
    Predict flip rate for baseline configuration using ML model.
    
    Args:
        pri_model: PRI model
        baseline: Baseline config string (hardware/precision/framework)
        workload: Workload type
        model: Model name (default: llama-3.1-8b)
    
    Returns:
        Predicted flip rate
    """
    # Parse baseline
    parts = baseline.split('/')
    if len(parts) != 3:
        raise ValueError(f"Baseline must be hardware/precision/framework, got: {baseline}")
    
    hardware, precision, framework = parts
    
    # Convert to features and predict
    features = _config_to_features(model, hardware, precision, framework, workload,
                                   pri_model['feature_columns'])
    ml_model = pri_model['model']
    predicted = float(ml_model.predict(features)[0])
    
    return max(0.0, min(100.0, predicted))


def calibrate_pri_model(empirical_data: Dict) -> Dict:
    """
    Retrain PRI model with new empirical drift measurements.
    
    This function would be used to update the model as more data is collected.
    
    Args:
        empirical_data: Dict mapping (model, hardware, precision, framework, workload) -> flip_rate
    
    Returns:
        Updated PRI model
    """
    raise NotImplementedError(
        "PRI model retraining not yet implemented. "
        "To retrain, add new data to flip_rates.csv and run experiments/train_pri_model_ml.py"
    )
