"""
Portability Risk Index (PRI) - Predict drift without running inference.

Enhanced version with interaction features for improved accuracy (R²=0.9981).
"""

import numpy as np
import pickle
import pandas as pd
from pathlib import Path
from typing import Dict, Optional

# Trained ML model path (Gradient Boosting with interactions, R²=0.9981)
MODEL_PATH = Path(__file__).parent.parent / "experiments" / "pri_model.pkl"
FEATURE_COLUMNS_PATH = Path(__file__).parent.parent / "experiments" / "pri_feature_columns.txt"

# Model metadata (trained with enhanced features including interactions)
MODEL_METADATA = {
    'r2_overall': 0.9981,
    'mae_pp': 0.22,
    'test_r2': 0.9866,
    'max_error': 4.22,
    'model_type': 'Gradient Boosting (deep) with interaction features',
    'n_features': 56,
    'data_version': 'corrected_safety_oct20',
    'trained_date': '2025-10-28'
}


def load_pri_model() -> Dict:
    """
    Load trained PRI model (Gradient Boosting with interactions).
    
    Returns:
        Dict with model object and metadata
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"PRI model not found at {MODEL_PATH}. "
            f"Please run experiments/train_pri_enhanced.py first."
        )
    
    # Load model
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    
    # Load feature columns
    if FEATURE_COLUMNS_PATH.exists():
        with open(FEATURE_COLUMNS_PATH, 'r') as f:
            feature_columns = [line.strip() for line in f.readlines()]
    else:
        raise FileNotFoundError(
            f"Feature columns not found at {FEATURE_COLUMNS_PATH}"
        )
    
    return {
        'version': '2.0-enhanced',
        'model': model,
        'feature_columns': feature_columns,
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
    # Convert config to features with interactions
    features = _config_to_features(
        model, hardware, precision, framework, workload,
        pri_model['feature_columns']
    )
    
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
            'test_r2': pri_model.get('test_r2', None),
            'n_features': pri_model['n_features']
        }
    }
    
    return result


def _config_to_features(model: str, hardware: str, precision: str,
                        framework: str, workload: str,
                        feature_columns: list) -> np.ndarray:
    """
    Convert configuration to feature vector matching training format.
    
    Generates base one-hot features + interaction features (56 total).
    Must exactly match the logic in train_pri_enhanced.py
    """
    # Start with all features as 0
    features_dict = {col: 0 for col in feature_columns}
    
    # Set base one-hot features
    features_dict[f'model_{model}'] = 1
    features_dict[f'hardware_{hardware}'] = 1
    features_dict[f'precision_{precision}'] = 1
    features_dict[f'framework_{framework}'] = 1
    features_dict[f'workload_{workload}'] = 1
    
    # Set precision × workload interactions
    interaction_key = f'precision_{precision}_workload_{workload}'
    if interaction_key in features_dict:
        features_dict[interaction_key] = 1
    
    # Set framework × workload interactions  
    interaction_key = f'framework_{framework}_workload_{workload}'
    if interaction_key in features_dict:
        features_dict[interaction_key] = 1
    
    # Set hardware × precision interactions
    interaction_key = f'hardware_{hardware}_precision_{precision}'
    if interaction_key in features_dict:
        features_dict[interaction_key] = 1
    
    # Convert to array in exact order of feature_columns
    feature_array = np.array([features_dict[col] for col in feature_columns]).reshape(1, -1)
    
    return feature_array
