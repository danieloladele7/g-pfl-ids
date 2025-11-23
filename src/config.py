# src/config.py
import os
import yaml
from pathlib import Path
from typing import Dict, Any

def get_run_config() -> Dict[str, Any]:
    """Get configuration from environment variables with fallbacks"""
    
    # Default configuration - matches your pyproject.toml structure
    config = {
        "num-server-rounds": int(os.getenv("FLWR_NUM_SERVER_ROUNDS", "2")),
        "fraction-fit": float(os.getenv("FLWR_FRACTION_FIT", "0.5")),
        "fraction-evaluate": float(os.getenv("FLWR_FRACTION_EVALUATE", "1.0")),
        "min-fit-clients": int(os.getenv("FLWR_MIN_FIT_CLIENTS", "2")),
        "min-evaluate-clients": int(os.getenv("FLWR_MIN_EVALUATE_CLIENTS", "2")),
        "min-available-clients": int(os.getenv("FLWR_MIN_AVAILABLE_CLIENTS", "10")),
        "local-epochs": int(os.getenv("FLWR_LOCAL_EPOCHS", "3")),
        "mode": os.getenv("FLWR_MODE", "oneclass"),
        "model_type": os.getenv("FLWR_MODEL_TYPE", "gae"), # gcn, gae, gcn_deepsvdd, gae_deepsvdd, gcn_personalized, gae_personalized
        "strategy": os.getenv("FLWR_STRATEGY", "fedavg"), # fedavg, fedprox
        "fedprox_mu": float(os.getenv("FLWR_FEDPROX_MU", "0.01")),
    }
    
    return config

def load_config() -> Dict[str, Any]:
    """Load experiment configuration from YAML file."""
    config_path = Path(__file__).parent.parent / "configs" / "experiments.yaml"
    
    if not config_path.exists():
        # Fallback to default config
        return get_default_config()
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def get_default_config() -> Dict[str, Any]:
    """Return default configuration if YAML file is not found."""
    return {
        "common": {
            "in_dim": 128,
            "dropout": 0.5,
            "batch_size": 32,
            "learning_rate": 0.01,
            "weight_decay": 5e-4,
        },
        "oneclass_models": {
            "deepsvdd": {
                "hidden_dims": [64, 32],
                "embed_dim": 16,
            },
            "gae": {
                "hidden_dims": [64, 32],
                "embed_dim": 16,
            }
        }
    }

def get_model_config(model_type: str, task_type: str = "oneclass") -> Dict[str, Any]:
    """Get configuration for specific model and task type."""
    config = load_config()
    
    # Get base configuration
    model_config = config['common'].copy()
    
    # Add model-specific configuration
    if task_type == 'oneclass':
        if model_type in config['oneclass_models']:
            model_config.update(config['oneclass_models'][model_type])
    
    return model_config

def save_experiment_config(experiment_name: str, config: Dict[str, Any]):
    """Save experiment configuration for reproducibility."""
    output_dir = Path("experiments") / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config_path = output_dir / "config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
