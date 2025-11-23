# src/server_app.py
"""fl-gcn-ids: A Flower / PyTorch app for One-Class Learning"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from flwr.common import Context, Metrics, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg, FedProx

from src.models import GCN, GAE, DeepSVDDGCN, GAEDeepSVDD, GCNWithPersonalizedHead, GAEWithPersonalizedHead
from src.utils import get_model_parameters, set_model_parameters

# Try to import config, provide fallback if missing
try:
    from src.config import get_run_config
except ImportError:
    print("Warning: src.config not found, using fallback configuration")
    def get_run_config():
        return {
            "num-server-rounds": int(os.getenv("FLWR_NUM_SERVER_ROUNDS", "20")),
            "fraction-fit": float(os.getenv("FLWR_FRACTION_FIT", "0.5")),
            "fraction-evaluate": 1.0,
            "min-fit-clients": int(os.getenv("FLWR_MIN_FIT_CLIENTS", "2")),
            "min-evaluate-clients": int(os.getenv("FLWR_MIN_EVALUATE_CLIENTS", "2")),
            "min-available-clients": int(os.getenv("FLWR_MIN_AVAILABLE_CLIENTS", "10")),
            "local-epochs": int(os.getenv("FLWR_LOCAL_EPOCHS", "30")),
            "mode": os.getenv("FLWR_MODE", "oneclass"),
            "model_type": os.getenv("FLWR_MODEL_TYPE", "deepsvdd"),
            "strategy": os.getenv("FLWR_STRATEGY", "fedavg"),
            "fedprox_mu": float(os.getenv("FLWR_FEDPROX_MU", "0.01")),
        }

import torch
import logging

logger = logging.getLogger(__name__)

class OneClassHistoryManager:
    """Manages training history for one-class learning with proper saving"""
    
    def __init__(self):
        self.history = {
            "losses_distributed": [],
            "metrics_distributed": {
                "auroc": [], "aupr": [], "f1": [], "precision": [], "recall": [],
                "accuracy": [], "best_f1": [], "tpr_at_fpr_0.01": [], 
                "tpr_at_fpr_0.05": [], "tpr_at_fpr_0.1": []
            },
            "config": {},
            "timestamp": datetime.now().isoformat()
        }
        self.current_round = 0
        self.best_auroc = 0.0
        self.best_model_path = None
        self.model_type = "deepsvdd"
        self.strategy = "fedavg"
        self.experiment_name = "default_experiment"
        
        # Initialize directories with defaults to avoid AttributeError
        self._initialize_directories()
    
    def _initialize_directories(self):
        """Initialize directory paths - called in __init__ to ensure they exist"""
        self.base_dir = Path("results") / self.experiment_name / self.model_type / self.strategy
        self.checkpoints_dir = Path("checkpoints") / self.experiment_name / self.model_type / self.strategy
        self.metrics_dir = self.base_dir / "metrics"
        
        # Create directories if they don't exist
        for dir_path in [self.base_dir, self.checkpoints_dir, self.metrics_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def initialize_results_manager(self, experiment_name, model_type, strategy):
        """Initialize results manager with experiment details"""
        self.experiment_name = experiment_name
        self.model_type = model_type
        self.strategy = strategy
        
        # Re-initialize directories with new parameters
        self._initialize_directories()
    
    def set_round(self, round_num):
        self.current_round = round_num
    
    def add_round_results(self, loss, metrics):
        """Add results from a round - FIXED to handle None values"""
        if loss is not None and not (isinstance(loss, float) and (loss is None or loss == 0.0)):
            self.history["losses_distributed"].append((self.current_round, float(loss)))
        
        if metrics:
            for metric_name, value in metrics.items():
                if metric_name in self.history["metrics_distributed"]:
                    # Handle None, NaN, and invalid values
                    if value is not None and not (isinstance(value, float) and (value is None or value == 0.0)):
                        try:
                            self.history["metrics_distributed"][metric_name].append((self.current_round, float(value)))
                        except (ValueError, TypeError):
                            # Skip invalid values
                            continue
            
            # Track best AUROC for model saving
            if "auroc" in metrics and metrics["auroc"] is not None:
                try:
                    auroc_val = float(metrics["auroc"])
                    if auroc_val > self.best_auroc:
                        self.best_auroc = auroc_val
                except (ValueError, TypeError):
                    pass
    
    def save_model(self, model, parameters, round_num, is_best=False):
        """Save model checkpoint - FIXED to handle different parameter formats"""
        # Ensure directories exist
        self._initialize_directories()
        
        # Convert parameters to the right format if needed
        if hasattr(parameters, 'tensors'):
            from flwr.common import parameters_to_ndarrays
            parameters_ndarrays = parameters_to_ndarrays(parameters)
        else:
            parameters_ndarrays = parameters
        
        # Set parameters and save
        try:
            set_model_parameters(model, parameters_ndarrays)
        except Exception as e:
            logger.warning(f"Error setting parameters: {e}")
            # Save parameters directly without setting to model
            if is_best:
                model_path = self.checkpoints_dir / "global_model_best.pt"
                torch.save({
                    'parameters': parameters_ndarrays,
                    'model_type': self.model_type,
                    'round': round_num,
                    'timestamp': datetime.now().isoformat(),
                    'strategy': self.strategy,
                    'experiment_name': self.experiment_name
                }, model_path)
                self.best_model_path = model_path
                logger.info(f"✅ Best model parameters saved: {model_path}")
            else:
                model_path = self.checkpoints_dir / f"global_model_round_{round_num}.pt"
                torch.save({'parameters': parameters_ndarrays}, model_path)
                logger.info(f"✅ Model parameters saved: {model_path}")
            return model_path
        
        # Save model with state dict
        if is_best:
            model_path = self.checkpoints_dir / "global_model_best.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_type': self.model_type,
                'round': round_num,
                'timestamp': datetime.now().isoformat(),
                'strategy': self.strategy,
                'experiment_name': self.experiment_name
            }, model_path)
            self.best_model_path = model_path
            logger.info(f"✅ Best model saved: {model_path}")
        else:
            model_path = self.checkpoints_dir / f"global_model_round_{round_num}.pt"
            torch.save(model.state_dict(), model_path)
            logger.info(f"✅ Model checkpoint saved: {model_path}")
        
        return model_path
    
    def save_results(self, round_num=None):
        """Save all results - FIXED to handle empty history"""
        # Ensure directories exist
        self._initialize_directories()
        
        if round_num is None:
            round_num = self.current_round
        
        # Save training history as JSON
        history_file = self.metrics_dir / "training_history.json"
        try:
            with open(history_file, 'w') as f:
                json.dump(self.history, f, indent=2, default=self._json_serializer)
            logger.info(f"✅ Training history saved: {history_file}")
        except Exception as e:
            logger.error(f"❌ Error saving history: {e}")
        
        # Save per-round results only if we have data
        try:
            if self.history["losses_distributed"]:
                latest_round = self.history["losses_distributed"][-1][0]
                round_data = {
                    "round": latest_round,
                    "loss": self.history["losses_distributed"][-1][1],
                    "metrics": {},
                    "timestamp": datetime.now().isoformat()
                }
                
                # Add latest metrics
                for metric_name, values in self.history["metrics_distributed"].items():
                    if values:
                        round_data["metrics"][metric_name] = values[-1][1]
                
                round_file = self.metrics_dir / f"round_{latest_round}_results.json"
                with open(round_file, 'w') as f:
                    json.dump(round_data, f, indent=2, default=self._json_serializer)
        except Exception as e:
            logger.error(f"❌ Error saving round results: {e}")
        
        # Save metrics as CSV if pandas is available
        try:
            import pandas as pd
            metrics_history = []
            for i, (round_idx, loss) in enumerate(self.history["losses_distributed"]):
                metric_entry = {"round": round_idx, "loss": loss}
                
                for metric_name, values in self.history["metrics_distributed"].items():
                    if i < len(values):
                        metric_entry[metric_name] = values[i][1]
                    else:
                        metric_entry[metric_name] = None
                
                metrics_history.append(metric_entry)
            
            if metrics_history:  # Only save if we have data
                csv_path = self.metrics_dir / "metrics_summary.csv"
                pd.DataFrame(metrics_history).to_csv(csv_path, index=False)
                logger.info(f"✅ Metrics CSV saved: {csv_path}")
        except ImportError:
            logger.warning("Pandas not available, skipping CSV export")
        except Exception as e:
            logger.error(f"❌ Error saving CSV: {e}")
        
        # Save best model info
        if self.best_model_path:
            try:
                best_model_info = {
                    "best_model_path": str(self.best_model_path),
                    "best_auroc": self.best_auroc,
                    "round_reached": round_num,
                    "timestamp": datetime.now().isoformat()
                }
                best_model_file = self.metrics_dir / "best_model_info.json"
                with open(best_model_file, 'w') as f:
                    json.dump(best_model_info, f, indent=2)
                logger.info(f"✅ Best model info saved: {best_model_file}")
            except Exception as e:
                logger.error(f"❌ Error saving best model info: {e}")
    
    def _json_serializer(self, obj):
        """JSON serializer for objects not serializable by default json code"""
        if isinstance(obj, (datetime,)):
            return obj.isoformat()
        raise TypeError(f"Type {type(obj)} not serializable")

# Global history manager
history_manager = OneClassHistoryManager()

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate metrics by weighted average"""
    if not metrics:
        return {}
    
    aggregated_metrics = {}
    all_metric_names = set()
    
    # Collect all metric names
    for num_examples, metric_dict in metrics:
        if metric_dict:
            all_metric_names.update(metric_dict.keys())
    
    # Aggregate each metric
    for metric_name in all_metric_names:
        values = []
        examples = []
        
        for num_examples, metric_dict in metrics:
            if metric_dict and metric_name in metric_dict and num_examples > 0:
                # Handle None values
                value = metric_dict[metric_name]
                if value is not None:
                    values.append(num_examples * value)
                    examples.append(num_examples)
        
        if values and examples:
            total_examples = sum(examples)
            aggregated_metrics[metric_name] = sum(values) / total_examples
    
    return aggregated_metrics

class HistoryFedAvg(FedAvg):
    """FedAvg strategy that tracks history and saves models"""
    
    def __init__(self, model, model_type, *args, **kwargs):
        # Remove model_type from kwargs before passing to parent
        kwargs_without_model_type = {k: v for k, v in kwargs.items() if k != 'model_type'}
        super().__init__(*args, **kwargs_without_model_type)
        self.model = model
        history_manager.model_type = model_type
    
    def aggregate_fit(self, server_round, results, failures):
        """Aggregate training results and save model - FIXED to update history"""
        # Call parent method
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        
        if aggregated_parameters is not None:
            # Save model checkpoint
            history_manager.save_model(
                self.model, 
                aggregated_parameters, 
                server_round,
                is_best=False
            )
        
        return aggregated_parameters, aggregated_metrics
    
    def aggregate_evaluate(self, server_round, results, failures):
        """Aggregate evaluation metrics and update history - FIXED"""
        # Call parent method
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(
            server_round, results, failures
        )
        
        # Update history - FIXED: Only update if we have valid results
        history_manager.set_round(server_round)
        
        if aggregated_loss is not None and aggregated_metrics is not None:
            # Convert loss to float and handle None/NaN
            try:
                loss_val = float(aggregated_loss) if aggregated_loss is not None else None
            except (ValueError, TypeError):
                loss_val = None
            
            history_manager.add_round_results(loss_val, aggregated_metrics)
            
            # Save best model - use any available metric for comparison
            if aggregated_metrics:
                # Use accuracy if available, otherwise use the first metric
                current_score = 0.0
                if "auroc" in aggregated_metrics and aggregated_metrics["auroc"] is not None:
                    current_score = aggregated_metrics["auroc"]
                elif "accuracy" in aggregated_metrics and aggregated_metrics["accuracy"] is not None:
                    current_score = aggregated_metrics["accuracy"]
                elif "f1" in aggregated_metrics and aggregated_metrics["f1"] is not None:
                    current_score = aggregated_metrics["f1"]
                elif len(aggregated_metrics) > 0:
                    # Get first non-None metric
                    for metric_value in aggregated_metrics.values():
                        if metric_value is not None:
                            current_score = metric_value
                            break
                
                if current_score > history_manager.best_auroc:
                    history_manager.best_auroc = current_score
                    current_parameters = get_model_parameters(self.model)
                    history_manager.save_model(
                        self.model,
                        current_parameters,
                        server_round,
                        is_best=True
                    )
                    logger.info(f"🎉 New best model saved with score: {current_score:.4f}")
        
        # Save history every round to ensure we don't lose data
        try:
            history_manager.save_results(server_round)
        except Exception as e:
            logger.error(f"Error saving results: {e}")
        
        return aggregated_loss, aggregated_metrics

class HistoryFedProx(FedProx):
    """FedProx strategy that tracks history and saves models"""
    
    def __init__(self, model, model_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        history_manager.model_type = model_type
    
    def aggregate_fit(self, server_round, results, failures):
        """Aggregate training results and save model"""
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        
        if aggregated_parameters is not None:
            history_manager.save_model(
                self.model, 
                aggregated_parameters, 
                server_round,
                is_best=False
            )
        
        return aggregated_parameters, aggregated_metrics
    
    def aggregate_evaluate(self, server_round, results, failures):
        """Aggregate evaluation metrics and update history"""
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(
            server_round, results, failures
        )
        
        # Update history
        history_manager.set_round(server_round)
        
        if aggregated_loss is not None and aggregated_metrics is not None:
            try:
                loss_val = float(aggregated_loss) if aggregated_loss is not None else None
            except (ValueError, TypeError):
                loss_val = None
            
            history_manager.add_round_results(loss_val, aggregated_metrics)
            
            # Save best model
            if aggregated_metrics:
                current_score = 0.0
                if "auroc" in aggregated_metrics and aggregated_metrics["auroc"] is not None:
                    current_score = aggregated_metrics["auroc"]
                elif "accuracy" in aggregated_metrics and aggregated_metrics["accuracy"] is not None:
                    current_score = aggregated_metrics["accuracy"]
                
                if current_score > history_manager.best_auroc:
                    history_manager.best_auroc = current_score
                    current_parameters = get_model_parameters(self.model)
                    history_manager.save_model(
                        self.model,
                        current_parameters,
                        server_round,
                        is_best=True
                    )
                    logger.info(f"🎉 New best model saved with score: {current_score:.4f}")
        
        # Save history every round
        try:
            history_manager.save_results(server_round)
        except Exception as e:
            logger.error(f"Error saving results: {e}")
        
        return aggregated_loss, aggregated_metrics

def server_fn(context: Context):
    # Use environment-based config
    run_config = get_run_config()
    
    num_rounds = run_config["num-server-rounds"]
    fraction_fit = run_config["fraction-fit"]
    model_type = run_config["model_type"]
    strategy_name = run_config["strategy"]
    fedprox_mu = run_config["fedprox_mu"]

    # Create experiment name
    experiment_name = f"{model_type}_{strategy_name}_{datetime.now().strftime('%Y%m%d_%H%M')}"
    
    # Initialize history manager
    history_manager.initialize_results_manager(experiment_name, model_type, strategy_name)

    # Create model based on configuration
    model = create_oneclass_server_model(model_type)
    
    # Store config in history
    history_manager.history["config"] = {
        "num_rounds": num_rounds,
        "fraction_fit": fraction_fit,
        "model_type": model_type,
        "strategy": strategy_name,
        "local_epochs": run_config["local-epochs"],
        "fedprox_mu": fedprox_mu,
        "experiment_name": experiment_name
    }

    parameters = ndarrays_to_parameters(get_model_parameters(model))

    # Use the correct strategy class
    common_kwargs = {
        "fraction_fit": fraction_fit,
        "fraction_evaluate": 1.0,
        "min_available_clients": run_config.get("min-available-clients", 10),
        "min_fit_clients": run_config.get("min-fit-clients", 2),
        "min_evaluate_clients": run_config.get("min-evaluate-clients", 2),
        "initial_parameters": parameters,
        "fit_metrics_aggregation_fn": weighted_average,
        "evaluate_metrics_aggregation_fn": weighted_average,
        "on_fit_config_fn": lambda round_idx: {
            "mode": "oneclass",
            "model_type": model_type,
            "local_epochs": run_config.get("local-epochs", 1)
        },
        "on_evaluate_config_fn": lambda round_idx: {
            "mode": "oneclass",
            "model_type": model_type
        }
    }
    
    if strategy_name == "fedprox":
        strategy = HistoryFedProx(
            model=model,
            model_type=model_type,
            proximal_mu=fedprox_mu,
            **common_kwargs
        )
    else:
        strategy = HistoryFedAvg(
            model=model,
            model_type=model_type,
            **common_kwargs
        )
    
    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)

def create_oneclass_server_model(model_type):
    """Create the appropriate one-class model for server"""
    if model_type == "gcn":
        return GCN(in_channels=128, hidden_channels=64, num_layers=2)
    elif model_type == "gae":
        return GAE(in_channels=128, hidden_channels=64, num_layers=2)
    elif model_type == "gcn_deepsvdd":
        return DeepSVDDGCN(in_channels=128, hidden_channels=64, num_layers=2)
    elif model_type == "gae_deepsvdd":
        return GAEDeepSVDD(in_channels=128, hidden_channels=64, num_layers=2)
    elif model_type == "gcn_personalized":
        return GCNWithPersonalizedHead(
            in_channels=128, hidden_channels=64, num_layers=2, num_clients=10
        )
    elif model_type == "gae_personalized":
        return GAEWithPersonalizedHead(
            in_channels=128, hidden_channels=64, num_layers=2, num_clients=10
        )
    else:
        return DeepSVDDGCN(in_channels=128, hidden_channels=64, num_layers=2)

app = ServerApp(server_fn=server_fn)

# Safe atexit registration with error handling
def safe_save_results():
    try:
        history_manager.save_results()
        print("✅ Results saved successfully on exit")
    except Exception as e:
        print(f"❌ Error saving results on exit: {e}")

import atexit
atexit.register(safe_save_results)
