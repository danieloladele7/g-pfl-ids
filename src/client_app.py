# src/client_app.py
"""fl-gcn-ids: A Flower / PyTorch app for One-Class Learning"""

import os
import torch
from pathlib import Path
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

from src.models import GCN, GAE, DeepSVDDGCN, GAEDeepSVDD, GCNWithPersonalizedHead, GAEWithPersonalizedHead
from src.utils import load_client_data, create_data_loaders, get_model_parameters, set_model_parameters, preprocess_graph_data
from src.training import train_oneclass_deepsvdd, train_oneclass_gae, train_oneclass_hybrid
from src.utils.oneclass_metrics import OneClassMetrics

# Try to import config, provide fallback if missing
try:
    from src.config import get_run_config
except ImportError:
    print("Warning: src.config not found, using fallback configuration")
    def get_run_config():
        return {
            "local-epochs": int(os.getenv("FLWR_LOCAL_EPOCHS", "30")),
            "model_type": os.getenv("FLWR_MODEL_TYPE", "gcn_deepsvdd"),  # Default to gcn_deepsvdd
            "strategy": os.getenv("FLWR_STRATEGY", "fedavg"),
            "fedprox_mu": float(os.getenv("FLWR_FEDPROX_MU", "0.01")),
        }

import logging
logger = logging.getLogger(__name__)

class OneClassFlowerClient(NumPyClient):
    def __init__(self, net, train_loader, test_loader, config):
        self.net = net
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
        self.center = None
        self.global_params = None

    def get_parameters(self, config):
        return get_model_parameters(self.net)

    def set_parameters(self, parameters):
        set_model_parameters(self.net, parameters)
        # Store global parameters for FedProx
        self.global_params = list(self.net.named_parameters())

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        
        model_type = self.config.get('model_type', 'gcn_deepsvdd')
        strategy = self.config.get('strategy', 'fedavg')
        fedprox_mu = self.config.get('fedprox_mu', 0.01) if strategy == 'fedprox' else 0.0
        
        # Handle all model types - FIXED
        if model_type in ['gcn_deepsvdd', 'deepsvdd']:
            # GCN with DeepSVDD
            result = train_oneclass_deepsvdd(
                self.net, self.train_loader, self.config['local_epochs'], 
                self.device, global_params=self.global_params, fedprox_mu=fedprox_mu
            )
            if isinstance(result, tuple):
                train_loss, self.center = result
            else:
                train_loss = result
                
        elif model_type in ['gae', 'gae_deepsvdd']:
            # GAE or GAE with DeepSVDD - use reconstruction training
            train_loss = train_oneclass_gae(
                self.net, self.train_loader, self.config['local_epochs'],
                self.device, global_params=self.global_params, fedprox_mu=fedprox_mu
            )
            
        elif model_type == 'hybrid':
            # Hybrid training (both reconstruction and DeepSVDD)
            result = train_oneclass_hybrid(
                self.net, self.train_loader, self.config['local_epochs'],
                self.device, global_params=self.global_params, fedprox_mu=fedprox_mu
            )
            if isinstance(result, tuple):
                train_loss, self.center = result
            else:
                train_loss = result
                
        elif model_type == 'gcn':
            # Pure GCN - use DeepSVDD training (distance-based)
            result = train_oneclass_deepsvdd(
                self.net, self.train_loader, self.config['local_epochs'], 
                self.device, global_params=self.global_params, fedprox_mu=fedprox_mu
            )
            if isinstance(result, tuple):
                train_loss, self.center = result
            else:
                train_loss = result
                
        elif model_type in ['gcn_personalized', 'gae_personalized']:
            # Personalized models - use standard training for now
            # Personalization happens after FL training
            if 'gae' in model_type:
                train_loss = train_oneclass_gae(
                    self.net, self.train_loader, self.config['local_epochs'],
                    self.device, global_params=self.global_params, fedprox_mu=fedprox_mu
                )
            else:
                result = train_oneclass_deepsvdd(
                    self.net, self.train_loader, self.config['local_epochs'], 
                    self.device, global_params=self.global_params, fedprox_mu=fedprox_mu
                )
                if isinstance(result, tuple):
                    train_loss, self.center = result
                else:
                    train_loss = result
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        return self.get_parameters({}), len(self.train_loader.dataset), {"train_loss": train_loss}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        
        model_type = self.config.get('model_type', 'gcn_deepsvdd')
        
        # Map model types to metrics computation types
        metrics_model_type = self._map_to_metrics_type(model_type)
        
        loss, num_examples, metrics = OneClassMetrics.evaluate_model(
            self.net, self.test_loader, self.device, metrics_model_type
        )
        
        return float(loss), int(num_examples), metrics
    
    def _map_to_metrics_type(self, model_type):
        """Map model type to metrics computation type"""
        if model_type in ['gcn', 'gcn_deepsvdd', 'gcn_personalized', 'deepsvdd']:
            return 'deepsvdd'
        elif model_type in ['gae', 'gae_personalized']:
            return 'gae'
        elif model_type in ['gae_deepsvdd', 'hybrid']:
            return 'hybrid'
        else:
            return 'deepsvdd'

def client_fn(context: Context):
    try:
        client_id = context.node_config["partition-id"]
        
        # Use environment-based config
        run_config = get_run_config()
        model_type = run_config["model_type"]
        strategy = run_config["strategy"]
        local_epochs = run_config["local-epochs"]
        fedprox_mu = run_config["fedprox_mu"]
        
        # Load one-class data only
        data = load_client_data(Path("data/graph_data/oneclass"), str(client_id))
        
        if not data:
            logger.info(f"Client {client_id} has no data, returning empty client")
            return OneClassFlowerClient(
                DeepSVDDGCN(in_channels=128, hidden_channels=64, num_layers=2),
                [], [], {}
            ).to_client()
        
        # Load and preprocess data
        if isinstance(data, dict) and 'graph' in data:
            graph = data['graph']
        else:
            graph = data
        
        graph = preprocess_graph_data(graph, client_id)
        train_loader, test_loader = create_data_loaders(graph)
        
        logger.info(f"Client {client_id} loaded with {graph.num_nodes} nodes")
        
        # Model selection based on configuration
        net = create_oneclass_model(model_type)
        
        config = {
            "local_epochs": local_epochs,
            "model_type": model_type,
            "strategy": strategy,
            "fedprox_mu": fedprox_mu
        }
        
        return OneClassFlowerClient(net, train_loader, test_loader, config).to_client()
    
    except Exception as e:
        logger.error(f"Error in client {client_id}: {e}")
        # Return a minimal client to avoid crashing
        return OneClassFlowerClient(
            DeepSVDDGCN(in_channels=128, hidden_channels=64, num_layers=2),
            [], [], {}
        ).to_client()

def create_oneclass_model(model_type):
    """Create the appropriate one-class model"""
    if model_type == "gcn":
        return GCN(in_channels=128, hidden_channels=64, num_layers=2)
    elif model_type == "gae":
        return GAE(in_channels=128, hidden_channels=64, num_layers=2)
    elif model_type in ["gcn_deepsvdd", "deepsvdd"]:  # Support old name for compatibility
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
        # Default to GCN with DeepSVDD
        return DeepSVDDGCN(in_channels=128, hidden_channels=64, num_layers=2)

app = ClientApp(client_fn=client_fn)
