# src/models/gcn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseGCN

class GCN(BaseGCN):
    """Standard GCN for binary classification - CAN be used for one-class with proper loss"""
    
    def __init__(self, in_channels, hidden_channels=64, num_layers=2, 
                 dropout=0.2, embed_dim=64, num_classes=2):
        super().__init__(in_channels, hidden_channels, num_layers, dropout, embed_dim, num_classes)
        self.classifier = nn.Linear(self.embed_dim, num_classes)
    
    def forward(self, x, edge_index, batch=None):
        node_embeddings, graph_embedding = self.encode(x, edge_index, batch)
        
        if graph_embedding is not None:
            logits = self.classifier(graph_embedding)
            return node_embeddings, logits
        else:
            logits = self.classifier(node_embeddings)
            return node_embeddings, logits

class DeepSVDDGCN(BaseGCN):
    """GCN with DeepSVDD for one-class classification"""
    
    def __init__(self, in_channels, hidden_channels=64, num_layers=2, 
                 dropout=0.2, embed_dim=64, num_classes=2):
        super().__init__(in_channels, hidden_channels, num_layers, dropout, embed_dim, num_classes)
        # NOTE: DeepSVDD doesn't need a classifier - uses distance from center
        
    def forward(self, x, edge_index, batch=None):
        node_embeddings, graph_embedding = self.encode(x, edge_index, batch)
        return node_embeddings, node_embeddings

class GCNWithPersonalizedHead(BaseGCN): # Not used
    """GCN with personalized heads for each client"""
    
    def __init__(self, in_channels, hidden_channels=64, num_layers=2, 
                 dropout=0.2, embed_dim=64, num_classes=2, num_clients=10, 
                 personalization_dim=32):
        super().__init__(in_channels, hidden_channels, num_layers, dropout, embed_dim, num_classes)
        
        self.personalization_dim = personalization_dim
        self.num_clients = num_clients
        
        # Global classifier (shared across all clients)
        self.global_classifier = nn.Linear(embed_dim, num_classes)
        
        # Personalized classification heads for each client
        self.personalized_heads = nn.ModuleList([
            nn.Linear(embed_dim, num_classes) for _ in range(num_clients)
        ])
        
        # Optional: Personalized projection layers for each client
        self.personalized_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, personalization_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for _ in range(num_clients)
        ])
    
    def forward_backbone(self, x, edge_index, batch=None):
        """Extract features using shared backbone"""
        return self.encode(x, edge_index, batch)
    
    def forward_global(self, x, edge_index, batch=None):
        """Use global classifier"""
        node_embeddings, graph_embedding = self.encode(x, edge_index, batch)
        
        if graph_embedding is not None:
            logits = self.global_classifier(graph_embedding)
        else:
            logits = self.global_classifier(node_embeddings)
        return node_embeddings, logits
    
    def forward_personalized(self, x, edge_index, client_id, batch=None, use_projection=False):
        """Use personalized head for specific client"""
        node_embeddings, graph_embedding = self.encode(x, edge_index, batch)
        
        if graph_embedding is not None:
            embeddings = graph_embedding
        else:
            embeddings = node_embeddings
        
        if use_projection and hasattr(self, 'personalized_projections'):
            # Apply personalized projection then classification
            projected = self.personalized_projections[client_id](embeddings)
            logits = self.personalized_heads[client_id](projected)
        else:
            # Direct personalized classification
            logits = self.personalized_heads[client_id](embeddings)
            
        return node_embeddings, logits
    
    def forward(self, x, edge_index, client_id=None, batch=None, use_personalized=False, use_projection=False):
        """
        Forward pass with optional personalization.
        
        Args:
            x: Input features
            edge_index: Graph structure
            client_id: Client identifier (required for personalized forward)
            batch: Batch indices for graph-level tasks
            use_personalized: Whether to use personalized components
            use_projection: Whether to use personalized projection layers
        """
        if use_personalized and client_id is not None:
            if client_id < 0 or client_id >= self.num_clients:
                raise ValueError(f"client_id must be between 0 and {self.num_clients-1}")
            return self.forward_personalized(x, edge_index, client_id, batch, use_projection)
        else:
            return self.forward_global(x, edge_index, batch)
