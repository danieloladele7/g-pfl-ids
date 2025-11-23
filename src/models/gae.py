# src/models/gae.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from .base import BaseGCN

class GAE(BaseGCN):
    """Pure Graph Autoencoder for one-class learning (reconstruction only)"""
    
    def __init__(self, in_channels, hidden_channels=64, num_layers=2, 
                 dropout=0.2, embed_dim=64, num_classes=2):
        super().__init__(in_channels, hidden_channels, num_layers, dropout, embed_dim, num_classes)
        
        # Decoder for reconstruction
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, in_channels)
        )
        
        # Classifier for compatibility (not used in one-class)
        self.classifier = nn.Linear(embed_dim, num_classes)
    
    def decode(self, z):
        """Decode embeddings back to feature space"""
        return self.decoder(z)
    
    def reconstruct_loss(self, x, z):
        """Reconstruction loss"""
        recon_x = self.decode(z)
        return F.mse_loss(recon_x, x)
    
    def forward(self, x, edge_index, batch=None):
        node_embeddings, graph_embedding = self.encode(x, edge_index, batch)
        reconstructed = self.decode(node_embeddings)
        
        if graph_embedding is not None:
            logits = self.classifier(graph_embedding)
            return reconstructed, logits
        else:
            logits = self.classifier(node_embeddings)
            return reconstructed, logits

class GAEDeepSVDD(GAE):
    """GAE with DeepSVDD for one-class classification"""
    
    def __init__(self, in_channels, hidden_channels=64, num_layers=2, 
                 dropout=0.2, embed_dim=64, num_classes=2):
        super().__init__(in_channels, hidden_channels, num_layers, dropout, embed_dim, num_classes)
        # Remove classifier for pure one-class
        del self.classifier
    
    def forward(self, x, edge_index, batch=None):
        node_embeddings, graph_embedding = self.encode(x, edge_index, batch)
        reconstructed = self.decode(node_embeddings)
        return reconstructed, node_embeddings

class GAEWithPersonalizedHead(GAE): # tested but not the best result
    """GAE with personalized heads for each client"""
    
    def __init__(self, in_channels, hidden_channels=64, num_layers=2, 
                 dropout=0.2, embed_dim=64, num_classes=2, num_clients=10, 
                 personalization_dim=32):
        super().__init__(in_channels, hidden_channels, num_layers, dropout, embed_dim, num_classes)
        
        self.personalization_dim = personalization_dim
        self.num_clients = num_clients
        
        # Global decoder (shared across all clients)
        self.global_decoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, in_channels)
        )
        
        # Personalized decoders for each client
        self.personalized_decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, in_channels)
            ) for _ in range(num_clients)
        ])
        
        # Personalized projection layers
        self.personalized_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, personalization_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for _ in range(num_clients)
        ])
    
    def decode_global(self, z):
        """Use global decoder"""
        return self.global_decoder(z)
    
    def decode_personalized(self, z, client_id, use_projection=False):
        """Use personalized decoder for specific client"""
        if use_projection and hasattr(self, 'personalized_projections'):
            projected = self.personalized_projections[client_id](z)
            return self.personalized_decoders[client_id](projected)
        else:
            return self.personalized_decoders[client_id](z)
    
    def forward(self, x, edge_index, client_id=None, batch=None, use_personalized=False, use_projection=False):
        """
        Forward pass with optional personalization
        """
        node_embeddings, graph_embedding = self.encode(x, edge_index, batch)
        
        if use_personalized and client_id is not None:
            if client_id < 0 or client_id >= self.num_clients:
                raise ValueError(f"client_id must be between 0 and {self.num_clients-1}")
            reconstructed = self.decode_personalized(node_embeddings, client_id, use_projection)
        else:
            reconstructed = self.decode_global(node_embeddings)
        
        return reconstructed, node_embeddings
