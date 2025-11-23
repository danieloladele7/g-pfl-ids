# src/models/base.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class BaseGCN(nn.Module):
    """Base GCN class with improved stability"""
    
    def __init__(self, in_channels, hidden_channels=64, num_layers=2, 
                 dropout=0.2, embed_dim=64, num_classes=2):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.dropout = dropout
        
        # Encoder layers with batch normalization
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # Projection head for embeddings with proper initialization
        self.projector = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, embed_dim)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
                
    def encode(self, x, edge_index, batch=None):
        """Extract node embeddings with normalization"""
        for conv, bn in zip(self.convs, self.batch_norms):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Project to embedding space with normalization
        embeddings = self.projector(x)
        embeddings = F.normalize(embeddings, p=2, dim=1)  # L2 normalize
        
        # Global pooling if batch is provided
        if batch is not None:
            graph_embedding = global_mean_pool(embeddings, batch)
            graph_embedding = F.normalize(graph_embedding, p=2, dim=1)
            return embeddings, graph_embedding
        return embeddings, None
