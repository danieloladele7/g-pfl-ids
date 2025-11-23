# src/training/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepSVDDLoss(nn.Module):
    """DeepSVDD loss for one-class classification with proper scaling"""
    
    def __init__(self, nu=0.1, center=None, feature_dim=64):
        super().__init__()
        self.nu = nu
        self.feature_dim = feature_dim
        
        # Initialize center properly
        if center is None:
            self.center = nn.Parameter(torch.zeros(feature_dim), requires_grad=False)
        else:
            self.center = nn.Parameter(center, requires_grad=False)
        
    def forward(self, embeddings, reduction='mean'):
        # Normalize embeddings first
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Calculate distances with proper scaling
        distances = torch.sum((embeddings - self.center) ** 2, dim=1)
        
        # Scale distances by feature dimension
        scaled_distances = distances / self.feature_dim
        
        # Hinge loss with proper scaling
        hinge = torch.clamp(scaled_distances - 1.0, min=0)
        loss = torch.mean(scaled_distances) + self.nu * torch.mean(hinge)
        
        return loss

    def update_center(self, embeddings, alpha=0.1):
        """Update center with exponential moving average"""
        with torch.no_grad():
            # Normalize embeddings before updating center
            normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
            batch_center = normalized_embeddings.mean(dim=0)
            
            if torch.norm(self.center) == 0:  # First update
                self.center.data.copy_(batch_center)
            else:
                self.center.data = alpha * batch_center + (1 - alpha) * self.center

class GAEReconstructionLoss(nn.Module):
    """Reconstruction loss with proper scaling"""
    
    def __init__(self, lambda_rec=1.0, feature_dim=128):
        super().__init__()
        self.lambda_rec = lambda_rec
        self.feature_dim = feature_dim
        self.mse_loss = nn.MSELoss()
        
    def forward(self, original_features, reconstructed_features):
        # Scale reconstruction loss by feature dimension
        base_loss = self.mse_loss(original_features, reconstructed_features)
        return self.lambda_rec * base_loss * self.feature_dim

class HybridLoss(nn.Module):
    """Combined loss with balanced scaling"""
    
    def __init__(self, nu=0.1, lambda_rec=0.1, lambda_svdd=1.0, feature_dim=128):
        super().__init__()
        self.svdd_loss = DeepSVDDLoss(nu=nu, feature_dim=64)  # embedding dim
        self.rec_loss = GAEReconstructionLoss(lambda_rec=1.0, feature_dim=feature_dim)
        self.lambda_rec = lambda_rec
        self.lambda_svdd = lambda_svdd
        
    def forward(self, embeddings, original_features, reconstructed_features):
        rec_loss = self.rec_loss(original_features, reconstructed_features)
        svdd_loss = self.svdd_loss(embeddings)
        
        # Balance the losses - reconstruction should have smaller weight
        total_loss = self.lambda_rec * rec_loss + self.lambda_svdd * svdd_loss
        return total_loss

class FedProxRegularizer(nn.Module):
    """FedProx regularization term"""
    
    def __init__(self, mu=0.01):
        super().__init__()
        self.mu = mu
        
    def forward(self, local_params, global_params):
        if global_params is None:
            return 0.0
            
        prox_term = 0.0
        for (name1, param1), (name2, param2) in zip(local_params, global_params):
            if name1 == name2:  # Ensure parameter alignment
                prox_term += torch.norm(param1 - param2) ** 2
        return self.mu * prox_term

class CompactnessLoss(nn.Module):
    """Encourage compact embeddings for one-class learning"""
    
    def __init__(self, lambda_var=0.1):
        super().__init__()
        self.lambda_var = lambda_var
        
    def forward(self, embeddings):
        """Minimize variance of embeddings"""
        return self.lambda_var * torch.var(embeddings, dim=0).mean()
