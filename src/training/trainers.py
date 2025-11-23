# src/training/trainers.py (UPDATED - Add training for all model types)
import torch
import torch.nn as nn
from .losses import DeepSVDDLoss, GAEReconstructionLoss, HybridLoss, FedProxRegularizer, CompactnessLoss

def initialize_model_weights(model):
    """Proper weight initialization for stable training"""
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)

def train_oneclass_deepsvdd(model, train_loader, epochs, device, optimizer=None, 
                           nu=0.1, lambda_compact=0.01, global_params=None, fedprox_mu=0.01):
    """Train one-class model with DeepSVDD loss (for GCN-based models)"""
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # Initialize model weights
    initialize_model_weights(model)
    
    svdd_loss = DeepSVDDLoss(nu=nu, feature_dim=model.embed_dim)
    compact_loss = CompactnessLoss(lambda_var=lambda_compact)
    prox_regularizer = FedProxRegularizer(mu=fedprox_mu)
    
    # Initialize center with first batch
    model.eval()
    with torch.no_grad():
        for data in train_loader:
            data = data.to(device)
            embeddings, _ = model(data.x, data.edge_index)
            svdd_loss.update_center(embeddings, alpha=1.0)  # Full update for first batch
            break
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        batch_count = 0
        
        for data in train_loader:
            try:
                data = data.to(device)
                optimizer.zero_grad()
                
                embeddings, _ = model(data.x, data.edge_index)
                
                loss_svdd = svdd_loss(embeddings)
                loss_compact = compact_loss(embeddings)
                
                # Add FedProx regularization if global parameters provided
                if global_params is not None:
                    local_params = list(model.named_parameters())
                    loss_prox = prox_regularizer(local_params, global_params)
                else:
                    loss_prox = 0.0
                
                loss = loss_svdd + loss_compact + loss_prox
                
                # Check for reasonable loss values
                if loss.item() > 1000:  # Unusually high loss
                    print(f"Warning: High loss {loss.item():.2f}, skipping batch")
                    continue
                
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                # Update center
                svdd_loss.update_center(embeddings, alpha=0.1)
                
                total_loss += loss.item()
                batch_count += 1
                
            except Exception as e:
                print(f"Error in training batch: {e}")
                continue
        
        if batch_count > 0:
            avg_loss = total_loss / batch_count
            print(f"Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.4f}")
        else:
            print(f"Epoch {epoch+1}/{epochs}: No batches processed")
    
    return total_loss / max(batch_count, 1), svdd_loss.center

def train_oneclass_gae(model, train_loader, epochs, device, optimizer=None, 
                      lambda_rec=0.1, global_params=None, fedprox_mu=0.01):
    """Train Graph Autoencoder for one-class learning (reconstruction-based)"""
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # Initialize model weights
    initialize_model_weights(model)
    
    rec_loss = GAEReconstructionLoss(lambda_rec=1.0, feature_dim=model.in_channels)
    prox_regularizer = FedProxRegularizer(mu=fedprox_mu)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        batch_count = 0
        
        for data in train_loader:
            try:
                data = data.to(device)
                optimizer.zero_grad()
                
                reconstructed, _ = model(data.x, data.edge_index)
                
                # Ensure dimensions match
                if data.x.shape != reconstructed.shape:
                    print(f"Dimension mismatch: input {data.x.shape}, output {reconstructed.shape}")
                    continue
                
                loss_rec = rec_loss(data.x, reconstructed)
                
                # Add FedProx regularization if global parameters provided
                if global_params is not None:
                    local_params = list(model.named_parameters())
                    loss_prox = prox_regularizer(local_params, global_params)
                else:
                    loss_prox = 0.0
                
                loss = lambda_rec * loss_rec + loss_prox
                
                # Check for reasonable loss
                if loss.item() > 1000:
                    print(f"Warning: High loss {loss.item():.2f}, skipping batch")
                    continue
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                batch_count += 1
                
            except Exception as e:
                print(f"Error in training batch: {e}")
                continue
        
        if batch_count > 0:
            avg_loss = total_loss / batch_count
            print(f"Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.4f}")
        else:
            print(f"Epoch {epoch+1}/{epochs}: No batches processed")
    
    return total_loss / max(batch_count, 1)

def train_oneclass_hybrid(model, train_loader, epochs, device, optimizer=None,
                         nu=0.1, lambda_rec=1.0, lambda_svdd=1.0, global_params=None, fedprox_mu=0.01):
    """Train hybrid model with both reconstruction and DeepSVDD losses"""
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # Initialize model weights
    initialize_model_weights(model)
    
    hybrid_loss = HybridLoss(nu=nu, lambda_rec=lambda_rec, lambda_svdd=lambda_svdd, feature_dim=model.in_channels)
    prox_regularizer = FedProxRegularizer(mu=fedprox_mu)
    
    # Initialize center with first batch
    model.eval()
    with torch.no_grad():
        for data in train_loader:
            data = data.to(device)
            reconstructed, embeddings = model(data.x, data.edge_index)
            hybrid_loss.svdd_loss.update_center(embeddings, alpha=1.0)
            break
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        batch_count = 0
        
        for data in train_loader:
            try:
                data = data.to(device)
                optimizer.zero_grad()
                
                reconstructed, embeddings = model(data.x, data.edge_index)
                loss_hybrid = hybrid_loss(embeddings, data.x, reconstructed)
                
                # Add FedProx regularization if global parameters provided
                if global_params is not None:
                    local_params = list(model.named_parameters())
                    loss_prox = prox_regularizer(local_params, global_params)
                else:
                    loss_prox = 0.0
                
                loss = loss_hybrid + loss_prox
                
                # Check for reasonable loss
                if loss.item() > 1000:
                    print(f"Warning: High loss {loss.item():.2f}, skipping batch")
                    continue
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                # Update center
                hybrid_loss.svdd_loss.update_center(embeddings, alpha=0.1)
                
                total_loss += loss.item()
                batch_count += 1
                
            except Exception as e:
                print(f"Error in training batch: {e}")
                continue
        
        if batch_count > 0:
            avg_loss = total_loss / batch_count
            print(f"Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.4f}")
        else:
            print(f"Epoch {epoch+1}/{epochs}: No batches processed")
    
    return total_loss / max(batch_count, 1), hybrid_loss.svdd_loss.center
