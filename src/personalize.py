# personalize.py
import torch
from pathlib import Path
from src.models import DeepSVDDGCN, GAEDeepSVDD, GCN, GAE
from src.utils import load_client_data, create_data_loaders, normalize_graph_features
from src.training.losses import DeepSVDDLoss, GAEReconstructionLoss
import json, os

def filter_state_dict(state_dict, model):
    """Filter state dict to only include keys that exist in the model"""
    model_state_dict = model.state_dict()
    filtered_state_dict = {}
    
    for key, value in state_dict.items():
        if key in model_state_dict:
            # Check if dimensions match
            if value.shape == model_state_dict[key].shape:
                filtered_state_dict[key] = value
            else:
                print(f"Warning: Dimension mismatch for {key}: {value.shape} vs {model_state_dict[key].shape}")
        else:
            print(f"Warning: Skipping unexpected key in state dict: {key}")
    
    return filtered_state_dict

def create_model_by_type(model_type, in_channels=128, hidden_channels=64, num_layers=2):
    """Create model based on type with proper initialization"""
    if model_type in ['gcn', 'gcn_deepsvdd', 'deepsvdd']:
        return DeepSVDDGCN(in_channels=in_channels, hidden_channels=hidden_channels, num_layers=num_layers)
    elif model_type in ['gae', 'gae_deepsvdd']:
        return GAEDeepSVDD(in_channels=in_channels, hidden_channels=hidden_channels, num_layers=num_layers)
    elif model_type == 'gcn_pure':
        return GCN(in_channels=in_channels, hidden_channels=hidden_channels, num_layers=num_layers)
    elif model_type == 'gae_pure':
        return GAE(in_channels=in_channels, hidden_channels=hidden_channels, num_layers=num_layers)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def personalize_oneclass_model(global_model, client_data, personalization_epochs=5, nu=0.1, fine_tune_encoder=False, model_type='gcn_deepsvdd'):
    """Personalize one-class model for a specific client - FIXED DIMENSION ISSUE"""
    # Create personalized model with same architecture
    personalized_model = create_model_by_type(model_type)
    
    # Load state dict with filtering
    filtered_state_dict = filter_state_dict(global_model.state_dict(), personalized_model)
    personalized_model.load_state_dict(filtered_state_dict, strict=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    personalized_model.to(device)
    
    # Determine embedding dimension from the model
    if hasattr(personalized_model, 'embed_dim'):
        embed_dim = personalized_model.embed_dim
    else:
        # Try to infer from model structure
        embed_dim = 64  # Default based on your architecture
    
    print(f"Personalizing {model_type} model with embedding dimension: {embed_dim}")
    
    # Freeze the encoder parameters initially
    for name, param in personalized_model.named_parameters():
        if any(layer in name for layer in ['convs', 'projector']):
            param.requires_grad = False
    
    # Optionally, fine-tune part of the encoder layers
    if fine_tune_encoder:
        # Fine-tune the last encoder layer
        for name, param in personalized_model.named_parameters():
            if any(layer in name for layer in ['convs.1', 'projector.4']):  # Last layers
                param.requires_grad = True
                print(f"Fine-tuning layer: {name}")
    
    # Always fine-tune the head layers if they exist
    for name, param in personalized_model.named_parameters():
        if any(head in name for head in ['head', 'classifier', 'decoder']):
            param.requires_grad = True
            print(f"Fine-tuning head layer: {name}")
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in personalized_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in personalized_model.parameters())
    print(f"Trainable parameters: {trainable_params}/{total_params} ({trainable_params/total_params*100:.2f}%)")

    # Compute client-specific center with correct dimension
    personalized_model.eval()
    client_embeddings = []
    with torch.no_grad():
        for data in client_data:
            data = data.to(device)
            
            if model_type in ['gcn', 'gcn_deepsvdd', 'deepsvdd', 'gcn_pure']:
                embeddings, _ = personalized_model(data.x, data.edge_index)
            elif model_type in ['gae', 'gae_deepsvdd', 'gae_pure']:
                if model_type == 'gae_pure':
                    _, embeddings = personalized_model(data.x, data.edge_index)
                else:
                    reconstructed, embeddings = personalized_model(data.x, data.edge_index)
            else:
                embeddings, _ = personalized_model(data.x, data.edge_index)
            
            client_embeddings.append(embeddings)
    
    if client_embeddings:
        all_embeddings = torch.cat(client_embeddings)
        client_center = all_embeddings.mean(dim=0)
        # Ensure center has correct dimension
        if client_center.shape[0] != embed_dim:
            print(f"Adjusting center dimension from {client_center.shape[0]} to {embed_dim}")
            if client_center.shape[0] < embed_dim:
                # Pad with zeros
                padding = torch.zeros(embed_dim - client_center.shape[0]).to(device)
                client_center = torch.cat([client_center, padding])
            else:
                # Truncate
                client_center = client_center[:embed_dim]
    else:
        client_center = torch.zeros(embed_dim).to(device)
    
    print(f"Client center computed with dimension: {client_center.shape}")
    
    # Fine-tune with client data
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, personalized_model.parameters()), 
        lr=0.001, weight_decay=1e-5
    )
    
    # Select appropriate loss function
    if model_type in ['gcn', 'gcn_deepsvdd', 'deepsvdd']:
        criterion = DeepSVDDLoss(nu=nu, feature_dim=embed_dim)
        criterion.center = client_center  # Set the computed center
        print("Using DeepSVDD loss")
    elif model_type in ['gae', 'gae_deepsvdd']:
        criterion = GAEReconstructionLoss(lambda_rec=1.0, feature_dim=128)
        print("Using GAE reconstruction loss")
    elif model_type == 'gcn_pure':
        # For pure GCN, we'll use a simple MSE loss on embeddings
        def criterion(embeddings):
            return torch.mean((embeddings - client_center) ** 2)
        print("Using MSE loss for GCN")
    elif model_type == 'gae_pure':
        criterion = GAEReconstructionLoss(lambda_rec=1.0, feature_dim=128)
        print("Using GAE reconstruction loss")
    else:
        criterion = DeepSVDDLoss(nu=nu, feature_dim=embed_dim)
        print("Using default DeepSVDD loss")
    
    personalized_model.train()
    for epoch in range(personalization_epochs):
        total_loss = 0
        batch_count = 0
        
        for data in client_data:
            data = data.to(device)
            optimizer.zero_grad()
            
            if model_type in ['gcn', 'gcn_deepsvdd', 'deepsvdd', 'gcn_pure']:
                embeddings, _ = personalized_model(data.x, data.edge_index)
                if model_type == 'gcn_pure':
                    loss = criterion(embeddings)
                else:
                    loss = criterion(embeddings)
            elif model_type in ['gae', 'gae_deepsvdd', 'gae_pure']:
                if model_type == 'gae_pure':
                    reconstructed, embeddings = personalized_model(data.x, data.edge_index)
                    loss = criterion(data.x, reconstructed)
                else:
                    reconstructed, embeddings = personalized_model(data.x, data.edge_index)
                    loss = criterion(data.x, reconstructed)
            else:
                embeddings, _ = personalized_model(data.x, data.edge_index)
                loss = criterion(embeddings)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
        
        if batch_count > 0:
            avg_loss = total_loss / batch_count
            print(f"Personalization Epoch {epoch+1}/{personalization_epochs}: Loss = {avg_loss:.6f}")
    
    return personalized_model, client_center

def personalize_clients_global(global_model_path, data_dir, output_dir, model_type='gcn_deepsvdd', personalization_epochs=5, fine_tune_encoder=False):
    """Personalize global model for all clients - FIXED"""
    print(f"Loading global model from: {global_model_path}")
    print(f"Model type: {model_type}")
    
    # Load global model with correct type
    global_model = create_model_by_type(model_type)
    
    # Load model state with proper error handling
    try:
        checkpoint = torch.load(global_model_path, map_location='cpu')
        print(f"Checkpoint keys: {list(checkpoint.keys())}")
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Filter state dict to remove incompatible keys
        filtered_state_dict = filter_state_dict(state_dict, global_model)
        
        # Load with strict=False to ignore missing keys
        missing_keys, unexpected_keys = global_model.load_state_dict(filtered_state_dict, strict=False)
        
        if missing_keys:
            print(f"Missing keys in model: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys in checkpoint: {unexpected_keys}")
            
        print("✅ Global model loaded successfully")
        
    except Exception as e:
        print(f"❌ Error loading global model: {e}")
        return {}
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    successful_clients = 0
    
    for client_id in range(10):  # Assuming 10 clients
        print(f"\n=== Processing Client {client_id} ===")
        
        data = load_client_data(Path(data_dir), str(client_id))
        if not data:
            print(f"Client {client_id}: No data available")
            continue
            
        # Extract graph data
        if isinstance(data, dict) and 'graph' in data:
            graph = data['graph']
        else:
            graph = data
            
        # Ensure the graph has the correct feature dimensions
        graph = normalize_graph_features(graph, expected_features=128)
        
        # Create data loaders
        try:
            train_loader, test_loader = create_data_loaders(graph)
            print(f"Client {client_id}: Loaded {len(train_loader)} training batches")
        except Exception as e:
            print(f"Client {client_id}: Error creating data loaders: {e}")
            continue
        
        print(f"Personalizing model for client {client_id}...")
        
        try:
            # Personalize model
            personalized_model, client_center = personalize_oneclass_model(
                global_model, train_loader, personalization_epochs, 
                fine_tune_encoder=fine_tune_encoder, model_type=model_type
            )
            
            # Save personalized model
            model_path = output_dir / f"personalized_model_{client_id}.pt"
            torch.save({
                'model_state_dict': personalized_model.state_dict(),
                'client_center': client_center,
                'client_id': client_id,
                'model_type': model_type
            }, model_path)
            
            # Save client center
            center_path = output_dir / f"client_center_{client_id}.pt"
            torch.save(client_center, center_path)
            
            print(f"✅ Personalized model saved: {model_path}")
            print(f"✅ Client center saved: {center_path}")
            
            results[client_id] = {
                "model_path": str(model_path),
                "center_path": str(center_path),
                "status": "success"
            }
            successful_clients += 1
            
        except Exception as e:
            print(f"❌ Error personalizing client {client_id}: {e}")
            results[client_id] = {
                "status": "failed",
                "error": str(e)
            }
    
    # Save results
    results_path = output_dir / "personalization_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n=== Personalization Summary ===")
    print(f"✅ Successful: {successful_clients}/10 clients")
    print(f"✅ Results saved: {results_path}")
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Personalize global model for clients")
    parser.add_argument("--global_model", type=str, required=True, help="Path to global model checkpoint")
    parser.add_argument("--data_dir", type=str, default="data/graph_data/oneclass")
    parser.add_argument("--output_dir", type=str, default="personalized_models")
    parser.add_argument("--model_type", type=str, default="gcn_deepsvdd", 
                       choices=["gcn", "gae", "gcn_deepsvdd", "gae_deepsvdd", "deepsvdd", "gcn_pure", "gae_pure"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--fine_tune_encoder", action="store_true", help="Fine-tune part of the encoder layers")

    args = parser.parse_args()
    
    results = personalize_clients_global(
        args.global_model, 
        args.data_dir, 
        args.output_dir,
        model_type=args.model_type,
        personalization_epochs=args.epochs,
        fine_tune_encoder=args.fine_tune_encoder
    )
