# personalized_evaluator.py
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from models import DeepSVDDGCN, GAEDeepSVDD
from utils import load_client_data, create_data_loaders, normalize_graph_features
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns

class PersonalizedMaliciousEvaluator:
    def __init__(self, personalized_models_dir, model_type='gcn', num_clients=10):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type
        self.personalized_models_dir = Path(personalized_models_dir)
        
        # Load all personalized models and centers
        self.client_models = {}
        self.client_centers = {}
        
        for client_id in range(num_clients):
            model_path = self.personalized_models_dir / f"personalized_model_{client_id}.pt"
            center_path = self.personalized_models_dir / f"client_center_{client_id}.pt"
            
            if model_path.exists() and center_path.exists():
                # Create model instance
                if model_type == 'gcn':
                    model = DeepSVDDGCN(in_channels=128, hidden_channels=64, num_layers=2)
                else:  # gae
                    model = GAEDeepSVDD(in_channels=128, hidden_channels=64, num_layers=2)
                
                # Load checkpoint
                try:
                    checkpoint = torch.load(model_path, map_location=self.device)
                    
                    # Check if it's a checkpoint dictionary or just state_dict
                    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['model_state_dict'])
                        # Try to get center from checkpoint first
                        if 'client_center' in checkpoint:
                            center = checkpoint['client_center'].to(self.device)
                        else:
                            center = torch.load(center_path).to(self.device)
                    else:
                        # It's just the state_dict
                        model.load_state_dict(checkpoint)
                        center = torch.load(center_path).to(self.device)
                    
                    model.to(self.device)
                    model.eval()
                    
                    self.client_models[client_id] = model
                    self.client_centers[client_id] = center
                    
                    print(f"Loaded personalized model and center for client {client_id}")
                    
                except Exception as e:
                    print(f"Error loading model for client {client_id}: {e}")
                    continue
            else:
                print(f"Model or center file missing for client {client_id}")
        
        print(f"Loaded {len(self.client_models)} personalized models")
    
    def _adjust_center_dimension(self, center, embeddings):
        """Adjust center dimension to match embeddings dimension"""
        if center.shape[0] == embeddings.shape[1]:
            return center
        elif center.shape[0] < embeddings.shape[1]:
            # Pad with zeros
            padding = torch.zeros(embeddings.shape[1] - center.shape[0]).to(self.device)
            return torch.cat([center, padding])
        else:
            # Truncate
            return center[:embeddings.shape[1]]

    def compute_tpr_at_fpr_thresholds(self, normal_distances, malicious_distances, fpr_thresholds=[0.01, 0.05, 0.10]):
        """Compute True Positive Rate (detection rate) at specific FPR thresholds"""
        results = {}
        
        # Convert to numpy arrays for easier manipulation
        normal_dists = np.array(normal_distances)
        malicious_dists = np.array(malicious_distances)
        
        for fpr_target in fpr_thresholds:
            # Find threshold that gives desired FPR (percentile from normal distances)
            threshold = np.percentile(normal_dists, (1 - fpr_target) * 100)
            
            # Calculate TPR (detection rate) at this threshold
            tpr = (malicious_dists > threshold).mean()
            
            # Calculate actual FPR at this threshold (for verification)
            actual_fpr = (normal_dists > threshold).mean()
            
            results[f'tpr_at_{int(fpr_target*100)}%_fpr'] = tpr
            results[f'actual_fpr_at_{int(fpr_target*100)}%_threshold'] = actual_fpr
            results[f'threshold_at_{int(fpr_target*100)}%_fpr'] = threshold
        
        return results

    def evaluate_client_specific(self, normal_data_dir, malicious_csv_path):
        """Evaluate each client's personalized model on their own data and malicious data"""
        client_results = {}
        
        for client_id, model in self.client_models.items():
            print(f"\nEvaluating client {client_id}...")
            center = self.client_centers[client_id]
            
            # Load client's normal data
            data = load_client_data(Path(normal_data_dir), str(client_id))
            if not data:
                print(f"Client {client_id}: No data available")
                continue
                
            graph = data['graph'] if isinstance(data, dict) else data
            graph = normalize_graph_features(graph, expected_features=128)
            _, test_loader = create_data_loaders(graph)
            
            # Calculate distances for client's normal data
            normal_distances = []
            with torch.no_grad():
                for batch in test_loader:
                    batch = batch.to(self.device)
                    embeddings, _ = model(batch.x, batch.edge_index)
                    # Adjust center dimension if needed
                    adjusted_center = self._adjust_center_dimension(center, embeddings)
                    dist = torch.norm(embeddings - adjusted_center, dim=1)
                    normal_distances.extend(dist.cpu().numpy())
            
            # Calculate distances for malicious data using this client's model
            malicious_distances = self.evaluate_malicious_with_client_model(model, center, malicious_csv_path)
            
            # Calculate metrics for this client
            y_true = [0] * len(normal_distances) + [1] * len(malicious_distances)
            y_scores = normal_distances + malicious_distances
            
            auroc = roc_auc_score(y_true, y_scores)
            aupr = average_precision_score(y_true, y_scores)
            
            # Find optimal threshold (maximizes F1 score)
            precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
            optimal_idx = np.argmax(f1_scores[:-1])
            optimal_threshold = thresholds[optimal_idx]
            
            # Calculate metrics at optimal threshold
            detection_rate = (np.array(malicious_distances) > optimal_threshold).mean()
            false_positive_rate = (np.array(normal_distances) > optimal_threshold).mean()
            
            # Calculate TPR at specific FPR thresholds (1%, 5%, 10%)
            tpr_at_fpr_metrics = self.compute_tpr_at_fpr_thresholds(normal_distances, malicious_distances)
            
            client_results[client_id] = {
                'auroc': auroc,
                'aupr': aupr,
                'detection_rate': detection_rate,  # TPR at optimal threshold
                'false_positive_rate': false_positive_rate,  # FPR at optimal threshold
                'optimal_threshold': optimal_threshold,
                'num_normal_samples': len(normal_distances),
                'num_malicious_samples': len(malicious_distances),
                'avg_normal_distance': np.mean(normal_distances),
                'avg_malicious_distance': np.mean(malicious_distances),
                **tpr_at_fpr_metrics  # Add TPR at specific FPR thresholds
            }
            
            print(f"Client {client_id} Results:")
            print(f"  AUROC: {auroc:.4f}")
            print(f"  AUPR: {aupr:.4f}")
            print(f"  Optimal threshold: {optimal_threshold:.4f}")
            print(f"  Detection Rate (TPR) at optimal threshold: {detection_rate:.4f}")
            print(f"  False Positive Rate at optimal threshold: {false_positive_rate:.4f}")
            print(f"  TPR at 1% FPR: {tpr_at_fpr_metrics['tpr_at_1%_fpr']:.4f}")
            print(f"  TPR at 5% FPR: {tpr_at_fpr_metrics['tpr_at_5%_fpr']:.4f}")
            print(f"  TPR at 10% FPR: {tpr_at_fpr_metrics['tpr_at_10%_fpr']:.4f}")
                    
        return client_results
    
    def evaluate_ensemble(self, normal_data_dir, malicious_csv_path, aggregation_method='mean'):
        """Evaluate using ensemble of all personalized models"""
        all_normal_distances = []
        all_malicious_distances = []
        
        # Load malicious features once
        malicious_features = self.load_malicious_features(malicious_csv_path)
        
        for client_id, model in self.client_models.items():
            center = self.client_centers[client_id]
            
            # Load client's normal data
            data = load_client_data(Path(normal_data_dir), str(client_id))
            if not data:
                continue
                
            graph = data['graph'] if isinstance(data, dict) else data
            graph = normalize_graph_features(graph, expected_features=128)
            _, test_loader = create_data_loaders(graph)
            
            # Calculate distances for client's normal data
            with torch.no_grad():
                for batch in test_loader:
                    batch = batch.to(self.device)
                    embeddings, _ = model(batch.x, batch.edge_index)
                    # Adjust center dimension if needed
                    adjusted_center = self._adjust_center_dimension(center, embeddings)
                    dist = torch.norm(embeddings - adjusted_center, dim=1)
                    all_normal_distances.append({
                        'client_id': client_id,
                        'distances': dist.cpu().numpy(),
                        'type': 'normal'
                    })
            
            # Calculate distances for malicious data
            malicious_dists = []
            with torch.no_grad():
                for features in malicious_features:
                    features = features.to(self.device).unsqueeze(0)
                    edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
                    embeddings, _ = model(features, edge_index)
                    # Adjust center dimension if needed
                    adjusted_center = self._adjust_center_dimension(center, embeddings)
                    dist = torch.norm(embeddings - adjusted_center, dim=1)
                    malicious_dists.extend(dist.cpu().numpy())
            
            all_malicious_distances.append({
                'client_id': client_id,
                'distances': np.array(malicious_dists),
                'type': 'malicious'
            })
        
        # Aggregate distances across clients
        ensemble_results = self.aggregate_distances(
            all_normal_distances, all_malicious_distances, aggregation_method
        )
        
        return ensemble_results
    
    def aggregate_distances(self, normal_distances, malicious_distances, method='mean'):
        """Aggregate distances from multiple clients"""
        if method == 'mean':
            # For evaluation, we need per-sample distances
            normal_flat = np.concatenate([item['distances'] for item in normal_distances])
            malicious_flat = np.concatenate([item['distances'] for item in malicious_distances])
        
        # Calculate metrics
        y_true = [0] * len(normal_flat) + [1] * len(malicious_flat)
        y_scores = normal_flat.tolist() + malicious_flat.tolist()
        
        auroc = roc_auc_score(y_true, y_scores)
        aupr = average_precision_score(y_true, y_scores)
        
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        optimal_idx = np.argmax(f1_scores[:-1])
        optimal_threshold = thresholds[optimal_idx]
        
        detection_rate = (np.array(malicious_flat) > optimal_threshold).mean()
        false_positive_rate = (np.array(normal_flat) > optimal_threshold).mean()
        
        # Calculate TPR at specific FPR thresholds
        tpr_at_fpr_metrics = self.compute_tpr_at_fpr_thresholds(normal_flat, malicious_flat)
        
        results = {
            'auroc': auroc,
            'aupr': aupr,
            'detection_rate': detection_rate,
            'false_positive_rate': false_positive_rate,
            'optimal_threshold': optimal_threshold,
            'aggregation_method': method,
            'num_normal_samples': len(normal_flat),
            'num_malicious_samples': len(malicious_flat),
            'num_clients_used': len(self.client_models),
            **tpr_at_fpr_metrics
        }
        
        print(f"\nEnsemble Results ({method} aggregation):")
        print(f"  AUROC: {auroc:.4f}")
        print(f"  AUPR: {aupr:.4f}")
        print(f"  Optimal threshold: {optimal_threshold:.4f}")
        print(f"  Detection Rate (TPR) at optimal threshold: {detection_rate:.4f}")
        print(f"  False Positive Rate at optimal threshold: {false_positive_rate:.4f}")
        print(f"  TPR at 1% FPR: {tpr_at_fpr_metrics['tpr_at_1%_fpr']:.4f}")
        print(f"  TPR at 5% FPR: {tpr_at_fpr_metrics['tpr_at_5%_fpr']:.4f}")
        print(f"  TPR at 10% FPR: {tpr_at_fpr_metrics['tpr_at_10%_fpr']:.4f}")
        
        return results
    
    def evaluate_malicious_with_client_model(self, model, center, malicious_csv_path):
        """Evaluate malicious data using a specific client's model"""
        malicious_features = self.load_malicious_features(malicious_csv_path)
        distances = []
        
        with torch.no_grad():
            for features in malicious_features:
                features = features.to(self.device).unsqueeze(0)
                edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
                embeddings, _ = model(features, edge_index)
                # Adjust center dimension if needed
                adjusted_center = self._adjust_center_dimension(center, embeddings)
                dist = torch.norm(embeddings - adjusted_center, dim=1)
                distances.extend(dist.cpu().numpy())
        
        return distances
    
    def load_malicious_features(self, csv_path):
        """Load malicious features from CSV and ensure 128 dimensions"""
        df = pd.read_csv(csv_path)
        features_list = []
        
        # Extract features (adjust based on your CSV structure)
        exclude_cols = ['label', 'malicious', 'id.orig_h', 'id.resp_h', 'target', 'class']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        print(f"Loading {len(df)} malicious samples with {len(feature_cols)} features")
        
        for _, row in df.iterrows():
            features = torch.tensor(row[feature_cols].values.astype(float)).float()
            
            # Ensure correct feature dimension (128)
            if len(features) < 128:
                padding = torch.zeros(128 - len(features))
                features = torch.cat([features, padding])
            elif len(features) > 128:
                features = features[:128]
            
            features_list.append(features)
        
        return features_list
    
    def plot_client_performance(self, client_results, output_path="personalized_client_performance.png"):
        """Plot performance metrics across clients"""
        client_ids = list(client_results.keys())
        metrics = ['auroc', 'detection_rate', 'false_positive_rate']
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for i, metric in enumerate(metrics):
            values = [client_results[cid][metric] for cid in client_ids]
            axes[i].bar(client_ids, values, color='skyblue', edgecolor='black')
            axes[i].set_title(f'Client {metric.upper()}')
            axes[i].set_xlabel('Client ID')
            axes[i].set_ylabel(metric.upper())
            axes[i].set_ylim(0, 1)
            
            # Add value labels on bars
            for j, v in enumerate(values):
                axes[i].text(j, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=600, bbox_inches='tight')
        plt.show()
        print(f"Client performance plot saved to {output_path}")
    
    def plot_distance_distributions(self, client_results, normal_data_dir, malicious_csv_path, output_path="personalized_distance_distributions.png"):
        """Plot distance distributions for each client"""
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.flatten()
        
        for idx, (client_id, model) in enumerate(self.client_models.items()):
            if idx >= 10:  # Only plot first 10 clients
                break
                
            center = self.client_centers[client_id]
            
            # Get normal distances
            data = load_client_data(Path(normal_data_dir), str(client_id))
            if not data:
                continue
                
            graph = data['graph'] if isinstance(data, dict) else data
            graph = normalize_graph_features(graph, expected_features=128)
            _, test_loader = create_data_loaders(graph)
            
            normal_distances = []
            with torch.no_grad():
                for batch in test_loader:
                    batch = batch.to(self.device)
                    embeddings, _ = model(batch.x, batch.edge_index)
                    dist = torch.norm(embeddings - center, dim=1)
                    normal_distances.extend(dist.cpu().numpy())
            
            # Get malicious distances
            malicious_distances = self.evaluate_malicious_with_client_model(model, center, malicious_csv_path)
            
            # Plot
            axes[idx].hist(normal_distances, alpha=0.7, label='Normal', bins=30, density=True)
            axes[idx].hist(malicious_distances, alpha=0.7, label='Malicious', bins=30, density=True)
            axes[idx].set_title(f'Client {client_id}\nAUROC: {client_results[client_id]["auroc"]:.3f}')
            axes[idx].set_xlabel('Distance from Center')
            axes[idx].set_ylabel('Density')
            axes[idx].legend()
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=600, bbox_inches='tight')
        plt.show()
        print(f"Distance distributions plot saved to {output_path}")

def convert_numpy_types(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    else:
        return obj

# Usage example
if __name__ == "__main__":
    import os, argparse
    
    parser = argparse.ArgumentParser(description="Evaluate personalized models for FL")
    parser.add_argument("--personalized_models_dir", type=str, default="personalized_models", 
                       help="Path to saved personalized models")
    parser.add_argument("--data_dir", type=str, default="data/graph_data/oneclass", 
                       help="Path to graph data")
    parser.add_argument("--malicious_csv_path", type=str, 
                       default="data/graph_data/oneclass/malicious.csv", 
                       help="Path to malicious csv data")
    parser.add_argument("--output", type=str, default="results/", 
                       help="Path to save results")
    parser.add_argument("--num_clients", type=int, default=10, 
                       help="Number of clients")
    parser.add_argument("--model_type", type=str, default="gcn", 
                       choices=["gcn", "gae"], 
                       help="Model type: 'gcn' or 'gae'")
    args = parser.parse_args()
    
    # Check if directory exists
    models_dir = Path(args.personalized_models_dir)
    if not models_dir.exists():
        print(f"Error: Directory {models_dir} does not exist!")
        print(f"Looking for models in: {models_dir.absolute()}")
        exit(1)
    
    # List files in directory for debugging
    print(f"\nFiles in {models_dir}:")
    for f in models_dir.glob("*.pt"):
        print(f"  - {f.name}")
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize evaluator
    evaluator = PersonalizedMaliciousEvaluator(
        personalized_models_dir=str(models_dir),
        model_type=args.model_type,
        num_clients=args.num_clients
    )
    
    # Check if any models were loaded
    if len(evaluator.client_models) == 0:
        print("\nERROR: No models were loaded!")
        print(f"Expected to find files like: personalized_model_0.pt and client_center_0.pt")
        print(f"in directory: {models_dir.absolute()}")
        print("\nMake sure:")
        print("1. The directory path is correct")
        print("2. The model files exist with correct naming")
        print("3. You're using the correct --model_type (gcn or gae)")
        exit(1)
    
    # Evaluate each client individually
    print("\n=== Evaluating Client-Specific Performance ===")
    client_results = evaluator.evaluate_client_specific(
        normal_data_dir=args.data_dir,
        malicious_csv_path=args.malicious_csv_path
    )
    
    # Evaluate ensemble performance
    print("\n=== Evaluating Ensemble Performance ===")
    ensemble_results = evaluator.evaluate_ensemble(
        normal_data_dir=args.data_dir,
        malicious_csv_path=args.malicious_csv_path,
        aggregation_method='mean'
    )
    
    # Generate plots
    plot_prefix = f"personalized_{args.model_type}"
    evaluator.plot_client_performance(client_results, 
                                     output_path=output_dir / f"{plot_prefix}_client_performance.png")
    evaluator.plot_distance_distributions(client_results, 
                                         args.data_dir,
                                         args.malicious_csv_path,
                                         output_path=output_dir / f"{plot_prefix}_distance_distributions.png")
    
    # Save results
    import json
    save_path = output_dir / f'personalized_{args.model_type}_evaluation_results.json'
    
    with open(save_path, 'w') as f:
        json.dump({
            'client_specific': convert_numpy_types(client_results),
            'ensemble': convert_numpy_types(ensemble_results)
        }, f, indent=2)
    
    print(f"\n=== Personalized Evaluation Complete ===")
    print(f"Results saved to '{save_path}'")