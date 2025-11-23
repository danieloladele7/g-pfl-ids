# malicious_evaluator.py
import os, argparse, json, datetime
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from src.models import DeepSVDDGCN, GAEDeepSVDD
from src.utils import load_client_data, create_data_loaders, normalize_graph_features
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score

class MaliciousEvaluator:
    def __init__(self, model_path, model_type='gcn'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if model_type == 'gcn':
            self.model = DeepSVDDGCN(in_channels=128, hidden_channels=64, num_layers=2)
        else:  # gae
            self.model = GAEDeepSVDD(in_channels=128, hidden_channels=64, num_layers=2)

# Load with strict=False to ignore classifier weights
        self.model.load_state_dict(torch.load(model_path), strict=False)
        self.model.to(self.device)
        self.model.eval() 
    
    def compute_center(self, normal_data_dir):
        """Compute center from normal client data"""
        centers = []
        for client_id in range(10):
            data = load_client_data(Path(normal_data_dir), str(client_id))
            if data:
                graph = data['graph'] if isinstance(data, dict) else data
                
                # Ensure the graph has the correct feature dimensions
                graph = normalize_graph_features(graph, expected_features=128)
                
                train_loader, _ = create_data_loaders(graph)
                
                client_embeddings = []
                with torch.no_grad():
                    for batch in train_loader:
                        batch = batch.to(self.device)
                        
                        # Check feature dimensions and fix if needed
                        if batch.x.size(1) != 128:
                            print(f"Warning: Batch features have {batch.x.size(1)} dimensions, normalizing to 128")
                            # Create a new batch with normalized features
                            from torch_geometric.data import Data
                            normalized_features = torch.zeros(batch.x.size(0), 128).to(self.device)
                            min_dim = min(batch.x.size(1), 128)
                            normalized_features[:, :min_dim] = batch.x[:, :min_dim]
                            batch.x = normalized_features
                        
                        embeddings, _ = self.model(batch.x, batch.edge_index)
                        client_embeddings.append(embeddings.cpu())
                
                if client_embeddings:
                    client_center = torch.cat(client_embeddings).mean(dim=0)
                    centers.append(client_center)
        
        return torch.stack(centers).mean(dim=0).to(self.device) if centers else torch.zeros(64).to(self.device)
    
    def evaluate_malicious_detection(self, normal_data_dir, malicious_csv_path):
        """Evaluate model on malicious detection"""
        center = self.compute_center(normal_data_dir)
        
        # Load and preprocess malicious data
        malicious_features = self.load_malicious_features(malicious_csv_path)
        
        # Calculate distances for normal data
        normal_distances = []
        for client_id in range(10):
            data = load_client_data(Path(normal_data_dir), str(client_id))
            if data:
                graph = data['graph'] if isinstance(data, dict) else data
                
                # Ensure the graph has the correct feature dimensions
                graph = normalize_graph_features(graph, expected_features=128)
                
                test_loader, _ = create_data_loaders(graph)
                
                with torch.no_grad():
                    for batch in test_loader:
                        batch = batch.to(self.device)
                        
                        # Check and fix feature dimensions
                        if batch.x.size(1) != 128:
                            normalized_features = torch.zeros(batch.x.size(0), 128).to(self.device)
                            min_dim = min(batch.x.size(1), 128)
                            normalized_features[:, :min_dim] = batch.x[:, :min_dim]
                            batch.x = normalized_features
                        
                        embeddings, _ = self.model(batch.x, batch.edge_index)
                        dist = torch.norm(embeddings - center, dim=1)
                        normal_distances.extend(dist.cpu().numpy())
        
        # Calculate distances for malicious data
        malicious_distances = []
        with torch.no_grad():
            for features in malicious_features:
                features = features.to(self.device).unsqueeze(0)  # Single sample
                
                # Create dummy edge_index for single node
                edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
                
                embeddings, _ = self.model(features, edge_index)
                dist = torch.norm(embeddings - center, dim=1)
                malicious_distances.extend(dist.cpu().numpy())
        
        # Calculate metrics
        y_true = [0] * len(normal_distances) + [1] * len(malicious_distances)
        y_scores = normal_distances + malicious_distances
        
        auroc = roc_auc_score(y_true, y_scores)
        aupr = average_precision_score(y_true, y_scores)
        
        # Find optimal threshold
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        optimal_idx = np.argmax(f1_scores[:-1])
        optimal_threshold = thresholds[optimal_idx]
        
        # Calculate detection rates
        detection_rate = (np.array(malicious_distances) > optimal_threshold).mean()
        false_positive_rate = (np.array(normal_distances) > optimal_threshold).mean()
        
        results = {
            'auroc': auroc,
            'aupr': aupr,
            'detection_rate': detection_rate,
            'false_positive_rate': false_positive_rate,
            'optimal_threshold': optimal_threshold,
            'num_normal_samples': len(normal_distances),
            'num_malicious_samples': len(malicious_distances)
        }
        
        print("Malicious Detection Results:")
        for key, value in results.items():
            print(f"  {key}: {value:.4f}")
        
        return results
    
    def load_malicious_features(self, csv_path):
        """Load malicious features from CSV and ensure 128 dimensions"""
        df = pd.read_csv(csv_path)
        features_list = []
        
        # Extract features (adjust based on CSV structure)
        # Exclude non-feature columns
        exclude_cols = ['label', 'malicious', 'id.orig_h', 'id.resp_h', 'target', 'class']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        print(f"Found {len(feature_cols)} feature columns in malicious CSV")
        print(f"Feature columns: {feature_cols}")
        
        for _, row in df.iterrows():
            features = torch.tensor(row[feature_cols].values.astype(float)).float()
            
            # Ensure correct feature dimension (128)
            if len(features) < 128:
                # Pad with zeros
                padding = torch.zeros(128 - len(features))
                features = torch.cat([features, padding])
                print(f"Padded features from {len(features) - (128 - len(features))} to 128 dimensions")
            elif len(features) > 128:
                # Truncate to 128
                features = features[:128]
                print(f"Truncated features from {len(features)} to 128 dimensions")
            
            features_list.append(features)
        
        print(f"Loaded {len(features_list)} malicious samples")
        return features_list

# Alternative version that builds graphs for malicious data
class MaliciousGraphEvaluator:
    def __init__(self, model_path, model_type='gcn'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if model_type == 'gcn':
            self.model = DeepSVDDGCN(in_channels=128, hidden_channels=64, num_layers=2)
        else:  # gae
            self.model = GAEDeepSVDD(in_channels=128, hidden_channels=64, num_layers=2)

        # Load with strict=False to ignore classifier weights
        self.model.load_state_dict(torch.load(model_path), strict=False)
        self.model.to(self.device)
        self.model.eval() 
    
    def csv_to_graph(self, csv_path, expected_features=128):
        """Convert malicious CSV to graph format"""
        import pandas as pd
        from torch_geometric.data import Data
        
        df = pd.read_csv(csv_path)
        graphs = []
        
        # Exclude non-feature columns
        exclude_cols = ['label', 'malicious', 'id.orig_h', 'id.resp_h', 'target', 'class']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        print(f"Converting {len(df)} malicious samples to graphs with {len(feature_cols)} features")
        
        for _, row in df.iterrows():
            # Extract features
            features = torch.tensor(row[feature_cols].values.astype(float)).float()
            
            # Ensure correct feature dimension
            if len(features) < expected_features:
                padding = torch.zeros(expected_features - len(features))
                features = torch.cat([features, padding])
            elif len(features) > expected_features:
                features = features[:expected_features]
            
            # Create single-node graph
            graph = Data(
                x=features.unsqueeze(0).float(),  # Shape: [1, 128]
                edge_index=torch.empty((2, 0), dtype=torch.long),  # No edges
                y=torch.tensor([1])  # Malicious label
            )
            graphs.append(graph)
        
        return graphs
    
    def compute_center(self, normal_data_dir):
        """Compute center from normal client data"""
        centers = []
        for client_id in range(10):
            data = load_client_data(Path(normal_data_dir), str(client_id))
            if data:
                graph = data['graph'] if isinstance(data, dict) else data
                graph = normalize_graph_features(graph, expected_features=128)
                train_loader, _ = create_data_loaders(graph)
                
                client_embeddings = []
                with torch.no_grad():
                    for batch in train_loader:
                        batch = batch.to(self.device)
                        embeddings, _ = self.model(batch.x, batch.edge_index)
                        client_embeddings.append(embeddings.cpu())
                
                if client_embeddings:
                    client_center = torch.cat(client_embeddings).mean(dim=0)
                    centers.append(client_center)
        
        return torch.stack(centers).mean(dim=0).to(self.device) if centers else torch.zeros(64).to(self.device)
    
    def evaluate_malicious_detection(self, normal_data_dir, malicious_csv_path):
        """Evaluate model on malicious detection using graph format"""
        center = self.compute_center(normal_data_dir)
        
        # Convert malicious CSV to graphs
        malicious_graphs = self.csv_to_graph(malicious_csv_path)
        malicious_loader, _ = create_data_loaders(malicious_graphs)
        
        # Calculate distances for normal data
        normal_distances = []
        for client_id in range(10):
            data = load_client_data(Path(normal_data_dir), str(client_id))
            if data:
                graph = data['graph'] if isinstance(data, dict) else data
                graph = normalize_graph_features(graph, expected_features=128)
                test_loader, _ = create_data_loaders(graph)
                
                with torch.no_grad():
                    for batch in test_loader:
                        batch = batch.to(self.device)
                        embeddings, _ = self.model(batch.x, batch.edge_index)
                        dist = torch.norm(embeddings - center, dim=1)
                        normal_distances.extend(dist.cpu().numpy())
        
        # Calculate distances for malicious data
        malicious_distances = []
        with torch.no_grad():
            for batch in malicious_loader:
                batch = batch.to(self.device)
                embeddings, _ = self.model(batch.x, batch.edge_index)
                dist = torch.norm(embeddings - center, dim=1)
                malicious_distances.extend(dist.cpu().numpy())
        
        # Calculate metrics
        y_true = [0] * len(normal_distances) + [1] * len(malicious_distances)
        y_scores = normal_distances + malicious_distances
        
        auroc = roc_auc_score(y_true, y_scores)
        aupr = average_precision_score(y_true, y_scores)
        
        # Find optimal threshold
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        optimal_idx = np.argmax(f1_scores[:-1])
        optimal_threshold = thresholds[optimal_idx]
        
        # Calculate detection rates
        detection_rate = (np.array(malicious_distances) > optimal_threshold).mean()
        false_positive_rate = (np.array(normal_distances) > optimal_threshold).mean()
        
        results = {
            'auroc': auroc,
            'aupr': aupr,
            'detection_rate': detection_rate,
            'false_positive_rate': false_positive_rate,
            'optimal_threshold': optimal_threshold,
            'num_normal_samples': len(normal_distances),
            'num_malicious_samples': len(malicious_distances)
        }
            
        print("Malicious Detection Results (Graph Format):")
        for key, value in results.items():
            print(f"  {key}: {value:.4f}")
        
        return results

def make_json_serializable(obj):
    """
    Convert obj into a form that can be JSON serialized.
    Handles numpy types, lists, dicts, datetime, and fallback to str.
    """
    # None, bool, int, float, str are already JSON serializable
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    # numpy scalar (e.g. np.int64, np.float32, etc.)
    if isinstance(obj, np.generic):
        return obj.item()

    # numpy array
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    # dict → recursively convert values
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}

    # list or tuple → recursively convert elements
    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(v) for v in obj]

    # datetime → ISO format string
    if isinstance(obj, datetime.datetime):
        return obj.isoformat()

    # If the object has `__dict__`, try serializing that
    if hasattr(obj, "__dict__"):
        return {k: make_json_serializable(v) for k, v in obj.__dict__.items()}

    # Fallback: convert to string
    return str(obj)


def save_result(results, model_type, save_path, personalized=False):
    os.makedirs(save_path, exist_ok=True)  # create folder if not exists
    if personalized:
        file_path = os.path.join(save_path, f"personalized_evaluation_{model_type}_result.json")
    else:
        file_path = os.path.join(save_path, f"evaluation_{model_type}_result.json")

    # Convert results into a JSON‑serializable object
    if isinstance(results, np.generic):
        obj = results.item()
    elif isinstance(results, dict):
        obj = {k: make_json_serializable(v) for k, v in results.items()}
    elif isinstance(results, (list, tuple)):
        obj = [make_json_serializable(v) for v in results]
    else:
        obj = make_json_serializable(results)
        # or simply: obj = results, if you expect it’s already serializable

    # Write JSON to file
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, sort_keys=True, ensure_ascii=False)

# Usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process IoT-23 data for FL training")
    parser.add_argument("--data_dir", type=str, default="data/graph_data/oneclass", help="Path to graph data")
    parser.add_argument("--malicious_csv_path", type=str, default="data/graph_data/oneclass/malicious.csv", help="Path to oneclass malicious csv data")
    parser.add_argument("--global_model_path", type=str, default="checkpoints", help="Path to oneclass malicious csv data")
    parser.add_argument("--output", type=str, default="results/", help="Path to results")
    parser.add_argument("--model_type", type=str, default="gcn", choices=["gcn", "gae"], help="Oneclass model type")
    args = parser.parse_args()
    
    if args.global_model_path.endswith(".pt"):
        global_model_path = args.global_model_path
    else:
        global_model_path = os.path.join(args.global_model_path, args.model_type, "global_model_best.pt")
    
    try:
        print("Testing regular MaliciousEvaluator...")
        evaluator = MaliciousEvaluator(global_model_path, model_type=args.model_type)
        results = evaluator.evaluate_malicious_detection(
            args.data_dir,
            args.malicious_csv_path
        )
        save_result(results, args.model_type, args.output)
    except Exception as e:
        print(f"Regular evaluator failed: {e}")
        print("\nTrying graph-based evaluator...")
        
        # Fall back to graph-based evaluator
        graph_evaluator = MaliciousGraphEvaluator(global_model_path, model_type=args.model_type)
        results = graph_evaluator.evaluate_malicious_detection(
            args.data_dir,
            args.malicious_csv_path
        )
        save_result(results, args.model_type, args.output)
