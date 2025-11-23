# src/utils/data_processing.py
"""
Data processing utilities for graph data.
"""

import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from src.utils.training_utils import set_seed, dirichlet_split_by_label
from pathlib import Path
import logging
import re

logger = logging.getLogger(__name__)

# Predefined mappings
CONN_STATE_MAPPING = {
    'SF': 'Normal', 'S0': 'No_Response', 'RSTOS0': 'Rejected', 'OTH': 'Other',
    'REJ': 'Rejected', 'SH': 'Half_Open', 'RSTO': 'Reset', 'RSTR': 'Reset',
    'S1': 'Established', 'S2': 'Established', 'S3': 'Established', 'RSTRH': 'Reset'
}

HISTORY_FEATURE_MAPPING = {
    'S': 'syn', 'I': 'init', 'D': 'data', 'F': 'fin', 'R': 'rst', 'T': 'timeout',
    'A': 'ack', 'Sr': 'syn_reply', 'Sh': 'handshake', 'Ar': 'ack_reply', 'Dr': 'data_reply',
    'Fa': 'fin_ack', 'Ha': 'handshake_ack', 'Dd': 'data_direction', 'DTT': 'data_timeout',
    '-': 'unknown'
}

def read_zeek_conn(path, usecols=None, dtype_map=None):
    path = Path(path)
    # Read until #fields to get fields and header_lines
    with path.open('r', errors='ignore') as f:
        for i, line in enumerate(f):
            if line.startswith('#fields'):
                fields = line.strip().split()[1:]
                header_lines = i + 1
                break
        else:
            raise ValueError("Can't find '#fields'")
    
    # If usecols specified, map names to positions maybe or pass names/usecols to read_csv
    df = pd.read_csv(
        path,
        sep=r'\s+',
        names=fields,
        comment='#',
        header=None,
        skiprows=header_lines,
        usecols=usecols,
        dtype=dtype_map,
        low_memory=True
    )
    return df

def process_flow_features(df: pd.DataFrame, include_graph_columns: bool = True) -> pd.DataFrame:
    """Process flow features including connection state and history"""
    # Define columns to preserve for graph building
    graph_columns = ['id.orig_h', 'id.resp_h'] if include_graph_columns else []
    
    # Define numeric columns
    numeric_cols = [
        'duration', 'orig_bytes', 'resp_bytes', 'orig_pkts', 'resp_pkts',
        'orig_ip_bytes', 'resp_ip_bytes', 'missed_bytes'
    ]
    
    # Filter to only available numeric columns
    numeric_cols = [col for col in numeric_cols if col in df.columns]
        
    # Create a copy to avoid modifying the original dataframe
    df_processed = df.copy()
    
    # Handle missing values in numeric columns
    for col in numeric_cols:
        # Replace non-numeric values with NaN
        df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
        # Fill NaN with 0
        df_processed[col] = df_processed[col].fillna(0)
        # Ensure no negative values for these features
        if col in [x for x in numeric_cols if x != 'duration']:
            df_processed[col] = df_processed[col].clip(lower=0)
    
    # Add time features
    if 'ts' in df_processed.columns:
        # Convert timestamp to datetime
        df_processed['timestamp'] = pd.to_datetime(df_processed['ts'], unit='s')
        
        # Time-based features
        df_processed['hour_of_day'] = df_processed['timestamp'].dt.hour
        df_processed['day_of_week'] = df_processed['timestamp'].dt.dayofweek
        df_processed['is_weekend'] = df_processed['day_of_week'].isin([5, 6]).astype(int)
        
        # Time since first connection
        if len(df_processed) > 0:
            df_processed['time_since_first'] = df_processed['ts'] - df_processed['ts'].min()
    
    # Add behavioral features
    df_processed['bytes_ratio'] = np.where(
        df_processed['resp_bytes'] > 0,
        df_processed['orig_bytes'] / df_processed['resp_bytes'],
        0
    )
    
    df_processed['packets_ratio'] = np.where(
        df_processed['resp_pkts'] > 0,
        df_processed['orig_pkts'] / df_processed['resp_pkts'],
        0
    )
    
    df_processed['bytes_per_packet_orig'] = np.where(
        df_processed['orig_pkts'] > 0,
        df_processed['orig_bytes'] / df_processed['orig_pkts'],
        0
    )
    
    df_processed['bytes_per_packet_resp'] = np.where(
        df_processed['resp_pkts'] > 0,
        df_processed['resp_bytes'] / df_processed['resp_pkts'],
        0
    )
    
    # Identify skewed features for log transformation
    skewed_features = [x for x in numeric_cols if x != 'duration'] + [
        'time_since_first', 'bytes_ratio', 'packets_ratio', 
        'bytes_per_packet_orig', 'bytes_per_packet_resp'
    ]
    
    # Apply log transformation to highly skewed features
    for feature in skewed_features:
        if feature in df_processed.columns:
            # Add 1 to avoid log(0)
            df_processed[feature] = np.log1p(df_processed[feature])
    
    # Add protocol features
    if 'proto' in df_processed.columns:
        df_processed['proto'] = df_processed['proto'].replace('-', 'unknown')
        proto_dummies = pd.get_dummies(df_processed['proto'], prefix='proto', dtype='uint8')
        df_processed = pd.concat([df_processed, proto_dummies], axis=1)
    
    if 'service' in df_processed.columns:
        df_processed['service'] = df_processed['service'].replace('-', 'unknown')
        service_dummies = pd.get_dummies(df_processed['service'], prefix='service', dtype='uint8')
        df_processed = pd.concat([df_processed, service_dummies], axis=1)
    
    # Connection duration characteristics
    if 'duration' in df_processed.columns:
        df_processed['log_duration'] = np.log1p(df_processed['duration'])
        
    # Process connection state
    df_processed['conn_state'] = df_processed['conn_state'].astype(str).str.upper().fillna('UNKNOWN')
    df_processed['conn_type'] = df_processed['conn_state'].map(CONN_STATE_MAPPING).fillna('Other')
    
    # One-hot encode connection type
    conn_type_dummies = pd.get_dummies(
        df_processed['conn_type'], 
        prefix='conn', 
        dtype='uint8'
    )
    df_processed = pd.concat([df_processed, conn_type_dummies], axis=1)
    
    # Process history features
    if 'history' in df_processed.columns:
        hist = df_processed['history'].fillna('').astype(str)
        history_features = pd.DataFrame({
            feature: hist.str.contains(code, case=False, na=False).astype('uint8')
            for code, feature in HISTORY_FEATURE_MAPPING.items()
        })
        df_processed = pd.concat([df_processed, history_features], axis=1)
    
    # Combine all feature columns
    preserved_cols = graph_columns + ['hour_of_day', 'day_of_week', 'is_weekend', 'log_duration', 'duration'] + \
                   skewed_features + \
                   list(conn_type_dummies.columns) + \
                   list(history_features.columns) + \
                   list(proto_dummies.columns) + \
                   list(service_dummies.columns) + ['label']

    # Filter to only existing columns
    preserved_cols = [col for col in preserved_cols if col in df_processed.columns]
    # Handle labels
    if 'label' in df_processed.columns:
        df_processed['label'] = df_processed['label'].astype(str).str.lower()
    
    df_processed = df_processed[preserved_cols]
    
    # print(f'preserved columns are: {df_processed.columns}')
    
    return df_processed

def filter_data_by_mode(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Filter data based on the selected mode (one-class or binary)."""
    if mode == "oneclass":
        # Keep only benign samples for one-class learning
        benign_mask = df['label'].str.contains('benign|normal', case=False, na=False)
        if benign_mask.sum() == 0:
            logger.warning("No benign samples found for one-class mode!")
            return df
        logger.info(f"One-class mode: Keeping {benign_mask.sum()} benign samples")
        return df[benign_mask].copy()
    elif mode == "binary":
        logger.info(f"Binary mode: {sum(df['label'] == 'benign')} benign, {sum(df['label'] == 'malicious')} malicious")
        return df
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'one-class' or 'binary'")

def create_per_flow_labels(df: pd.DataFrame) -> np.ndarray:
    """Create per-flow labels (0 for benign, 1 for malicious)"""
    return df['label'].apply(lambda x: 0 if 'benign' in x else 1).values

def create_per_host_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Create per-host labels by aggregating flow information"""
    # Group by source IP
    src_agg = df.groupby('id.orig_h').agg({
        'label': lambda x: 1 if any('malicious' in str(l).lower() for l in x) else 0
    }).rename(columns={'label': 'is_malicious_src'})
    
    # Group by destination IP
    dst_agg = df.groupby('id.resp_h').agg({
        'label': lambda x: 1 if any('malicious' in str(l).lower() for l in x) else 0
    }).rename(columns={'label': 'is_malicious_dst'})
    
    # Combine and create final host labels
    host_labels = pd.concat([src_agg, dst_agg], axis=1).fillna(0)

    # Convert the src/dst malicious flags to integers 0/1 (from floats) first
    host_labels['is_malicious_src'] = host_labels['is_malicious_src'].astype(int)
    host_labels['is_malicious_dst'] = host_labels['is_malicious_dst'].astype(int)

    # Then compute the combined malicious flag
    host_labels['is_malicious'] = ((host_labels['is_malicious_src']) | (host_labels['is_malicious_dst'])).astype(int)
    
    return host_labels[['is_malicious']]  # Return only the final label column

def load_client_data(data_dir: Path, client_id: str):
    """Load graph data for a client with improved error handling"""
    data_dir = Path(data_dir)
    file_path = data_dir / f"client_graph_{client_id}.pt"
    
    if not file_path.exists():
        return None
        
    try:
        data = torch.load(file_path, weights_only=False)
        # Ensure data is in proper format
        if hasattr(data, 'num_nodes') or (isinstance(data, dict) and 'graph' in data):
            return data
        else:
            logger.error(f"Invalid data format for client {client_id}")
            return None
    except Exception as e:
        logger.error(f"Error loading data for client {client_id}: {e}")
        return None

def split_graph_data(graph, train_ratio=0.7, test_ratio=0.3, random_state=42):
    """Split graph data into train and test sets with proper separation"""
    num_nodes = graph.num_nodes
    
    # Ensure we have labels
    if not hasattr(graph, 'y') or graph.y is None:
        # Create dummy labels if none exist
        graph.y = torch.zeros(num_nodes, dtype=torch.long)
    
    # Use stratified splitting to maintain class balance
    from sklearn.model_selection import train_test_split
    
    indices = np.arange(num_nodes)
    labels = graph.y.numpy()
    
    # Handle case where all labels are the same
    if len(np.unique(labels)) == 1:
        # Random split if only one class
        train_idx, test_idx = train_test_split(
            indices, 
            test_size=test_ratio, 
            random_state=random_state
        )
    else:
        # Stratified split for multiple classes
        train_idx, test_idx = train_test_split(
            indices, 
            test_size=test_ratio, 
            stratify=labels,
            random_state=random_state
        )
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[train_idx] = True
    test_mask[test_idx] = True
    
    return train_mask, test_mask

def create_data_loaders(graph, batch_size=32):
    """Create data loaders for train and test sets"""
    train_mask, test_mask = split_graph_data(graph)
    
    # Create subsets
    train_data = graph.clone()
    test_data = graph.clone()
    
    # Apply masks
    train_data.train_mask = train_mask
    test_data.test_mask = test_mask
    
    # Create data loaders
    train_loader = DataLoader([train_data], batch_size=batch_size, shuffle=True)
    test_loader = DataLoader([test_data], batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def preprocess_graph_data(graph, client_id=None):
    """Preprocess graph data to ensure consistency"""
    # Ensure feature matrix exists
    if not hasattr(graph, 'x') or graph.x is None:
        graph.x = extract_features_from_graph(graph)
    
    # Ensure consistent feature dimensions
    graph = normalize_graph_features(graph, expected_features=128)
    
    # Validate and fix data with client ID for better logging
    graph = validate_data(graph, client_id)
    
    return graph

def normalize_graph_features(graph, expected_features=128, normalize=True):
    """Ensure graph has consistent feature dimensions with optional normalization"""
    if hasattr(graph, 'x') and graph.x is not None:
        current_features = graph.x.shape[1]
        if current_features < expected_features:
            # Pad with zeros
            padding = torch.zeros((graph.x.shape[0], expected_features - current_features))
            graph.x = torch.cat([graph.x, padding], dim=1)
        elif current_features > expected_features:
            # Truncate excess features
            graph.x = graph.x[:, :expected_features]
        
        # Normalize features to have zero mean and unit variance
        if normalize and graph.x.numel() > 0:
            # Skip if all zeros
            if torch.any(graph.x != 0):
                # Standard scaling
                mean = graph.x.mean(dim=0, keepdim=True)
                std = graph.x.std(dim=0, keepdim=True) + 1e-8  # Avoid division by zero
                graph.x = (graph.x - mean) / std
    
    return graph

# def normalize_graph_features(graph, expected_features=128):
#     """Ensure graph has consistent feature dimensions"""
#     if hasattr(graph, 'x') and graph.x is not None:
#         current_features = graph.x.shape[1]
#         if current_features < expected_features:
#             # Pad with zeros
#             padding = torch.zeros((graph.x.shape[0], expected_features - current_features))
#             graph.x = torch.cat([graph.x, padding], dim=1)
#         elif current_features > expected_features:
#             # Truncate excess features
#             graph.x = graph.x[:, :expected_features]
#     else:
#         # Create feature matrix from individual attributes
#         feature_tensors = []
#         reserved_attrs = ['edge_index', 'y', 'num_nodes', 'batch']
#
#         for attr_name in dir(graph):
#             if not attr_name.startswith('_') and attr_name not in reserved_attrs:
#                 attr = getattr(graph, attr_name)
#                 if isinstance(attr, torch.Tensor) and attr.dim() == 1:
#                     feature_tensors.append(attr)
#
#         if feature_tensors:
#             graph.x = torch.stack(feature_tensors, dim=1)
#             # Ensure correct dimension
#             current_features = graph.x.shape[1]
#             if current_features < expected_features:
#                 padding = torch.zeros((graph.x.shape[0], expected_features - current_features))
#                 graph.x = torch.cat([graph.x, padding], dim=1)
#             elif current_features > expected_features:
#                 graph.x = graph.x[:, :expected_features]
#         else:
#             # Create default features if none exist
#             graph.x = torch.zeros((graph.num_nodes, expected_features))
#
#     return graph

def extract_features_from_graph(graph):
    """Extract features from graph object, handling both unified and individual feature formats"""
    if hasattr(graph, 'x') and graph.x is not None:
        # Unified feature matrix exists
        return graph.x
    else:
        # Extract features from individual attributes
        feature_tensors = []
        reserved_attrs = ['edge_index', 'y', 'num_nodes']
        
        # Get all attributes that are likely features
        for attr_name in dir(graph):
            if not attr_name.startswith('_') and attr_name not in reserved_attrs:
                attr = getattr(graph, attr_name)
                if isinstance(attr, torch.Tensor) and attr.dim() == 1:
                    feature_tensors.append(attr)
        
        if feature_tensors:
            # Stack individual features into a matrix
            return torch.stack(feature_tensors, dim=1)
        else:
            return None

def validate_data(data, client_id=None):
    """Check for NaN or Inf in features and labels, and fix if possible"""
    if client_id:
        client_info = f" for client {client_id}"
    else:
        client_info = ""
    
    # Check features
    if hasattr(data, 'x') and data.x is not None:
        if torch.isnan(data.x).any() or torch.isinf(data.x).any():
            nan_count = torch.isnan(data.x).sum().item()
            inf_count = torch.isinf(data.x).sum().item()
            print(f"Warning{client_info}: Data features contain {nan_count} NaN and {inf_count} Inf values, replacing with zeros")
            data.x = torch.nan_to_num(data.x, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Check labels if they exist
    if hasattr(data, 'y') and data.y is not None:
        if torch.isnan(data.y).any() or torch.isinf(data.y).any():
            nan_count = torch.isnan(data.y).sum().item()
            inf_count = torch.isinf(data.y).sum().item()
            print(f"Warning{client_info}: Labels contain {nan_count} NaN and {inf_count} Inf values, replacing with zeros")
            data.y = torch.nan_to_num(data.y, nan=0.0, posinf=0.0, neginf=0.0)
    
    return data

def fix_graph_consistency(graph_data):
    """Attempt to fix graph consistency issues."""
    if graph_data is None:
        return None
    
    num_nodes = graph_data.num_nodes
    edge_index = graph_data.edge_index
    
    # Filter out invalid edges
    if edge_index.size(0) > 0:
        valid_mask = (edge_index[0] < num_nodes) & (edge_index[1] < num_nodes)
        if not valid_mask.all():
            print(f"Fixing {(~valid_mask).sum().item()} invalid edges")
            graph_data.edge_index = edge_index[:, valid_mask]
            
            # Also filter edge attributes if they exist
            if hasattr(graph_data, 'edge_attr') and graph_data.edge_attr is not None:
                graph_data.edge_attr = graph_data.edge_attr[valid_mask]
    
    return graph_data

def validate_graph_consistency(graph_data):
    """Validate that graph edges reference valid nodes."""
    if graph_data is None:
        return False
    
    num_nodes = graph_data.num_nodes
    edge_index = graph_data.edge_index
    
    # Check if edge indices are within bounds
    if edge_index.size(0) > 0:
        max_index = edge_index.max().item()
        if max_index >= num_nodes:
            print(f"ERROR: Graph contains invalid edge references. Max index: {max_index}, Num nodes: {num_nodes}")
            return False
    
    return True

def process_iot23_data(input_path: Path, output_dir: Path, n_clients: int = 10, alpha: float = 0.5, min_samples_per_client: int = 986, labeling_strategy: str = "per_flow", mode: str = "oneclass", fraction: float = 0.3, seed: int = 42):
    """Process IoT-23 data and create non-IID client splits"""
    set_seed(seed)
    
    # Read and preprocess the data
    logger.info("Reading Zeek conn log...")
    df = read_zeek_conn(input_path)
    total_size = int(len(df) * fraction)
    df_frac = df.iloc[:total_size]
    
    logger.info("Processing flow features...")
    df_processed = process_flow_features(df_frac)
    
    # Filter data based on mode
    logger.info(f"Filtering data for {mode} mode...")
    if mode == "oneclass":
        # Keep only benign samples for one-class learning
        benign_mask = df_processed['label'].str.contains('benign|normal', case=False, na=False)
        if benign_mask.sum() == 0:
            logger.warning("No benign samples found for one-class mode!")
        else:
            df_processed = df_processed[benign_mask]
            logger.info(f"One-class mode: Keeping {benign_mask.sum()} benign samples")
    
    # Create labels based on the selected strategy
    if labeling_strategy == "per_flow":
        labels = create_per_flow_labels(df_processed)
        # For per-flow, we'll use all the data
        data_to_split = df_processed
    else:  # per_host
        logger.info("Creating per-host labels...")
        host_labels = create_per_host_labels(df_processed)
        
        # For per-host, we need to aggregate features per host
        # First, identify numeric columns only
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col not in ['id.orig_h', 'id.resp_h', 'label']]
        
        # Just a simplified feature aggregation - TODO: implement proper feature aggregation
        src_features = df_processed.groupby('id.orig_h')[numeric_cols].mean().add_prefix('src_')
        # Aggregate destination features
        dst_features = df_processed.groupby('id.resp_h')[numeric_cols].mean().add_prefix('dst_')
        
        # Combine features
        host_features = src_features.join(dst_features, how='outer').fillna(0)
        
        # Merge with labels
        data_to_split = host_features.join(host_labels)
        labels = data_to_split['is_malicious'].values
    
    # Create non-IID splits using Dirichlet distribution
    logger.info("Creating non-IID client splits...")
    client_indices = dirichlet_split_by_label(labels, n_clients, alpha=alpha, min_samples_per_client=min_samples_per_client, seed=seed)
    
    # Save client data
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, indices in enumerate(client_indices):
        if labeling_strategy == "per_flow":
            client_data = df_processed.iloc[indices]
        else:
            # For per-host, we need to handle the indices differently
            host_names = data_to_split.index[indices]
            client_data = df_processed[
                df_processed['id.orig_h'].isin(host_names) | 
                df_processed['id.resp_h'].isin(host_names)
            ]
        
        # Save client data
        torch.save({
            'data': client_data,
            'labeling_strategy': labeling_strategy,
            'client_id': i
        }, output_dir / f"client_{i}.pt")
    
    # Save dataset info
    dataset_info = {
        'labeling_strategy': labeling_strategy,
        'n_clients': n_clients,
        'alpha': alpha,
        'total_samples': len(df_processed)
    }
    torch.save(dataset_info, output_dir / "dataset_info.pt")
    
    logger.info(f"Processed data saved to {output_dir}")

def load_malicious_data(data_dir: Path, client_id: str):
    """Load malicious data for evaluation"""
    malicious_dir = data_dir / "malicious"
    if not malicious_dir.exists():
        print(f"❌ Malicious data directory not found: {malicious_dir}")
        return None
    
    file_path = malicious_dir / f"client_graph_{client_id}.pt"
    if not file_path.exists():
        return None
        
    try:
        data = torch.load(file_path, weights_only=False)
        return data
    except Exception as e:
        print(f"Error loading malicious data for client {client_id}: {e}")
        return None

def create_malicious_evaluation_loaders(malicious_graph, batch_size=32):
    """Create data loaders for malicious evaluation"""
    if malicious_graph is None:
        return None, None
    
    # Use all nodes for evaluation (no train/test split for malicious)
    malicious_graph.test_mask = torch.ones(malicious_graph.num_nodes, dtype=torch.bool)
    test_loader = DataLoader([malicious_graph], batch_size=batch_size, shuffle=False)
    
    return test_loader, test_loader  # Return same for train/test compatibility

def personalize_any_model(global_model, client_data, personalization_epochs=5, 
                         model_type='gcn_deepsvdd', personalization_lr=0.001):
    """Personalize any model type after global training"""
    
    # Create personalized copy
    personalized_model = type(global_model)(
        in_channels=128, hidden_channels=64, num_layers=2
    )
    personalized_model.load_state_dict(global_model.state_dict())
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    personalized_model.to(device)
    
    # Different optimization for different models
    if 'personalized' in model_type:
        # Only optimize personalized components
        params_to_optimize = []
        for name, param in personalized_model.named_parameters():
            if 'personalized' in name or 'head' in name:
                param.requires_grad = True
                params_to_optimize.append(param)
            else:
                param.requires_grad = False
    else:
        # Fine-tune all parameters with lower LR
        params_to_optimize = personalized_model.parameters()
    
    optimizer = torch.optim.Adam(params_to_optimize, lr=personalization_lr)
    
    # Train on client data
    for epoch in range(personalization_epochs):
        total_loss = 0
        for data in client_data:
            data = data.to(device)
            optimizer.zero_grad()
            
            if model_type in ['gcn', 'gcn_deepsvdd']:
                embeddings, _ = personalized_model(data.x, data.edge_index)
                loss = compute_deepsvdd_loss(embeddings, center)
            elif model_type in ['gae', 'gae_deepsvdd']:
                reconstructed, embeddings = personalized_model(data.x, data.edge_index)
                loss = compute_hybrid_loss(embeddings, data.x, reconstructed)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Personalization Epoch {epoch+1}: Loss = {total_loss:.4f}")
    
    return personalized_model
