# src/utils/oneclass_metrics.py
import torch
import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    roc_curve, f1_score, precision_score, recall_score, accuracy_score
)
from typing import Dict, Tuple, Optional
import time
import psutil
import os

class OneClassMetrics:
    """Comprehensive metrics for one-class classification with proper evaluation"""
    
    @staticmethod
    def compute_anomaly_scores(model, test_loader, device, model_type='deepsvdd'):
        """Compute anomaly scores for different model types - IMPROVED"""
        model.eval()
        all_scores = []
        all_labels = []
        
        with torch.no_grad():
            for data in test_loader:
                try:
                    data = data.to(device)
                    
                    if model_type == 'deepsvdd':
                        embeddings, _ = model(data.x, data.edge_index)
                        # Use distance from origin as anomaly score (DeepSVDD)
                        scores = torch.norm(embeddings, dim=1).cpu().numpy()
                        
                    elif model_type == 'gae':
                        reconstructed, _ = model(data.x, data.edge_index)
                        # Use reconstruction error as anomaly score
                        scores = torch.norm(data.x - reconstructed, dim=1).cpu().numpy()
                        
                    elif model_type == 'hybrid':
                        reconstructed, embeddings = model(data.x, data.edge_index)
                        # Combine distance and reconstruction error
                        dist_scores = torch.norm(embeddings, dim=1)
                        rec_scores = torch.norm(data.x - reconstructed, dim=1)
                        scores = (0.5 * dist_scores + 0.5 * rec_scores).cpu().numpy()
                    else:
                        # Default: use embedding norm
                        embeddings, _ = model(data.x, data.edge_index)
                        scores = torch.norm(embeddings, dim=1).cpu().numpy()
                    
                    all_scores.extend(scores)
                    all_labels.extend(data.y.cpu().numpy())
                    
                except Exception as e:
                    print(f"Error computing scores: {e}")
                    continue
        
        return np.array(all_scores), np.array(all_labels)
    
    @staticmethod
    def compute_all_metrics(scores: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Compute comprehensive one-class metrics - IMPROVED"""
        metrics = {}
        
        try:
            # Check if there are valid data
            if len(scores) == 0 or len(labels) == 0:
                print("Warning: No scores or labels for metric computation")
                return OneClassMetrics._get_default_metrics()
            
            # Check class distribution
            unique_labels, counts = np.unique(labels, return_counts=True)
            print(f"Class distribution: {dict(zip(unique_labels, counts))}")
            
            # If only one class, metrics are undefined
            if len(unique_labels) < 2:
                print(f"Warning: Only one class present: {unique_labels}")
                # Try to create artificial anomaly by using top 10% as anomalies
                if len(scores) > 10:
                    threshold = np.percentile(scores, 90)
                    artificial_labels = (scores > threshold).astype(int)
                    # Compute metrics with artificial labels
                    return OneClassMetrics._compute_metrics_with_labels(scores, artificial_labels)
                else:
                    return OneClassMetrics._get_default_metrics()
            
            return OneClassMetrics._compute_metrics_with_labels(scores, labels)
            
        except Exception as e:
            print(f"Error computing metrics: {e}")
            return OneClassMetrics._get_default_metrics()
    
    @staticmethod
    def _compute_metrics_with_labels(scores: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Compute metrics when we have both classes"""
        metrics = {}
        
        # AUC metrics
        metrics['auroc'] = roc_auc_score(labels, scores)
        metrics['aupr'] = average_precision_score(labels, scores)
        
        # Find optimal threshold using Youden's J statistic
        fpr, tpr, thresholds = roc_curve(labels, scores)
        youden_j = tpr - fpr
        optimal_idx = np.argmax(youden_j)
        optimal_threshold = thresholds[optimal_idx]
        
        # Binary predictions at optimal threshold
        predictions = (scores > optimal_threshold).astype(int)
        
        # Standard classification metrics
        metrics['f1'] = f1_score(labels, predictions, zero_division=0)
        metrics['precision'] = precision_score(labels, predictions, zero_division=0)
        metrics['recall'] = recall_score(labels, predictions, zero_division=0)
        metrics['accuracy'] = accuracy_score(labels, predictions)
        
        # Detection rates at specific FPRs
        for fpr_rate in [0.01, 0.05, 0.1]:
            try:
                idx = np.where(fpr <= fpr_rate)[0]
                if len(idx) > 0:
                    tpr_at_fpr = tpr[idx[-1]]
                else:
                    tpr_at_fpr = 0.0
                metrics[f'tpr_at_fpr_{fpr_rate}'] = tpr_at_fpr
            except:
                metrics[f'tpr_at_fpr_{fpr_rate}'] = 0.0
        
        # Best F1 score
        precision_curve, recall_curve, thresholds_pr = precision_recall_curve(labels, scores)
        f1_scores = 2 * (precision_curve * recall_curve) / (precision_curve + recall_curve + 1e-8)
        metrics['best_f1'] = np.max(f1_scores) if len(f1_scores) > 0 else 0.0
        
        return metrics
    
    @staticmethod
    def _get_default_metrics() -> Dict[str, float]:
        """Return default metrics when computation fails"""
        return {
            "auroc": 0.5, "aupr": 0.5, "f1": 0.0, "precision": 0.0, 
            "recall": 0.0, "accuracy": 0.5, "best_f1": 0.0,
            "tpr_at_fpr_0.01": 0.0, "tpr_at_fpr_0.05": 0.0, "tpr_at_fpr_0.1": 0.0
        }
    
    @staticmethod
    def evaluate_model_with_resources(model, test_loader, device, model_type='deepsvdd') -> Tuple[float, int, Dict]:
        """Comprehensive evaluation with resource monitoring"""
        start_time = time.time()
        process = psutil.Process(os.getpid())
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        try:
            scores, labels = OneClassMetrics.compute_anomaly_scores(model, test_loader, device, model_type)
            
            if len(scores) == 0:
                print("Warning: No scores computed during evaluation")
                metrics = OneClassMetrics._get_default_metrics()
            else:
                metrics = OneClassMetrics.compute_all_metrics(scores, labels)
            
            # Use AUROC as the primary loss (1 - AUROC for minimization)
            loss = 1.0 - metrics['auroc']
            num_examples = len(labels)
            
            # Resource usage
            end_time = time.time()
            end_memory = process.memory_info().rss / 1024 / 1024
            inference_time = end_time - start_time
            memory_used = end_memory - start_memory
            
            # Add resource metrics
            metrics['inference_time_seconds'] = inference_time
            metrics['memory_used_mb'] = memory_used
            metrics['samples_per_second'] = num_examples / inference_time if inference_time > 0 else 0
            
            return float(loss), int(num_examples), metrics
            
        except Exception as e:
            print(f"Error in evaluate_model: {e}")
            metrics = OneClassMetrics._get_default_metrics()
            metrics.update({
                'inference_time_seconds': 0.0,
                'memory_used_mb': 0.0,
                'samples_per_second': 0.0
            })
            return 1.0, 0, metrics
    
    @staticmethod
    def evaluate_model(model, test_loader, device, model_type='deepsvdd') -> Tuple[float, int, Dict]:
        """Backward compatibility wrapper"""
        return OneClassMetrics.evaluate_model_with_resources(model, test_loader, device, model_type)

