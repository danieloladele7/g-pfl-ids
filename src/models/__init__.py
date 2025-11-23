# src/models/__init__.py (UPDATED)
from .base import BaseGCN
from .gcn import GCN, DeepSVDDGCN, GCNWithPersonalizedHead
from .gae import GAE, GAEDeepSVDD, GAEWithPersonalizedHead

__all__ = [
    'BaseGCN', 
    'GCN', 
    'DeepSVDDGCN', 
    'GAE', 
    'GAEDeepSVDD',
    'GCNWithPersonalizedHead',
    'GAEWithPersonalizedHead'
]
