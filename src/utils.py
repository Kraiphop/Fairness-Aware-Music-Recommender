"""
Utility functions for the Fairness-Aware Music Recommender system.
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ensure_dir(directory: str) -> None:
    """Ensure directory exists, create if it doesn't."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory: {directory}")

def save_pickle(obj: any, filepath: str) -> None:
    """Save object to pickle file."""
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)
    logger.info(f"Saved object to: {filepath}")

def load_pickle(filepath: str) -> any:
    """Load object from pickle file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    with open(filepath, 'rb') as f:
        obj = pickle.load(f)
    logger.info(f"Loaded object from: {filepath}")
    return obj

def get_project_root() -> str:
    """Get the project root directory."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_data_dir() -> str:
    """Get the data directory path."""
    return os.path.join(get_project_root(), 'data')

def get_models_dir() -> str:
    """Get the models directory path."""
    return os.path.join(get_project_root(), 'models')

def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Normalize scores to [0, 1] range."""
    if len(scores) == 0:
        return scores
    
    min_score = np.min(scores)
    max_score = np.max(scores)
    
    if max_score == min_score:
        return np.ones_like(scores)
    
    return (scores - min_score) / (max_score - min_score)

def calculate_popularity_score(play_counts: np.ndarray) -> np.ndarray:
    """Calculate popularity scores from play counts."""
    # Use log scaling to reduce the impact of extremely popular items
    return np.log1p(play_counts)

def calculate_diversity_score(recommendations: List[int], 
                            item_features: np.ndarray = None,
                            feature_similarity_matrix: np.ndarray = None) -> float:
    """
    Calculate diversity score for a list of recommendations.
    
    Args:
        recommendations: List of item indices
        item_features: Feature matrix for items (optional)
        feature_similarity_matrix: Pre-computed similarity matrix (optional)
    
    Returns:
        Diversity score (higher = more diverse)
    """
    if len(recommendations) <= 1:
        return 1.0
    
    if feature_similarity_matrix is not None:
        # Use pre-computed similarity matrix
        similarities = []
        for i in range(len(recommendations)):
            for j in range(i + 1, len(recommendations)):
                item1, item2 = recommendations[i], recommendations[j]
                if item1 < feature_similarity_matrix.shape[0] and item2 < feature_similarity_matrix.shape[1]:
                    similarities.append(feature_similarity_matrix[item1, item2])
        
        if similarities:
            avg_similarity = np.mean(similarities)
            return 1.0 - avg_similarity  # Diversity = 1 - similarity
    
    # Fallback: assume items are diverse if no similarity info
    return 1.0

def calculate_coverage_score(recommended_items: List[int], 
                           total_items: int) -> float:
    """Calculate catalog coverage for recommendations."""
    unique_items = len(set(recommended_items))
    return unique_items / total_items

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero."""
    return numerator / denominator if denominator != 0 else default

class MetricsTracker:
    """Track and compute various metrics over time."""
    
    def __init__(self):
        self.metrics = {}
    
    def add_metric(self, name: str, value: float, step: int = None) -> None:
        """Add a metric value."""
        if name not in self.metrics:
            self.metrics[name] = []
        
        if step is not None:
            self.metrics[name].append((step, value))
        else:
            self.metrics[name].append(value)
    
    def get_metric(self, name: str) -> List[float]:
        """Get all values for a metric."""
        return self.metrics.get(name, [])
    
    def get_latest_metric(self, name: str) -> float:
        """Get the latest value for a metric."""
        values = self.get_metric(name)
        return values[-1] if values else 0.0
    
    def get_metric_summary(self, name: str) -> Dict[str, float]:
        """Get summary statistics for a metric."""
        values = self.get_metric(name)
        if not values:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        
        return {
            "mean": np.mean(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values)
        }

