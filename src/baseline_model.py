"""
Baseline recommender models for music recommendation.
Implements collaborative filtering using matrix factorization.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity
import logging
from tqdm import tqdm

from utils import save_pickle, load_pickle, get_models_dir, normalize_scores

logger = logging.getLogger(__name__)

class BaselineRecommender:
    """Baseline collaborative filtering recommender using matrix factorization."""
    
    def __init__(self, n_components: int = 50, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        self.model = None
        self.user_factors = None
        self.item_factors = None
        self.interaction_matrix = None
        self.user_to_idx = None
        self.item_to_idx = None
        self.idx_to_user = None
        self.idx_to_item = None
        self.is_trained = False
        
    def train(self, interaction_matrix: np.ndarray, 
              user_to_idx: Dict, item_to_idx: Dict) -> None:
        """
        Train the baseline recommender model.
        
        Args:
            interaction_matrix: User-item interaction matrix
            user_to_idx: Mapping from user IDs to matrix indices
            item_to_idx: Mapping from item IDs to matrix indices
        """
        logger.info("Training baseline recommender...")
        
        self.interaction_matrix = interaction_matrix
        self.user_to_idx = user_to_idx
        self.item_to_idx = item_to_idx
        self.idx_to_user = {idx: user for user, idx in user_to_idx.items()}
        self.idx_to_item = {idx: item for item, idx in item_to_idx.items()}
        
        # Use Non-negative Matrix Factorization
        self.model = NMF(
            n_components=self.n_components,
            random_state=self.random_state,
            max_iter=200,
            alpha_W=0.1,
            alpha_H=0.1,
            l1_ratio=0.1
        )
        
        # Fit the model
        self.user_factors = self.model.fit_transform(interaction_matrix)
        self.item_factors = self.model.components_.T
        
        self.is_trained = True
        logger.info(f"Training completed. Model shape: {self.user_factors.shape} x {self.item_factors.shape}")
    
    def predict_scores(self, user_id: str) -> np.ndarray:
        """
        Predict scores for all items for a given user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Array of predicted scores for all items
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if user_id not in self.user_to_idx:
            logger.warning(f"Unknown user: {user_id}")
            return np.zeros(len(self.item_to_idx))
        
        user_idx = self.user_to_idx[user_id]
        user_vector = self.user_factors[user_idx]
        
        # Compute dot product with all item factors
        scores = np.dot(self.item_factors, user_vector)
        
        return scores
    
    def recommend(self, user_id: str, n_recommendations: int = 10, 
                  exclude_known: bool = True) -> List[Tuple[str, float]]:
        """
        Generate recommendations for a user.
        
        Args:
            user_id: User identifier
            n_recommendations: Number of recommendations to generate
            exclude_known: Whether to exclude items the user has already interacted with
            
        Returns:
            List of (item_id, score) tuples
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making recommendations")
        
        # Get predicted scores
        scores = self.predict_scores(user_id)
        
        # Get item indices sorted by score
        item_indices = np.argsort(scores)[::-1]
        
        # Exclude known items if requested
        if exclude_known and user_id in self.user_to_idx:
            user_idx = self.user_to_idx[user_id]
            known_items = np.where(self.interaction_matrix[user_idx] > 0)[0]
            item_indices = [idx for idx in item_indices if idx not in known_items]
        
        # Get top recommendations
        recommendations = []
        for idx in item_indices[:n_recommendations]:
            item_id = self.idx_to_item[idx]
            score = scores[idx]
            recommendations.append((item_id, score))
        
        return recommendations
    
    def get_similar_items(self, item_id: str, n_similar: int = 10) -> List[Tuple[str, float]]:
        """
        Find items similar to a given item.
        
        Args:
            item_id: Item identifier
            n_similar: Number of similar items to return
            
        Returns:
            List of (item_id, similarity_score) tuples
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before finding similar items")
        
        if item_id not in self.item_to_idx:
            logger.warning(f"Unknown item: {item_id}")
            return []
        
        item_idx = self.item_to_idx[item_id]
        item_vector = self.item_factors[item_idx].reshape(1, -1)
        
        # Compute cosine similarity with all other items
        similarities = cosine_similarity(item_vector, self.item_factors)[0]
        
        # Get top similar items (excluding the item itself)
        similar_indices = np.argsort(similarities)[::-1]
        similar_items = []
        
        for idx in similar_indices:
            if idx != item_idx:  # Exclude the item itself
                similar_item_id = self.idx_to_item[idx]
                similarity = similarities[idx]
                similar_items.append((similar_item_id, similarity))
                
                if len(similar_items) >= n_similar:
                    break
        
        return similar_items
    
    def save_model(self, filepath: str = None) -> None:
        """Save the trained model."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        if filepath is None:
            models_dir = get_models_dir()
            filepath = f"{models_dir}/baseline_model.pkl"
        
        model_data = {
            'user_factors': self.user_factors,
            'item_factors': self.item_factors,
            'user_to_idx': self.user_to_idx,
            'item_to_idx': self.item_to_idx,
            'idx_to_user': self.idx_to_user,
            'idx_to_item': self.idx_to_item,
            'n_components': self.n_components,
            'interaction_matrix': self.interaction_matrix
        }
        
        save_pickle(model_data, filepath)
        logger.info(f"Model saved to: {filepath}")
    
    def load_model(self, filepath: str = None) -> None:
        """Load a trained model."""
        if filepath is None:
            models_dir = get_models_dir()
            filepath = f"{models_dir}/baseline_model.pkl"
        
        model_data = load_pickle(filepath)
        
        self.user_factors = model_data['user_factors']
        self.item_factors = model_data['item_factors']
        self.user_to_idx = model_data['user_to_idx']
        self.item_to_idx = model_data['item_to_idx']
        self.idx_to_user = model_data['idx_to_user']
        self.idx_to_item = model_data['idx_to_item']
        self.n_components = model_data['n_components']
        self.interaction_matrix = model_data['interaction_matrix']
        
        self.is_trained = True
        logger.info(f"Model loaded from: {filepath}")

class PopularityRecommender:
    """Simple popularity-based recommender for comparison."""
    
    def __init__(self):
        self.item_popularity = None
        self.item_to_idx = None
        self.idx_to_item = None
        self.is_trained = False
    
    def train(self, interaction_matrix: np.ndarray, 
              item_to_idx: Dict) -> None:
        """Train the popularity-based model."""
        logger.info("Training popularity-based recommender...")
        
        self.item_to_idx = item_to_idx
        self.idx_to_item = {idx: item for item, idx in item_to_idx.items()}
        
        # Calculate item popularity (total play counts)
        self.item_popularity = np.sum(interaction_matrix, axis=0)
        
        self.is_trained = True
        logger.info("Popularity-based model trained")
    
    def recommend(self, user_id: str, n_recommendations: int = 10,
                  exclude_known: bool = True) -> List[Tuple[str, float]]:
        """Generate popularity-based recommendations."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making recommendations")
        
        # Get items sorted by popularity
        popular_items = np.argsort(self.item_popularity)[::-1]
        
        recommendations = []
        for idx in popular_items[:n_recommendations]:
            item_id = self.idx_to_item[idx]
            popularity = self.item_popularity[idx]
            recommendations.append((item_id, popularity))
        
        return recommendations

class RandomRecommender:
    """Random recommender for baseline comparison."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.item_to_idx = None
        self.idx_to_item = None
        self.is_trained = False
        np.random.seed(random_state)
    
    def train(self, interaction_matrix: np.ndarray, 
              item_to_idx: Dict) -> None:
        """Train the random model (just store mappings)."""
        self.item_to_idx = item_to_idx
        self.idx_to_item = {idx: item for item, idx in item_to_idx.items()}
        self.is_trained = True
        logger.info("Random model trained")
    
    def recommend(self, user_id: str, n_recommendations: int = 10,
                  exclude_known: bool = True) -> List[Tuple[str, float]]:
        """Generate random recommendations."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making recommendations")
        
        n_items = len(self.item_to_idx)
        random_indices = np.random.choice(n_items, size=n_recommendations, replace=False)
        
        recommendations = []
        for idx in random_indices:
            item_id = self.idx_to_item[idx]
            random_score = np.random.random()
            recommendations.append((item_id, random_score))
        
        return recommendations

def train_baseline_models(processed_data: Dict) -> Dict[str, object]:
    """Train all baseline models."""
    logger.info("Training baseline models...")
    
    models = {}
    
    # Train baseline collaborative filtering model
    baseline_model = BaselineRecommender(n_components=50)
    baseline_model.train(
        processed_data['interaction_matrix'],
        processed_data['user_to_idx'],
        processed_data['artist_to_idx']
    )
    baseline_model.save_model()
    models['baseline'] = baseline_model
    
    # Train popularity-based model
    popularity_model = PopularityRecommender()
    popularity_model.train(
        processed_data['interaction_matrix'],
        processed_data['artist_to_idx']
    )
    models['popularity'] = popularity_model
    
    # Train random model
    random_model = RandomRecommender()
    random_model.train(
        processed_data['interaction_matrix'],
        processed_data['artist_to_idx']
    )
    models['random'] = random_model
    
    logger.info("All baseline models trained successfully")
    return models

if __name__ == "__main__":
    from data_processing import LastFMDataProcessor
    
    # Process data and train models
    processor = LastFMDataProcessor()
    processed_data = processor.process_all()
    models = train_baseline_models(processed_data)
    
    # Test recommendations
    test_user = list(processed_data['user_to_idx'].keys())[0]
    baseline_recs = models['baseline'].recommend(test_user, n_recommendations=5)
    popularity_recs = models['popularity'].recommend(test_user, n_recommendations=5)
    
    print(f"Baseline recommendations for {test_user}: {baseline_recs}")
    print(f"Popularity recommendations for {test_user}: {popularity_recs}")
