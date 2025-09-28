"""
Fairness-aware re-ranking methods for music recommendation.
Implements MMR (Maximal Marginal Relevance) and constrained re-ranking approaches.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional, Callable
import logging
from collections import Counter, defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import minimize

from utils import normalize_scores

logger = logging.getLogger(__name__)

class FairnessAwareReranker:
    """Fairness-aware re-ranking system using various approaches."""
    
    def __init__(self, artists_df: pd.DataFrame, bias_detector=None):
        """
        Initialize fairness-aware re-ranker.
        
        Args:
            artists_df: DataFrame with artist metadata
            bias_detector: BiasDetector instance for analyzing recommendations
        """
        self.artists_df = artists_df
        
        # Create mappings
        self.artist_to_genre = dict(zip(artists_df['artist_id'], artists_df['genre']))
        self.artist_to_popularity = dict(zip(artists_df['artist_id'], artists_df['popularity']))
        
        # Genre information
        self.unique_genres = artists_df['genre'].unique().tolist()
        self.genre_to_idx = {genre: idx for idx, genre in enumerate(self.unique_genres)}
        
        # Popularity thresholds
        self.popularity_threshold = np.percentile(artists_df['popularity'], 80)
        
        self.bias_detector = bias_detector
    
    def mmr_rerank(self, recommendations: List[Tuple[str, float]], 
                   diversity_weight: float = 0.3, 
                   n_recommendations: int = 10) -> List[Tuple[str, float]]:
        """
        Re-rank recommendations using Maximal Marginal Relevance (MMR).
        
        Args:
            recommendations: List of (artist_id, score) tuples
            diversity_weight: Weight for diversity (0 = relevance only, 1 = diversity only)
            n_recommendations: Number of final recommendations
            
        Returns:
            Re-ranked list of (artist_id, score) tuples
        """
        if len(recommendations) <= n_recommendations:
            return recommendations
        
        # Extract artist IDs and scores
        artist_ids = [rec[0] for rec in recommendations]
        scores = np.array([rec[1] for rec in recommendations])
        
        # Normalize scores
        scores = normalize_scores(scores)
        
        # Create genre similarity matrix
        genre_similarity = self._create_genre_similarity_matrix(artist_ids)
        
        # MMR algorithm
        selected = []
        remaining = list(range(len(artist_ids)))
        
        # Select first item (highest relevance)
        best_idx = np.argmax(scores)
        selected.append(best_idx)
        remaining.remove(best_idx)
        
        while len(selected) < n_recommendations and remaining:
            best_mmr_score = -np.inf
            best_idx = None
            
            for idx in remaining:
                # Relevance score
                relevance = scores[idx]
                
                # Diversity score (1 - max similarity to selected items)
                max_similarity = 0.0
                for sel_idx in selected:
                    similarity = genre_similarity[idx, sel_idx]
                    max_similarity = max(max_similarity, similarity)
                
                diversity = 1.0 - max_similarity
                
                # MMR score
                mmr_score = (1 - diversity_weight) * relevance + diversity_weight * diversity
                
                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_idx = idx
            
            if best_idx is not None:
                selected.append(best_idx)
                remaining.remove(best_idx)
            else:
                break
        
        # Return re-ranked recommendations
        reranked = [(artist_ids[idx], scores[idx]) for idx in selected]
        return reranked
    
    def constrained_rerank(self, recommendations: List[Tuple[str, float]], 
                          genre_constraints: Dict[str, int] = None,
                          popularity_constraints: Dict[str, float] = None,
                          n_recommendations: int = 10) -> List[Tuple[str, float]]:
        """
        Re-rank recommendations with fairness constraints.
        
        Args:
            recommendations: List of (artist_id, score) tuples
            genre_constraints: Dict mapping genre to minimum number of artists
            popularity_constraints: Dict with 'min_long_tail' and 'max_popular' ratios
            n_recommendations: Number of final recommendations
            
        Returns:
            Re-ranked list of (artist_id, score) tuples
        """
        if len(recommendations) <= n_recommendations:
            return recommendations
        
        # Default constraints
        if genre_constraints is None:
            genre_constraints = {}
        
        if popularity_constraints is None:
            popularity_constraints = {
                'min_long_tail': 0.3,  # At least 30% long-tail artists
                'max_popular': 0.7     # At most 70% popular artists
            }
        
        # Extract data
        artist_ids = [rec[0] for rec in recommendations]
        scores = np.array([rec[1] for rec in recommendations])
        
        # Normalize scores
        scores = normalize_scores(scores)
        
        # Solve constrained optimization problem
        selected_indices = self._solve_constrained_selection(
            artist_ids, scores, genre_constraints, popularity_constraints, n_recommendations
        )
        
        # Return re-ranked recommendations
        reranked = [(artist_ids[idx], scores[idx]) for idx in selected_indices]
        return reranked
    
    def diversity_boost_rerank(self, recommendations: List[Tuple[str, float]], 
                              diversity_boost: float = 0.5,
                              n_recommendations: int = 10) -> List[Tuple[str, float]]:
        """
        Re-rank recommendations by boosting diversity scores.
        
        Args:
            recommendations: List of (artist_id, score) tuples
            diversity_boost: Factor to boost diversity (0 = no boost, 1 = full boost)
            n_recommendations: Number of final recommendations
            
        Returns:
            Re-ranked list of (artist_id, score) tuples
        """
        if len(recommendations) <= n_recommendations:
            return recommendations
        
        # Extract data
        artist_ids = [rec[0] for rec in recommendations]
        scores = np.array([rec[1] for rec in recommendations])
        
        # Calculate diversity scores
        diversity_scores = self._calculate_diversity_scores(artist_ids)
        
        # Combine relevance and diversity
        combined_scores = (1 - diversity_boost) * scores + diversity_boost * diversity_scores
        
        # Select top items
        top_indices = np.argsort(combined_scores)[::-1][:n_recommendations]
        
        # Return re-ranked recommendations
        reranked = [(artist_ids[idx], scores[idx]) for idx in top_indices]
        return reranked
    
    def genre_balanced_rerank(self, recommendations: List[Tuple[str, float]], 
                            balance_weight: float = 0.4,
                            n_recommendations: int = 10) -> List[Tuple[str, float]]:
        """
        Re-rank recommendations to balance genre representation.
        
        Args:
            recommendations: List of (artist_id, score) tuples
            balance_weight: Weight for genre balance (0 = no balance, 1 = full balance)
            n_recommendations: Number of final recommendations
            
        Returns:
            Re-ranked list of (artist_id, score) tuples
        """
        if len(recommendations) <= n_recommendations:
            return recommendations
        
        # Extract data
        artist_ids = [rec[0] for rec in recommendations]
        scores = np.array([rec[1] for rec in recommendations])
        
        # Group by genre
        genre_groups = defaultdict(list)
        for i, artist_id in enumerate(artist_ids):
            genre = self.artist_to_genre.get(artist_id, 'Unknown')
            genre_groups[genre].append((i, scores[i]))
        
        # Calculate target number per genre
        n_genres = len(genre_groups)
        target_per_genre = max(1, n_recommendations // n_genres)
        
        # Select items from each genre
        selected = []
        for genre, items in genre_groups.items():
            # Sort by score within genre
            items.sort(key=lambda x: x[1], reverse=True)
            
            # Select items from this genre
            n_select = min(target_per_genre, len(items))
            for i in range(n_select):
                selected.append((items[i][0], items[i][1], genre))
        
        # Sort selected items by score
        selected.sort(key=lambda x: x[1], reverse=True)
        
        # Take top n_recommendations
        top_selected = selected[:n_recommendations]
        
        # Return re-ranked recommendations
        reranked = [(artist_ids[idx], score) for idx, score, _ in top_selected]
        return reranked
    
    def _create_genre_similarity_matrix(self, artist_ids: List[str]) -> np.ndarray:
        """Create similarity matrix based on genres."""
        n = len(artist_ids)
        similarity_matrix = np.zeros((n, n))
        
        for i, artist_id_i in enumerate(artist_ids):
            genre_i = self.artist_to_genre.get(artist_id_i, 'Unknown')
            for j, artist_id_j in enumerate(artist_ids):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    genre_j = self.artist_to_genre.get(artist_id_j, 'Unknown')
                    # Binary similarity: 1 if same genre, 0 if different
                    similarity_matrix[i, j] = 1.0 if genre_i == genre_j else 0.0
        
        return similarity_matrix
    
    def _calculate_diversity_scores(self, artist_ids: List[str]) -> np.ndarray:
        """Calculate diversity scores for artists."""
        n = len(artist_ids)
        diversity_scores = np.zeros(n)
        
        # Count genres
        genre_counts = Counter([self.artist_to_genre.get(artist_id, 'Unknown') 
                               for artist_id in artist_ids])
        
        for i, artist_id in enumerate(artist_ids):
            genre = self.artist_to_genre.get(artist_id, 'Unknown')
            
            # Diversity score is inverse of genre frequency
            genre_frequency = genre_counts[genre] / n
            diversity_scores[i] = 1.0 / (1.0 + genre_frequency)
        
        return normalize_scores(diversity_scores)
    
    def _solve_constrained_selection(self, artist_ids: List[str], scores: np.ndarray,
                                   genre_constraints: Dict[str, int],
                                   popularity_constraints: Dict[str, float],
                                   n_recommendations: int) -> List[int]:
        """Solve constrained selection problem using greedy approach."""
        
        # Create genre and popularity mappings
        artist_genres = [self.artist_to_genre.get(artist_id, 'Unknown') 
                        for artist_id in artist_ids]
        artist_popularities = [self.artist_to_popularity.get(artist_id, 0.0) 
                             for artist_id in artist_ids]
        
        # Track selections
        selected = []
        remaining = list(range(len(artist_ids)))
        
        # Sort by score
        sorted_indices = sorted(remaining, key=lambda x: scores[x], reverse=True)
        
        # Greedy selection with constraints
        for idx in sorted_indices:
            if len(selected) >= n_recommendations:
                break
            
            # Check genre constraints
            genre = artist_genres[idx]
            current_genre_count = sum(1 for sel_idx in selected 
                                    if artist_genres[sel_idx] == genre)
            
            if genre in genre_constraints:
                if current_genre_count >= genre_constraints[genre]:
                    continue
            
            # Check popularity constraints
            if popularity_constraints:
                current_popular_count = sum(1 for sel_idx in selected 
                                          if artist_popularities[sel_idx] >= self.popularity_threshold)
                current_long_tail_count = len(selected) - current_popular_count
                
                # Check max popular constraint
                if 'max_popular' in popularity_constraints:
                    max_popular = int(n_recommendations * popularity_constraints['max_popular'])
                    if artist_popularities[idx] >= self.popularity_threshold and current_popular_count >= max_popular:
                        continue
                
                # Check min long-tail constraint
                if 'min_long_tail' in popularity_constraints:
                    min_long_tail = int(n_recommendations * popularity_constraints['min_long_tail'])
                    if artist_popularities[idx] < self.popularity_threshold and current_long_tail_count >= min_long_tail:
                        continue
            
            selected.append(idx)
        
        # Fill remaining slots if needed
        while len(selected) < n_recommendations and remaining:
            for idx in sorted_indices:
                if idx not in selected:
                    selected.append(idx)
                    break
        
        return selected[:n_recommendations]

class HybridReranker:
    """Hybrid re-ranker combining multiple fairness approaches."""
    
    def __init__(self, artists_df: pd.DataFrame, bias_detector=None):
        self.artists_df = artists_df
        self.bias_detector = bias_detector
        self.base_reranker = FairnessAwareReranker(artists_df, bias_detector)
    
    def adaptive_rerank(self, recommendations: List[Tuple[str, float]], 
                       user_id: str = None,
                       user_history: List[str] = None,
                       n_recommendations: int = 10) -> List[Tuple[str, float]]:
        """
        Adaptively re-rank based on user's bias profile.
        
        Args:
            recommendations: List of (artist_id, score) tuples
            user_id: User ID for context
            user_history: User interaction history
            n_recommendations: Number of final recommendations
            
        Returns:
            Re-ranked list of (artist_id, score) tuples
        """
        if not self.bias_detector:
            # Fallback to MMR if no bias detector
            return self.base_reranker.mmr_rerank(recommendations, 0.3, n_recommendations)
        
        # Analyze user's historical bias
        user_bias_profile = self._analyze_user_bias(user_history)
        
        # Choose re-ranking strategy based on bias profile
        if user_bias_profile['popularity_bias'] > 0.7:
            # High popularity bias -> use diversity boost
            return self.base_reranker.diversity_boost_rerank(
                recommendations, diversity_boost=0.6, n_recommendations=n_recommendations
            )
        elif user_bias_profile['genre_diversity'] < 0.3:
            # Low genre diversity -> use genre balancing
            return self.base_reranker.genre_balanced_rerank(
                recommendations, balance_weight=0.5, n_recommendations=n_recommendations
            )
        else:
            # Balanced user -> use constrained re-ranking
            genre_constraints = {'Rock': 2, 'Pop': 2}  # Example constraints
            popularity_constraints = {'min_long_tail': 0.4, 'max_popular': 0.6}
            
            return self.base_reranker.constrained_rerank(
                recommendations, genre_constraints, popularity_constraints, n_recommendations
            )
    
    def _analyze_user_bias(self, user_history: List[str]) -> Dict[str, float]:
        """Analyze user's historical bias patterns."""
        if not user_history or len(user_history) < 5:
            return {'popularity_bias': 0.5, 'genre_diversity': 0.5}
        
        # Analyze popularity bias
        popularities = [self.base_reranker.artist_to_popularity.get(artist_id, 0.0) 
                       for artist_id in user_history]
        popularity_bias = np.mean([1.0 if p >= self.base_reranker.popularity_threshold 
                                 else 0.0 for p in popularities])
        
        # Analyze genre diversity
        genres = [self.base_reranker.artist_to_genre.get(artist_id, 'Unknown') 
                 for artist_id in user_history]
        unique_genres = len(set(genres))
        genre_diversity = unique_genres / len(genres)
        
        return {
            'popularity_bias': popularity_bias,
            'genre_diversity': genre_diversity
        }

def create_fairness_reranker(artists_df: pd.DataFrame, 
                           method: str = 'mmr',
                           bias_detector=None) -> FairnessAwareReranker:
    """
    Factory function to create fairness-aware re-ranker.
    
    Args:
        artists_df: Artist metadata DataFrame
        method: Re-ranking method ('mmr', 'constrained', 'diversity_boost', 'genre_balanced', 'hybrid')
        bias_detector: BiasDetector instance
        
    Returns:
        Configured re-ranker instance
    """
    if method == 'hybrid':
        return HybridReranker(artists_df, bias_detector)
    else:
        return FairnessAwareReranker(artists_df, bias_detector)

if __name__ == "__main__":
    from data_processing import LastFMDataProcessor
    from baseline_model import BaselineRecommender
    from bias_detection import BiasDetector
    
    # Load data and models
    processor = LastFMDataProcessor()
    processed_data = processor.process_all()
    
    # Create bias detector
    bias_detector = BiasDetector(processed_data['artists'], processed_data['interactions'])
    
    # Create fairness re-ranker
    reranker = FairnessAwareReranker(processed_data['artists'], bias_detector)
    
    # Train baseline model
    baseline_model = BaselineRecommender()
    baseline_model.train(
        processed_data['interaction_matrix'],
        processed_data['user_to_idx'],
        processed_data['artist_to_idx']
    )
    
    # Test user
    test_user = list(processed_data['user_to_idx'].keys())[0]
    
    # Get baseline recommendations
    baseline_recs = baseline_model.recommend(test_user, n_recommendations=20)
    
    # Apply different re-ranking methods
    mmr_recs = reranker.mmr_rerank(baseline_recs, diversity_weight=0.3)
    constrained_recs = reranker.constrained_rerank(baseline_recs)
    diversity_recs = reranker.diversity_boost_rerank(baseline_recs, diversity_boost=0.5)
    
    print(f"Baseline recommendations: {baseline_recs[:5]}")
    print(f"MMR re-ranked: {mmr_recs[:5]}")
    print(f"Constrained re-ranked: {constrained_recs[:5]}")
    print(f"Diversity boost re-ranked: {diversity_recs[:5]}")
    
    # Analyze bias
    baseline_analysis = bias_detector.analyze_recommendations(baseline_recs[:10])
    mmr_analysis = bias_detector.analyze_recommendations(mmr_recs[:10])
    
    print(f"\nBaseline bias analysis: {baseline_analysis['popularity_bias']['popularity_bias_score']:.3f}")
    print(f"MMR bias analysis: {mmr_analysis['popularity_bias']['popularity_bias_score']:.3f}")

