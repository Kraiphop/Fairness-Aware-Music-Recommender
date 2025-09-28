"""
Bias detection and fairness metrics for music recommendation.
Implements various bias detection methods including popularity bias, genre diversity, and exposure fairness.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import logging
from collections import Counter
from scipy.stats import entropy
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

class BiasDetector:
    """Detect various types of bias in music recommendations."""
    
    def __init__(self, artists_df: pd.DataFrame, interactions_df: pd.DataFrame):
        """
        Initialize bias detector with artist and interaction data.
        
        Args:
            artists_df: DataFrame with artist metadata (genre, popularity, etc.)
            interactions_df: DataFrame with user-artist interactions
        """
        self.artists_df = artists_df
        self.interactions_df = interactions_df
        
        # Create mappings
        self.artist_to_genre = dict(zip(artists_df['artist_id'], artists_df['genre']))
        self.artist_to_popularity = dict(zip(artists_df['artist_id'], artists_df['popularity']))
        
        # Calculate artist play counts
        self.artist_play_counts = interactions_df.groupby('artist_id')['play_count'].sum()
        
        # Define popularity thresholds
        self.popularity_threshold = np.percentile(list(self.artist_to_popularity.values()), 80)
        
    def detect_popularity_bias(self, recommendations: List[Tuple[str, float]], 
                             user_id: str = None) -> Dict[str, float]:
        """
        Detect popularity bias in recommendations.
        
        Args:
            recommendations: List of (artist_id, score) tuples
            user_id: User ID for context (optional)
            
        Returns:
            Dictionary with popularity bias metrics
        """
        if not recommendations:
            return {
                'popularity_bias_score': 0.0,
                'popular_artists_ratio': 0.0,
                'avg_popularity': 0.0,
                'popularity_entropy': 0.0
            }
        
        artist_ids = [rec[0] for rec in recommendations]
        popularities = [self.artist_to_popularity.get(artist_id, 0.0) for artist_id in artist_ids]
        
        # Calculate metrics
        popular_artists_ratio = sum(1 for p in popularities if p >= self.popularity_threshold) / len(popularities)
        avg_popularity = np.mean(popularities)
        
        # Popularity bias score (higher = more biased towards popular items)
        popularity_bias_score = popular_artists_ratio
        
        # Popularity entropy (higher = more diverse in popularity)
        if len(set(popularities)) > 1:
            popularity_entropy = entropy([popularities.count(p) for p in set(popularities)])
        else:
            popularity_entropy = 0.0
        
        return {
            'popularity_bias_score': popularity_bias_score,
            'popular_artists_ratio': popular_artists_ratio,
            'avg_popularity': avg_popularity,
            'popularity_entropy': popularity_entropy
        }
    
    def detect_genre_diversity_bias(self, recommendations: List[Tuple[str, float]]) -> Dict[str, float]:
        """
        Detect genre diversity bias in recommendations.
        
        Args:
            recommendations: List of (artist_id, score) tuples
            
        Returns:
            Dictionary with genre diversity metrics
        """
        if not recommendations:
            return {
                'genre_diversity_score': 0.0,
                'num_unique_genres': 0,
                'genre_entropy': 0.0,
                'genre_distribution': {},
                'dominant_genre_ratio': 0.0
            }
        
        artist_ids = [rec[0] for rec in recommendations]
        genres = [self.artist_to_genre.get(artist_id, 'Unknown') for artist_id in artist_ids]
        
        # Calculate genre distribution
        genre_counts = Counter(genres)
        genre_distribution = dict(genre_counts)
        
        # Calculate metrics
        num_unique_genres = len(set(genres))
        genre_diversity_score = num_unique_genres / len(artist_ids)
        
        # Genre entropy (higher = more diverse)
        if len(genre_counts) > 1:
            counts = list(genre_counts.values())
            genre_entropy = entropy(counts) / np.log(len(counts))  # Normalized entropy
        else:
            genre_entropy = 0.0
        
        # Dominant genre ratio
        if genre_counts:
            dominant_genre_count = max(genre_counts.values())
            dominant_genre_ratio = dominant_genre_count / len(artist_ids)
        else:
            dominant_genre_ratio = 0.0
        
        return {
            'genre_diversity_score': genre_diversity_score,
            'num_unique_genres': num_unique_genres,
            'genre_entropy': genre_entropy,
            'genre_distribution': genre_distribution,
            'dominant_genre_ratio': dominant_genre_ratio
        }
    
    def detect_exposure_fairness(self, recommendations: List[Tuple[str, float]], 
                               user_history: List[str] = None) -> Dict[str, float]:
        """
        Detect exposure fairness bias in recommendations.
        
        Args:
            recommendations: List of (artist_id, score) tuples
            user_history: List of artist IDs the user has interacted with (optional)
            
        Returns:
            Dictionary with exposure fairness metrics
        """
        if not recommendations:
            return {
                'exposure_fairness_score': 0.0,
                'long_tail_ratio': 0.0,
                'coverage_score': 0.0,
                'novelty_score': 0.0
            }
        
        artist_ids = [rec[0] for rec in recommendations]
        
        # Long-tail ratio (artists with low popularity)
        long_tail_artists = sum(1 for artist_id in artist_ids 
                               if self.artist_to_popularity.get(artist_id, 0.0) < self.popularity_threshold)
        long_tail_ratio = long_tail_artists / len(artist_ids)
        
        # Coverage score (unique artists recommended)
        coverage_score = len(set(artist_ids)) / len(artist_ids)
        
        # Novelty score (artists not in user history)
        novelty_score = 1.0
        if user_history:
            new_artists = sum(1 for artist_id in artist_ids if artist_id not in user_history)
            novelty_score = new_artists / len(artist_ids)
        
        # Overall exposure fairness score
        exposure_fairness_score = (long_tail_ratio + coverage_score + novelty_score) / 3.0
        
        return {
            'exposure_fairness_score': exposure_fairness_score,
            'long_tail_ratio': long_tail_ratio,
            'coverage_score': coverage_score,
            'novelty_score': novelty_score
        }
    
    def detect_gender_bias(self, recommendations: List[Tuple[str, float]]) -> Dict[str, float]:
        """
        Detect gender bias in recommendations (if gender information is available).
        
        Args:
            recommendations: List of (artist_id, score) tuples
            
        Returns:
            Dictionary with gender bias metrics
        """
        if 'gender' not in self.artists_df.columns:
            return {
                'gender_bias_score': 0.0,
                'gender_distribution': {},
                'gender_entropy': 0.0
            }
        
        artist_ids = [rec[0] for rec in recommendations]
        
        # Get gender information
        artist_to_gender = dict(zip(self.artists_df['artist_id'], self.artists_df['gender']))
        genders = [artist_to_gender.get(artist_id, 'Unknown') for artist_id in artist_ids]
        
        # Calculate gender distribution
        gender_counts = Counter(genders)
        gender_distribution = dict(gender_counts)
        
        # Calculate gender entropy
        if len(gender_counts) > 1:
            counts = list(gender_counts.values())
            gender_entropy = entropy(counts) / np.log(len(counts))  # Normalized entropy
        else:
            gender_entropy = 0.0
        
        # Gender bias score (higher = more balanced)
        gender_bias_score = gender_entropy
        
        return {
            'gender_bias_score': gender_bias_score,
            'gender_distribution': gender_distribution,
            'gender_entropy': gender_entropy
        }
    
    def analyze_recommendations(self, recommendations: List[Tuple[str, float]], 
                              user_id: str = None, user_history: List[str] = None) -> Dict[str, Dict]:
        """
        Comprehensive bias analysis of recommendations.
        
        Args:
            recommendations: List of (artist_id, score) tuples
            user_id: User ID for context
            user_history: List of artist IDs the user has interacted with
            
        Returns:
            Dictionary with all bias metrics
        """
        analysis = {
            'popularity_bias': self.detect_popularity_bias(recommendations, user_id),
            'genre_diversity': self.detect_genre_diversity_bias(recommendations),
            'exposure_fairness': self.detect_exposure_fairness(recommendations, user_history),
            'gender_bias': self.detect_gender_bias(recommendations)
        }
        
        return analysis
    
    def compare_bias_metrics(self, baseline_recs: List[Tuple[str, float]], 
                           fair_recs: List[Tuple[str, float]], 
                           user_id: str = None, 
                           user_history: List[str] = None) -> pd.DataFrame:
        """
        Compare bias metrics between baseline and fairness-aware recommendations.
        
        Args:
            baseline_recs: Baseline recommendations
            fair_recs: Fairness-aware recommendations
            user_id: User ID for context
            user_history: User interaction history
            
        Returns:
            DataFrame comparing bias metrics
        """
        baseline_analysis = self.analyze_recommendations(baseline_recs, user_id, user_history)
        fair_analysis = self.analyze_recommendations(fair_recs, user_id, user_history)
        
        # Flatten the nested dictionaries
        baseline_flat = {}
        fair_flat = {}
        
        for bias_type, metrics in baseline_analysis.items():
            for metric, value in metrics.items():
                baseline_flat[f"{bias_type}_{metric}"] = value
        
        for bias_type, metrics in fair_analysis.items():
            for metric, value in metrics.items():
                fair_flat[f"{bias_type}_{metric}"] = value
        
        # Create comparison DataFrame
        comparison_data = {
            'Metric': list(baseline_flat.keys()),
            'Baseline': list(baseline_flat.values()),
            'Fairness-Aware': list(fair_flat.values())
        }
        
        df = pd.DataFrame(comparison_data)
        
        # Calculate improvement
        df['Improvement'] = df['Fairness-Aware'] - df['Baseline']
        df['Improvement_%'] = (df['Improvement'] / df['Baseline'] * 100).round(2)
        
        return df

class BiasVisualizer:
    """Visualize bias metrics and distributions."""
    
    def __init__(self, bias_detector: BiasDetector):
        self.bias_detector = bias_detector
    
    def plot_popularity_distribution(self, recommendations: List[Tuple[str, float]], 
                                   title: str = "Popularity Distribution") -> plt.Figure:
        """Plot popularity distribution of recommendations."""
        if not recommendations:
            return None
        
        artist_ids = [rec[0] for rec in recommendations]
        popularities = [self.bias_detector.artist_to_popularity.get(artist_id, 0.0) 
                       for artist_id in artist_ids]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(popularities, bins=20, alpha=0.7, edgecolor='black')
        ax.axvline(self.bias_detector.popularity_threshold, color='red', 
                  linestyle='--', label='Popularity Threshold')
        ax.set_xlabel('Artist Popularity')
        ax.set_ylabel('Count')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def plot_genre_distribution(self, recommendations: List[Tuple[str, float]], 
                              title: str = "Genre Distribution") -> plt.Figure:
        """Plot genre distribution of recommendations."""
        if not recommendations:
            return None
        
        artist_ids = [rec[0] for rec in recommendations]
        genres = [self.bias_detector.artist_to_genre.get(artist_id, 'Unknown') 
                 for artist_id in artist_ids]
        
        genre_counts = Counter(genres)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        genres_list = list(genre_counts.keys())
        counts_list = list(genre_counts.values())
        
        ax.bar(genres_list, counts_list, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Genre')
        ax.set_ylabel('Count')
        ax.set_title(title)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def plot_bias_comparison(self, comparison_df: pd.DataFrame, 
                           title: str = "Bias Metrics Comparison") -> plt.Figure:
        """Plot comparison of bias metrics between baseline and fairness-aware models."""
        # Select key metrics for visualization
        key_metrics = [
            'popularity_bias_popularity_bias_score',
            'genre_diversity_genre_diversity_score',
            'exposure_fairness_exposure_fairness_score',
            'genre_diversity_genre_entropy'
        ]
        
        available_metrics = [m for m in key_metrics if m in comparison_df['Metric'].values]
        
        if not available_metrics:
            return None
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        x = np.arange(len(available_metrics))
        width = 0.35
        
        baseline_values = [comparison_df[comparison_df['Metric'] == m]['Baseline'].iloc[0] 
                          for m in available_metrics]
        fair_values = [comparison_df[comparison_df['Metric'] == m]['Fairness-Aware'].iloc[0] 
                      for m in available_metrics]
        
        ax.bar(x - width/2, baseline_values, width, label='Baseline', alpha=0.7)
        ax.bar(x + width/2, fair_values, width, label='Fairness-Aware', alpha=0.7)
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Score')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('_', ' ').title() for m in available_metrics], 
                          rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig

def analyze_model_bias(models: Dict, processed_data: Dict, 
                      sample_users: List[str] = None) -> Dict:
    """
    Analyze bias across multiple models and users.
    
    Args:
        models: Dictionary of trained models
        processed_data: Processed dataset
        sample_users: List of user IDs to analyze (optional)
        
    Returns:
        Dictionary with bias analysis results
    """
    logger.info("Analyzing model bias...")
    
    bias_detector = BiasDetector(
        processed_data['artists'], 
        processed_data['interactions']
    )
    
    if sample_users is None:
        # Select random sample of users
        np.random.seed(42)
        all_users = list(processed_data['user_to_idx'].keys())
        sample_users = np.random.choice(all_users, size=min(100, len(all_users)), replace=False)
    
    results = {}
    
    for model_name, model in models.items():
        logger.info(f"Analyzing bias for {model_name} model...")
        
        model_results = {
            'popularity_bias_scores': [],
            'genre_diversity_scores': [],
            'exposure_fairness_scores': [],
            'overall_bias_scores': []
        }
        
        for user_id in sample_users:
            try:
                recommendations = model.recommend(user_id, n_recommendations=10)
                
                # Get user history
                user_interactions = processed_data['interactions'][
                    processed_data['interactions']['user_id'] == user_id
                ]
                user_history = user_interactions['artist_id'].tolist()
                
                # Analyze bias
                analysis = bias_detector.analyze_recommendations(
                    recommendations, user_id, user_history
                )
                
                # Store results
                model_results['popularity_bias_scores'].append(
                    analysis['popularity_bias']['popularity_bias_score']
                )
                model_results['genre_diversity_scores'].append(
                    analysis['genre_diversity']['genre_diversity_score']
                )
                model_results['exposure_fairness_scores'].append(
                    analysis['exposure_fairness']['exposure_fairness_score']
                )
                
                # Overall bias score (lower is better)
                overall_bias = (
                    analysis['popularity_bias']['popularity_bias_score'] +  # Higher is worse
                    (1 - analysis['genre_diversity']['genre_diversity_score']) +  # Lower diversity is worse
                    (1 - analysis['exposure_fairness']['exposure_fairness_score'])  # Lower fairness is worse
                ) / 3.0
                
                model_results['overall_bias_scores'].append(overall_bias)
                
            except Exception as e:
                logger.warning(f"Error analyzing user {user_id}: {e}")
                continue
        
        # Calculate average metrics
        results[model_name] = {
            'avg_popularity_bias': np.mean(model_results['popularity_bias_scores']),
            'avg_genre_diversity': np.mean(model_results['genre_diversity_scores']),
            'avg_exposure_fairness': np.mean(model_results['exposure_fairness_scores']),
            'avg_overall_bias': np.mean(model_results['overall_bias_scores']),
            'detailed_results': model_results
        }
    
    return results

if __name__ == "__main__":
    from data_processing import LastFMDataProcessor
    from baseline_model import train_baseline_models
    
    # Process data and train models
    processor = LastFMDataProcessor()
    processed_data = processor.process_all()
    models = train_baseline_models(processed_data)
    
    # Analyze bias
    bias_results = analyze_model_bias(models, processed_data)
    
    print("Bias Analysis Results:")
    for model_name, results in bias_results.items():
        print(f"\n{model_name}:")
        print(f"  Average Popularity Bias: {results['avg_popularity_bias']:.3f}")
        print(f"  Average Genre Diversity: {results['avg_genre_diversity']:.3f}")
        print(f"  Average Exposure Fairness: {results['avg_exposure_fairness']:.3f}")
        print(f"  Average Overall Bias: {results['avg_overall_bias']:.3f}")

