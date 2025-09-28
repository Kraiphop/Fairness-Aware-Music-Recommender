"""
Evaluation metrics for music recommendation systems.
Implements accuracy metrics (Precision@k, Recall@k, NDCG@k) and fairness metrics.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import logging
from collections import defaultdict
from sklearn.metrics import ndcg_score
from scipy.stats import entropy

logger = logging.getLogger(__name__)

class RecommendationEvaluator:
    """Comprehensive evaluation of recommendation systems."""
    
    def __init__(self, artists_df: pd.DataFrame, interactions_df: pd.DataFrame):
        """
        Initialize evaluator with artist and interaction data.
        
        Args:
            artists_df: DataFrame with artist metadata
            interactions_df: DataFrame with user-artist interactions
        """
        self.artists_df = artists_df
        self.interactions_df = interactions_df
        
        # Create mappings
        self.artist_to_genre = dict(zip(artists_df['artist_id'], artists_df['genre']))
        self.artist_to_popularity = dict(zip(artists_df['artist_id'], artists_df['popularity']))
        
        # Calculate popularity threshold
        self.popularity_threshold = np.percentile(artists_df['popularity'], 80)
        
        # Create user-item interaction matrix for evaluation
        self._create_interaction_matrix()
    
    def _create_interaction_matrix(self):
        """Create binary interaction matrix for evaluation."""
        # Create user-item pairs with binary ratings (1 if interacted, 0 if not)
        self.user_item_matrix = self.interactions_df.pivot_table(
            index='user_id', 
            columns='artist_id', 
            values='play_count', 
            fill_value=0
        )
        
        # Convert to binary (1 if play_count > 0, 0 otherwise)
        self.user_item_matrix = (self.user_item_matrix > 0).astype(int)
        
        self.user_ids = self.user_item_matrix.index.tolist()
        self.item_ids = self.user_item_matrix.columns.tolist()
    
    def precision_at_k(self, recommendations: List[str], user_id: str, k: int = 10) -> float:
        """
        Calculate Precision@k for a user.
        
        Args:
            recommendations: List of recommended item IDs
            user_id: User ID
            k: Number of top recommendations to consider
            
        Returns:
            Precision@k score
        """
        if user_id not in self.user_item_matrix.index:
            return 0.0
        
        # Get user's actual interactions
        user_interactions = self.user_item_matrix.loc[user_id]
        relevant_items = set(user_interactions[user_interactions == 1].index)
        
        # Get top-k recommendations
        top_k_recs = recommendations[:k]
        
        # Calculate precision
        relevant_recommended = len(set(top_k_recs) & relevant_items)
        return relevant_recommended / k if k > 0 else 0.0
    
    def recall_at_k(self, recommendations: List[str], user_id: str, k: int = 10) -> float:
        """
        Calculate Recall@k for a user.
        
        Args:
            recommendations: List of recommended item IDs
            user_id: User ID
            k: Number of top recommendations to consider
            
        Returns:
            Recall@k score
        """
        if user_id not in self.user_item_matrix.index:
            return 0.0
        
        # Get user's actual interactions
        user_interactions = self.user_item_matrix.loc[user_id]
        relevant_items = set(user_interactions[user_interactions == 1].index)
        
        if len(relevant_items) == 0:
            return 0.0
        
        # Get top-k recommendations
        top_k_recs = recommendations[:k]
        
        # Calculate recall
        relevant_recommended = len(set(top_k_recs) & relevant_items)
        return relevant_recommended / len(relevant_items)
    
    def ndcg_at_k(self, recommendations: List[str], user_id: str, k: int = 10) -> float:
        """
        Calculate NDCG@k for a user.
        
        Args:
            recommendations: List of recommended item IDs
            user_id: User ID
            k: Number of top recommendations to consider
            
        Returns:
            NDCG@k score
        """
        if user_id not in self.user_item_matrix.index:
            return 0.0
        
        # Get user's actual interactions
        user_interactions = self.user_item_matrix.loc[user_id]
        relevant_items = user_interactions[user_interactions == 1]
        
        if len(relevant_items) == 0:
            return 0.0
        
        # Create relevance scores (1 for relevant items, 0 for others)
        y_true = np.zeros(len(recommendations))
        for i, item_id in enumerate(recommendations):
            if item_id in relevant_items.index:
                y_true[i] = 1.0
        
        # Calculate NDCG
        y_score = np.ones(len(recommendations))  # Assume all recommendations have equal relevance
        
        if len(y_true) > 0:
            return ndcg_score([y_true], [y_score], k=k)
        else:
            return 0.0
    
    def coverage(self, recommendations: List[str], total_items: int) -> float:
        """
        Calculate catalog coverage.
        
        Args:
            recommendations: List of recommended item IDs
            total_items: Total number of items in catalog
            
        Returns:
            Coverage score
        """
        unique_items = len(set(recommendations))
        return unique_items / total_items if total_items > 0 else 0.0
    
    def intra_list_diversity(self, recommendations: List[str]) -> float:
        """
        Calculate intra-list diversity based on genres.
        
        Args:
            recommendations: List of recommended item IDs
            
        Returns:
            Intra-list diversity score
        """
        if len(recommendations) <= 1:
            return 0.0
        
        # Get genres for recommendations
        genres = [self.artist_to_genre.get(item_id, 'Unknown') for item_id in recommendations]
        
        # Calculate genre distribution
        genre_counts = defaultdict(int)
        for genre in genres:
            genre_counts[genre] += 1
        
        # Calculate entropy (higher entropy = more diverse)
        if len(genre_counts) > 1:
            counts = list(genre_counts.values())
            total = sum(counts)
            probabilities = [count / total for count in counts]
            return entropy(probabilities) / np.log(len(probabilities))  # Normalized entropy
        else:
            return 0.0
    
    def popularity_bias_score(self, recommendations: List[str]) -> float:
        """
        Calculate popularity bias score.
        
        Args:
            recommendations: List of recommended item IDs
            
        Returns:
            Popularity bias score (higher = more biased towards popular items)
        """
        if not recommendations:
            return 0.0
        
        popularities = [self.artist_to_popularity.get(item_id, 0.0) for item_id in recommendations]
        popular_items = sum(1 for p in popularities if p >= self.popularity_threshold)
        
        return popular_items / len(recommendations)
    
    def genre_diversity_score(self, recommendations: List[str]) -> float:
        """
        Calculate genre diversity score.
        
        Args:
            recommendations: List of recommended item IDs
            
        Returns:
            Genre diversity score (higher = more diverse)
        """
        if not recommendations:
            return 0.0
        
        genres = [self.artist_to_genre.get(item_id, 'Unknown') for item_id in recommendations]
        unique_genres = len(set(genres))
        
        return unique_genres / len(recommendations)
    
    def evaluate_user(self, recommendations: List[str], user_id: str, 
                     k_values: List[int] = [5, 10, 20]) -> Dict[str, float]:
        """
        Evaluate recommendations for a single user.
        
        Args:
            recommendations: List of recommended item IDs
            user_id: User ID
            k_values: List of k values for precision/recall/ndcg
            
        Returns:
            Dictionary with evaluation metrics
        """
        metrics = {}
        
        # Accuracy metrics
        for k in k_values:
            metrics[f'precision@{k}'] = self.precision_at_k(recommendations, user_id, k)
            metrics[f'recall@{k}'] = self.recall_at_k(recommendations, user_id, k)
            metrics[f'ndcg@{k}'] = self.ndcg_at_k(recommendations, user_id, k)
        
        # Diversity and fairness metrics
        metrics['intra_list_diversity'] = self.intra_list_diversity(recommendations)
        metrics['popularity_bias'] = self.popularity_bias_score(recommendations)
        metrics['genre_diversity'] = self.genre_diversity_score(recommendations)
        
        return metrics
    
    def evaluate_model(self, model, user_ids: List[str] = None, 
                      k_values: List[int] = [5, 10, 20],
                      n_recommendations: int = 20) -> Dict[str, float]:
        """
        Evaluate a recommendation model on multiple users.
        
        Args:
            model: Recommendation model with recommend() method
            user_ids: List of user IDs to evaluate (optional)
            k_values: List of k values for precision/recall/ndcg
            n_recommendations: Number of recommendations to generate
            
        Returns:
            Dictionary with average evaluation metrics
        """
        if user_ids is None:
            # Use all users or a sample
            user_ids = self.user_ids[:min(1000, len(self.user_ids))]
        
        logger.info(f"Evaluating model on {len(user_ids)} users...")
        
        all_metrics = defaultdict(list)
        
        for user_id in user_ids:
            try:
                # Get recommendations
                recommendations = model.recommend(user_id, n_recommendations=n_recommendations)
                rec_ids = [rec[0] for rec in recommendations]  # Extract item IDs
                
                # Evaluate user
                user_metrics = self.evaluate_user(rec_ids, user_id, k_values)
                
                # Accumulate metrics
                for metric, value in user_metrics.items():
                    all_metrics[metric].append(value)
                    
            except Exception as e:
                logger.warning(f"Error evaluating user {user_id}: {e}")
                continue
        
        # Calculate average metrics
        avg_metrics = {}
        for metric, values in all_metrics.items():
            avg_metrics[f'avg_{metric}'] = np.mean(values) if values else 0.0
            avg_metrics[f'std_{metric}'] = np.std(values) if values else 0.0
        
        logger.info("Model evaluation completed")
        return avg_metrics
    
    def compare_models(self, models: Dict, user_ids: List[str] = None,
                      k_values: List[int] = [5, 10, 20]) -> pd.DataFrame:
        """
        Compare multiple recommendation models.
        
        Args:
            models: Dictionary of model_name -> model
            user_ids: List of user IDs to evaluate
            k_values: List of k values for evaluation
            
        Returns:
            DataFrame with comparison results
        """
        logger.info(f"Comparing {len(models)} models...")
        
        results = []
        
        for model_name, model in models.items():
            logger.info(f"Evaluating {model_name}...")
            
            # Evaluate model
            metrics = self.evaluate_model(model, user_ids, k_values)
            
            # Add model name
            metrics['model'] = model_name
            
            results.append(metrics)
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(results)
        
        # Reorder columns
        model_col = comparison_df['model']
        comparison_df = comparison_df.drop('model', axis=1)
        comparison_df.insert(0, 'model', model_col)
        
        logger.info("Model comparison completed")
        return comparison_df
    
    def analyze_fairness_tradeoffs(self, baseline_model, fair_model, 
                                  user_ids: List[str] = None,
                                  k_values: List[int] = [10]) -> Dict:
        """
        Analyze trade-offs between accuracy and fairness.
        
        Args:
            baseline_model: Baseline recommendation model
            fair_model: Fairness-aware recommendation model
            user_ids: List of user IDs to evaluate
            k_values: List of k values for evaluation
            
        Returns:
            Dictionary with trade-off analysis
        """
        logger.info("Analyzing fairness-accuracy trade-offs...")
        
        if user_ids is None:
            user_ids = self.user_ids[:min(500, len(self.user_ids))]
        
        tradeoff_results = {
            'baseline_metrics': [],
            'fair_metrics': [],
            'differences': []
        }
        
        for user_id in user_ids:
            try:
                # Get recommendations from both models
                baseline_recs = baseline_model.recommend(user_id, n_recommendations=20)
                fair_recs = fair_model.recommend(user_id, n_recommendations=20)
                
                baseline_rec_ids = [rec[0] for rec in baseline_recs]
                fair_rec_ids = [rec[0] for rec in fair_recs]
                
                # Evaluate both
                baseline_user_metrics = self.evaluate_user(baseline_rec_ids, user_id, k_values)
                fair_user_metrics = self.evaluate_user(fair_rec_ids, user_id, k_values)
                
                # Calculate differences
                differences = {}
                for metric in baseline_user_metrics:
                    differences[f'{metric}_diff'] = (
                        fair_user_metrics[metric] - baseline_user_metrics[metric]
                    )
                
                # Store results
                tradeoff_results['baseline_metrics'].append(baseline_user_metrics)
                tradeoff_results['fair_metrics'].append(fair_user_metrics)
                tradeoff_results['differences'].append(differences)
                
            except Exception as e:
                logger.warning(f"Error analyzing trade-offs for user {user_id}: {e}")
                continue
        
        # Calculate summary statistics
        summary = {}
        
        # Average baseline metrics
        baseline_avg = defaultdict(list)
        for user_metrics in tradeoff_results['baseline_metrics']:
            for metric, value in user_metrics.items():
                baseline_avg[metric].append(value)
        
        for metric, values in baseline_avg.items():
            summary[f'baseline_avg_{metric}'] = np.mean(values)
        
        # Average fair metrics
        fair_avg = defaultdict(list)
        for user_metrics in tradeoff_results['fair_metrics']:
            for metric, value in user_metrics.items():
                fair_avg[metric].append(value)
        
        for metric, values in fair_avg.items():
            summary[f'fair_avg_{metric}'] = np.mean(values)
        
        # Average differences
        diff_avg = defaultdict(list)
        for user_diffs in tradeoff_results['differences']:
            for metric, value in user_diffs.items():
                diff_avg[metric].append(value)
        
        for metric, values in diff_avg.items():
            summary[f'avg_{metric}'] = np.mean(values)
            summary[f'std_{metric}'] = np.std(values)
        
        logger.info("Trade-off analysis completed")
        return summary

def create_evaluation_report(models: Dict, processed_data: Dict, 
                           evaluator: RecommendationEvaluator = None,
                           sample_size: int = 500) -> Dict:
    """
    Create comprehensive evaluation report.
    
    Args:
        models: Dictionary of model_name -> model
        processed_data: Processed dataset
        evaluator: RecommendationEvaluator instance (optional)
        sample_size: Number of users to sample for evaluation
        
    Returns:
        Dictionary with evaluation report
    """
    logger.info("Creating evaluation report...")
    
    if evaluator is None:
        evaluator = RecommendationEvaluator(
            processed_data['artists'], 
            processed_data['interactions']
        )
    
    # Sample users for evaluation
    all_users = list(processed_data['user_to_idx'].keys())
    np.random.seed(42)
    sample_users = np.random.choice(all_users, size=min(sample_size, len(all_users)), replace=False)
    
    # Compare models
    comparison_df = evaluator.compare_models(models, sample_users)
    
    # Create report
    report = {
        'model_comparison': comparison_df,
        'sample_size': len(sample_users),
        'total_users': len(all_users),
        'total_items': len(processed_data['artist_to_idx']),
        'total_interactions': len(processed_data['interactions'])
    }
    
    # Add detailed analysis if we have baseline and fair models
    if 'baseline' in models and 'fair' in models:
        tradeoff_analysis = evaluator.analyze_fairness_tradeoffs(
            models['baseline'], models['fair'], sample_users
        )
        report['tradeoff_analysis'] = tradeoff_analysis
    
    logger.info("Evaluation report created")
    return report

if __name__ == "__main__":
    from data_processing import LastFMDataProcessor
    from baseline_model import train_baseline_models
    
    # Process data and train models
    processor = LastFMDataProcessor()
    processed_data = processor.process_all()
    models = train_baseline_models(processed_data)
    
    # Create evaluator
    evaluator = RecommendationEvaluator(
        processed_data['artists'], 
        processed_data['interactions']
    )
    
    # Evaluate models
    comparison_df = evaluator.compare_models(models)
    print("Model Comparison Results:")
    print(comparison_df)
    
    # Create evaluation report
    report = create_evaluation_report(models, processed_data, evaluator)
    print(f"\nEvaluation Report Summary:")
    print(f"Sample Size: {report['sample_size']}")
    print(f"Total Users: {report['total_users']}")
    print(f"Total Items: {report['total_items']}")
    print(f"Total Interactions: {report['total_interactions']}")

