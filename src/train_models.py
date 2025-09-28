"""
Training script for the Fairness-Aware Music Recommender system.
"""

import os
import sys
import logging
import argparse
from typing import Dict

# Add src directory to path
sys.path.append(os.path.dirname(__file__))

from data_processing import LastFMDataProcessor
from baseline_model import train_baseline_models
from bias_detection import analyze_model_bias
from fairness_reranking import FairnessAwareReranker
from evaluation import create_evaluation_report
from utils import save_pickle, get_models_dir, get_data_dir

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Fairness-Aware Music Recommender')
    parser.add_argument('--data-dir', type=str, default=None, help='Data directory path')
    parser.add_argument('--models-dir', type=str, default=None, help='Models directory path')
    parser.add_argument('--sample-size', type=int, default=1000, help='Sample size for evaluation')
    parser.add_argument('--force-retrain', action='store_true', help='Force retraining even if models exist')
    
    args = parser.parse_args()
    
    logger.info("Starting Fairness-Aware Music Recommender training...")
    
    # Initialize data processor
    processor = LastFMDataProcessor(data_dir=args.data_dir)
    
    # Process data
    logger.info("Processing data...")
    processed_data = processor.process_all()
    
    logger.info(f"Data processed: {processed_data['n_users']} users, {processed_data['n_artists']} artists")
    logger.info(f"Total interactions: {len(processed_data['interactions'])}")
    
    # Train baseline models
    logger.info("Training baseline models...")
    models = train_baseline_models(processed_data)
    
    # Analyze bias in baseline models
    logger.info("Analyzing bias in baseline models...")
    bias_results = analyze_model_bias(models, processed_data)
    
    # Save bias analysis results
    bias_results_path = os.path.join(get_data_dir(), "bias_analysis_results.pkl")
    save_pickle(bias_results, bias_results_path)
    
    # Create fairness-aware reranker
    logger.info("Creating fairness-aware reranker...")
    from bias_detection import BiasDetector
    bias_detector = BiasDetector(processed_data['artists'], processed_data['interactions'])
    
    reranker = FairnessAwareReranker(processed_data['artists'], bias_detector)
    
    # Save reranker
    reranker_path = os.path.join(get_models_dir(), "fairness_reranker.pkl")
    save_pickle(reranker, reranker_path)
    
    # Create evaluation report
    logger.info("Creating evaluation report...")
    from evaluation import RecommendationEvaluator
    evaluator = RecommendationEvaluator(processed_data['artists'], processed_data['interactions'])
    
    evaluation_report = create_evaluation_report(
        models, processed_data, evaluator, sample_size=args.sample_size
    )
    
    # Save evaluation report
    eval_report_path = os.path.join(get_data_dir(), "evaluation_report.pkl")
    save_pickle(evaluation_report, eval_report_path)
    
    # Print summary
    logger.info("Training completed successfully!")
    logger.info("Summary:")
    logger.info(f"  - Users: {processed_data['n_users']}")
    logger.info(f"  - Artists: {processed_data['n_artists']}")
    logger.info(f"  - Interactions: {len(processed_data['interactions'])}")
    logger.info(f"  - Models trained: {len(models)}")
    
    # Print bias analysis results
    logger.info("Bias Analysis Results:")
    for model_name, results in bias_results.items():
        logger.info(f"  {model_name}:")
        logger.info(f"    Popularity Bias: {results['avg_popularity_bias']:.3f}")
        logger.info(f"    Genre Diversity: {results['avg_genre_diversity']:.3f}")
        logger.info(f"    Exposure Fairness: {results['avg_exposure_fairness']:.3f}")
        logger.info(f"    Overall Bias: {results['avg_overall_bias']:.3f}")
    
    # Print evaluation results
    logger.info("Evaluation Results:")
    comparison_df = evaluation_report['model_comparison']
    for _, row in comparison_df.iterrows():
        logger.info(f"  {row['model']}:")
        logger.info(f"    Precision@10: {row.get('avg_precision@10', 0):.3f}")
        logger.info(f"    Recall@10: {row.get('avg_recall@10', 0):.3f}")
        logger.info(f"    NDCG@10: {row.get('avg_ndcg@10', 0):.3f}")

if __name__ == "__main__":
    main()






