"""
Streamlit demo application for Fairness-Aware Music Recommender.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
from typing import Dict, List, Tuple

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_processing import LastFMDataProcessor
from baseline_model import BaselineRecommender, PopularityRecommender
from bias_detection import BiasDetector
from fairness_reranking import FairnessAwareReranker
from evaluation import RecommendationEvaluator
from utils import load_pickle, get_data_dir, get_models_dir

# Import components
from components.recommendation_components import (
    display_recommendations,
    display_bias_metrics,
    plot_recommendation_comparison,
    plot_bias_comparison,
    display_user_stats,
    display_explanation,
    create_summary_metrics
)

# Page configuration
st.set_page_config(
    page_title="Fairness-Aware Music Recommender",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .explanation-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff7f0e;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache the processed data."""
    try:
        # Try to load preprocessed data
        data_dir = get_data_dir()
        processed_interactions = os.path.join(data_dir, "processed_interactions.pkl")
        processed_users = os.path.join(data_dir, "processed_users.pkl")
        processed_artists = os.path.join(data_dir, "processed_artists.pkl")
        interaction_matrix = os.path.join(data_dir, "interaction_matrix.pkl")
        
        if all(os.path.exists(f) for f in [processed_interactions, processed_users, 
                                         processed_artists, interaction_matrix]):
            st.info("Loading preprocessed data...")
            interactions_df = load_pickle(processed_interactions)
            users_df = load_pickle(processed_users)
            artists_df = load_pickle(processed_artists)
            interaction_matrix = load_pickle(interaction_matrix)
            
            # Create mappings
            unique_users = sorted(interactions_df['user_id'].unique())
            unique_artists = sorted(interactions_df['artist_id'].unique())
            
            user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
            artist_to_idx = {artist: idx for idx, artist in enumerate(unique_artists)}
            
            processed_data = {
                'interactions': interactions_df,
                'users': users_df,
                'artists': artists_df,
                'interaction_matrix': interaction_matrix,
                'user_to_idx': user_to_idx,
                'artist_to_idx': artist_to_idx,
                'n_users': len(unique_users),
                'n_artists': len(unique_artists)
            }
            
            return processed_data
        else:
            raise FileNotFoundError("Preprocessed data not found")
            
    except Exception as e:
        st.warning(f"Could not load preprocessed data: {e}")
        st.info("Processing data from scratch...")
        
        # Process data from scratch
        processor = LastFMDataProcessor()
        processed_data = processor.process_all()
        
        return processed_data

@st.cache_resource
def load_models(processed_data: Dict):
    """Load and cache the trained models."""
    try:
        # Try to load pre-trained models
        models_dir = get_models_dir()
        baseline_model_path = os.path.join(models_dir, "baseline_model.pkl")
        
        models = {}
        
        if os.path.exists(baseline_model_path):
            st.info("Loading pre-trained models...")
            baseline_model = BaselineRecommender()
            baseline_model.load_model(baseline_model_path)
            models['baseline'] = baseline_model
        else:
            st.info("Training models...")
            from baseline_model import train_baseline_models
            models = train_baseline_models(processed_data)
        
        return models
        
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.info("Training models from scratch...")
        
        from baseline_model import train_baseline_models
        models = train_baseline_models(processed_data)
        return models

def main():
    """Main application function."""
    
    # Header
    st.markdown('<h1 class="main-header">🎵 Fairness-Aware Music Recommender</h1>', 
                unsafe_allow_html=True)
    
    # Introduction section
    with st.expander("📖 What is this? Click here for an introduction", expanded=True):
        st.markdown("""
        ### Welcome! This demo shows how to make music recommendations more fair and diverse.
        
        **The Problem**: Traditional recommendation systems often show bias - they recommend popular artists to everyone, 
        making it harder for smaller artists to get discovered.
        
        **The Solution**: This system balances accuracy with fairness by:
        - 🎯 **Detecting bias** in recommendations (popularity bias, genre imbalance)
        - ⚖️ **Re-ranking recommendations** to include diverse artists
        - 📊 **Showing you the difference** between traditional and fair recommendations
        
        **How to use this demo**:
        1. **Select a user** from the sidebar (or pick "Random")
        2. **Compare the two columns** - see how fairness-aware recommendations differ
        3. **Adjust the diversity slider** to see how it affects recommendations
        4. **Look at the charts** to understand bias and improvements
        
        **Key Metrics Explained**:
        - **Popularity Bias**: Lower is better (less focus on mainstream artists)
        - **Genre Diversity**: Higher is better (more variety in music styles)
        - **Exposure Fairness**: Higher is better (more artists get recommended)
        """)
    
    # Load data
    with st.spinner("Loading data..."):
        processed_data = load_data()
    
    # Load models
    with st.spinner("Loading models..."):
        models = load_models(processed_data)
    
    # Sidebar
    st.sidebar.title("🎛️ Controls")
    
    # User selection
    st.sidebar.subheader("Select User")
    st.sidebar.caption("Choose a user to see their personalized recommendations")
    
    # Get sample users
    sample_users = list(processed_data['user_to_idx'].keys())[:100]  # First 100 users
    
    user_option = st.sidebar.selectbox(
        "Choose a user:",
        options=["Random"] + sample_users[:20],  # Show first 20 users + random option
        index=0,
        help="Select a specific user or 'Random' to see different recommendation patterns"
    )
    
    if user_option == "Random":
        selected_user = np.random.choice(sample_users)
        st.sidebar.write(f"Selected: {selected_user}")
    else:
        selected_user = user_option
    
    # Recommendation parameters
    st.sidebar.subheader("Recommendation Settings")
    
    n_recommendations = st.sidebar.slider(
        "Number of recommendations:",
        min_value=5,
        max_value=20,
        value=10,
        step=1
    )
    
    diversity_weight = st.sidebar.slider(
        "Diversity weight:",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.1
    )
    
    # Create bias detector and reranker
    bias_detector = BiasDetector(
        processed_data['artists'], 
        processed_data['interactions']
    )
    
    reranker = FairnessAwareReranker(
        processed_data['artists'], 
        bias_detector
    )
    
    # Main content
    if selected_user in processed_data['user_to_idx']:
        
        # User statistics
        display_user_stats(
            selected_user, 
            processed_data['interactions'], 
            processed_data['artists']
        )
        
        st.divider()
        
        # Generate recommendations
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Baseline Recommendations")
            
            # Get baseline recommendations
            baseline_recs = models['baseline'].recommend(
                selected_user, 
                n_recommendations=n_recommendations * 2  # Get more for re-ranking
            )
            
            # Display baseline recommendations
            display_recommendations(
                baseline_recs, 
                processed_data['artists'], 
                "Baseline Model",
                n_recommendations
            )
            
            # Analyze baseline bias
            baseline_analysis = bias_detector.analyze_recommendations(
                baseline_recs[:n_recommendations], selected_user
            )
            
            display_bias_metrics(baseline_analysis, "Baseline Bias Analysis")
        
        with col2:
            st.subheader("⚖️ Fairness-Aware Recommendations")
            
            # Apply fairness re-ranking
            fair_recs = reranker.mmr_rerank(
                baseline_recs, 
                diversity_weight=diversity_weight,
                n_recommendations=n_recommendations
            )
            
            # Display fair recommendations
            display_recommendations(
                fair_recs, 
                processed_data['artists'], 
                "Fairness-Aware Model",
                n_recommendations
            )
            
            # Analyze fair bias
            fair_analysis = bias_detector.analyze_recommendations(
                fair_recs, selected_user
            )
            
            display_bias_metrics(fair_analysis, "Fairness-Aware Bias Analysis")
        
        # Comparison section
        st.divider()
        st.subheader("📊 Comparison Analysis")
        
        # Summary metrics
        create_summary_metrics(baseline_analysis, fair_analysis)
        
        # Detailed comparison
        plot_recommendation_comparison(
            baseline_recs[:n_recommendations], 
            fair_recs, 
            processed_data['artists']
        )
        
        plot_bias_comparison(baseline_analysis, fair_analysis)
        
        # Explanations
        st.divider()
        display_explanation(
            baseline_recs[:n_recommendations], 
            fair_recs, 
            processed_data['artists']
        )
        
    else:
        st.error(f"User {selected_user} not found in the dataset.")
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Fairness-Aware Music Recommender Demo | Built with Streamlit</p>
        <p>This system demonstrates how to balance recommendation accuracy with fairness and diversity.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()






