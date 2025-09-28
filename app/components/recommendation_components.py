"""
Streamlit components for recommendation display and visualization.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Tuple

def display_recommendations(recommendations: List[Tuple[str, float]], 
                          artists_df: pd.DataFrame,
                          title: str = "Recommendations",
                          max_items: int = 10) -> None:
    """
    Display recommendations in a formatted table.
    
    Args:
        recommendations: List of (artist_id, score) tuples
        artists_df: DataFrame with artist metadata
        title: Title for the recommendations section
        max_items: Maximum number of items to display
    """
    st.subheader(title)
    
    if not recommendations:
        st.write("No recommendations available.")
        return
    
    # Create recommendations DataFrame
    rec_data = []
    for artist_id, score in recommendations[:max_items]:
        artist_info = artists_df[artists_df['artist_id'] == artist_id]
        if not artist_info.empty:
            # Handle different column names
            artist_name = artist_info.get('artist_name', artist_info.get('name', artist_id)).iloc[0]
            genre = artist_info.get('genre', 'Unknown').iloc[0]
            popularity = artist_info.get('popularity', 0.0).iloc[0]
            
            rec_data.append({
                'Artist': artist_name,
                'Genre': genre,
                'Popularity': f"{popularity:.3f}",
                'Score': f"{score:.3f}"
            })
    
    if rec_data:
        rec_df = pd.DataFrame(rec_data)
        st.dataframe(rec_df, use_container_width=True)
    else:
        st.write("No valid recommendations found.")

def display_bias_metrics(bias_analysis: Dict, title: str = "Bias Analysis") -> None:
    """
    Display bias metrics in a formatted way.
    
    Args:
        bias_analysis: Dictionary with bias metrics
        title: Title for the bias analysis section
    """
    st.subheader(title)
    
    # Create metrics DataFrame
    metrics_data = []
    
    for bias_type, metrics in bias_analysis.items():
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                metrics_data.append({
                    'Bias Type': bias_type.replace('_', ' ').title(),
                    'Metric': metric_name.replace('_', ' ').title(),
                    'Value': f"{value:.3f}"
                })
    
    if metrics_data:
        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df, use_container_width=True)
    else:
        st.write("No bias metrics available.")

def plot_recommendation_comparison(baseline_recs: List[Tuple[str, float]], 
                                 fair_recs: List[Tuple[str, float]],
                                 artists_df: pd.DataFrame) -> None:
    """
    Plot comparison between baseline and fairness-aware recommendations.
    
    Args:
        baseline_recs: Baseline recommendations
        fair_recs: Fairness-aware recommendations
        artists_df: DataFrame with artist metadata
    """
    st.subheader("Recommendation Comparison")
    
    # Create comparison data
    comparison_data = []
    
    # Baseline recommendations
    for i, (artist_id, score) in enumerate(baseline_recs[:10]):
        artist_info = artists_df[artists_df['artist_id'] == artist_id]
        if not artist_info.empty:
            comparison_data.append({
                'Rank': i + 1,
                'Artist': artist_info.get('artist_name', artist_info.get('name', artist_id)).iloc[0],
                'Genre': artist_info.get('genre', 'Unknown').iloc[0],
                'Popularity': artist_info.get('popularity', 0.0).iloc[0],
                'Score': score,
                'Model': 'Baseline'
            })
    
    # Fair recommendations
    for i, (artist_id, score) in enumerate(fair_recs[:10]):
        artist_info = artists_df[artists_df['artist_id'] == artist_id]
        if not artist_info.empty:
            comparison_data.append({
                'Rank': i + 1,
                'Artist': artist_info.get('artist_name', artist_info.get('name', artist_id)).iloc[0],
                'Genre': artist_info.get('genre', 'Unknown').iloc[0],
                'Popularity': artist_info.get('popularity', 0.0).iloc[0],
                'Score': score,
                'Model': 'Fairness-Aware'
            })
    
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Popularity Distribution', 'Genre Distribution', 
                          'Score Distribution', 'Rank Comparison'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Popularity distribution
        for model in ['Baseline', 'Fairness-Aware']:
            model_data = comparison_df[comparison_df['Model'] == model]
            fig.add_trace(
                go.Histogram(x=model_data['Popularity'], name=f'{model} Popularity', 
                           opacity=0.7),
                row=1, col=1
            )
        
        # Genre distribution
        genre_counts = comparison_df.groupby(['Model', 'Genre']).size().reset_index(name='Count')
        for model in ['Baseline', 'Fairness-Aware']:
            model_data = genre_counts[genre_counts['Model'] == model]
            fig.add_trace(
                go.Bar(x=model_data['Genre'], y=model_data['Count'], 
                      name=f'{model} Genres'),
                row=1, col=2
            )
        
        # Score distribution
        for model in ['Baseline', 'Fairness-Aware']:
            model_data = comparison_df[comparison_df['Model'] == model]
            fig.add_trace(
                go.Box(y=model_data['Score'], name=f'{model} Scores'),
                row=2, col=1
            )
        
        # Rank comparison
        for model in ['Baseline', 'Fairness-Aware']:
            model_data = comparison_df[comparison_df['Model'] == model]
            fig.add_trace(
                go.Scatter(x=model_data['Rank'], y=model_data['Score'], 
                          mode='lines+markers', name=f'{model}'),
                row=2, col=2
            )
        
        fig.update_layout(height=800, showlegend=True, 
                         title_text="Baseline vs Fairness-Aware Recommendations")
        
        st.plotly_chart(fig, use_container_width=True)

def plot_bias_comparison(baseline_analysis: Dict, fair_analysis: Dict) -> None:
    """
    Plot comparison of bias metrics between baseline and fair models.
    
    Args:
        baseline_analysis: Baseline bias analysis
        fair_analysis: Fair bias analysis
    """
    st.subheader("Bias Metrics Comparison")
    
    # Extract key metrics for comparison
    metrics_data = []
    
    # Popularity bias
    baseline_pop_bias = baseline_analysis.get('popularity_bias', {}).get('popularity_bias_score', 0)
    fair_pop_bias = fair_analysis.get('popularity_bias', {}).get('popularity_bias_score', 0)
    
    metrics_data.extend([
        {'Metric': 'Popularity Bias', 'Baseline': baseline_pop_bias, 'Fair': fair_pop_bias},
    ])
    
    # Genre diversity
    baseline_genre_div = baseline_analysis.get('genre_diversity', {}).get('genre_diversity_score', 0)
    fair_genre_div = fair_analysis.get('genre_diversity', {}).get('genre_diversity_score', 0)
    
    metrics_data.extend([
        {'Metric': 'Genre Diversity', 'Baseline': baseline_genre_div, 'Fair': fair_genre_div},
    ])
    
    # Exposure fairness
    baseline_exposure = baseline_analysis.get('exposure_fairness', {}).get('exposure_fairness_score', 0)
    fair_exposure = fair_analysis.get('exposure_fairness', {}).get('exposure_fairness_score', 0)
    
    metrics_data.extend([
        {'Metric': 'Exposure Fairness', 'Baseline': baseline_exposure, 'Fair': fair_exposure},
    ])
    
    if metrics_data:
        metrics_df = pd.DataFrame(metrics_data)
        
        # Create comparison chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Baseline',
            x=metrics_df['Metric'],
            y=metrics_df['Baseline'],
            marker_color='lightblue'
        ))
        
        fig.add_trace(go.Bar(
            name='Fairness-Aware',
            x=metrics_df['Metric'],
            y=metrics_df['Fair'],
            marker_color='lightcoral'
        ))
        
        fig.update_layout(
            title='Bias Metrics Comparison',
            xaxis_title='Metrics',
            yaxis_title='Score',
            barmode='group'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display improvement metrics
        st.subheader("Improvement Analysis")
        improvements = []
        for _, row in metrics_df.iterrows():
            improvement = row['Fair'] - row['Baseline']
            improvement_pct = (improvement / row['Baseline'] * 100) if row['Baseline'] > 0 else 0
            improvements.append({
                'Metric': row['Metric'],
                'Improvement': f"{improvement:.3f}",
                'Improvement %': f"{improvement_pct:.1f}%"
            })
        
        improvement_df = pd.DataFrame(improvements)
        st.dataframe(improvement_df, use_container_width=True)

def display_user_stats(user_id: str, interactions_df: pd.DataFrame, 
                      artists_df: pd.DataFrame) -> None:
    """
    Display user statistics and preferences.
    
    Args:
        user_id: User ID
        interactions_df: DataFrame with user interactions
        artists_df: DataFrame with artist metadata
    """
    st.subheader(f"User Profile: {user_id}")
    
    # Get user interactions
    user_interactions = interactions_df[interactions_df['user_id'] == user_id]
    
    if user_interactions.empty:
        st.write("No interaction history found for this user.")
        return
    
    # Calculate user statistics
    total_interactions = len(user_interactions)
    total_playtime = user_interactions['play_count'].sum()
    unique_artists = user_interactions['artist_id'].nunique()
    
    # Genre preferences
    user_artists = user_interactions.merge(artists_df, on='artist_id', how='left')
    genre_counts = user_artists['genre'].value_counts()
    
    # Display statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Interactions", total_interactions)
    
    with col2:
        st.metric("Total Playtime", f"{total_playtime:,}")
    
    with col3:
        st.metric("Unique Artists", unique_artists)
    
    # Genre preferences chart
    if not genre_counts.empty:
        st.subheader("Genre Preferences")
        fig = px.pie(values=genre_counts.values, names=genre_counts.index, 
                    title="User Genre Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    # Top artists
    top_artists = user_interactions.nlargest(10, 'play_count')
    
    # Check available columns and merge accordingly
    available_cols = ['artist_id']
    if 'artist_name' in artists_df.columns:
        available_cols.append('artist_name')
    elif 'name' in artists_df.columns:
        available_cols.append('name')
    
    if 'genre' in artists_df.columns:
        available_cols.append('genre')
    
    top_artists = top_artists.merge(artists_df[available_cols], 
                                  on='artist_id', how='left')
    
    st.subheader("Top Artists")
    
    # Display with available columns
    display_cols = []
    if 'artist_name' in top_artists.columns:
        display_cols.append('artist_name')
    elif 'name' in top_artists.columns:
        display_cols.append('name')
    
    if 'genre' in top_artists.columns:
        display_cols.append('genre')
    
    display_cols.append('play_count')
    
    if display_cols:
        st.dataframe(top_artists[display_cols], use_container_width=True)
    else:
        st.dataframe(top_artists[['artist_id', 'play_count']], use_container_width=True)

def display_explanation(baseline_recs: List[Tuple[str, float]], 
                       fair_recs: List[Tuple[str, float]],
                       artists_df: pd.DataFrame) -> None:
    """
    Display explanations for fairness-aware recommendations.
    
    Args:
        baseline_recs: Baseline recommendations
        fair_recs: Fairness-aware recommendations
        artists_df: DataFrame with artist metadata
    """
    st.subheader("Fairness Explanations")
    
    # Find differences between baseline and fair recommendations
    baseline_artists = set(rec[0] for rec in baseline_recs[:5])
    fair_artists = set(rec[0] for rec in fair_recs[:5])
    
    # Artists only in fair recommendations
    fair_only = fair_artists - baseline_artists
    baseline_only = baseline_artists - fair_artists
    
    if fair_only:
        st.write("**Artists added for fairness:**")
        for artist_id in fair_only:
            artist_info = artists_df[artists_df['artist_id'] == artist_id]
            if not artist_info.empty:
                artist_name = artist_info.get('artist_name', artist_info.get('name', artist_id)).iloc[0]
                genre = artist_info.get('genre', 'Unknown').iloc[0]
                popularity = artist_info.get('popularity', 0.0).iloc[0]
                
                if popularity < 0.3:  # Low popularity
                    st.write(f"• **{artist_name}** ({genre}) - Added to promote long-tail artists")
                else:
                    st.write(f"• **{artist_name}** ({genre}) - Added for genre diversity")
    
    if baseline_only:
        st.write("**Artists removed for fairness:**")
        for artist_id in baseline_only:
            artist_info = artists_df[artists_df['artist_id'] == artist_id]
            if not artist_info.empty:
                artist_name = artist_info.get('artist_name', artist_info.get('name', artist_id)).iloc[0]
                genre = artist_info.get('genre', 'Unknown').iloc[0]
                popularity = artist_info.get('popularity', 0.0).iloc[0]
                
                if popularity > 0.7:  # High popularity
                    st.write(f"• **{artist_name}** ({genre}) - Removed to reduce popularity bias")
                else:
                    st.write(f"• **{artist_name}** ({genre}) - Removed for better genre balance")
    
    # Overall explanation
    st.write("**Overall Strategy:**")
    st.write("The fairness-aware system balances recommendation accuracy with:")
    st.write("• **Diversity**: Including artists from different genres")
    st.write("• **Exposure**: Promoting less popular (long-tail) artists")
    st.write("• **Balance**: Ensuring fair representation across artist groups")

def create_summary_metrics(baseline_analysis: Dict, fair_analysis: Dict) -> None:
    """
    Create summary metrics display.
    
    Args:
        baseline_analysis: Baseline bias analysis
        fair_analysis: Fair bias analysis
    """
    st.subheader("Summary Metrics")
    
    # Extract key metrics
    baseline_pop_bias = baseline_analysis.get('popularity_bias', {}).get('popularity_bias_score', 0)
    fair_pop_bias = fair_analysis.get('popularity_bias', {}).get('popularity_bias_score', 0)
    
    baseline_genre_div = baseline_analysis.get('genre_diversity', {}).get('genre_diversity_score', 0)
    fair_genre_div = fair_analysis.get('genre_diversity', {}).get('genre_diversity_score', 0)
    
    baseline_exposure = baseline_analysis.get('exposure_fairness', {}).get('exposure_fairness_score', 0)
    fair_exposure = fair_analysis.get('exposure_fairness', {}).get('exposure_fairness_score', 0)
    
    # Display metrics in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Popularity Bias", f"{baseline_pop_bias:.3f}", 
                 f"{fair_pop_bias - baseline_pop_bias:.3f}")
        st.caption("Lower is better")
    
    with col2:
        st.metric("Genre Diversity", f"{baseline_genre_div:.3f}", 
                 f"{fair_genre_div - baseline_genre_div:.3f}")
        st.caption("Higher is better")
    
    with col3:
        st.metric("Exposure Fairness", f"{baseline_exposure:.3f}", 
                 f"{fair_exposure - baseline_exposure:.3f}")
        st.caption("Higher is better")

