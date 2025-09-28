"""
Data processing module for the Last.fm 360K dataset.
Handles loading, cleaning, and preprocessing of user-artist interaction data.
"""

import os
import pandas as pd
import numpy as np
import tarfile
import requests
from typing import Tuple, Dict, List
import logging
from tqdm import tqdm

from utils import ensure_dir, save_pickle, load_pickle, get_data_dir

logger = logging.getLogger(__name__)

class LastFMDataProcessor:
    """Process Last.fm 360K dataset for music recommendation."""
    
    def __init__(self, data_dir: str = None):
        self.data_dir = data_dir or get_data_dir()
        ensure_dir(self.data_dir)
        
        # File paths
        self.dataset_url = "https://www.dtic.upf.edu/~ocelma/MusicRecommendationDataset/lastfm-dataset-360K.tar.gz"
        self.tar_file = os.path.join(self.data_dir, "lastfm-dataset-360K.tar.gz")
        self.extracted_dir = os.path.join(self.data_dir, "lastfm-dataset-360K")
        self.users_file = os.path.join(self.extracted_dir, "usersha1-artmbid-artname-plays.tsv")
        self.userid_file = os.path.join(self.extracted_dir, "usersha1-profile.tsv")
        self.artistid_file = os.path.join(self.extracted_dir, "artists.tsv")
        
        # Processed data paths
        self.processed_interactions = os.path.join(self.data_dir, "processed_interactions.pkl")
        self.processed_users = os.path.join(self.data_dir, "processed_users.pkl")
        self.processed_artists = os.path.join(self.data_dir, "processed_artists.pkl")
        self.interaction_matrix = os.path.join(self.data_dir, "interaction_matrix.pkl")
        
    def download_dataset(self) -> bool:
        """Download the Last.fm dataset if not already present."""
        if os.path.exists(self.tar_file):
            logger.info("Dataset already downloaded")
            return True
        
        logger.info("Downloading Last.fm dataset...")
        try:
            response = requests.get(self.dataset_url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(self.tar_file, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            logger.info("Dataset downloaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download dataset: {e}")
            return False
    
    def extract_dataset(self) -> bool:
        """Extract the downloaded tar file."""
        if os.path.exists(self.extracted_dir):
            logger.info("Dataset already extracted")
            return True
        
        if not os.path.exists(self.tar_file):
            logger.error("Tar file not found. Please download the dataset first.")
            return False
        
        logger.info("Extracting dataset...")
        try:
            with tarfile.open(self.tar_file, 'r:gz') as tar:
                tar.extractall(self.data_dir)
            logger.info("Dataset extracted successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to extract dataset: {e}")
            return False
    
    def create_sample_dataset(self) -> bool:
        """Create a sample dataset for testing when the full dataset is not available."""
        logger.info("Creating sample dataset...")
        
        # Create sample data
        np.random.seed(42)
        n_users = 10000
        n_artists = 5000
        
        # Generate sample user-artist interactions
        user_ids = [f"user_{i:06d}" for i in range(n_users)]
        artist_ids = [f"artist_{i:06d}" for i in range(n_artists)]
        
        # Create interactions with power-law distribution (realistic for music)
        interactions = []
        for user_id in user_ids:
            # Each user interacts with 10-100 artists
            n_interactions = np.random.poisson(30)
            n_interactions = max(10, min(100, n_interactions))
            
            # Sample artists with popularity bias
            artist_probs = np.random.power(0.7, n_artists)
            artist_probs = artist_probs / artist_probs.sum()
            
            selected_artists = np.random.choice(
                range(n_artists), 
                size=n_interactions, 
                replace=False, 
                p=artist_probs
            )
            
            for artist_idx in selected_artists:
                play_count = np.random.poisson(10) + 1
                interactions.append({
                    'user_id': user_id,
                    'artist_id': artist_ids[artist_idx],
                    'artist_name': f"Artist {artist_idx}",
                    'play_count': play_count
                })
        
        # Save sample interactions
        interactions_df = pd.DataFrame(interactions)
        interactions_df.to_csv(os.path.join(self.data_dir, "sample_interactions.tsv"), 
                              sep='\t', index=False)
        
        # Save sample users
        users_df = pd.DataFrame({
            'user_id': user_ids,
            'country': np.random.choice(['US', 'UK', 'CA', 'DE', 'FR'], n_users),
            'age': np.random.choice(['18-24', '25-34', '35-44', '45+'], n_users),
            'gender': np.random.choice(['M', 'F', 'Other'], n_users)
        })
        users_df.to_csv(os.path.join(self.data_dir, "sample_users.tsv"), 
                       sep='\t', index=False)
        
        # Save sample artists with genres
        genres = ['Rock', 'Pop', 'Hip-Hop', 'Electronic', 'Jazz', 'Classical', 'Country', 'R&B']
        artists_df = pd.DataFrame({
            'artist_id': artist_ids,
            'artist_name': [f"Artist {i}" for i in range(n_artists)],
            'genre': np.random.choice(genres, n_artists),
            'popularity': np.random.power(0.5, n_artists)  # Power law distribution
        })
        artists_df.to_csv(os.path.join(self.data_dir, "sample_artists.tsv"), 
                         sep='\t', index=False)
        
        logger.info("Sample dataset created successfully")
        return True
    
    def load_interactions(self) -> pd.DataFrame:
        """Load user-artist interactions from the dataset."""
        # Try to load real dataset first
        if os.path.exists(self.users_file):
            logger.info("Loading real Last.fm interactions...")
            interactions_df = pd.read_csv(
                self.users_file, 
                sep='\t', 
                header=None,
                names=['user_id', 'artist_mbid', 'artist_name', 'play_count']
            )
        else:
            # Fall back to sample dataset
            sample_file = os.path.join(self.data_dir, "sample_interactions.tsv")
            if os.path.exists(sample_file):
                logger.info("Loading sample interactions...")
                interactions_df = pd.read_csv(sample_file, sep='\t')
            else:
                logger.warning("No dataset found. Creating sample dataset...")
                self.create_sample_dataset()
                interactions_df = pd.read_csv(sample_file, sep='\t')
        
        # Clean the data
        interactions_df = self.clean_interactions(interactions_df)
        return interactions_df
    
    def clean_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess the interactions data."""
        logger.info("Cleaning interactions data...")
        
        # Remove rows with missing data
        initial_count = len(df)
        df = df.dropna()
        
        # Convert play_count to numeric
        df['play_count'] = pd.to_numeric(df['play_count'], errors='coerce')
        df = df.dropna()
        
        # Remove duplicate user-artist pairs, keep the one with highest play count
        df = df.sort_values('play_count', ascending=False).drop_duplicates(
            subset=['user_id', 'artist_name'], keep='first'
        )
        
        # Filter out very low play counts (likely noise)
        df = df[df['play_count'] >= 2]
        
        # Filter users and artists with very few interactions
        user_counts = df['user_id'].value_counts()
        artist_counts = df['artist_name'].value_counts()
        
        df = df[df['user_id'].isin(user_counts[user_counts >= 5].index)]
        df = df[df['artist_name'].isin(artist_counts[artist_counts >= 5].index)]
        
        # Create artist_id if it doesn't exist
        if 'artist_id' not in df.columns:
            artist_mapping = {name: f"artist_{i:06d}" for i, name in 
                            enumerate(df['artist_name'].unique())}
            df['artist_id'] = df['artist_name'].map(artist_mapping)
        
        logger.info(f"Cleaned data: {initial_count} -> {len(df)} interactions")
        return df
    
    def load_users(self) -> pd.DataFrame:
        """Load user metadata."""
        if os.path.exists(self.userid_file):
            logger.info("Loading real user data...")
            users_df = pd.read_csv(
                self.userid_file, 
                sep='\t', 
                header=None,
                names=['user_id', 'gender', 'age', 'country', 'signup_date']
            )
        else:
            sample_file = os.path.join(self.data_dir, "sample_users.tsv")
            if os.path.exists(sample_file):
                logger.info("Loading sample user data...")
                users_df = pd.read_csv(sample_file, sep='\t')
            else:
                logger.warning("No user data found. Creating sample data...")
                self.create_sample_dataset()
                users_df = pd.read_csv(sample_file, sep='\t')
        
        return users_df
    
    def load_artists(self) -> pd.DataFrame:
        """Load artist metadata."""
        if os.path.exists(self.artistid_file):
            logger.info("Loading real artist data...")
            artists_df = pd.read_csv(
                self.artistid_file, 
                sep='\t', 
                header=None,
                names=['artist_id', 'artist_name', 'mbid']
            )
        else:
            sample_file = os.path.join(self.data_dir, "sample_artists.tsv")
            if os.path.exists(sample_file):
                logger.info("Loading sample artist data...")
                artists_df = pd.read_csv(sample_file, sep='\t')
            else:
                logger.warning("No artist data found. Creating sample data...")
                self.create_sample_dataset()
                artists_df = pd.read_csv(sample_file, sep='\t')
        
        # Add genre information if not present
        if 'genre' not in artists_df.columns:
            genres = ['Rock', 'Pop', 'Hip-Hop', 'Electronic', 'Jazz', 'Classical', 'Country', 'R&B']
            np.random.seed(42)
            artists_df['genre'] = np.random.choice(genres, len(artists_df))
        
        # Add popularity if not present
        if 'popularity' not in artists_df.columns:
            np.random.seed(42)
            artists_df['popularity'] = np.random.power(0.5, len(artists_df))
        
        return artists_df
    
    def create_interaction_matrix(self, interactions_df: pd.DataFrame) -> Tuple[np.ndarray, Dict, Dict]:
        """Create user-item interaction matrix."""
        logger.info("Creating interaction matrix...")
        
        # Create mappings
        unique_users = sorted(interactions_df['user_id'].unique())
        unique_artists = sorted(interactions_df['artist_id'].unique())
        
        user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
        artist_to_idx = {artist: idx for idx, artist in enumerate(unique_artists)}
        
        # Create matrix
        n_users = len(unique_users)
        n_artists = len(unique_artists)
        interaction_matrix = np.zeros((n_users, n_artists))
        
        # Fill matrix with play counts
        for _, row in tqdm(interactions_df.iterrows(), total=len(interactions_df), desc="Building matrix"):
            user_idx = user_to_idx[row['user_id']]
            artist_idx = artist_to_idx[row['artist_id']]
            interaction_matrix[user_idx, artist_idx] = row['play_count']
        
        logger.info(f"Created interaction matrix: {n_users} users x {n_artists} artists")
        return interaction_matrix, user_to_idx, artist_to_idx
    
    def process_all(self) -> Dict:
        """Process all data and return processed datasets."""
        logger.info("Starting data processing...")
        
        # Try to download and extract dataset
        if not self.extract_dataset():
            if not self.download_dataset():
                logger.warning("Could not download dataset, using sample data")
                self.create_sample_dataset()
            else:
                self.extract_dataset()
        
        # Load and process data
        interactions_df = self.load_interactions()
        users_df = self.load_users()
        artists_df = self.load_artists()
        
        # Create interaction matrix
        interaction_matrix, user_to_idx, artist_to_idx = self.create_interaction_matrix(interactions_df)
        
        # Save processed data
        save_pickle(interactions_df, self.processed_interactions)
        save_pickle(users_df, self.processed_users)
        save_pickle(artists_df, self.processed_artists)
        save_pickle(interaction_matrix, self.interaction_matrix)
        
        # Create mappings for easy access
        idx_to_user = {idx: user for user, idx in user_to_idx.items()}
        idx_to_artist = {idx: artist for artist, idx in artist_to_idx.items()}
        
        processed_data = {
            'interactions': interactions_df,
            'users': users_df,
            'artists': artists_df,
            'interaction_matrix': interaction_matrix,
            'user_to_idx': user_to_idx,
            'artist_to_idx': artist_to_idx,
            'idx_to_user': idx_to_user,
            'idx_to_artist': idx_to_artist,
            'n_users': len(user_to_idx),
            'n_artists': len(artist_to_idx)
        }
        
        logger.info("Data processing completed successfully")
        return processed_data

def main():
    """Main function to process the dataset."""
    processor = LastFMDataProcessor()
    processed_data = processor.process_all()
    
    print(f"Processed {processed_data['n_users']} users and {processed_data['n_artists']} artists")
    print(f"Total interactions: {len(processed_data['interactions'])}")
    
    return processed_data

if __name__ == "__main__":
    main()
