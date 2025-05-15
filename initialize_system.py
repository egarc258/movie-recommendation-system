#!/usr/bin/env python3
import os
import pickle
import torch
import pandas as pd
import numpy as np
import zipfile
import urllib.request
import nltk
from app.recommender.models import BERTMovieEmbeddings
from app.recommender.engine import MovieRecommender
from app.utils.data_processor import extract_movie_features

def download_movielens_dataset(size='small'):
    """Download the MovieLens dataset"""
    data_dir = 'app/data'
    os.makedirs(data_dir, exist_ok=True)
    
    # Define URLs for different dataset sizes
    if size == 'small':
        url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
        zip_path = os.path.join(data_dir, "ml-latest-small.zip")
        extract_dir = os.path.join(data_dir, "ml-latest-small")
    elif size == '1m':
        url = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
        zip_path = os.path.join(data_dir, "ml-1m.zip")
        extract_dir = os.path.join(data_dir, "ml-1m")
    else:  # 'full' or '20m'
        url = "https://files.grouplens.org/datasets/movielens/ml-20m.zip"
        zip_path = os.path.join(data_dir, "ml-20m.zip")
        extract_dir = os.path.join(data_dir, "ml-20m")
    
    # Check if data already exists
    if os.path.exists(os.path.join(data_dir, f"movies_{size}.csv")) and        os.path.exists(os.path.join(data_dir, f"ratings_{size}.csv")):
        print(f"Dataset already exists. Loading from {data_dir}...")
        return pd.read_csv(os.path.join(data_dir, f"movies_{size}.csv")),                pd.read_csv(os.path.join(data_dir, f"ratings_{size}.csv"))
    
    # Download dataset if needed
    if not os.path.exists(extract_dir):
        print(f"Downloading MovieLens {size} dataset...")
        try:
            urllib.request.urlretrieve(url, zip_path)
            
            # Extract zip file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            
            # Remove zip file
            os.remove(zip_path)
            
            print(f"Downloaded and extracted to {extract_dir}")
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            return None, None
    
    # Load data based on size
    if size == 'small':
        movies_path = os.path.join(extract_dir, "movies.csv")
        ratings_path = os.path.join(extract_dir, "ratings.csv")
        
        movies = pd.read_csv(movies_path)
        ratings = pd.read_csv(ratings_path)
    elif size == '1m':
        movies_path = os.path.join(extract_dir, "movies.dat")
        ratings_path = os.path.join(extract_dir, "ratings.dat")
        
        movies = pd.read_csv(movies_path, sep='::', 
                            names=['movieId', 'title', 'genres'],
                            engine='python', encoding='latin-1')
        ratings = pd.read_csv(ratings_path, sep='::',
                             names=['userId', 'movieId', 'rating', 'timestamp'],
                             engine='python')
    else:  # 'full' or '20m'
        movies_path = os.path.join(extract_dir, "movies.csv")
        ratings_path = os.path.join(extract_dir, "ratings.csv")
        
        movies = pd.read_csv(movies_path)
        ratings = pd.read_csv(ratings_path)
    
    # Save processed files for easier access
    movies.to_csv(os.path.join(data_dir, f"movies_{size}.csv"), index=False)
    ratings.to_csv(os.path.join(data_dir, f"ratings_{size}.csv"), index=False)
    
    return movies, ratings

def initialize_system(rebuild=False):
    """Initialize the recommendation system"""
    print("Initializing movie recommendation system...")
    
    # Ensure NLTK resources are downloaded
    print("Checking NLTK resources...")
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        print("Downloading required NLTK resources...")
        nltk.download('punkt')
        nltk.download('stopwords')
        print("NLTK resources downloaded successfully!")
    
    # Create directories
    os.makedirs('app/models', exist_ok=True)
    os.makedirs('app/data', exist_ok=True)
    
    # Check if we have pre-built models
    embeddings_file = 'app/models/movie_embeddings.pkl'
    
    if os.path.exists(embeddings_file) and not rebuild:
        print("Loading pre-built models...")
        # Load movies and ratings
        try:
            movies = pd.read_csv('app/data/movies_processed.csv')
            ratings = pd.read_csv('app/data/ratings.csv')
            
            # Load embeddings
            bert_embedder = BERTMovieEmbeddings()
            movie_embeddings = bert_embedder.load_embeddings(embeddings_file)
            
            print("Models loaded successfully.")
            print(f"Loaded {len(movies)} movies and {len(ratings)} ratings.")
            
            # Initialize recommender
            recommender = MovieRecommender(movies, ratings, movie_embeddings)
            
            return recommender, bert_embedder
            
        except Exception as e:
            print(f"Error loading models: {e}")
            print("Rebuilding models from scratch...")
            rebuild = True
    
    if rebuild or not os.path.exists(embeddings_file):
        print("Building models from scratch...")
        # Download and load data
        print("Downloading MovieLens dataset...")
        movies, ratings = download_movielens_dataset(size='small')
        
        if movies is None or ratings is None:
            print("Failed to download or process dataset.")
            return None, None
        
        # Save raw data
        movies.to_csv('app/data/movies_raw.csv', index=False)
        ratings.to_csv('app/data/ratings.csv', index=False)
        
        # Process movie features
        print("Processing movie features...")
        movies_processed = extract_movie_features(movies)
        movies_processed.to_csv('app/data/movies_processed.csv', index=False)
        
        # Generate movie embeddings
        print("Generating BERT embeddings (this might take a while)...")
        bert_embedder = BERTMovieEmbeddings()
        movie_embeddings = bert_embedder.get_embeddings(movies_processed['nlp_features'].tolist())
        
        # Save embeddings
        print("Saving embeddings...")
        bert_embedder.save_embeddings(movie_embeddings, embeddings_file)
        
        # Initialize recommender
        recommender = MovieRecommender(movies_processed, ratings, movie_embeddings)
        
        # Save recommender
        recommender.save()
        
        print("System initialized successfully.")
        print(f"Processed {len(movies_processed)} movies and {len(ratings)} ratings.")
        
        return recommender, bert_embedder

if __name__ == "__main__":
    # Parse command-line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Initialize the movie recommendation system')
    parser.add_argument('--rebuild', action='store_true', help='Rebuild the models from scratch')
    args = parser.parse_args()
    
    # Initialize system
    initialize_system(rebuild=args.rebuild)
    
    print("System is ready to use. Run 'python run.py' to start the web application.")
