import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

def load_movielens_data(size='small'):
    """
    Download and load MovieLens dataset
    size: 'small' (100k), 'full' (20M) or '1m' (1M)
    """
    data_dir = os.path.join('app', 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    # URLs for different dataset sizes
    if size == 'small':
        url_movies = "https://files.grouplens.org/datasets/movielens/ml-latest-small/movies.csv"
        url_ratings = "https://files.grouplens.org/datasets/movielens/ml-latest-small/ratings.csv"
    elif size == '1m':
        url_movies = "https://files.grouplens.org/datasets/movielens/ml-1m/movies.dat"
        url_ratings = "https://files.grouplens.org/datasets/movielens/ml-1m/ratings.dat"
    else:  # full
        url_movies = "https://files.grouplens.org/datasets/movielens/ml-20m/movies.csv"
        url_ratings = "https://files.grouplens.org/datasets/movielens/ml-20m/ratings.csv"
    
    # Define file paths
    movies_file = os.path.join(data_dir, f'movies_{size}.csv')
    ratings_file = os.path.join(data_dir, f'ratings_{size}.csv')
    
    # Check if files already exist
    if os.path.exists(movies_file) and os.path.exists(ratings_file):
        print(f"Loading existing {size} dataset...")
        
        if size == '1m':
            movies = pd.read_csv(movies_file, sep='::', 
                                names=['movieId', 'title', 'genres'],
                                engine='python', encoding='latin-1')
            ratings = pd.read_csv(ratings_file, sep='::',
                                 names=['userId', 'movieId', 'rating', 'timestamp'],
                                 engine='python')
        else:
            movies = pd.read_csv(movies_file)
            ratings = pd.read_csv(ratings_file)
    else:
        print(f"Downloading {size} dataset...")
        
        if size == '1m':
            movies = pd.read_csv(url_movies, sep='::', 
                                names=['movieId', 'title', 'genres'],
                                engine='python', encoding='latin-1')
            ratings = pd.read_csv(url_ratings, sep='::',
                                 names=['userId', 'movieId', 'rating', 'timestamp'],
                                 engine='python')
        else:
            movies = pd.read_csv(url_movies)
            ratings = pd.read_csv(url_ratings)
        
        # Save to files
        movies.to_csv(movies_file, index=False)
        ratings.to_csv(ratings_file, index=False)
    
    return movies, ratings

def preprocess_text(text):
    """
    Preprocess text by tokenizing and removing stopwords
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    
    return ' '.join(filtered_tokens)

def extract_movie_features(movies_df):
    """
    Extract and preprocess movie titles and genres
    """
    # Extract year from title
    movies_df['year'] = movies_df['title'].str.extract(r'\((\d{4})\)$')
    movies_df['title_clean'] = movies_df['title'].str.replace(r'\s*\(\d{4}\)\s*$', '', regex=True)
    
    # Preprocess title
    movies_df['title_processed'] = movies_df['title_clean'].apply(preprocess_text)
    
    # Process genres
    movies_df['genres_processed'] = movies_df['genres'].apply(
        lambda x: preprocess_text(x.replace('|', ' ')) if isinstance(x, str) else ''
    )
    
    # Create a combined feature for NLP
    movies_df['nlp_features'] = movies_df['title_processed'] + ' ' + movies_df['genres_processed']
    
    return movies_df
