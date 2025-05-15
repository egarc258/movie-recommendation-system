from flask import Blueprint, render_template, request, jsonify, session, current_app
import os
import json
import pickle
import pandas as pd
import numpy as np
from app.utils.data_processor import load_movielens_data, extract_movie_features
from app.recommender.engine import MovieRecommender
from app.recommender.models import BERTMovieEmbeddings, BERTSentimentAnalyzer
from app.recommender.continuous_learning import ContinuousLearning

main_bp = Blueprint('main', __name__)

# Global variables
recommender = None
user_sessions = {}
next_user_id = 1000

# Initialize the recommender
def load_recommender():
    global recommender, next_user_id
    
    # Check if we have pre-built models
    if os.path.exists('app/models/movie_embeddings.pkl'):
        try:
            # Load movies and ratings
            movies = pd.read_csv('app/data/movies_processed.csv')
            ratings = pd.read_csv('app/data/ratings.csv')
            
            # Load embeddings
            bert_embedder = BERTMovieEmbeddings()
            movie_embeddings = bert_embedder.load_embeddings('app/models/movie_embeddings.pkl')
            
            # Initialize recommender
            recommender = MovieRecommender(movies, ratings, movie_embeddings)
            
            # Update next_user_id
            next_user_id = max(ratings['userId'].max() + 1, 1000)
            
            return True
        except Exception as e:
            print(f"Error loading recommender: {e}")
            return False
    
    return False

# Try to load recommender on startup
recommender_loaded = load_recommender()

def initialize_app(app):
    """Initialize the application (to be called from create_app)"""
    global recommender_loaded, recommender
    
    if not recommender_loaded:
        from initialize_system import initialize_system
        recommender, _ = initialize_system(rebuild=False)
        recommender_loaded = True

@main_bp.route('/')
def index():
    """Render the home page"""
    global recommender
    
    # Lazy loading of the recommender if not loaded yet
    if recommender is None:
        load_recommender()
        
        # If still None, initialize from scratch
        if recommender is None:
            from initialize_system import initialize_system
            recommender, _ = initialize_system(rebuild=False)
    
    return render_template('index.html')

@main_bp.route('/register', methods=['POST'])
def register():
    """Register a new user"""
    global next_user_id
    
    # Get username from request
    data = request.get_json()
    username = data.get('username')
    
    if not username:
        return jsonify({'error': 'Username is required'}), 400
    
    # Check if user already exists
    if username in user_sessions:
        # Return existing user
        return jsonify({
            'user_id': user_sessions[username]['user_id'],
            'success': True
        })
    
    # Create new user ID
    user_id = next_user_id
    next_user_id += 1
    
    # Store user session
    user_sessions[username] = {
        'user_id': user_id,
        'ratings': {},
        'reviews': {}
    }
    
    return jsonify({
        'user_id': user_id,
        'success': True
    })

@main_bp.route('/recommend', methods=['GET'])
def recommend():
    """Get movie recommendations for a user"""
    # Get username from request
    username = request.args.get('username')
    
    if not username:
        return jsonify({'error': 'Username is required'}), 400
    
    # Check if user exists
    if username not in user_sessions:
        return jsonify({'error': 'User not found'}), 404
    
    user_id = user_sessions[username]['user_id']
    
    # Get hybrid recommendations
    try:
        rec_indices = recommender.hybrid_recommendations(user_id, n=10)
        
        # Get movie details
        recommendations = recommender.get_movie_details(rec_indices)
        
        return jsonify({'recommendations': recommendations})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@main_bp.route('/search', methods=['GET'])
def search():
    """Search for movies by title or genre"""
    # Get search query
    query = request.args.get('query', '').lower()
    
    if not query:
        return jsonify({'movies': []})
    
    # Search in titles and genres
    try:
        results = recommender.movies_df[
            recommender.movies_df['title_clean'].str.lower().str.contains(query) |
            recommender.movies_df['genres'].str.lower().str.contains(query)
        ]
        
        # Limit to 20 results
        results = results.head(20)
        
        # Convert to list of dictionaries
        movies_list = results.to_dict('records')
        
        return jsonify({'movies': movies_list})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@main_bp.route('/popular', methods=['GET'])
def popular():
    """Get popular movies based on average rating"""
    try:
        # Get average ratings per movie
        avg_ratings = recommender.ratings_df.groupby('movieId')['rating'].agg(['mean', 'count']).reset_index()
        
        # Filter movies with at least 10 ratings
        popular = avg_ratings[avg_ratings['count'] >= 10].sort_values('mean', ascending=False)
        
        # Get top 20 movies
        top_movies = popular.head(20)
        
        # Get movie details
        movies_list = []
        for _, row in top_movies.iterrows():
            movie = recommender.movies_df[recommender.movies_df['movieId'] == row['movieId']]
            if len(movie) > 0:
                movie_dict = movie.iloc[0].to_dict()
                movie_dict['avg_rating'] = row['mean']
                movie_dict['num_ratings'] = row['count']
                movies_list.append(movie_dict)
        
        return jsonify({'movies': movies_list})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@main_bp.route('/rate', methods=['POST'])
def rate():
    """Rate a movie"""
    # Get parameters from request
    data = request.get_json()
    username = data.get('username')
    movie_id = data.get('movie_id')
    rating = data.get('rating')
    
    if not all([username, movie_id, rating]):
        return jsonify({'error': 'Username, movie ID, and rating are required'}), 400
    
    # Validate rating
    try:
        rating = float(rating)
        if rating < 0.5 or rating > 5:
            return jsonify({'error': 'Rating must be between 0.5 and 5'}), 400
    except ValueError:
        return jsonify({'error': 'Rating must be a number'}), 400
    
    # Check if user exists
    if username not in user_sessions:
        return jsonify({'error': 'User not found'}), 404
    
    user_id = user_sessions[username]['user_id']
    
    # Update user's ratings
    user_sessions[username]['ratings'][movie_id] = rating
    
    # Update recommender
    try:
        recommender.update_user(user_id, movie_id, rating)
        
        # Save the ratings
        recommender.save()
        
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@main_bp.route('/feedback', methods=['POST'])
def feedback():
    """Submit detailed feedback about a movie"""
    # Get parameters from request
    data = request.get_json()
    username = data.get('username')
    movie_id = data.get('movie_id')
    rating = data.get('rating')
    review = data.get('review', '')
    
    if not all([username, movie_id, rating]):
        return jsonify({'error': 'Username, movie ID, and rating are required'}), 400
    
    # Validate rating
    try:
        rating = float(rating)
        if rating < 0.5 or rating > 5:
            return jsonify({'error': 'Rating must be between 0.5 and 5'}), 400
    except ValueError:
        return jsonify({'error': 'Rating must be a number'}), 400
    
    # Check if user exists
    if username not in user_sessions:
        return jsonify({'error': 'User not found'}), 404
    
    user_id = user_sessions[username]['user_id']
    
    # Update user's ratings and reviews
    user_sessions[username]['ratings'][movie_id] = rating
    user_sessions[username]['reviews'][movie_id] = review
    
    # Create feedback data
    feedback_data = pd.DataFrame([{
        'userId': user_id,
        'movieId': movie_id,
        'rating': rating,
        'review': review,
        'timestamp': int(pd.Timestamp.now().timestamp())
    }])
    
    # Update ratings in recommender
    try:
        recommender.update_user(user_id, movie_id, rating)
        
        # Add review to feedback database (for sentiment analysis training)
        feedback_file = 'app/data/user_feedback.csv'
        
        if os.path.exists(feedback_file):
            # Append to existing file
            feedback_data.to_csv(feedback_file, mode='a', header=False, index=False)
        else:
            # Create new file
            feedback_data.to_csv(feedback_file, index=False)
        
        # Save the ratings
        recommender.save()
        
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@main_bp.route('/update-models', methods=['POST'])
def update_models():
    """Trigger a model update (for admin use)"""
    # Check admin password (simple auth for example)
    data = request.get_json()
    password = data.get('password')
    
    if password != 'admin123':  # In production, use proper authentication
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        # Initialize sentiment analyzer
        sentiment_analyzer = BERTSentimentAnalyzer()
        
        # Initialize continuous learning
        cl = ContinuousLearning(recommender, sentiment_analyzer)
        
        # Load BERT embedder
        bert_embedder = BERTMovieEmbeddings()
        
        # Force model update
        success = cl.perform_update(bert_embedder=bert_embedder)
        
        if success:
            return jsonify({'success': True, 'message': 'Models updated successfully'})
        else:
            return jsonify({'error': 'Failed to update models'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500
