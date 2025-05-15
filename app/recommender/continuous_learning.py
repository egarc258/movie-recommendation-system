import pandas as pd
import os
import time
import pickle
from datetime import datetime

class ContinuousLearning:
    def __init__(self, recommender, sentiment_analyzer=None, update_interval=7):
        """
        Initialize continuous learning component
        update_interval: Number of days between model updates
        """
        self.recommender = recommender
        self.sentiment_analyzer = sentiment_analyzer
        self.update_interval = update_interval
        self.last_update = datetime.now()
        
        # Create a directory for model snapshots
        os.makedirs('app/models/snapshots', exist_ok=True)
        
        # Load last update time if exists
        self._load_state()
    
    def _load_state(self):
        """Load continuous learning state"""
        state_path = 'app/models/cl_state.pkl'
        if os.path.exists(state_path):
            with open(state_path, 'rb') as f:
                state = pickle.load(f)
                self.last_update = state.get('last_update', datetime.now())
    
    def _save_state(self):
        """Save continuous learning state"""
        state_path = 'app/models/cl_state.pkl'
        state = {
            'last_update': self.last_update
        }
        with open(state_path, 'wb') as f:
            pickle.dump(state, f)
    
    def collect_user_feedback(self):
        """
        Collect user feedback for continuous learning
        """
        # Get all user ratings
        return self.recommender.ratings_df
    
    def update_sentiment_model(self, reviews_data):
        """
        Update the sentiment analysis model based on user reviews
        reviews_data: DataFrame with movie reviews and ratings
        """
        if self.sentiment_analyzer is None:
            return
        
        # Convert ratings to sentiment labels (0=negative, 1=neutral, 2=positive)
        def rating_to_sentiment(rating):
            if rating <= 2:
                return 0  # Negative
            elif rating <= 3.5:
                return 1  # Neutral
            else:
                return 2  # Positive
        
        reviews_data['sentiment'] = reviews_data['rating'].apply(rating_to_sentiment)
        
        # Extract review texts and sentiments
        texts = reviews_data['review'].tolist()
        labels = reviews_data['sentiment'].tolist()
        
        # Fine-tune the sentiment model
        self.sentiment_analyzer.fine_tune(texts, labels, epochs=1)
        
        # Save the updated model
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.sentiment_analyzer.save_model(f'app/models/snapshots/sentiment_model_{timestamp}')
    
    def update_embeddings(self, bert_embedder):
        """
        Update movie embeddings based on new data
        """
        # Re-calculate embeddings for all movies
        new_embeddings = bert_embedder.get_embeddings(self.recommender.movies_df['nlp_features'].tolist())
        
        # Update the recommender with new embeddings
        self.recommender.embeddings = new_embeddings
        
        # Save the updated embeddings
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        bert_embedder.save_embeddings(new_embeddings, f'app/models/snapshots/movie_embeddings_{timestamp}.pkl')
        
        # Also save to standard location
        bert_embedder.save_embeddings(new_embeddings, 'app/models/movie_embeddings.pkl')
    
    def check_for_update(self):
        """
        Check if it's time to update the models
        """
        now = datetime.now()
        days_since_update = (now - self.last_update).days
        
        if days_since_update >= self.update_interval:
            return True
        return False
    
    def perform_update(self, bert_embedder=None):
        """
        Perform a model update
        """
        if not self.check_for_update():
            return False
        
        print("Performing model update...")
        
        # Collect user feedback
        feedback_data = self.collect_user_feedback()
        
        # Update models
        try:
            # Update embeddings if bert_embedder is provided
            if bert_embedder is not None:
                self.update_embeddings(bert_embedder)
            
            # Update user profiles
            self.recommender._create_user_profiles()
            
            # If we have review text and a sentiment analyzer, update sentiment model
            if self.sentiment_analyzer is not None and 'review' in feedback_data.columns:
                self.update_sentiment_model(feedback_data)
            
            # Save the recommender
            self.recommender.save()
            
            # Update last update time
            self.last_update = datetime.now()
            self._save_state()
            
            print("Model update completed successfully.")
            return True
        
        except Exception as e:
            print(f"Error during model update: {e}")
            return False
