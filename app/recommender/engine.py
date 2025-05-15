import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import pickle

class MovieRecommender:
    def __init__(self, movies_df, ratings_df, embeddings):
        self.movies_df = movies_df
        self.ratings_df = ratings_df
        self.embeddings = embeddings
        self.user_profiles = {}
        self.item_similarity = cosine_similarity(embeddings)
        
        # Create user profiles
        self._create_user_profiles()
    
    def _create_user_profiles(self):
        """
        Create user profiles based on ratings
        """
        for user_id in self.ratings_df['userId'].unique():
            user_ratings = self.ratings_df[self.ratings_df['userId'] == user_id]
            
            if len(user_ratings) > 0:
                # Calculate weighted average embedding
                user_movie_indices = []
                for movie_id in user_ratings['movieId']:
                    movie_indices = self.movies_df[self.movies_df['movieId'] == movie_id].index
                    if len(movie_indices) > 0:
                        user_movie_indices.append(movie_indices[0])
                
                if len(user_movie_indices) > 0:
                    # Get ratings for these movies
                    user_movie_ratings = []
                    for idx in user_movie_indices:
                        movie_id = self.movies_df.iloc[idx]['movieId']
                        ratings = user_ratings[user_ratings['movieId'] == movie_id]['rating']
                        if len(ratings) > 0:
                            user_movie_ratings.append(ratings.values[0])
                    
                    # Normalize ratings to be between 0 and 1
                    normalized_ratings = np.array(user_movie_ratings) / 5.0
                    
                    # Get embeddings for these movies
                    movie_embeddings = self.embeddings[user_movie_indices]
                    
                    # Calculate weighted average
                    self.user_profiles[user_id] = np.average(movie_embeddings, axis=0, weights=normalized_ratings)
    
    def add_new_user(self, user_id, movie_ratings):
        """
        Add a new user to the system based on their movie ratings
        movie_ratings: Dictionary of {movie_id: rating}
        """
        # Add ratings to the ratings dataframe
        new_ratings = []
        for movie_id, rating in movie_ratings.items():
            new_ratings.append({
                'userId': user_id,
                'movieId': movie_id,
                'rating': rating,
                'timestamp': int(pd.Timestamp.now().timestamp())
            })
        
        new_ratings_df = pd.DataFrame(new_ratings)
        self.ratings_df = pd.concat([self.ratings_df, new_ratings_df], ignore_index=True)
        
        # Update user profile
        self._create_user_profiles()
    
    def update_user(self, user_id, movie_id, rating):
        """
        Update a user's rating for a movie
        """
        # Check if user already rated this movie
        existing_rating = self.ratings_df[(self.ratings_df['userId'] == user_id) & 
                                         (self.ratings_df['movieId'] == movie_id)]
        
        if len(existing_rating) > 0:
            # Update existing rating
            self.ratings_df.loc[existing_rating.index, 'rating'] = rating
            self.ratings_df.loc[existing_rating.index, 'timestamp'] = int(pd.Timestamp.now().timestamp())
        else:
            # Add new rating
            new_rating = pd.DataFrame([{
                'userId': user_id,
                'movieId': movie_id,
                'rating': rating,
                'timestamp': int(pd.Timestamp.now().timestamp())
            }])
            self.ratings_df = pd.concat([self.ratings_df, new_rating], ignore_index=True)
        
        # Update user profile
        self._create_user_profiles()
    
    def get_similar_users(self, user_id, n=10):
        """
        Find n most similar users to the given user
        """
        if user_id not in self.user_profiles:
            return []
        
        user_embedding = self.user_profiles[user_id]
        
        # Calculate similarity to all other users
        similarities = []
        for uid, embedding in self.user_profiles.items():
            if uid != user_id:
                sim = cosine_similarity([user_embedding], [embedding])[0][0]
                similarities.append((uid, sim))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:n]
    
    def content_based_recommendations(self, movie_indices, n=10):
        """
        Recommend movies similar to the given movies based on content
        movie_indices: List of indices in the movies_df
        """
        if not movie_indices:
            return []
        
        # Calculate average similarity to all movies
        avg_sim = np.zeros(len(self.embeddings))
        for idx in movie_indices:
            avg_sim += self.item_similarity[idx]
        
        avg_sim /= len(movie_indices)
        
        # Get top n similar movies
        top_indices = np.argsort(avg_sim)[::-1]
        
        # Filter out already seen movies
        recommendations = []
        for idx in top_indices:
            if idx not in movie_indices:
                recommendations.append(idx)
                if len(recommendations) >= n:
                    break
        
        return recommendations
    
    def user_based_recommendations(self, user_id, n=10):
        """
        Recommend movies based on similar users' preferences
        """
        if user_id not in self.user_profiles:
            return []
        
        # Get similar users
        similar_users = self.get_similar_users(user_id)
        
        if not similar_users:
            return []
        
        # Get movies rated by the user
        user_movies = set(self.ratings_df[self.ratings_df['userId'] == user_id]['movieId'])
        
        # Collect recommendations from similar users
        recommendations = {}
        for sim_user_id, similarity in similar_users:
            sim_user_ratings = self.ratings_df[self.ratings_df['userId'] == sim_user_id]
            
            for _, row in sim_user_ratings.iterrows():
                movie_id = row['movieId']
                
                # Skip movies the user has already rated
                if movie_id in user_movies:
                    continue
                
                # Weight the rating by user similarity
                weighted_rating = row['rating'] * similarity
                
                if movie_id not in recommendations:
                    recommendations[movie_id] = {'weighted_sum': weighted_rating, 'similarity_sum': similarity}
                else:
                    recommendations[movie_id]['weighted_sum'] += weighted_rating
                    recommendations[movie_id]['similarity_sum'] += similarity
        
        # Calculate the final score
        for movie_id in recommendations:
            recommendations[movie_id]['score'] = recommendations[movie_id]['weighted_sum'] / recommendations[movie_id]['similarity_sum']
        
        # Sort by score
        sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1]['score'], reverse=True)
        
        # Get the top n movie indices
        top_movie_ids = [movie_id for movie_id, _ in sorted_recommendations[:n]]
        top_movie_indices = []
        
        for movie_id in top_movie_ids:
            movie_idx = self.movies_df[self.movies_df['movieId'] == movie_id].index
            if len(movie_idx) > 0:
                top_movie_indices.append(movie_idx[0])
        
        return top_movie_indices
    
    def hybrid_recommendations(self, user_id, n=10, content_weight=0.5):
        """
        Generate hybrid recommendations combining content-based and collaborative filtering
        """
        # Get user's rated movies
        user_movies = self.ratings_df[self.ratings_df['userId'] == user_id]
        
        if len(user_movies) == 0:
            # If new user, rely solely on content-based
            return self.content_based_recommendations([], n)
        
        # Get movie indices the user has rated
        user_movie_indices = []
        for movie_id in user_movies['movieId']:
            movie_idx = self.movies_df[self.movies_df['movieId'] == movie_id].index
            if len(movie_idx) > 0:
                user_movie_indices.append(movie_idx[0])
        
        # Get content-based recommendations
        content_recs = self.content_based_recommendations(user_movie_indices, n)
        
        # Get user-based recommendations
        user_recs = self.user_based_recommendations(user_id, n)
        
        # Combine the two lists with weights
        all_recs = {}
        
        for idx, rec in enumerate(content_recs):
            all_recs[rec] = content_weight * (n - idx) / n
        
        for idx, rec in enumerate(user_recs):
            if rec in all_recs:
                all_recs[rec] += (1 - content_weight) * (n - idx) / n
            else:
                all_recs[rec] = (1 - content_weight) * (n - idx) / n
        
        # Sort by score
        sorted_recs = sorted(all_recs.items(), key=lambda x: x[1], reverse=True)
        
        # Return the top n recommendations
        return [rec for rec, _ in sorted_recs[:n]]
    
    def get_movie_details(self, movie_indices):
        """
        Get movie details for the given indices
        """
        if not movie_indices:
            return []
        
        movies = []
        for idx in movie_indices:
            movie = self.movies_df.iloc[idx].to_dict()
            movies.append(movie)
        
        return movies
    
    def save(self, directory='app/models'):
        """Save the recommender to files"""
        os.makedirs(directory, exist_ok=True)
        
        # Save ratings
        self.ratings_df.to_csv(os.path.join(directory, 'ratings.csv'), index=False)
        
        # Save user profiles
        with open(os.path.join(directory, 'user_profiles.pkl'), 'wb') as f:
            pickle.dump(self.user_profiles, f)
    
    def load(self, directory='app/models', movies_df=None):
        """Load the recommender from files"""
        # Load ratings
        ratings_path = os.path.join(directory, 'ratings.csv')
        if os.path.exists(ratings_path):
            self.ratings_df = pd.read_csv(ratings_path)
        
        # Load user profiles
        profiles_path = os.path.join(directory, 'user_profiles.pkl')
        if os.path.exists(profiles_path):
            with open(profiles_path, 'rb') as f:
                self.user_profiles = pickle.load(f)
        
        # Update movies_df if provided
        if movies_df is not None:
            self.movies_df = movies_df
