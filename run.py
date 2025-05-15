from app import create_app
import threading
import time
import os
from app.recommender.continuous_learning import ContinuousLearning
from app.recommender.models import BERTSentimentAnalyzer, BERTMovieEmbeddings

app = create_app()

# Get recommender from views
from app.views import recommender

# Schedule continuous learning updates
def update_models_periodically():
    """Update models periodically in a background thread"""
    while True:
        # Sleep for a day
        time.sleep(86400)  # 24 hours
        
        try:
            # Make sure recommender is loaded
            if recommender is None:
                continue
                
            print("Checking for scheduled model update...")
            
            # Initialize sentiment analyzer
            sentiment_analyzer = BERTSentimentAnalyzer()
            
            # Initialize continuous learning
            cl = ContinuousLearning(recommender, sentiment_analyzer)
            
            # Check if update is needed
            if cl.check_for_update():
                print("Performing scheduled model update...")
                
                # Load BERT embedder
                bert_embedder = BERTMovieEmbeddings()
                
                # Perform update
                cl.perform_update(bert_embedder=bert_embedder)
            else:
                print("No update needed at this time.")
        
        except Exception as e:
            print(f"Error during scheduled update: {e}")

if __name__ == '__main__':
    # Check if models directory exists
    if not os.path.exists('app/models'):
        os.makedirs('app/models', exist_ok=True)
    
    # Start update thread
    update_thread = threading.Thread(target=update_models_periodically, daemon=True)
    update_thread.start()
    
    # Run the app using a different port (8080 instead of 5000)
    app.run(debug=True, host='0.0.0.0', port=8080)
