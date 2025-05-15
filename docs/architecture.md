# System Architecture

This document describes the architecture of the movie recommendation system.

## Components

### 1. Data Processing

- Located in `app/utils/data_processor.py`
- Handles loading and preprocessing of the MovieLens dataset
- Extracts movie features (title, genres, year)
- Processes text using NLP techniques

### 2. NLP with BERT

- Located in `app/recommender/models.py`
- Implements BERTMovieEmbeddings for text embedding generation
- Implements BERTSentimentAnalyzer for review sentiment analysis
- Uses transformers from Hugging Face

### 3. Recommendation Engine

- Located in `app/recommender/engine.py`
- Implements content-based, collaborative, and hybrid filtering
- Generates movie recommendations based on user preferences
- Uses cosine similarity for content matching

### 4. Continuous Learning

- Located in `app/recommender/continuous_learning.py`
- Updates models based on user feedback
- Performs periodic retraining of sentiment models
- Adapts to changing user preferences

### 5. Web Interface

- Located in `app/templates/` and `app/static/`
- Provides a responsive web UI for interacting with the system
- Allows users to browse, search, rate, and get recommendations
- Built with Flask, HTML, CSS, and JavaScript
