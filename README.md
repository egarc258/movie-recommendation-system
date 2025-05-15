# PersonaFilm: AI Movie Recommendation System

A personalized movie recommendation system using Natural Language Processing (NLP) and Machine Learning (ML) techniques. This system leverages the MovieLens dataset, BERT transformers, and collaborative filtering to provide tailored movie recommendations.

## Features

- **Data Processing**: Transforms movie data using tokenization and stop word removal
- **NLP with Transformers**: Uses BERT for sentiment analysis and text embeddings
- **Hybrid Recommendation Algorithm**: Combines content-based and collaborative filtering
- **Continuous Learning**: System improves as users interact with it
- **Web Interface**: Interactive UI for browsing, searching, and rating movies

## Tech Stack

- **Backend**: Python, Flask
- **NLP**: BERT, Transformers, NLTK
- **ML**: Scikit-learn, PyTorch
- **Data Processing**: Pandas, NumPy
- **Frontend**: HTML, CSS, JavaScript
- **Deployment**: Docker

## Installation

### Prerequisites

- Python 3.9+
- pip
- Virtual environment (recommended)

### Setup

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd movie-recommender
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Initialize the system:
   ```bash
   python initialize_system.py
   ```
   
   Note: The first initialization will take some time as it downloads the MovieLens dataset and generates BERT embeddings.

5. Run the application:
   ```bash
   python run.py
   ```

6. Open your browser and go to http://localhost:5000

## Docker Deployment

1. Build and run using Docker Compose:
   ```bash
   docker-compose up -d
   ```

2. Access the application at http://localhost:5000

## Usage

1. **Register/Login**: Enter a username to get started
2. **Browse Movies**: View popular movies or search for titles
3. **Rate Movies**: Rate movies on a scale from 0.5 to 5 stars
4. **Get Recommendations**: Once you've rated several movies, go to the "For You" tab
5. **Provide Feedback**: Leave detailed reviews to help improve recommendations

## System Architecture

### 1. Data Collection & Preprocessing

- Uses the MovieLens dataset (100K, 1M, or 20M versions)
- Extracts movie features (title, genres, year)
- Tokenizes and removes stop words to improve NLP performance

### 2. NLP with Transformers

- BERT generates semantic embeddings of movie content
- Sentiment analysis of user reviews enhances recommendations

### 3. Recommendation Engine

- Content-based filtering identifies similar movies
- Collaborative filtering finds users with similar tastes
- Hybrid approach combines both methods for optimal recommendations

### 4. Continuous Learning

- System periodically updates models based on new user data
- Sentiment analysis model improves through fine-tuning
- User profiles evolve as more ratings are collected

### 5. Web Interface

- Responsive design works on desktop and mobile
- Intuitive UI for discovering and rating movies
- Detailed feedback options for better personalization

## Project Structure

```
movie-recommender/
│
├── app/
│   ├── data/                  # Datasets and processed data
│   ├── models/                # Saved models and embeddings
│   ├── recommender/           # Recommendation system components
│   │   ├── engine.py          # Main recommendation logic
│   │   ├── models.py          # BERT models for NLP
│   │   └── continuous_learning.py # Learning components
│   ├── static/                # Static assets
│   │   ├── css/               # CSS styles
│   │   └── js/                # JavaScript files
│   ├── templates/             # HTML templates
│   ├── utils/                 # Utility functions
│   │   └── data_processor.py  # Data processing utilities
│   ├── views.py               # Flask routes
│   └── __init__.py            # Flask app initialization
│
├── config.py                  # Configuration settings
├── initialize_system.py       # System initialization script
├── run.py                     # Main application runner
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Docker configuration
├── docker-compose.yml         # Docker Compose configuration
└── README.md                  # Project documentation
```

## Further Improvements

- Add diversity enhancement to recommendations
- Implement explainable AI features to show why movies were recommended
- Add content enrichment from external APIs (e.g., TMDB)
- Implement A/B testing for recommendation algorithms
- Add user clustering for better cold-start recommendations

## License

MIT

## Acknowledgments

- [MovieLens](https://grouplens.org/datasets/movielens/) for the dataset
- [Hugging Face](https://huggingface.co/) for the transformer models
