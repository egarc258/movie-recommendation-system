# Setup Guide

## Prerequisites

- Python 3.9+
- pip
- Virtual environment (recommended)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/movie-recommendation-system.git
   cd movie-recommendation-system
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

6. Open your browser and go to http://localhost:8080
