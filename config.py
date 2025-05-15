import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Flask configuration
SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-key-for-development-only')
DEBUG = os.environ.get('FLASK_ENV') == 'development'

# Application configuration
DATASET_SIZE = 'small'  # 'small', '1m', or 'full'
MODEL_UPDATE_INTERVAL = 7  # days
