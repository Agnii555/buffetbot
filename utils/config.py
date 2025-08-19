"""Configuration settings for BuffettBot."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project root directory
ROOT_DIR = Path(__file__).parent.parent.absolute()

# Data paths
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "saved_models"
DATASET_PATH = DATA_DIR / "Dataset_Warren_Buffet_Clean.csv"

# Model configurations
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
BASE_MODEL = os.getenv("BASE_MODEL", "google/flan-t5-base")
FINE_TUNED_MODEL_PATH = MODELS_DIR / "fine_tuned"
EMBEDDINGS_MODEL_PATH = MODELS_DIR / "embeddings"
FAISS_INDEX_PATH = MODELS_DIR / "faiss_index" / "buffett_index.faiss"

# Generation settings
MAX_CONTEXT_LENGTH = int(os.getenv("MAX_CONTEXT_LENGTH", "512"))
MAX_RESPONSE_LENGTH = int(os.getenv("MAX_RESPONSE_LENGTH", "200"))

# Retrieval settings
TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "3"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))

# Training settings
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "5e-5"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "8"))
NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", "3"))

# App settings
APP_TITLE = os.getenv("APP_TITLE", "BuffettBot - Investment Wisdom")
PAGE_ICON = os.getenv("PAGE_ICON", "💰")
LAYOUT = os.getenv("LAYOUT", "wide")

def create_directories():
    """Create necessary directories for the project."""
    directories = [
        DATA_DIR,
        MODELS_DIR,
        MODELS_DIR / "embeddings",
        MODELS_DIR / "fine_tuned", 
        MODELS_DIR / "faiss_index"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    
    print("✅ Project directories created successfully!")

if __name__ == "__main__":
    create_directories()