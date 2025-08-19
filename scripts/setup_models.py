"""Setup script to download and prepare all models for BuffettBot."""

import sys
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.config import *
from utils.data_processor import BuffettDataProcessor
from models.embeddings import BuffettEmbedder
from models.generator import BuffettGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_directories():
    """Create all necessary directories."""
    logger.info("📁 Creating project directories...")
    create_directories()

def download_base_models():
    """Download and cache base models."""
    logger.info("📥 Downloading base models...")
    
    # Download embedding model
    embedder = BuffettEmbedder(EMBEDDING_MODEL)
    embedder.save_model(EMBEDDINGS_MODEL_PATH)
    
    # Download base generation model
    generator = BuffettGenerator(BASE_MODEL)
    
    logger.info("✅ Base models downloaded and cached")

def prepare_data():
    """Load and validate the dataset."""
    logger.info("📊 Preparing dataset...")
    
    if not DATASET_PATH.exists():
        logger.error(f"❌ Dataset not found at {DATASET_PATH}")
        logger.info("Please place 'Dataset_Warren_Buffet_Clean.csv' in the data/ folder")
        return False
    
    # Load and validate data
    processor = BuffettDataProcessor(DATASET_PATH)
    df = processor.load_data()
    
    logger.info(f"✅ Dataset loaded: {len(df)} Q&A pairs")
    logger.info(f"Categories: {processor.get_categories()}")
    
    return True

def main():
    """Main setup function."""
    logger.info("🚀 Setting up BuffettBot...")
    
    # Step 1: Create directories
    setup_directories()
    
    # Step 2: Validate data
    if not prepare_data():
        return False
    
    # Step 3: Download models
    download_base_models()
    
    logger.info("🎉 BuffettBot setup complete!")
    logger.info("Next steps:")
    logger.info("1. Run 'python scripts/build_index.py' to create the search index")
    logger.info("2. Optional: Run 'python scripts/train_model.py' to fine-tune")
    logger.info("3. Run 'streamlit run app/streamlit_app.py' to start the app")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)