"""Build FAISS index for BuffettBot retrieval system."""

import sys
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.config import *
from utils.data_processor import BuffettDataProcessor
from models.embeddings import BuffettEmbedder
from models.retriever import BuffettRetriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def build_retrieval_index():
    """Build the FAISS index for semantic search."""
    logger.info("🔍 Building retrieval index...")
    
    # Load data
    processor = BuffettDataProcessor(DATASET_PATH)
    qa_pairs = processor.get_qa_pairs()
    
    # Prepare texts and metadata for indexing
    texts = []
    metadata = []
    
    for qa in qa_pairs:
        # Add answer for retrieval
        texts.append(qa['answer'])
        metadata.append({
            'type': 'answer',
            'question': qa['question'],
            'answer': qa['answer'],
            'category': qa['category']
        })
    
    logger.info(f"📝 Prepared {len(texts)} texts for indexing")
    
    # Load embedder
    if EMBEDDINGS_MODEL_PATH.exists():
        embedder = BuffettEmbedder.load_model(EMBEDDINGS_MODEL_PATH)
    else:
        embedder = BuffettEmbedder(EMBEDDING_MODEL)
    
    # Build retriever
    retriever = BuffettRetriever(embedder)
    retriever.build_index(texts, metadata)
    
    # Save index
    retriever.save_index(FAISS_INDEX_PATH)
    
    # Test the index
    test_query = "Why is margin of safety important?"
    results = retriever.search(test_query, k=3)
    
    logger.info("🧪 Testing retrieval system:")
    logger.info(f"Query: {test_query}")
    for i, result in enumerate(results[:2]):
        logger.info(f"Result {i+1}: {result['text'][:100]}...")
    
    logger.info("✅ FAISS index built and saved successfully!")

if __name__ == "__main__":
    build_retrieval_index()