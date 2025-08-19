"""Embedding models for semantic search in BuffettBot."""

import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Union
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class BuffettEmbedder:
    """Handles text embeddings for semantic search."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model."""
        try:
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"✅ Loaded embedding model: {self.model_name}")
        except Exception as e:
            logger.error(f"❌ Failed to load embedding model: {e}")
            raise
    
    def encode_texts(self, texts: List[str], convert_to_numpy: bool = True) -> np.ndarray:
        """Encode a list of texts into embeddings."""
        if not texts:
            raise ValueError("No texts provided for encoding")
        
        try:
            embeddings = self.model.encode(
                texts, 
                convert_to_numpy=convert_to_numpy,
                show_progress_bar=True
            )
            logger.info(f"✅ Encoded {len(texts)} texts into {embeddings.shape} embeddings")
            return embeddings
        except Exception as e:
            logger.error(f"❌ Failed to encode texts: {e}")
            raise
    
    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text into embedding."""
        return self.encode_texts([text])[0]
    
    def get_similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts."""
        embeddings = self.encode_texts([text1, text2])
        
        # Calculate cosine similarity
        dot_product = np.dot(embeddings[0], embeddings[1])
        norms = np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        similarity = dot_product / norms
        
        return float(similarity)
    
    def save_model(self, save_path: Path):
        """Save the embedding model to disk."""
        try:
            save_path.mkdir(parents=True, exist_ok=True)
            self.model.save(str(save_path))
            logger.info(f"✅ Saved embedding model to {save_path}")
        except Exception as e:
            logger.error(f"❌ Failed to save model: {e}")
            raise
    
    @classmethod
    def load_model(cls, model_path: Path) -> 'BuffettEmbedder':
        """Load a saved embedding model."""
        try:
            embedder = cls.__new__(cls)
            embedder.model = SentenceTransformer(str(model_path))
            embedder.model_name = str(model_path)
            logger.info(f"✅ Loaded embedding model from {model_path}")
            return embedder
        except Exception as e:
            logger.error(f"❌ Failed to load model from {model_path}: {e}")
            raise