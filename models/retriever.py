"""Retrieval system for BuffettBot using FAISS."""

import faiss
import numpy as np
import pickle
from typing import List, Dict, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class BuffettRetriever:
    """FAISS-based retrieval system for Warren Buffett Q&A pairs."""
    
    def __init__(self, embedder):
        self.embedder = embedder
        self.index = None
        self.documents = None
        self.metadata = None
    
    def build_index(self, texts: List[str], metadata: List[Dict] = None):
        """Build FAISS index from text documents."""
        logger.info("🏗️ Building FAISS index...")
        
        # Generate embeddings
        embeddings = self.embedder.encode_texts(texts)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype(np.float32))
        
        # Store documents and metadata
        self.documents = texts
        self.metadata = metadata or [{} for _ in texts]
        
        logger.info(f"✅ Built FAISS index with {self.index.ntotal} documents")
    
    def search(self, query: str, k: int = 3) -> List[Dict]:
        """Search for most similar documents to the query."""
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Encode query
        query_embedding = self.embedder.encode_texts([query])
        
        # Search FAISS index
        distances, indices = self.index.search(query_embedding.astype(np.float32), k)
        
        # Prepare results
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx >= 0:  # Valid index
                result = {
                    'text': self.documents[idx],
                    'metadata': self.metadata[idx],
                    'distance': float(distance),
                    'similarity': 1 / (1 + distance),  # Convert distance to similarity
                    'rank': i + 1
                }
                results.append(result)
        
        return results
    
    def retrieve_context(self, query: str, k: int = 3) -> str:
        """Retrieve and format context for generation."""
        results = self.search(query, k=k)
        
        context_parts = []
        for result in results:
            context_parts.append(result['text'])
        
        return "\n\n".join(context_parts)
    
    def save_index(self, save_path: Path):
        """Save the FAISS index and associated data."""
        if self.index is None:
            raise ValueError("No index to save. Build index first.")
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(save_path))
        
        # Save documents and metadata
        data_path = save_path.parent / "index_data.pkl"
        with open(data_path, 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'metadata': self.metadata
            }, f)
        
        logger.info(f"✅ Saved FAISS index to {save_path}")
    
    def load_index(self, index_path: Path):
        """Load a saved FAISS index."""
        try:
            # Load FAISS index
            self.index = faiss.read_index(str(index_path))
            
            # Load documents and metadata
            data_path = index_path.parent / "index_data.pkl"
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
                self.documents = data['documents']
                self.metadata = data['metadata']
            
            logger.info(f"✅ Loaded FAISS index from {index_path}")
        except Exception as e:
            logger.error(f"❌ Failed to load index: {e}")
            raise
    
    def get_stats(self) -> Dict:
        """Get statistics about the retrieval system."""
        if self.index is None:
            return {"status": "No index built"}
        
        return {
            "total_documents": self.index.ntotal,
            "embedding_dimension": self.index.d,
            "index_type": type(self.index).__name__
        }