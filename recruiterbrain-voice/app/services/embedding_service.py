"""
Optimized Embedding Service using E5-Base-V2
Singleton pattern with batch processing support
"""

import logging
from typing import List, Literal
import torch
from sentence_transformers import SentenceTransformer

from app.config.settings import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Generate 768-dim embeddings with GPU support and batch processing"""
    
    def __init__(self, model_name: str = None):
        self.model = None
        self.model_name = model_name or settings.embedding_model  # 'intfloat/e5-base-v2'
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"🎯 EmbeddingService ready (device: {self.device})")
    
    def load_model(self):
        """Lazy load model on first use"""
        if self.model is None:
            logger.info(f"📦 Loading {self.model_name}...")
            self.model = SentenceTransformer(self.model_name, device=self.device)
            logger.info(f"✅ Model loaded on {self.device}")
    
    def generate_embedding(
        self, 
        text: str, 
        prefix: Literal["passage", "query"] = "passage"
    ) -> List[float]:
        """
        Generate 768-dim embedding for single text
        
        Args:
            text: Text to embed
            prefix: 'passage' for documents, 'query' for search queries
        
        Returns:
            768-dimensional embedding vector
        """
        self.load_model()
        
        prefixed_text = f"{prefix}: {text}"
        embedding = self.model.encode(
            prefixed_text,
            normalize_embeddings=True,
            convert_to_numpy=True,  # Direct to NumPy (efficient)
            show_progress_bar=False
        )
        
        return embedding.tolist()
    
    def batch_generate_embeddings(
        self,
        texts: List[str],
        prefix: Literal["passage", "query"] = "passage",
        batch_size: int = 32
    ) -> List[List[float]]:
        """
        Batch generation (MUCH faster for multiple texts)
        
        Args:
            texts: List of texts to embed
            prefix: 'passage' for documents, 'query' for search queries
            batch_size: Number of texts to process at once
        
        Returns:
            List of 768-dimensional embedding vectors
        
        Performance:
            - 10 texts: ~3x faster than individual calls
            - 50 texts: ~10x faster than individual calls
        """
        self.load_model()
        
        # Add prefix to all texts
        prefixed_texts = [f"{prefix}: {text}" for text in texts]
        
        # Batch encode (much faster than loop)
        embeddings = self.model.encode(
            prefixed_texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 10  # Only show progress for large batches
        )
        
        return [emb.tolist() for emb in embeddings]
    
    def generate_query_embedding(self, query: str) -> List[float]:
        """
        Generate embedding for search query (convenience method)
        
        Args:
            query: Search query text
        
        Returns:
            768-dimensional embedding vector
        """
        return self.generate_embedding(query, prefix="query")


# Global singleton instance
_embedding_service = None

def get_embedding_service() -> EmbeddingService:
    """Get singleton embedding service"""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service