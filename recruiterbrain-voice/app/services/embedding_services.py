"""
Embedding generation using intfloat/e5-base-v2
"""

import torch
from sentence_transformers import SentenceTransformer
from typing import List
import logging

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Generate 768-dim embeddings for text"""
    
    def __init__(self):
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Lazy load the embedding model"""
        if self.model is None:
            logger.info("📦 Loading embedding model: intfloat/e5-base-v2...")
            self.model = SentenceTransformer('intfloat/e5-base-v2')
            logger.info("✅ Embedding model loaded")
    
    def generate_embedding(self, text: str, prefix: str = "passage") -> List[float]:
        """
        Generate 768-dim embedding for text
        
        Args:
            text: Text to embed
            prefix: 'passage' for documents, 'query' for search queries
        
        Returns:
            List of 768 floats
        """
        # E5 models require prefix
        prefixed_text = f"{prefix}: {text}"
        
        with torch.no_grad():
            embedding = self.model.encode(
                prefixed_text,
                convert_to_tensor=True,
                normalize_embeddings=True  # Cosine similarity
            )
            
            # Convert to list
            embedding_list = embedding.cpu().numpy().tolist()
        
        return embedding_list
    
    def generate_batch_embeddings(
        self, 
        texts: List[str], 
        prefix: str = "passage"
    ) -> List[List[float]]:
        """Generate embeddings for multiple texts (faster)"""
        prefixed_texts = [f"{prefix}: {text}" for text in texts]
        
        with torch.no_grad():
            embeddings = self.model.encode(
                prefixed_texts,
                convert_to_tensor=True,
                normalize_embeddings=True,
                batch_size=32
            )
            
            # Convert to list of lists
            embeddings_list = embeddings.cpu().numpy().tolist()
        
        return embeddings_list


# Global instance (loaded once)
_embedding_service = None

def get_embedding_service() -> EmbeddingService:
    """Get singleton embedding service"""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service