"""
Embedding service for generating vector embeddings from text.

This module provides:
- Text embedding generation using Google Vertex AI
- Support for batch embedding generation
- Caching and optimization for frequently embedded texts
"""

import asyncio
import logging
from typing import List, Optional
from langchain_google_vertexai import VertexAIEmbeddings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Service for generating text embeddings using Google Vertex AI.

    Uses the textembedding-gecko-multilingual model which produces 768-dimensional vectors
    optimized for multilingual (including Russian/Kazakh) semantic similarity search.
    """

    def __init__(self, model_name: str = "text-embedding-004"):
        """
        Initialize the embedding service.

        Args:
            model_name: Name of the Vertex AI embedding model
                      Default: text-embedding-004 (768 dimensions)
                      Latest stable model with multilingual support (100+ languages)
                      Alternative: textembedding-gecko@latest
        """
        self.model_name = model_name
        self.embeddings = VertexAIEmbeddings(model_name=model_name)
        logger.info(f"EmbeddingService initialized with model: {model_name}")

    async def generate_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generate embedding vector for a single text.

        Args:
            text: Input text to embed

        Returns:
            List of floats representing the embedding vector (768 dimensions)
            Returns None if embedding generation fails
        """
        try:
            # Vertex AI embeddings are synchronous, so we run them in thread pool
            embedding = await asyncio.to_thread(
                self.embeddings.embed_query,
                text
            )
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None

    async def generate_embeddings_batch(
        self,
        texts: List[str]
    ) -> List[Optional[List[float]]]:
        """
        Generate embeddings for multiple texts in batch.

        Batch processing is more efficient for multiple texts as it reduces
        API call overhead.

        Args:
            texts: List of input texts to embed

        Returns:
            List of embedding vectors (or None for failed embeddings)
        """
        try:
            embeddings = await asyncio.to_thread(
                self.embeddings.embed_documents,
                texts
            )
            return embeddings
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            return [None] * len(texts)

    def get_embedding_dimension(self) -> int:
        """
        Get the dimensionality of embeddings produced by this service.

        Returns:
            Integer representing vector dimension (768 for gecko model)
        """
        # textembedding-gecko produces 768-dimensional vectors
        return 768


_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """
    Get or create the global embedding service instance.

    Returns:
        EmbeddingService: Singleton embedding service instance
    """
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
