"""
Vector query engine for semantic similarity search using pgvector.

This module provides:
- Similarity search for document tags using cosine distance
- Efficient vector operations using pgvector and HNSW indexing
- Integration with SQLAlchemy for database operations
"""

import asyncio
import logging
from typing import List, Optional, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.infrastructure.persistence.database.models import DocumentTag
from app.infrastructure.ai.embedding.vertex_ai_embedding import get_embedding_service

logger = logging.getLogger(__name__)


class VectorQueryEngine:
    """
    Engine for performing vector similarity searches on document tags.

    Uses pgvector extension for efficient similarity search with HNSW indexing.
    """

    def __init__(self):
        """Initialize the vector query engine."""
        self.embedding_service = get_embedding_service()
        logger.info("VectorQueryEngine initialized")

    async def find_closest_tag(
        self,
        query_text: str,
        db: Session,
        top_k: int = 1,
        similarity_threshold: float = 0.5
    ) -> Optional[Tuple[str, float]]:
        """
        Find the closest matching tag for a given text using semantic similarity.

        Args:
            query_text: Input text to match against tags
            db: Database session
            top_k: Number of top results to return (default: 1)
            similarity_threshold: Minimum similarity score (0-1) to consider a match
                                Default: 0.5 (50% similarity)

        Returns:
            Tuple of (tag_name, similarity_score) for the best match
            Returns None if no match exceeds the threshold
        """
        try:
            query_embedding = await self.embedding_service.generate_embedding(query_text)

            if query_embedding is None:
                logger.error("Failed to generate embedding for query text")
                return None

            embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"

            # Perform similarity search using cosine distance
            # Note: pgvector's <=> operator computes cosine distance (1 - cosine_similarity)
            # So we need to convert it back: similarity = 1 - distance
            # Use raw SQL interpolation since pgvector syntax doesn't work with bound parameters
            query_sql = f"""
                SELECT
                    tag_name,
                    1 - (embedding <=> '{embedding_str}'::vector) AS similarity
                FROM document_tags
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> '{embedding_str}'::vector
                LIMIT {top_k}
            """

            result = await asyncio.to_thread(
                db.execute,
                text(query_sql)
            )

            rows = result.fetchall()

            if not rows:
                logger.warning("No tags found in database")
                return None

            best_match = rows[0]
            tag_name = best_match[0]
            similarity = float(best_match[1])

            logger.info(f"Best match: {tag_name} (similarity: {similarity:.3f})")

            if similarity < similarity_threshold:
                logger.info(
                    f"Best match similarity {similarity:.3f} below threshold {similarity_threshold}"
                )
                return None

            return (tag_name, similarity)

        except Exception as e:
            logger.error(f"Error in find_closest_tag: {e}")
            return None

    async def find_top_k_tags(
        self,
        query_text: str,
        db: Session,
        top_k: int = 5,
        similarity_threshold: float = 0.3
    ) -> List[Tuple[str, float]]:
        """
        Find top-k closest matching tags for a given text.

        Args:
            query_text: Input text to match against tags
            db: Database session
            top_k: Number of top results to return
            similarity_threshold: Minimum similarity score to include

        Returns:
            List of tuples (tag_name, similarity_score) sorted by similarity
        """
        try:
            query_embedding = await self.embedding_service.generate_embedding(query_text)

            if query_embedding is None:
                logger.error("Failed to generate embedding for query text")
                return []

            embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"

            query_sql = f"""
                SELECT
                    tag_name,
                    1 - (embedding <=> '{embedding_str}'::vector) AS similarity
                FROM document_tags
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> '{embedding_str}'::vector
                LIMIT {top_k}
            """

            result = await asyncio.to_thread(
                db.execute,
                text(query_sql)
            )

            rows = result.fetchall()

            # Filter by threshold and return
            matches = [
                (row[0], float(row[1]))
                for row in rows
                if float(row[1]) >= similarity_threshold
            ]

            logger.info(f"Found {len(matches)} tags above threshold {similarity_threshold}")
            return matches

        except Exception as e:
            logger.error(f"Error in find_top_k_tags: {e}")
            return []

    async def add_tag_with_embedding(
        self,
        tag_name: str,
        document_type: Optional[str],
        semantic_description: Optional[str],
        search_terms: Optional[List[str]],
        db: Session
    ) -> bool:
        """
        Add a new tag to the database with its embedding.

        Args:
            tag_name: Specific document title/name
            document_type: General document category (e.g., "Архитектурно-планировочное задание")
            semantic_description: Description of document purpose/function
            search_terms: List of search terms for matching
            db: Database session

        Returns:
            True if successful, False otherwise
        """
        try:
            # Create optimized text for embedding
            # The tag_name is the REQUIREMENT (what we're looking for)
            # The document_type, semantic_description, and search_terms are extracted
            # from EXAMPLE documents that match this requirement
            # This helps the model understand what documents of this type look like

            parts = []

            parts.append(tag_name)

            if document_type:
                parts.append(document_type)

            if semantic_description:
                parts.append(semantic_description)

            if search_terms and len(search_terms) > 0:
                parts.append(', '.join(search_terms)) 

            text_to_embed = ". ".join(parts)

            logger.info(f"Embedding text ({len(text_to_embed)} chars): {text_to_embed[:200]}...")

            embedding = await self.embedding_service.generate_embedding(text_to_embed)

            if embedding is None:
                logger.error(f"Failed to generate embedding for tag: {tag_name}")
                return False

            embedding_str = "[" + ",".join(map(str, embedding)) + "]"

            from datetime import datetime, timezone
            new_tag = DocumentTag(
                tag_name=tag_name,
                description=semantic_description or "",
                keywords=search_terms or [],
                embedding=embedding_str,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc)
            )

            await asyncio.to_thread(db.add, new_tag)
            await asyncio.to_thread(db.commit)

            logger.info(f"Successfully added tag: {tag_name}")
            return True

        except Exception as e:
            logger.error(f"Error adding tag: {e}")
            await asyncio.to_thread(db.rollback)
            return False


_vector_query_engine: Optional[VectorQueryEngine] = None


def get_vector_query_engine() -> VectorQueryEngine:
    """
    Get or create the global vector query engine instance.

    Returns:
        VectorQueryEngine: Singleton vector query engine instance
    """
    global _vector_query_engine
    if _vector_query_engine is None:
        _vector_query_engine = VectorQueryEngine()
    return _vector_query_engine
