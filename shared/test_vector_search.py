"""
Script to test vector similarity search on document tags.

This script:
1. Connects to the database
2. Takes a search query from command line
3. Performs vector similarity search
4. Displays top matching tags with similarity scores

Usage:
    python scripts/test_vector_search.py "search query text"
    python scripts/test_vector_search.py "архитектурно-планировочное задание" --top-k 5
"""

import sys
import asyncio
import argparse
import logging
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from app.infrastructure.persistence.database.models import SessionLocal
from app.infrastructure.ai.vector_search.vertex_ai_vector_engine import VectorQueryEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_vector_search(
    query: str,
    top_k: int = 5,
    threshold: float = 0.3
) -> None:
    """
    Test vector similarity search with a query.

    Args:
        query: Search query text
        top_k: Number of top results to return
        threshold: Minimum similarity threshold (0-1)
    """
    db = SessionLocal()

    try:
        logger.info("=" * 80)
        logger.info("VECTOR SIMILARITY SEARCH TEST")
        logger.info("=" * 80)
        logger.info(f"Query: {query}")
        logger.info(f"Top-K: {top_k}")
        logger.info(f"Threshold: {threshold}")
        logger.info("=" * 80)

        # Get vector query engine
        engine = VectorQueryEngine()

        # Perform search
        logger.info("\nSearching...")
        results = await engine.find_top_k_tags(
            query_text=query,
            db=db,
            top_k=top_k,
            similarity_threshold=threshold
        )

        # Display results
        logger.info("\n" + "=" * 80)
        logger.info("RESULTS")
        logger.info("=" * 80)

        if not results:
            logger.info("No matches found above threshold.")
        else:
            logger.info(f"Found {len(results)} matches:\n")
            for idx, (tag_name, similarity) in enumerate(results, 1):
                similarity_pct = similarity * 100
                logger.info(f"{idx}. [{similarity_pct:.1f}%] {tag_name}")

        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Error during search: {e}")
        raise
    finally:
        db.close()


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Test vector similarity search on document tags'
    )
    parser.add_argument(
        'query',
        type=str,
        help='Search query text'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=5,
        help='Number of top results to return (default: 5)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.3,
        help='Minimum similarity threshold 0-1 (default: 0.3)'
    )

    args = parser.parse_args()

    try:
        asyncio.run(test_vector_search(
            query=args.query,
            top_k=args.top_k,
            threshold=args.threshold
        ))
    except KeyboardInterrupt:
        logger.info("\nOperation cancelled by user (Ctrl+C)")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Search failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
