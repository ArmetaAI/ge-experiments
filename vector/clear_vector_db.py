"""
Script to remove all entries from the vector database (document_tags table).

This script:
1. Connects to the database using the same configuration as other scripts
2. Deletes all records from the document_tags table
3. Displays confirmation and statistics

Usage:
    python scripts/clear_vector_db.py
    python scripts/clear_vector_db.py --confirm  # Skip confirmation prompt
"""

import sys
import argparse
import logging
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from sqlalchemy.orm import Session
from app.core.database import SessionLocal, DocumentTag

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def clear_vector_database(confirm: bool = False) -> None:
    """
    Remove all entries from the document_tags table.

    Args:
        confirm: If True, skip confirmation prompt
    """
    db: Session = SessionLocal()

    try:
        # Count existing records
        total_count = db.query(DocumentTag).count()

        logger.info("=" * 80)
        logger.info("VECTOR DATABASE CLEANUP")
        logger.info("=" * 80)
        logger.info(f"Current records in document_tags table: {total_count}")

        if total_count == 0:
            logger.info("Database is already empty. Nothing to delete.")
            return

        # Confirmation prompt
        if not confirm:
            logger.warning("\nWARNING: This will delete ALL entries from the document_tags table!")
            response = input(f"Are you sure you want to delete {total_count} records? (yes/no): ")

            if response.lower() not in ['yes', 'y']:
                logger.info("Operation cancelled by user.")
                return

        # Delete all records
        logger.info("\nDeleting all records...")
        deleted_count = db.query(DocumentTag).delete()
        db.commit()

        logger.info("=" * 80)
        logger.info("CLEANUP COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Successfully deleted {deleted_count} records from document_tags table")
        logger.info("The vector database is now empty.")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
        db.rollback()
        raise
    finally:
        db.close()


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Clear all entries from the vector database (document_tags table)'
    )
    parser.add_argument(
        '--confirm',
        action='store_true',
        help='Skip confirmation prompt and delete immediately'
    )

    args = parser.parse_args()

    try:
        clear_vector_database(confirm=args.confirm)
    except KeyboardInterrupt:
        logger.info("\nOperation cancelled by user (Ctrl+C)")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to clear database: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
