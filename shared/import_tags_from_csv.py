"""
Import document tags from CSV file into the document_tags table.

This script:
1. Reads the CSV file with extracted titles and keywords
2. Generates embeddings for each tag
3. Inserts into document_tags table

Usage:
    python scripts/import_tags_from_csv.py --input results/document_tags_20251029_150352.csv
"""

import asyncio
import argparse
import csv
import logging
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from app.core.database import SessionLocal
from app.services.vector_query_engine import get_vector_query_engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def import_tags_from_csv(csv_path: str, skip_errors: bool = True):
    """
    Import tags from CSV file into database.

    Args:
        csv_path: Path to CSV file
        skip_errors: Whether to skip rows with errors
    """
    logger.info(f"Reading CSV: {csv_path}")

    # Read CSV file
    tags_to_import = []
    error_count = 0

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Skip rows with errors if specified
            if row.get('error') and skip_errors:
                error_count += 1
                logger.warning(f"Skipping {row['file_path']}: {row['error']}")
                continue

            # Skip rows without title
            if not row.get('title'):
                error_count += 1
                logger.warning(f"Skipping {row['file_path']}: No title extracted")
                continue

            # Parse keywords
            keywords_str = row.get('keywords', '')
            keywords = [k.strip() for k in keywords_str.split(',') if k.strip()]

            # Build rich description for better embeddings
            # Combine title + keywords for semantic understanding
            description = row['title']
            if keywords:
                description += f". Ключевые слова: {', '.join(keywords)}"

            tags_to_import.append({
                'tag_name': row['title'],
                'description': description,
                'keywords': keywords
            })

    logger.info(f"Found {len(tags_to_import)} valid tags to import")
    if error_count > 0:
        logger.info(f"Skipped {error_count} rows with errors/missing data")

    if not tags_to_import:
        logger.error("No valid tags to import!")
        return

    # Import into database
    db = SessionLocal()
    vector_engine = get_vector_query_engine()

    success_count = 0
    fail_count = 0

    for idx, tag_data in enumerate(tags_to_import, 1):
        logger.info(f"[{idx}/{len(tags_to_import)}] Adding: {tag_data['tag_name']}")

        try:
            result = await vector_engine.add_tag_with_embedding(
                tag_name=tag_data['tag_name'],
                description=tag_data['description'],
                keywords=tag_data['keywords'],
                db=db
            )

            if result:
                success_count += 1
                logger.info(f"  ✓ Success")
            else:
                fail_count += 1
                logger.warning(f"  ✗ Failed")

        except Exception as e:
            fail_count += 1
            logger.error(f"  ✗ Error: {e}")

        # Small delay to avoid rate limiting on embedding API
        await asyncio.sleep(0.3)

    db.close()

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("IMPORT SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total tags in CSV: {len(tags_to_import)}")
    logger.info(f"Successfully imported: {success_count}")
    logger.info(f"Failed: {fail_count}")
    logger.info("=" * 80)


async def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Import document tags from CSV into database'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to CSV file (e.g., results/document_tags_20251029_150352.csv)'
    )
    parser.add_argument(
        '--include-errors',
        action='store_true',
        help='Include rows with errors (default: skip them)'
    )

    args = parser.parse_args()

    csv_path = Path(args.input)
    if not csv_path.exists():
        logger.error(f"CSV file not found: {csv_path}")
        return

    logger.info("=" * 80)
    logger.info("IMPORTING TAGS FROM CSV")
    logger.info("=" * 80)
    logger.info(f"Input file: {csv_path}")
    logger.info(f"Skip errors: {not args.include_errors}")
    logger.info("=" * 80 + "\n")

    await import_tags_from_csv(
        str(csv_path),
        skip_errors=not args.include_errors
    )


if __name__ == "__main__":
    asyncio.run(main())