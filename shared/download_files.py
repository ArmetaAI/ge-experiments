import os
from pathlib import Path
from google.cloud import storage
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_files_by_type(
    bucket_name: str,
    file_types: List[str],
    base_download_dir: str = ".",
    project_id: str = None
) -> Dict[str, int]:
    """
    Download all files from GCS bucket organized by their metadata file_type.

    Args:
        bucket_name: Name of the GCS bucket
        file_types: List of file types to download (e.g., ["ИРД", "ПСД"])
        base_download_dir: Base directory where files will be downloaded
        project_id: GCS project ID (optional)

    Returns:
        Dict with file type counts
    """
    client = storage.Client(project=project_id)
    bucket = client.bucket(bucket_name)

    logger.info(f"Connecting to bucket: {bucket_name}")

    # Get all blobs
    blobs = list(bucket.list_blobs())
    logger.info(f"Found {len(blobs)} total files in bucket")

    # Organize files by type
    files_by_type = {}
    for blob in blobs:
        file_type = blob.metadata.get('file_type') if blob.metadata else None
        if file_type and file_type in file_types:
            if file_type not in files_by_type:
                files_by_type[file_type] = []
            files_by_type[file_type].append(blob)

    # Download files
    download_counts = {}

    for file_type, blobs in files_by_type.items():
        logger.info(f"\nDownloading {len(blobs)} files for type: {file_type}")

        # Create directory for this file type
        type_dir = Path(base_download_dir) / file_type
        type_dir.mkdir(parents=True, exist_ok=True)

        download_counts[file_type] = 0

        for i, blob in enumerate(blobs, 1):
            # Get filename from blob name (last part of path)
            filename = Path(blob.name).name
            local_path = type_dir / filename

            logger.info(f"  [{i}/{len(blobs)}] Downloading: {blob.name} -> {local_path}")

            try:
                blob.download_to_filename(str(local_path))
                download_counts[file_type] += 1
            except Exception as e:
                logger.error(f"  Failed to download {blob.name}: {e}")

    return download_counts


if __name__ == "__main__":
    # Configuration from .env
    BUCKET_NAME = "gosexpert_test_project"
    PROJECT_ID = "gosexpert"
    FILE_TYPES = ["ИРД", "ПСД"]  # IRD and PSD in Cyrillic

    logger.info("="*60)
    logger.info("Starting file download from GCS")
    logger.info(f"Bucket: {BUCKET_NAME}")
    logger.info(f"File types: {FILE_TYPES}")
    logger.info("="*60)

    counts = download_files_by_type(
        bucket_name=BUCKET_NAME,
        file_types=FILE_TYPES,
        base_download_dir=".",
        project_id=PROJECT_ID
    )

    logger.info("\n" + "="*60)
    logger.info("Download Summary:")
    for file_type, count in counts.items():
        logger.info(f"  {file_type}: {count} files downloaded")
    logger.info("="*60)
