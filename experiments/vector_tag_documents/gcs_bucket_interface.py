"""
GCS Bucket Interface.

This module provides a comprehensive interface for interacting with a specific
GCS bucket, including listing files, retrieving metadata, filtering by metadata,
and downloading files.
"""

from typing import Optional, List, Dict, Any
from pathlib import Path
import os

from google.cloud import storage


class GCSBucketInterface:
    """
    Interface for interacting with a specific GCS bucket.

    This class provides methods for:
    - Listing all files in the bucket with their metadata
    - Getting specific file information
    - Filtering files by metadata key-value pairs
    - Downloading files to a local directory
    """

    def __init__(self, bucket_name: str = "gosexpert_categorize"):
        """
        Initialize the GCS Bucket Interface.

        Args:
            bucket_name: Name of the GCS bucket to interact with
        """
        self.client = storage.Client()
        self.bucket_name = bucket_name
        self.bucket = self.client.bucket(bucket_name)
        print(f"[GCSBucketInterface] Initialized for bucket: {bucket_name}")

    def list(self, prefix: Optional[str] = None, max_results: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        List all files in the bucket with their metadata.

        Args:
            prefix: Optional prefix to filter files (e.g., "projects/123/")
            max_results: Optional maximum number of results to return

        Returns:
            List[Dict[str, Any]]: List of dictionaries containing file information:
                - name: File path in bucket
                - size: File size in bytes
                - content_type: MIME type
                - time_created: Creation timestamp
                - updated: Last update timestamp
                - metadata: Custom metadata dictionary
                - md5_hash: MD5 hash of the file
                - etag: Entity tag
                - generation: Object generation number
                - metageneration: Metadata generation number
                - storage_class: Storage class
                - gcs_uri: Full GCS URI (gs://bucket/path)
        """
        try:
            blobs = self.bucket.list_blobs(prefix=prefix, max_results=max_results)

            files = []
            for blob in blobs:
                file_info = self._blob_to_dict(blob)
                files.append(file_info)

            print(f"[GCSBucketInterface] Listed {len(files)} files" +
                  (f" with prefix '{prefix}'" if prefix else ""))

            return files

        except Exception as e:
            print(f"[GCSBucketInterface] Failed to list files: {str(e)}")
            raise Exception(f"Failed to list files from bucket: {str(e)}")

    def get(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific file including its metadata.

        Args:
            file_path: Path to the file in the bucket (e.g., "projects/123/psd/file.pdf")

        Returns:
            Optional[Dict[str, Any]]: Dictionary containing file information, or None if not found
        """
        try:
            blob = self.bucket.blob(file_path)

            # Check if blob exists
            if not blob.exists():
                print(f"[GCSBucketInterface] File not found: {file_path}")
                return None

            # Reload to get all metadata
            blob.reload()

            file_info = self._blob_to_dict(blob)

            print(f"[GCSBucketInterface] Retrieved file info: {file_path}")

            return file_info

        except Exception as e:
            print(f"[GCSBucketInterface] Failed to get file {file_path}: {str(e)}")
            raise Exception(f"Failed to get file information: {str(e)}")

    def get_files_by_metadata(
        self,
        metadata_key: str,
        metadata_value: str,
        prefix: Optional[str] = None,
        max_results: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get all files that have a specific metadata key-value pair.

        Args:
            metadata_key: The metadata key to filter by
            metadata_value: The metadata value to match
            prefix: Optional prefix to filter files before checking metadata
            max_results: Optional maximum number of results to return

        Returns:
            List[Dict[str, Any]]: List of dictionaries containing file information
                                  for files matching the metadata criteria
        """
        try:
            # Get all files (with optional prefix filter)
            all_files = self.list(prefix=prefix, max_results=max_results)

            # Filter by metadata
            matching_files = []
            for file_info in all_files:
                metadata = file_info.get("metadata", {})
                if metadata and metadata.get(metadata_key) == metadata_value:
                    matching_files.append(file_info)

            print(f"[GCSBucketInterface] Found {len(matching_files)} files with " +
                  f"metadata {metadata_key}={metadata_value}")

            return matching_files

        except Exception as e:
            print(f"[GCSBucketInterface] Failed to filter by metadata: {str(e)}")
            raise Exception(f"Failed to filter files by metadata: {str(e)}")

    def download_files(
        self,
        files: List[Dict[str, Any]],
        download_folder: Optional[str] = None,
        preserve_structure: bool = True
    ) -> List[str]:
        """
        Download a list of files to a local directory.

        Args:
            files: List of file dictionaries (from list() or get_files_by_metadata())
            download_folder: Local folder to download files to.
                           Defaults to "./downloads" in current directory
            preserve_structure: If True, preserves the GCS folder structure locally.
                              If False, downloads all files to the root of download_folder

        Returns:
            List[str]: List of local file paths where files were downloaded

        Raises:
            Exception: If download fails
        """
        try:
            # Set default download folder
            if download_folder is None:
                download_folder = os.path.join(os.getcwd(), "downloads")

            # Create download folder if it doesn't exist
            Path(download_folder).mkdir(parents=True, exist_ok=True)

            downloaded_paths = []

            for file_info in files:
                file_path = file_info["name"]
                blob = self.bucket.blob(file_path)

                # Determine local file path
                if preserve_structure:
                    # Keep the folder structure
                    local_path = os.path.join(download_folder, file_path)
                    # Create subdirectories if needed
                    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
                else:
                    # Flatten structure - just use filename
                    filename = os.path.basename(file_path)
                    local_path = os.path.join(download_folder, filename)

                # Download the file
                blob.download_to_filename(local_path)
                downloaded_paths.append(local_path)

                print(f"[GCSBucketInterface] Downloaded: {file_path} -> {local_path}")

            print(f"[GCSBucketInterface] Successfully downloaded {len(downloaded_paths)} files " +
                  f"to {download_folder}")

            return downloaded_paths

        except Exception as e:
            print(f"[GCSBucketInterface] Failed to download files: {str(e)}")
            raise Exception(f"Failed to download files: {str(e)}")

    def _blob_to_dict(self, blob: storage.Blob) -> Dict[str, Any]:
        """
        Convert a GCS blob to a dictionary with all relevant information.

        Args:
            blob: GCS blob object

        Returns:
            Dict[str, Any]: Dictionary containing blob information
        """
        return {
            "name": blob.name,
            "size": blob.size,
            "content_type": blob.content_type,
            "time_created": blob.time_created,
            "updated": blob.updated,
            "metadata": blob.metadata or {},
            "md5_hash": blob.md5_hash,
            "etag": blob.etag,
            "generation": blob.generation,
            "metageneration": blob.metageneration,
            "storage_class": blob.storage_class,
            "gcs_uri": f"gs://{self.bucket_name}/{blob.name}"
        }


def get_bucket_interface(bucket_name: str = "gosexpert_categorize") -> GCSBucketInterface:
    """
    Get a GCSBucketInterface instance.

    Args:
        bucket_name: Name of the GCS bucket (default: "gosexpert_categorize")

    Returns:
        GCSBucketInterface: Initialized bucket interface
    """
    return GCSBucketInterface(bucket_name)
