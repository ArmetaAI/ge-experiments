"""
Google Cloud Storage Service.

This module provides a clean service layer for all GCS interactions.
It abstracts away the complexity of GCS client management and provides
simple, high-level methods for file operations.
"""

from typing import Optional
import io

from google.cloud import storage
from fastapi import UploadFile

from app.shared.config.settings import settings


class StorageService:
    """
    Service class for Google Cloud Storage operations.
    
    This class handles all interactions with GCS, including:
    - File uploads
    - File downloads
    - File deletion
    - URL generation
    """
    
    def __init__(self):
        """
        Initialize the Storage Service.
        
        Creates a GCS client instance using default credentials.
        In production (Cloud Run), this will automatically use the
        service account attached to the Cloud Run service.
        """
        self.client = storage.Client()
        self.bucket_name = settings.GCS_BUCKET_NAME
        self.bucket = self.client.bucket(self.bucket_name)
        
    def upload_file(
        self,
        project_id: str,
        file: UploadFile,
        package_type: Optional[str] = None,
        content_type: Optional[str] = None
    ) -> str:
        """
        Upload a file to Google Cloud Storage.

        Files are organized by project ID and package type in the bucket:
        projects/{project_id}/{package_type}/{filename}

        Args:
            project_id: Unique project identifier for organizing files
            file: FastAPI UploadFile object to upload
            package_type: Package type ('PSD', 'IRD', etc.) for folder organization
            content_type: Optional content type override

        Returns:
            str: Object path within the bucket (e.g., "projects/{project_id}/{package_type}/{filename}")

        Raises:
            Exception: If upload fails
        """
        try:
            # Construct blob path with package type subfolder
            if package_type:
                # Normalize package type to lowercase for consistent folder names
                package_folder = package_type.lower()
                blob_name = f"projects/{project_id}/{package_folder}/{file.filename}"
            else:
                # Fallback to old structure if package_type not provided
                blob_name = f"projects/{project_id}/{file.filename}"

            blob = self.bucket.blob(blob_name)

            # Set content type
            if content_type:
                blob.content_type = content_type
            elif file.content_type:
                blob.content_type = file.content_type
            else:
                # Default to application/octet-stream
                blob.content_type = "application/octet-stream"

            # Upload file
            # Reset file pointer to beginning
            file.file.seek(0)
            blob.upload_from_file(file.file, content_type=blob.content_type)

            # Return just the object path (standardized format)
            print(f"[StorageService] Uploaded: {file.filename} -> gs://{self.bucket_name}/{blob_name}")

            return blob_name
            
        except Exception as e:
            print(f"[StorageService] Upload failed for {file.filename}: {str(e)}")
            raise Exception(f"Failed to upload file to GCS: {str(e)}")
    
    def download_file(self, path: str) -> bytes:
        """
        Download a file from Google Cloud Storage.

        Args:
            path: Either a full GCS URI (gs://bucket/path) or an object path within the bucket

        Returns:
            bytes: File contents

        Raises:
            Exception: If download fails
        """
        try:
            # Handle both full URI and object path formats
            if path.startswith("gs://"):
                # Parse full GCS URI
                path_parts = path[5:].split("/", 1)
                bucket_name = path_parts[0]
                blob_name = path_parts[1] if len(path_parts) > 1 else ""
                bucket = self.client.bucket(bucket_name)
            else:
                # Use object path with configured bucket
                blob_name = path
                bucket = self.bucket

            blob = bucket.blob(blob_name)

            # Download to bytes
            contents = blob.download_as_bytes()

            print(f"[StorageService] Downloaded: {path} ({len(contents)} bytes)")

            return contents

        except Exception as e:
            print(f"[StorageService] Download failed for {path}: {str(e)}")
            raise Exception(f"Failed to download file from GCS: {str(e)}")
    
    def download_to_file(self, path: str, local_path: str) -> None:
        """
        Download a file from GCS to a local file path.

        Args:
            path: Either a full GCS URI (gs://bucket/path) or an object path within the bucket
            local_path: Local file path to save to

        Raises:
            Exception: If download fails
        """
        try:
            # Handle both full URI and object path formats
            if path.startswith("gs://"):
                # Parse full GCS URI
                path_parts = path[5:].split("/", 1)
                bucket_name = path_parts[0]
                blob_name = path_parts[1] if len(path_parts) > 1 else ""
                bucket = self.client.bucket(bucket_name)
            else:
                # Use object path with configured bucket
                blob_name = path
                bucket = self.bucket

            blob = bucket.blob(blob_name)

            # Download to file
            blob.download_to_filename(local_path)

            print(f"[StorageService] Downloaded: {path} -> {local_path}")

        except Exception as e:
            print(f"[StorageService] Download failed for {path}: {str(e)}")
            raise Exception(f"Failed to download file from GCS: {str(e)}")
    
    def delete_file(self, path: str) -> None:
        """
        Delete a file from Google Cloud Storage.

        Args:
            path: Either a full GCS URI (gs://bucket/path) or an object path within the bucket

        Raises:
            Exception: If deletion fails
        """
        try:
            # Handle both full URI and object path formats
            if path.startswith("gs://"):
                # Parse full GCS URI
                path_parts = path[5:].split("/", 1)
                bucket_name = path_parts[0]
                blob_name = path_parts[1] if len(path_parts) > 1 else ""
                bucket = self.client.bucket(bucket_name)
            else:
                # Use object path with configured bucket
                blob_name = path
                bucket = self.bucket

            blob = bucket.blob(blob_name)

            # Delete blob
            blob.delete()

            print(f"[StorageService] Deleted: {path}")

        except Exception as e:
            print(f"[StorageService] Deletion failed for {path}: {str(e)}")
            raise Exception(f"Failed to delete file from GCS: {str(e)}")
    
    def delete_project_files(self, project_id: str) -> int:
        """
        Delete all files for a specific project.
        
        Args:
            project_id: Project identifier
            
        Returns:
            int: Number of files deleted
            
        Raises:
            Exception: If deletion fails
        """
        try:
            prefix = f"projects/{project_id}/"
            blobs = list(self.bucket.list_blobs(prefix=prefix))
            
            deleted_count = 0
            for blob in blobs:
                blob.delete()
                deleted_count += 1
            
            print(f"[StorageService] Deleted {deleted_count} files for project {project_id}")
            
            return deleted_count
            
        except Exception as e:
            print(f"[StorageService] Batch deletion failed for project {project_id}: {str(e)}")
            raise Exception(f"Failed to delete project files from GCS: {str(e)}")
    
    def delete_package_files(self, project_id: str, package_type: str) -> int:
        """
        Delete all files for a specific package within a project.
        
        Args:
            project_id: Project identifier
            package_type: Package type ('PSD', 'IRD', etc.)
            
        Returns:
            int: Number of files deleted
            
        Raises:
            Exception: If deletion fails
        """
        try:
            package_folder = package_type.lower()
            prefix = f"projects/{project_id}/{package_folder}/"
            blobs = list(self.bucket.list_blobs(prefix=prefix))
            
            deleted_count = 0
            for blob in blobs:
                blob.delete()
                deleted_count += 1
            
            print(f"[StorageService] Deleted {deleted_count} {package_type} files for project {project_id}")
            
            return deleted_count
            
        except Exception as e:
            print(f"[StorageService] Package deletion failed for project {project_id}/{package_type}: {str(e)}")
            raise Exception(f"Failed to delete {package_type} package files from GCS: {str(e)}")

    
    def generate_signed_url(
        self,
        gcs_uri: str,
        expiration_minutes: int = 60
    ) -> str:
        """
        Generate a signed URL for temporary public access to a file.
        
        Args:
            gcs_uri: Full GCS URI (gs://bucket/path)
            expiration_minutes: URL expiration time in minutes (default: 60)
            
        Returns:
            str: Signed URL
            
        Raises:
            Exception: If URL generation fails
        """
        try:
            # Parse GCS URI
            if not gcs_uri.startswith("gs://"):
                raise ValueError(f"Invalid GCS URI: {gcs_uri}")
            
            path_parts = gcs_uri[5:].split("/", 1)
            bucket_name = path_parts[0]
            blob_name = path_parts[1] if len(path_parts) > 1 else ""
            
            # Get blob
            bucket = self.client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            
            # Generate signed URL
            from datetime import timedelta
            url = blob.generate_signed_url(
                version="v4",
                expiration=timedelta(minutes=expiration_minutes),
                method="GET"
            )
            
            return url
            
        except Exception as e:
            print(f"[StorageService] Failed to generate signed URL for {gcs_uri}: {str(e)}")
            raise Exception(f"Failed to generate signed URL: {str(e)}")
    
    def file_exists(self, gcs_uri: str) -> bool:
        """
        Check if a file exists in GCS.
        
        Args:
            gcs_uri: Full GCS URI (gs://bucket/path)
            
        Returns:
            bool: True if file exists, False otherwise
        """
        try:
            # Parse GCS URI
            if not gcs_uri.startswith("gs://"):
                return False
            
            path_parts = gcs_uri[5:].split("/", 1)
            bucket_name = path_parts[0]
            blob_name = path_parts[1] if len(path_parts) > 1 else ""
            
            # Get blob
            bucket = self.client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            
            # Check existence
            return blob.exists()
            
        except Exception:
            return False


# Convenience function to get a storage service instance
def get_storage_service() -> StorageService:
    """
    Get a StorageService instance.
    
    Returns:
        StorageService: Initialized storage service
    """
    return StorageService()
