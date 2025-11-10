"""Base service for all compliance checks."""
from abc import ABC, abstractmethod
from typing import List
from google.cloud import storage
from app.shared.config.settings import settings
from app.shared.utils.pdf_cache import get_pdf_cache
import logging

logger = logging.getLogger(__name__)


class BaseComplianceService(ABC):
    """
    Base class for all compliance services.
    
    Provides shared functionality:
    - PDF cache access (singleton)
    - GCS bucket and storage client
    - Logger instance
    """
    
    def __init__(self, bucket_name: str):
        """
        Initialize service with GCS bucket.
        
        Args:
            bucket_name: Name of the GCS bucket
        """
        self.bucket_name = bucket_name
        self.storage_client = storage.Client(
            project=settings.GCS_PROJECT_ID or None
        )
        self.bucket = self.storage_client.bucket(bucket_name)
        self.pdf_cache = get_pdf_cache()
        self.logger = logger
    
    @abstractmethod
    async def process(self, files: List[str]) -> dict:
        """
        Process compliance check for list of files.
        
        Args:
            files: List of GCS file paths
            
        Returns:
            dict: Results keyed by file_path
        """
        pass

