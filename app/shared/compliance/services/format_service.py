"""Format validation service for PDF files."""
import asyncio
import fitz
from urllib.parse import unquote
from typing import List
from google.api_core import exceptions as google_exceptions

from .base_service import BaseComplianceService


class FormatService(BaseComplianceService):
    """Service for validating PDF file format."""
    
    async def process(self, files: List[str]) -> dict:
        """
        Check if files are valid PDF format.
        
        Args:
            files: List of file paths in GCS
            
        Returns:
            dict: {file_path: "pdf" or "not pdf"}
        """
        tasks = [self._check_file_format(fp) for fp in files]
        results = await asyncio.gather(*tasks)
        return dict(results)
    
    async def _check_file_format(self, file_path: str) -> tuple:
        """Check if single file is valid PDF."""
        try:
            await asyncio.to_thread(
                self.pdf_cache.get_or_load,
                unquote(file_path),
                None,
                self.bucket
            )
            return file_path, "pdf"
        
        except (google_exceptions.NotFound,
                google_exceptions.Forbidden,
                fitz.FileDataError,
                ValueError,
                Exception):
            return file_path, "not pdf"

