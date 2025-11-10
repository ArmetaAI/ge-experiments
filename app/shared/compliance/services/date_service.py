"""Date extraction service for compliance checks."""
import asyncio
import fitz
import re
from urllib.parse import unquote
from typing import List

from .base_service import BaseComplianceService


class DateService(BaseComplianceService):
    """Service for extracting dates from PDF documents."""
    
    DATE_PATTERNS = [
        r'\b(\d{1,2}[./-]\d{1,2}[./-]\d{4})\b',
        r'\b(\d{4}[./-]\d{1,2}[./-]\d{1,2})\b',
        r'\b(\d{1,2}\s+(?:января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)\s+\d{4})\b',
    ]
    
    async def process(self, files: List[str]) -> dict:
        """
        Extract dates from each file.
        
        Args:
            files: List of file paths in GCS
            
        Returns:
            dict: {file_path: [(page_num, date_string), ...]}
        """
        self.logger.info(f"Starting date_check for {len(files)} files")
        
        tasks = [self._process_file(fp) for fp in files]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        files_dates = {
            file_path: result if not isinstance(result, Exception) else []
            for file_path, result in zip(files, results)
        }
        
        for file_path, result in zip(files, results):
            if isinstance(result, Exception):
                self.logger.error(f"Error processing dates in {file_path}: {result}")
        
        total_dates = sum(len(dates) for dates in files_dates.values())
        self.logger.info(f"Completed date_check: found {total_dates} dates across {len(files)} files")
        
        return files_dates
    
    async def _process_file(self, file_path: str) -> list:
        """Process a single file to find dates."""
        try:
            doc = await asyncio.to_thread(
                self.pdf_cache.get_or_load,
                unquote(file_path),
                None,
                self.bucket
            )
            
            total_pages = len(doc)
            pages_to_check = list(range(total_pages))
            
            images_dict = await self._check_images(doc, pages_to_check)
            
            page_tasks = [
                self._extract_date_from_page(
                    doc[page_idx], 
                    page_idx + 1, 
                    images_dict.get(page_idx + 1, False)
                )
                for page_idx in pages_to_check
            ]
            
            page_results = await asyncio.gather(*page_tasks)
            
            dates_found = [(page_num, date) for page_num, date in page_results if date is not None]
            
            self.logger.debug(f"Found {len(dates_found)} dates in {file_path} ({total_pages} pages)")
            
            return dates_found
        
        except Exception as e:
            self.logger.error(f"Error processing file {file_path} for dates: {e}")
            return []
    
    async def _extract_date_from_page(self, page, page_num: int, has_images: bool) -> tuple:
        """Extract date from a single page."""
        try:
            text = await asyncio.to_thread(page.get_text)
            
            date_found = await asyncio.to_thread(
                self._search_dates_in_text,
                text,
                self.DATE_PATTERNS
            )
            return (page_num, date_found)
        
        except Exception as e:
            self.logger.error(f"Error extracting date from page {page_num}: {e}")
            return (page_num, None)
    
    @staticmethod
    def _search_dates_in_text(text: str, date_patterns: list) -> str:
        """Search for date patterns in text."""
        for pattern in date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        return None
    
    @staticmethod
    async def _check_images(doc: fitz.Document, pages_to_check: list) -> dict:
        """
        Check which pages have images.
        
        Args:
            doc: PDF document
            pages_to_check: List of page indices to check
            
        Returns:
            dict: {page_num: has_images_bool}
        """
        async def check_page_images(page_idx: int) -> tuple:
            def get_images():
                page = doc.load_page(page_idx)
                return len(page.get_images()) > 0
            
            has_images = await asyncio.to_thread(get_images)
            return (page_idx + 1, has_images)
        
        tasks = [check_page_images(idx) for idx in pages_to_check]
        results = await asyncio.gather(*tasks)
        
        return dict(results)

