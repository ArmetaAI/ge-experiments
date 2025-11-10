"""Page analysis service for empty pages and page counting."""
import asyncio
import fitz
from urllib.parse import unquote
from typing import List
from google.api_core import exceptions as google_exceptions

from .base_service import BaseComplianceService


class PageService(BaseComplianceService):
    """Service for page-related checks: empty pages and page counting."""
    
    async def process(self, files: List[str]) -> dict:
        """Alias for find_empty_pages for compatibility with base class."""
        return await self.find_empty_pages(files)
    
    async def find_empty_pages(self, files: List[str]) -> dict:
        """
        Count number of empty pages in the files.
        
        Args:
            files: List of file paths in GCS
            
        Returns:
            dict: {file_path: (count, (empty_page_nums,), error_msg)}
        """
        tasks = [self._process_file_for_empty_pages(fp) for fp in files]
        results = await asyncio.gather(*tasks)
        return dict(zip(files, results))
    
    async def count_pages(self, files: List[str], max_concurrent: int = 5) -> dict:
        """
        Count pages and analyze document characteristics.
        
        Args:
            files: List of file paths in GCS
            max_concurrent: Maximum concurrent file processing
            
        Returns:
            dict: {file_path: (page_count, flag, memo)}
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_semaphore(file_path: str):
            async with semaphore:
                return await self._process_file_for_page_count(file_path)
        
        tasks = [process_with_semaphore(fp) for fp in files]
        results = await asyncio.gather(*tasks)
        return dict(zip(files, results))
    
    async def _process_file_for_empty_pages(self, file_path: str) -> tuple:
        """Process a single file for empty pages."""
        try:
            pdf = await asyncio.to_thread(
                self.pdf_cache.get_or_load,
                unquote(file_path),
                None,
                self.bucket
            )
            
            async def process_one_page(page_num):
                def get_text():
                    page = pdf.load_page(page_num)
                    return page.get_text("text").strip()
                
                text = await asyncio.to_thread(get_text)
                if not text:
                    return page_num + 1
                return None
            
            tasks = [process_one_page(page_num) for page_num in range(pdf.page_count)]
            results = await asyncio.gather(*tasks)
            
            empty_pages = [page for page in results if page is not None]
            
            return (len(empty_pages), tuple(empty_pages), "")
        
        except google_exceptions.NotFound:
            self.logger.error(f"File not found: {file_path}")
            return (0, (), "File not found")
        except fitz.FileDataError:
            self.logger.error(f"Corrupted PDF: {file_path}")
            return (0, (), "Corrupted PDF")
        except Exception as e:
            self.logger.error(f"Unknown error processing {file_path}: {e}")
            return (0, (), "Unknown error")
    
    async def _process_file_for_page_count(self, file_path: str) -> tuple:
        """Process a single file for page count analysis."""
        try:
            pdf = await asyncio.to_thread(
                self.pdf_cache.get_or_load,
                unquote(file_path),
                None,
                self.bucket
            )
            
            def analyze_pdf():
                num_pages = pdf.page_count
                flag = "нормальный"
                memo = ""
                
                if num_pages == 0:
                    return 0, "короткий", "пустой файл"
                
                if num_pages == 1:
                    page = pdf.load_page(0)
                    text = page.get_text("text").strip()
                    if not text:
                        memo = "пустая страница"
                
                check_pages = min(num_pages, 3)
                is_scan = all(
                    len(pdf.load_page(i).get_images()) > 0
                    for i in range(check_pages)
                )
                if is_scan:
                    memo = "скан"
                
                if num_pages <= 2:
                    flag = "короткий"
                
                return num_pages, flag, memo
            
            return await asyncio.to_thread(analyze_pdf)
        
        except google_exceptions.NotFound:
            self.logger.error(f"File not found: {file_path}")
            return 0, "ошибка", "File not found"
        except fitz.FileDataError:
            self.logger.error(f"Corrupted PDF: {file_path}")
            return 0, "ошибка", "Corrupted PDF"
        except Exception as e:
            self.logger.error(f"Error processing {file_path}: {e}")
            return 0, "ошибка", str(e)

