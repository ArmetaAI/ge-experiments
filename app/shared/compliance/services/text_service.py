"""Text sufficiency check service."""
import asyncio
import fitz
from typing import List
from google.cloud.exceptions import NotFound, Forbidden, GoogleCloudError

from .base_service import BaseComplianceService


class TextService(BaseComplianceService):
    """Service for checking text sufficiency in PDF documents."""
    
    async def process(self, files: List[str], min_symbols: int = 50) -> dict:
        """
        Check text sufficiency in each file.
        
        Args:
            files: List of file paths in GCS
            min_symbols: Minimum symbol threshold per page
            
        Returns:
            dict: {file_path: ({page: (symbols, deficit)}, memo) or ({}, "Error: message")}
        """
        tasks = [self._process_file(fp, min_symbols) for fp in files]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return {
            file_path: ({}, f"Error: {str(result)}") if isinstance(result, Exception) else result
            for file_path, result in zip(files, results)
        }
    
    async def _process_file(self, file_path: str, min_symbols: int = 50) -> tuple:
        """Process a single PDF file for text sufficiency."""
        try:
            doc = await asyncio.to_thread(
                self.pdf_cache.get_or_load,
                file_path,
                None,
                self.bucket
            )
        except NotFound:
            self.logger.error(f"File not found in GCS: {file_path}")
            raise Exception("file not found")
        except Forbidden:
            self.logger.error(f"Access denied for file in GCS: {file_path}")
            raise Exception("access denied")
        except GoogleCloudError as e:
            self.logger.error(f"Google Cloud error loading file {file_path}: {str(e)}")
            raise Exception(f"error loading from GCS - {str(e)}")
        except fitz.FileDataError:
            self.logger.error(f"File is damaged or has invalid PDF format: {file_path}")
            raise Exception("file is damaged or has invalid PDF format")
        except Exception as e:
            self.logger.error(f"Failed to load PDF {file_path}: {str(e)}")
            raise Exception(f"failed to load PDF - {str(e)}")
        
        try:
            page_tasks = [
                self._process_page(doc[page_num], page_num + 1, min_symbols)
                for page_num in range(len(doc))
            ]
            
            page_results = await asyncio.gather(*page_tasks)
            
            page_info = {}
            has_scans = False
            for page_num, symbol_count, deficit, has_images in page_results:
                page_info[page_num] = (symbol_count, deficit)
                if has_images:
                    has_scans = True
            
            memo = "OCR is required" if has_scans else ""
            
            return (page_info, memo)
        
        except Exception as e:
            self.logger.error(f"Error processing document pages for file {file_path}: {str(e)}")
            raise Exception(f"error processing document pages - {str(e)}")
    
    @staticmethod
    async def _process_page(page, page_num: int, min_symbols: int) -> tuple:
        """Process a single page for text sufficiency."""
        try:
            image_list, text = await asyncio.gather(
                asyncio.to_thread(page.get_images),
                asyncio.to_thread(page.get_text)
            )
            
            has_images = len(image_list) > 0
            symbol_count = len(''.join(text.split()))
            
            deficit = symbol_count < min_symbols
            
            return (page_num, symbol_count, deficit, has_images)
        
        except Exception as e:
            raise Exception(f"error processing page {page_num} - {str(e)}")

