"""Signature and stamp detection service using Vision LLM."""
import asyncio
import fitz
import io
from urllib.parse import unquote
from typing import List
from google.cloud.exceptions import NotFound, Forbidden, GoogleCloudError
from langchain_google_vertexai import ChatVertexAI
from PIL import Image

from app.shared.utils.processing_utils import parse_json_response, image_llm_call
from app.shared.compliance.prompts import SIGNATURE_AND_STAMP_DETECTION_PROMPT
from app.shared.config.settings import settings
from .base_service import BaseComplianceService


class SignatureService(BaseComplianceService):
    """Service for detecting signatures and stamps using Vision LLM."""
    
    def __init__(self, bucket_name: str):
        super().__init__(bucket_name)
        self.vision_llm = ChatVertexAI(model_name="gemini-2.5-pro")
    
    async def process(self, files: List[str]) -> dict:
        """
        Detect signatures and stamps in each file.
        
        Args:
            files: List of file paths in GCS
            
        Returns:
            dict: {file_path: {"signatures": count, "stamps": count} or "Error: message"}
        """
        tasks = [self._process_file(fp) for fp in files]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return {
            file_path: f"Error: {str(result)}" if isinstance(result, Exception) else result
            for file_path, result in zip(files, results)
        }
    
    async def _process_file(self, file_path: str) -> dict:
        """Process a single file for signature and stamp detection."""
        try:
            doc = await asyncio.to_thread(
                self.pdf_cache.get_or_load,
                unquote(file_path),
                None,
                self.bucket
            )
        except NotFound:
            self.logger.error(f"File not found in GCS for signature detection: {file_path}")
            raise Exception("file not found")
        except Forbidden:
            self.logger.error(f"Access denied for file in GCS for signature detection: {file_path}")
            raise Exception("access denied")
        except GoogleCloudError as e:
            self.logger.error(f"Google Cloud error loading file for signature detection {file_path}: {str(e)}")
            raise Exception(f"error loading from GCS - {str(e)}")
        except fitz.FileDataError:
            self.logger.error(f"File is damaged or has invalid PDF format for signature detection: {file_path}")
            raise Exception("file is damaged or has invalid PDF format")
        except Exception as e:
            self.logger.error(f"Unknown error loading file for signature detection {file_path}: {str(e)}")
            raise Exception(f"unknown error loading file - {str(e)}")
        
        try:
            total_pages = len(doc)
            
            detection_tasks = [
                self._detect_on_page(doc[page_idx], page_idx + 1)
                for page_idx in range(total_pages)
            ]
            
            results = await asyncio.gather(*detection_tasks)
            
            total_signatures = sum(result["signatures"] for result in results)
            total_stamps = sum(result["stamps"] for result in results)
            
            return {"signatures": total_signatures, "stamps": total_stamps}
        
        except Exception as e:
            self.logger.error(f"Error processing signature and stamp detection for file {file_path}: {str(e)}")
            raise Exception(f"error processing signature and stamp detection - {str(e)}")
    
    async def _detect_on_page(self, page, page_num: int) -> dict:
        """Detect signatures and stamps on a single page."""
        try:
            pix = await asyncio.to_thread(page.get_pixmap, dpi=150)
            img_bytes = await asyncio.to_thread(pix.tobytes, "png")
            
            image = await asyncio.to_thread(Image.open, io.BytesIO(img_bytes))
            
            response = await asyncio.to_thread(
                image_llm_call,
                image,
                self.vision_llm,
                SIGNATURE_AND_STAMP_DETECTION_PROMPT
            )
            
            detection_result = parse_json_response(response.content)
            
            signatures = detection_result.get("signature_count", 0) if detection_result.get("has_signature", False) else 0
            stamps = detection_result.get("stamp_count", 0) if detection_result.get("has_stamp", False) else 0
            
            return {"signatures": signatures, "stamps": stamps}
        
        except Exception as e:
            self.logger.warning(f"Error detecting signatures/stamps on page {page_num}: {str(e)}")
            return {"signatures": 0, "stamps": 0}

