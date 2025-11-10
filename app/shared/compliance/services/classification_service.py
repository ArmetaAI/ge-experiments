"""Document classification service."""
import asyncio
import fitz
import io
import re
from urllib.parse import unquote
from typing import List, Optional, Tuple
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_vertexai import ChatVertexAI
from PIL import Image

from app.shared.compliance import prompts
from app.shared.utils.processing_utils import parse_json_response, image_llm_call
from app.infrastructure.persistence.database.models import SessionLocal
from app.infrastructure.ai.vector_search.vertex_ai_vector_engine import get_vector_query_engine
from .base_service import BaseComplianceService


class ClassificationService(BaseComplianceService):
    """Service for classifying documents based on type_project."""
    
    MIN_TEXT_LENGTH = 50
    
    async def process(self, files: List[str], type_project: str = None) -> dict:
        """
        Classify documents.
        
        Args:
            files: List of file paths in GCS
            type_project: Project type (unused for now)
            
        Returns:
            dict: {file_path: (title, tag) or (tome_number, title) or ("Error: message", None)}
        """
        tasks = [self._classify_file(fp) for fp in files]
        results = await asyncio.gather(*tasks)
        return dict(results)
    
    async def _classify_file(self, file_path: str) -> Tuple[str, tuple]:
        """Classify a single document."""
        try:
            doc = await asyncio.to_thread(
                self.pdf_cache.get_or_load,
                unquote(file_path),
                None,
                self.bucket
            )
            
            text = ""
            max_pages = min(3, doc.page_count)
            for page_num in range(max_pages):
                page = doc.load_page(page_num)
                page_text = await asyncio.to_thread(page.get_text, "text")
                text += page_text
            
            is_scanned = len(text.strip()) < self.MIN_TEXT_LENGTH
            
            if not is_scanned:
                tome_result = self._is_tome_present(text)
                if tome_result:
                    tome_number, title = tome_result
                    return (file_path, (tome_number, title))
            
            title = await self._extract_title(text, doc, is_scanned)
            tag = await self._get_closest_tag(text)
            
            return (file_path, (title, tag))
        
        except fitz.FileDataError:
            self.logger.error(f"Classify: Corrupted or invalid PDF: {file_path}")
            return (file_path, ("Error: corrupted PDF", None))
        except ValueError as e:
            self.logger.error(f"Classify: Value error processing {file_path}: {str(e)}")
            return (file_path, (f"Error: {str(e)}", None))
        except Exception as e:
            self.logger.error(f"Classify: Unexpected error processing {file_path}: {str(e)}", exc_info=True)
            return (file_path, (f"Error: {str(e)}", None))
    
    @staticmethod
    def _is_tome_present(text: str) -> Optional[Tuple[str, str]]:
        """Check if tome pattern is present in text."""
        pattern = r'Том\s+(\d+(?:\.\d+)?)\s*\n\s*(.+?)(?:\n|$)'
        
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            tome_number = match.group(1)
            title = match.group(2).strip()
            return (tome_number, title)
        
        return None
    
    async def _extract_title(self, text: str, doc, is_scanned: bool, dpi=150) -> str:
        """Extract document title using LLM."""
        llm = ChatVertexAI(model_name="gemini-2.5-flash", temperature=0)
        
        if is_scanned:
            page = doc.load_page(0)
            pix = await asyncio.to_thread(page.get_pixmap, dpi=dpi)
            img_bytes = await asyncio.to_thread(pix.tobytes, "png")
            image = await asyncio.to_thread(Image.open, io.BytesIO(img_bytes))
            response = await asyncio.to_thread(image_llm_call, image, llm, prompts.SCANNED_DOCUMENT_TITLE_PROMPT)
            result = parse_json_response(response.content)
            
            doc_type = result.get("type")
            if doc_type:
                return doc_type
            
            title = result.get("title")
            if title:
                return title
            
            return "Не удалось определить"
        
        else:
            system_prompt = SystemMessage(content=prompts.TEXT_DOCUMENT_TITLE_SYSTEM_PROMPT)
            user_prompt = HumanMessage(content=prompts.TEXT_DOCUMENT_TITLE_USER_PROMPT.format(text=text[:2000]))
            
            response = await asyncio.to_thread(llm.invoke, [system_prompt, user_prompt])
            result = parse_json_response(response.content)
            
            if not result.get("title"):
                raise ValueError(f"LLM could not find title in the text")
            
            return result.get("title", "Дефолтный ответ")
    
    async def _get_closest_tag(self, text: str, top_k=1, similarity_threshold=0.3) -> str:
        """Get closest matching tag using vector similarity search."""
        db = None
        try:
            db = SessionLocal()
            
            vector_engine = get_vector_query_engine()
            
            result = await vector_engine.find_closest_tag(
                query_text=text[:2000],
                db=db,
                top_k=top_k,
                similarity_threshold=similarity_threshold
            )
            
            if result:
                tag_name, similarity = result
                self.logger.info(f"Found matching tag: {tag_name} (similarity: {similarity:.3f})")
                return tag_name
            else:
                self.logger.error("No matching tag found, using default")
                return "Прочее"
        
        except Exception as e:
            self.logger.error(f"Error during tag matching: {str(e)}", exc_info=True)
            return "Прочее"
        
        finally:
            if db is not None:
                db.close()

