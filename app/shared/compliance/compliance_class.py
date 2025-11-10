"""
ComplianceClass - Facade pattern for backward compatibility.

DEPRECATED: This class delegates to specialized services.
New code should use services directly from app.shared.compliance.services

Migration example:
    # Old way:
    compliance = ComplianceClass(files, type_project, bucket_name)
    qr_results = await compliance.qr_code_number()
    
    # New way:
    from app.shared.compliance.services import QRCodeService
    qr_service = QRCodeService(bucket_name)
    qr_results = await qr_service.process(files)
"""
import logging
from typing import List
from google.cloud import storage
from langchain_google_vertexai import ChatVertexAI

from app.shared.config.settings import settings
from app.shared.utils.pdf_cache import get_pdf_cache
from .services import (
    QRCodeService,
    PageService,
    DateService,
    SignatureService,
    TextService,
    FormatService,
    ClassificationService,
)

logger = logging.getLogger(__name__)


class ComplianceClass:
    """
    DEPRECATED: Facade for backward compatibility.
    
    This class delegates to specialized services.
    Use services directly for new code.
    """
    
    def __init__(self, files: List[str], type_project: str, bucket_name: str):
        """
        Initialize ComplianceClass.
        
        Args:
            files: List of file paths in GCS
            type_project: Project type
            bucket_name: GCS bucket name
        """
        self.files = files
        self.type_project = type_project
        self.bucket_name = bucket_name
        
        self.storage_client = storage.Client(
            project=settings.GCS_PROJECT_ID or None
        )
        self.bucket = self.storage_client.bucket(bucket_name)
        self.vision_llm = ChatVertexAI(model_name="gemini-2.5-pro")
        self.pdf_cache = get_pdf_cache()
        
        self._qr_service = None
        self._page_service = None
        self._date_service = None
        self._signature_service = None
        self._text_service = None
        self._format_service = None
        self._classification_service = None
    
    @property
    def qr_service(self):
        if self._qr_service is None:
            self._qr_service = QRCodeService(self.bucket_name)
        return self._qr_service
    
    @property
    def page_service(self):
        if self._page_service is None:
            self._page_service = PageService(self.bucket_name)
        return self._page_service
    
    @property
    def date_service(self):
        if self._date_service is None:
            self._date_service = DateService(self.bucket_name)
        return self._date_service
    
    @property
    def signature_service(self):
        if self._signature_service is None:
            self._signature_service = SignatureService(self.bucket_name)
        return self._signature_service
    
    @property
    def text_service(self):
        if self._text_service is None:
            self._text_service = TextService(self.bucket_name)
        return self._text_service
    
    @property
    def format_service(self):
        if self._format_service is None:
            self._format_service = FormatService(self.bucket_name)
        return self._format_service
    
    @property
    def classification_service(self):
        if self._classification_service is None:
            self._classification_service = ClassificationService(self.bucket_name)
        return self._classification_service
    
    async def qr_code_number(self) -> dict:
        """Count QR codes. Delegates to QRCodeService."""
        return await self.qr_service.process(self.files)
    
    async def empty_lists(self) -> dict:
        """Find empty pages. Delegates to PageService."""
        return await self.page_service.find_empty_pages(self.files)
    
    async def page_number(self, max_concurrent: int = 5) -> dict:
        """Count pages. Delegates to PageService."""
        return await self.page_service.count_pages(self.files, max_concurrent)
    
    async def date_check(self) -> dict:
        """Extract dates. Delegates to DateService."""
        return await self.date_service.process(self.files)
    
    async def signature_and_stamp_number(self) -> dict:
        """Detect signatures/stamps. Delegates to SignatureService."""
        return await self.signature_service.process(self.files)
    
    async def insufficient_files(self) -> dict:
        """Check insufficient text. Delegates to TextService."""
        return await self.text_service.process(self.files)
    
    async def check_format(self) -> dict:
        """Check PDF format. Delegates to FormatService."""
        return await self.format_service.process(self.files)
    
    async def classify_documents(self) -> dict:
        """Classify documents. Delegates to ClassificationService."""
        return await self.classification_service.process(
            self.files,
            self.type_project
        )
    
    async def document_existence(self) -> dict:
        """Not implemented yet."""
        pass
    
    def get_cache_stats(self) -> dict:
        """Get PDF cache statistics."""
        return self.pdf_cache.get_stats()

