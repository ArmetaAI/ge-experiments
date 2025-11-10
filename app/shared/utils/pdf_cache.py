"""
PDF Cache Singleton - кэширование parsed PDF для избежания повторных загрузок.

Использует OrderedDict с LRU для управления памятью.
Thread-safe для concurrent операций.
"""

import io
import fitz
import logging
from typing import Optional, Dict, Any, Protocol
from collections import OrderedDict
from threading import Lock

logger = logging.getLogger(__name__)

DEFAULT_CACHE_SIZE = 100


class BucketProtocol(Protocol):
    """Protocol for GCS bucket-like objects."""
    name: str
    
    def blob(self, path: str) -> Any:
        """Return a blob object for the given path."""
        ...


class PDFCache:
    """
    Singleton кэш для parsed PDF документов.
    
    Особенности:
    - OrderedDict с LRU: автоматически удаляет старые PDF при превышении лимита
    - Thread-safe: защита от race conditions
    - Статистика: hits/misses для мониторинга
    """
    
    _instance: Optional['PDFCache'] = None
    _lock = Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, max_size: int = DEFAULT_CACHE_SIZE):
        """
        Initialize cache on first creation.
        
        Args:
            max_size: Maximum number of PDFs to cache
        """
        if self._initialized:
            return
        
        self._cache: OrderedDict = OrderedDict()
        self._max_size = max_size
        self._stats = {
            'hits': 0,
            'misses': 0,
            'errors': 0,
            'evictions': 0
        }
        self._initialized = True
        logger.info("[PDFCache] Singleton initialized", extra={"max_size": max_size})
    
    def _close_document(self, doc: fitz.Document, context: str = "") -> None:
        """Safely close a PDF document with error handling."""
        try:
            doc.close()
        except Exception as e:
            logger.warning(f"[PDFCache] Failed to close document {context}: {e}")
    
    def get_or_load(
        self, 
        file_path: str, 
        pdf_bytes: Optional[bytes] = None,
        bucket: Optional[BucketProtocol] = None
    ) -> fitz.Document:
        """
        Получить PDF из кэша или загрузить и закэшировать.
        
        Args:
            file_path: Путь к файлу (используется как ключ кэша)
            pdf_bytes: Опционально - байты PDF (если уже загружены)
            bucket: Опционально - GCS bucket для загрузки
            
        Returns:
            fitz.Document: Parsed PDF документ
            
        Raises:
            ValueError: Если ни pdf_bytes ни bucket не предоставлены
        """
        cache_key = self._make_key(file_path, bucket)
        
        with self._lock:
            if cache_key in self._cache:
                self._stats['hits'] += 1
                logger.debug(f"[PDFCache] HIT: {file_path}")
                self._cache.move_to_end(cache_key)
                return self._cache[cache_key]
            
            self._stats['misses'] += 1
        
        logger.debug(f"[PDFCache] MISS: {file_path}")
        
        try:
            if pdf_bytes is None:
                if bucket is None:
                    raise ValueError("Either pdf_bytes or bucket must be provided")
                
                blob = bucket.blob(file_path)
                pdf_bytes = blob.download_as_bytes()
                logger.debug(f"[PDFCache] Downloaded {len(pdf_bytes)} bytes for {file_path}")
            
            doc = fitz.open(stream=io.BytesIO(pdf_bytes), filetype="pdf")
            
            with self._lock:
                if len(self._cache) >= self._max_size:
                    oldest_key, oldest_doc = self._cache.popitem(last=False)
                    self._close_document(oldest_doc, f"during eviction of {oldest_key}")
                    self._stats['evictions'] += 1
                    logger.debug(f"[PDFCache] Evicted oldest: {oldest_key}")
                
                self._cache[cache_key] = doc
                logger.info(f"[PDFCache] Cached {file_path} ({doc.page_count} pages)")
            
            return doc
            
        except Exception as e:
            with self._lock:
                self._stats['errors'] += 1
            logger.error(f"[PDFCache] Error loading {file_path}: {e}")
            raise
    
    def invalidate(self, file_path: str, bucket: Optional[BucketProtocol] = None):
        """
        Удалить PDF из кэша (например, после обновления файла).
        
        Args:
            file_path: Путь к файлу
            bucket: Опционально - bucket для формирования ключа
        """
        cache_key = self._make_key(file_path, bucket)
        
        with self._lock:
            if cache_key in self._cache:
                doc = self._cache[cache_key]
                self._close_document(doc, f"during invalidation of {file_path}")
                del self._cache[cache_key]
                logger.info(f"[PDFCache] Invalidated: {file_path}")
    
    def clear(self):
        """Очистить весь кэш."""
        with self._lock:
            for doc in list(self._cache.values()):
                self._close_document(doc, "during clear")
            
            self._cache.clear()
            logger.info("[PDFCache] Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Получить статистику кэша.
        
        Returns:
            dict: {hits, misses, errors, evictions, size, hit_rate}
        """
        with self._lock:
            total = self._stats['hits'] + self._stats['misses']
            hit_rate = (self._stats['hits'] / total * 100) if total > 0 else 0.0
            
            return {
                'hits': self._stats['hits'],
                'misses': self._stats['misses'],
                'errors': self._stats['errors'],
                'evictions': self._stats['evictions'],
                'size': len(self._cache),
                'hit_rate': round(hit_rate, 2)
            }
    
    def _make_key(self, file_path: str, bucket: Optional[BucketProtocol] = None) -> str:
        if bucket:
            return f"{bucket.name}/{file_path}"
        return file_path


def get_pdf_cache() -> PDFCache:
    return PDFCache()

