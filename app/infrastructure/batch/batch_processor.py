"""
Batch Processor для обработки файлов батчами с real-time прогрессом.
"""
import asyncio
import uuid
from typing import List, Dict, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    """Статус обработки job."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BatchJob:
    """Job для batch обработки."""
    job_id: str
    files: List[str]
    total_files: int
    processed_files: int = 0
    status: JobStatus = JobStatus.PENDING
    progress: float = 0.0
    results: Dict = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    def update_progress(self):
        """Обновить прогресс."""
        if self.total_files > 0:
            self.progress = (self.processed_files / self.total_files) * 100


class BatchProcessor:
    """
    Процессор для batch обработки файлов.
    
    Features:
    - Обработка батчами (по N файлов)
    - Real-time прогресс через callbacks
    - Partial results
    - Error handling и fault tolerance
    """
    
    def __init__(
        self, 
        batch_size: int = 5,
        max_concurrent_batches: int = 2
    ):
        """
        Initialize batch processor.
        
        Args:
            batch_size: Количество файлов в одном батче
            max_concurrent_batches: Максимум батчей параллельно
        """
        self.batch_size = batch_size
        self.semaphore = asyncio.Semaphore(max_concurrent_batches)
        self.jobs: Dict[str, BatchJob] = {}
        self.progress_callbacks: Dict[str, List[Callable]] = {}
        logger.info(f"BatchProcessor initialized: batch_size={batch_size}, max_concurrent={max_concurrent_batches}")
    
    def create_job(self, files: List[str]) -> str:
        """
        Создать новый job.
        
        Args:
            files: Список путей к файлам
            
        Returns:
            str: job_id
        """
        job_id = str(uuid.uuid4())
        job = BatchJob(
            job_id=job_id,
            files=files,
            total_files=len(files)
        )
        self.jobs[job_id] = job
        logger.info(f"Created job {job_id} with {len(files)} files")
        return job_id
    
    def register_progress_callback(
        self, 
        job_id: str, 
        callback: Callable[[Dict], Any]
    ):
        """
        Зарегистрировать callback для уведомлений о прогрессе.
        
        Args:
            job_id: ID job
            callback: Async функция, принимающая dict с прогрессом
        """
        if job_id not in self.progress_callbacks:
            self.progress_callbacks[job_id] = []
        self.progress_callbacks[job_id].append(callback)
    
    async def _notify_progress(self, job_id: str):
        """Уведомить все callbacks о прогрессе."""
        job = self.jobs.get(job_id)
        if not job:
            return
        
        progress_data = {
            "job_id": job_id,
            "status": job.status.value,
            "progress": round(job.progress, 2),
            "processed_files": job.processed_files,
            "total_files": job.total_files,
            "partial_results": job.results,
            "errors": job.errors
        }
        
        callbacks = self.progress_callbacks.get(job_id, [])
        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(progress_data)
                else:
                    callback(progress_data)
            except Exception as e:
                logger.error(f"Error in progress callback for job {job_id}: {e}")
    
    def _create_batches(self, files: List[str]) -> List[List[str]]:
        """
        Разбить файлы на батчи.
        
        Args:
            files: Список файлов
            
        Returns:
            List[List[str]]: Список батчей
        """
        batches = []
        for i in range(0, len(files), self.batch_size):
            batch = files[i:i + self.batch_size]
            batches.append(batch)
        return batches
    
    async def process_batch(
        self,
        job_id: str,
        batch: List[str],
        batch_num: int,
        process_func: Callable[[List[str]], Dict]
    ) -> Dict:
        """
        Обработать один батч файлов.
        
        Args:
            job_id: ID job
            batch: Список файлов в батче
            batch_num: Номер батча (для логирования)
            process_func: Функция обработки, принимает список файлов, возвращает dict
            
        Returns:
            dict: Результаты обработки батча
        """
        job = self.jobs[job_id]
        
        async with self.semaphore:
            logger.info(f"[Job {job_id}] Processing batch {batch_num}: {len(batch)} files")
            
            try:
                batch_results = await process_func(batch)
                
                job.results.update(batch_results)
                job.processed_files += len(batch)
                job.update_progress()
                
                await self._notify_progress(job_id)
                
                logger.info(
                    f"[Job {job_id}] Batch {batch_num} completed. "
                    f"Progress: {job.progress:.1f}% ({job.processed_files}/{job.total_files})"
                )
                
                return batch_results
                
            except Exception as e:
                error_msg = f"Batch {batch_num} error: {str(e)}"
                logger.error(f"[Job {job_id}] {error_msg}")
                job.errors.append(error_msg)
                
                return {}
    
    async def process_job(
        self,
        job_id: str,
        process_func: Callable[[List[str]], Dict]
    ):
        """
        Обработать весь job (все батчи).
        
        Args:
            job_id: ID job
            process_func: Async функция обработки compliance checks
        """
        job = self.jobs.get(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")
        
        try:
            job.status = JobStatus.PROCESSING
            job.started_at = datetime.utcnow()
            await self._notify_progress(job_id)
            
            batches = self._create_batches(job.files)
            logger.info(f"[Job {job_id}] Created {len(batches)} batches from {len(job.files)} files")
            
            tasks = [
                self.process_batch(job_id, batch, i, process_func)
                for i, batch in enumerate(batches)
            ]
            
            await asyncio.gather(*tasks, return_exceptions=True)
            
            job.status = JobStatus.COMPLETED
            job.completed_at = datetime.utcnow()
            await self._notify_progress(job_id)
            
            duration = (job.completed_at - job.started_at).total_seconds()
            logger.info(
                f"[Job {job_id}] Completed in {duration:.1f}s. "
                f"Processed {job.processed_files}/{job.total_files} files"
            )
            
        except Exception as e:
            job.status = JobStatus.FAILED
            job.errors.append(f"Job failed: {str(e)}")
            logger.error(f"[Job {job_id}] Failed: {e}")
            await self._notify_progress(job_id)
            raise
    
    def get_job_status(self, job_id: str) -> Optional[Dict]:
        """
        Получить статус job.
        
        Args:
            job_id: ID job
            
        Returns:
            dict: Статус job или None если не найден
        """
        job = self.jobs.get(job_id)
        if not job:
            return None
        
        return {
            "job_id": job.job_id,
            "status": job.status.value,
            "progress": round(job.progress, 2),
            "processed_files": job.processed_files,
            "total_files": job.total_files,
            "results": job.results,
            "errors": job.errors,
            "created_at": job.created_at.isoformat(),
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None
        }
    
    def cancel_job(self, job_id: str) -> bool:
        """
        Отменить job.
        
        Args:
            job_id: ID job
            
        Returns:
            bool: True если job был отменен
        """
        job = self.jobs.get(job_id)
        if job and job.status == JobStatus.PROCESSING:
            job.status = JobStatus.CANCELLED
            logger.info(f"[Job {job_id}] Cancelled")
            return True
        return False
    
    def cleanup_old_jobs(self, max_age_hours: int = 24):
        """
        Очистить старые завершенные jobs.
        
        Args:
            max_age_hours: Максимальный возраст job в часах
        """
        now = datetime.utcnow()
        to_remove = []
        
        for job_id, job in self.jobs.items():
            if job.completed_at:
                age = (now - job.completed_at).total_seconds() / 3600
                if age > max_age_hours:
                    to_remove.append(job_id)
        
        for job_id in to_remove:
            del self.jobs[job_id]
            if job_id in self.progress_callbacks:
                del self.progress_callbacks[job_id]
        
        if to_remove:
            logger.info(f"Cleaned up {len(to_remove)} old jobs")


_batch_processor: Optional[BatchProcessor] = None


def get_batch_processor() -> BatchProcessor:
    """
    Get singleton batch processor instance.
    
    Returns:
        BatchProcessor: Singleton instance
    """
    global _batch_processor
    if _batch_processor is None:
        _batch_processor = BatchProcessor(
            batch_size=5,
            max_concurrent_batches=2
        )
    return _batch_processor

