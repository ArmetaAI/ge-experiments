"""This module is designed to encapsulate compliance functions class"""
import asyncio
import cv2
import fitz
import io
import re
import base64
import logging
import numpy

from app.shared.config.settings import settings
from google.api_core import exceptions as google_exceptions
from google.cloud import storage
from google.cloud.exceptions import NotFound, Forbidden, GoogleCloudError
from pyzbar import pyzbar
from urllib.parse import unquote
from langchain_core.messages import HumanMessage, SystemMessage
from typing import Optional, Tuple, Dict, List
from datetime import datetime, date
from enum import Enum
from app.infrastructure.persistence.database.models import SessionLocal
from app.infrastructure.ai.vector_search.vertex_ai_vector_engine import get_vector_query_engine
from app.shared.compliance import prompts
from app.shared.utils.processing_utils import parse_json_response
from langchain_google_vertexai import ChatVertexAI
from PIL import Image
from app.shared.utils.processing_utils import parse_json_response, image_llm_call
from app.shared.compliance.prompts import SIGNATURE_AND_STAMP_DETECTION_PROMPT
from app.shared.utils.pdf_cache import get_pdf_cache

logger = logging.getLogger(__name__)


class ComplianceClass:
    """
    This class is designed to aggregate compliance functions

    Attributes:
        files: list[str], list of input files (addresses) of the project
        type_project: str, type of the incoming project
        storage_client: google storage client
        bucket_name: str, the name of the projects bucket
    """

    # Date patterns for date extraction (supports DD.MM.YYYY, Kazakh and Russian months)
    _DATE_PATTERNS = [
        r'\b(\d{1,2})[./-](\d{1,2})[./-](\d{4})\b',  # DD.MM.YYYY or DD/MM/YYYY or DD-MM-YYYY
        r'\b(\d{4})[./-](\d{1,2})[./-](\d{1,2})\b',  # YYYY-MM-DD or YYYY/MM/DD or YYYY.MM.DD
        r'\b(\d{1,2})\s+(января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря|қантар|ақпан|наурыз|сәуір|мамыр|маусым|шілде|тамыз|қыркүйек|қазан|қараша|желтоқсан)\s+(\d{4})\b'
    ]

    # Document categories and expiration settings
    _EXPIRABLE_KEYWORDS = {
        'лицензия': True, 'разрешение': True, 'изыскания': True, 'изысканий': True, 'сертификат': True,
        'паспорт': True, 'технические условия': True, 'договор': True, 'заключение': True
    }
    
    # Month name mappings for Kazakh and Russian
    _MONTH_NAMES = {
        # Kazakh months
        'қантар': 1, 'ақпан': 2, 'наурыз': 3, 'сәуір': 4, 'мамыр': 5, 'маусым': 6,
        'шілде': 7, 'тамыз': 8, 'қыркүйек': 9, 'қазан': 10, 'қараша': 11, 'желтоқсан': 12,
        # Russian months
        'января': 1, 'февраля': 2, 'марта': 3, 'апреля': 4, 'мая': 5, 'июня': 6,
        'июля': 7, 'августа': 8, 'сентября': 9, 'октября': 10, 'ноября': 11, 'декабря': 12
    }

    def __init__(self, files: list[str], type_project: str, bucket_name: str):
        self.files = files
        self.type_project = type_project
        self.storage_client = storage.Client(project=settings.GCS_PROJECT_ID or None)
        self.bucket = self.storage_client.bucket(bucket_name)
        self.vision_llm = ChatVertexAI(model_name="gemini-2.5-pro")
        self.pdf_cache = get_pdf_cache()

    async def document_existence(self) -> dict:
        """
        This function is designated to check if basic documents
        exist among the files of the project
        Args:

        Returns:
            dict, dictionary with key - basic document, value - (file, flag, memo)

        """
        pass

    async def empty_lists(self) -> dict:
        """
        This function is designated to count number of empty lists of the files
        Args:

        Returns:
            dict, dictionary with key - a file address of the project,
            value - (number of empty pages, (pages), error message)

        """
        # Process all files concurrently
        tasks = [self.__process_single_file(file_path) for file_path in self.files]
        results = await asyncio.gather(*tasks)

        files_empty_pages = {file_path: result for file_path, result in zip(self.files, results)}
        return files_empty_pages

    async def qr_code_number(self) -> dict:
        """
        This function is designated to count number of QR codes in the files
        Args:

        Returns:
            dict, dictionary with key - a file address of the project,
            value - (number of QR codes)

        """
        # Process all files concurrently
        tasks = [self.__process_file_for_qr(file_path) for file_path in self.files]
        results = await asyncio.gather(*tasks)

        # Build the final dictionary
        files_qr_counts = dict(zip(self.files, results))

        return files_qr_counts

    async def __process_file_for_qr(self, file_path: str) -> int:
        """
        Process a single file to count QR codes

        Args:
            file_path: Path to the file in GCS

        Returns:
            int: number of QR codes found
        """
        total_qr_codes = await asyncio.to_thread(
            self.__process_file_for_qr_sync,
            file_path
        )
        return total_qr_codes

    def __process_file_for_qr_sync(self, file_path: str) -> int:
        """
        Performs all blocking I/O and CPU-bound work for a single file.

        Args:
            file_path: Path to the file in GCS

        Returns:
            int: number of QR codes found
        """
        try:
            doc = self.pdf_cache.get_or_load(
                file_path=unquote(file_path),
                bucket=self.bucket
            )
            total_qr_codes = self.__qr_code_count_in_doc(doc, file_path)
            return total_qr_codes
        except google_exceptions.NotFound:
            logger.warning(f"QR Check: File not found in GCS: {file_path}")
            return 0
        except fitz.FileDataError:
            logger.warning(f"QR Check: Failed to open corrupt PDF: {file_path}")
            return 0
        except Exception as e:
            logger.error(
                f"QR Check: Unhandled error processing file: {file_path}. Error: {e}",
                exc_info=True  # full stack trace
            )
            return 0

    @staticmethod
    def __qr_code_count_in_doc(doc: fitz.Document, file_path: str) -> int:
        """
        Processes all pages in the document to count QR codes.

        Args:
            doc: document object
            file_path: (str) The path for logging context

        Returns:
            int: total number of QR codes found
        """
        total_qr_codes = 0
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            image_list = page.get_images(full=True)
            if not image_list:
                continue

            total_qr_codes += ComplianceClass.__process_images_on_page(
                doc, page_num, file_path, image_list
            )
        return total_qr_codes

    @staticmethod
    def __process_images_on_page(doc: fitz.Document, page_num: int, file_path: str, image_list: list) -> int:
        """
        Processes all images on a single page and returns their QR code count
        This method contains the single inner loop over images
        It is a nested loop but its time complexity is O(pages + images),
        which is best we could come up with (30.10.2025).

        Args:
            doc: The document object (to extract images)
            page_num: The current page number (for logging)
            file_path: The file path (for logging)
            image_list: The list of images to process

        Returns:
            int: number of QR codes found on this page
        """
        page_qr_codes = 0
        for img_info in image_list:
            xref = img_info[0]
            if xref == 0:
                continue
            try:
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                np_arr = numpy.frombuffer(image_bytes, numpy.uint8)
                img_cv = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if img_cv is None:
                    logger.warning(
                        f"QR Check: OpenCV failed to decode image (xref={xref}) on page {page_num + 1} in {file_path}")
                    continue
                gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                decoded_objects = pyzbar.decode(thresh, symbols=[pyzbar.ZBarSymbol.QRCODE])
                page_qr_codes += len(decoded_objects)
            except Exception as e:
                logger.warning(
                    f"QR Check: Failed to process one image (xref={xref}) on page {page_num + 1} in {file_path}. Error: {e}"
                )
                continue
        return page_qr_codes

    async def __process_single_file(self, file_path: str) -> tuple:
        """Process a single file and return (count, pages, error_msg)"""

        async def read_and_process():
            try:
                pdf = await asyncio.to_thread(
                    self.pdf_cache.get_or_load,
                    unquote(file_path),
                    None,
                    self.bucket
                )

                # One loop iteration as an async function
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
                logger.error(f"File not found: {file_path}")
                return (0, (), "File not found")
            except google_exceptions.Forbidden:
                logger.error(f"Access denied to file: {file_path}")
                return (0, (), "Access denied to file")
            except fitz.FileDataError:
                logger.error(f"Corrupted PDF: {file_path}")
                return (0, (), "Corrupted PDF")
            except ValueError:
                logger.error(f"Invalid file path: {file_path}")
                return (0, (), "Invalid file path")
            except Exception as e:
                logger.error(f"Unknown error processing {file_path}: {e}")
                return (0, (), "Unknown error")

        result = await read_and_process()
        return result
    
    async def date_check(self) -> dict:
        """
        This function checks for the presence of dates in each file
        Goes through files, finds dates on each page, and checks if there are images

        Returns:
            dict: dictionary with key - file path,
                    value - list of tuples [(page_number, date_found), ...]
        """
        logger.info(f"Starting date_check for {len(self.files)} files")
        
        tasks = [
            self.__process_file_for_dates(file_path)
            for file_path in self.files
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        files_dates = {
            file_path: result if not isinstance(result, Exception) else []
            for file_path, result in zip(self.files, results)
        }
        
        # Log exceptions
        for file_path, result in zip(self.files, results):
            if isinstance(result, Exception):
                logger.error(f"Error processing dates in {file_path}: {result}")
        
        total_dates = sum(len(dates) for dates in files_dates.values())
        logger.info(f"Completed date_check: found {total_dates} dates across {len(self.files)} files")

        return files_dates

    async def __process_file_for_dates(self, file_path: str) -> list:
        """
        Process a single file to find dates on each page

        Args:
            file_path: Path to the file in GCS

        Returns:
            list: list of tuples [(page_number, date_found), ...]
        """
        try:
            doc = await asyncio.to_thread(
                self.pdf_cache.get_or_load,
                unquote(file_path),
                None,
                self.bucket
            )

            total_pages = len(doc)
            pages_to_check = list(range(total_pages))

            # Check which pages have images using __check_images method
            images_dict = await ComplianceClass.__check_images(doc, pages_to_check)

            # Process each page to find dates
            page_tasks = [
                self.__extract_date_from_page(doc[page_idx], page_idx + 1, images_dict.get(page_idx + 1, False))
                for page_idx in pages_to_check
            ]

            page_results = await asyncio.gather(*page_tasks)

            # Filter out None results (pages without dates)
            dates_found = [(page_num, date) for page_num, date in page_results if date is not None]
            
            logger.debug(f"Found {len(dates_found)} dates in {file_path} ({total_pages} pages)")

            return dates_found

        except Exception as e:
            logger.error(f"Error processing file {file_path} for dates: {e}")
            return []

    @staticmethod
    async def __extract_date_from_page(page, page_num: int, has_images: bool) -> tuple:
        """
        Extract date from a single page
        
        Args:
            page: PyMuPDF page object
            page_num: Page number
            has_images: Whether the page contains images
            
        Returns:
            tuple: (page_num, date_string) or (page_num, None) if no date found
        """
        try:
            text = await asyncio.to_thread(page.get_text)
        
            date_found = await asyncio.to_thread(
                ComplianceClass.__search_dates_in_text,
                text,
                ComplianceClass._DATE_PATTERNS
            )
            return (page_num, date_found)
        
        except Exception as e:
            logger.error(f"Error extracting date from page {page_num}: {e}")
            return (page_num, None)

    @staticmethod
    def __search_dates_in_text(text: str, date_patterns: list) -> str:
        """
        Search for date patterns in text

        Args:
            text: Text content to search in
            date_patterns: List of regex patterns to search for

        Returns:
            str: First matched date string or None if no date found
        """
        for pattern in date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        return None


    async def signature_and_stamp_number(self) -> dict:
        """
        Detect both signatures and stamps in each file using Vision LLM
        
        Returns:
            dict: {file_path: {"signatures": count, "stamps": count} or "Error: error_message"}
        """
        tasks = [
            self.__detect_signatures_and_stamps_in_file(file_path, self.bucket, self.vision_llm)
            for file_path in self.files
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        result = {
            file_path: f"Error: {str(file_result)}" if isinstance(file_result, Exception) else file_result
            for file_path, file_result in zip(self.files, results)
        }
        
        return result

    async def __detect_signatures_and_stamps_in_file(self, file_path: str, bucket, vision_llm) -> dict:
        """
        Detect signatures and stamps in a single file using Vision LLM
        
        Args:
            file_path: Path to the file in GCS
            bucket: GCS bucket object
            vision_llm: Vision LLM model
            
        Returns:
            dict: {"signatures": count, "stamps": count}
        """
        try:
            doc = await asyncio.to_thread(
                self.pdf_cache.get_or_load,
                unquote(file_path),
                None,
                bucket
            )
            
        except NotFound:
            logger.error(f"File not found in GCS for signature detection: {file_path}")
            raise Exception("file not found")
        except Forbidden:
            logger.error(f"Access denied for file in GCS for signature detection: {file_path}")
            raise Exception("access denied")
        except GoogleCloudError as e:
            logger.error(f"Google Cloud error loading file for signature detection {file_path}: {str(e)}")
            raise Exception(f"error loading from GCS - {str(e)}")
        except fitz.FileDataError:
            logger.error(f"File is damaged or has invalid PDF format for signature detection: {file_path}")
            raise Exception("file is damaged or has invalid PDF format")
        except Exception as e:
            logger.error(f"Unknown error loading file for signature detection {file_path}: {str(e)}")
            raise Exception(f"unknown error loading file - {str(e)}")

        try:
            total_pages = len(doc)
            
            detection_tasks = [
                ComplianceClass.__detect_on_page(doc[page_idx], vision_llm, page_idx + 1)
                for page_idx in range(total_pages)
            ]
            
            results = await asyncio.gather(*detection_tasks)
            
            total_signatures = sum(result["signatures"] for result in results)
            total_stamps = sum(result["stamps"] for result in results)
            
            return {"signatures": total_signatures, "stamps": total_stamps}
            
        except Exception as e:
            logger.error(f"Error processing signature and stamp detection for file {file_path}: {str(e)}")
            raise Exception(f"error processing signature and stamp detection - {str(e)}")
    
    @staticmethod
    async def __detect_on_page(page, vision_llm, page_num: int) -> dict:
        """
        Detect signatures and stamps on a page using Vision LLM
        
        Args:
            page: PyMuPDF page object
            vision_llm: Vision LLM model
            page_num: Page number (1-indexed)
            
        Returns:
            dict: {"signatures": count, "stamps": count}
        """
        try:
            pix = await asyncio.to_thread(page.get_pixmap, dpi=150)
            img_bytes = await asyncio.to_thread(pix.tobytes, "png")
            
            image = await asyncio.to_thread(Image.open, io.BytesIO(img_bytes))
            
            response = await asyncio.to_thread(
                image_llm_call,
                image,
                vision_llm,
                SIGNATURE_AND_STAMP_DETECTION_PROMPT
            )
            
            detection_result = parse_json_response(response.content)
            
            signatures = detection_result.get("signature_count", 0) if detection_result.get("has_signature", False) else 0
            stamps = detection_result.get("stamp_count", 0) if detection_result.get("has_stamp", False) else 0
            
            return {"signatures": signatures, "stamps": stamps}
            
        except Exception as e:
            logger.warning(f"Error detecting signatures/stamps on page {page_num}: {str(e)}")
            return {"signatures": 0, "stamps": 0}

    @staticmethod
    async def __check_images(doc, pages_to_check: list) -> dict:
        """
        Check if pages contain images
        
        Args:
            doc: PyMuPDF document object
            pages_to_check: List of page indices (0-indexed) to check
            
        Returns:
            dict: {page_number (1-indexed): has_images (bool)}
        """
        tasks = [
            ComplianceClass.__check_page_images(doc[page_idx], page_idx + 1)
            for page_idx in pages_to_check
        ]
        
        results = await asyncio.gather(*tasks)
        
        return {page_num: has_images for page_num, has_images in results}

    @staticmethod
    async def __check_page_images(page, page_num: int) -> tuple:
        """
        Check if a page contains images
        
        Args:
            page: PyMuPDF page object
            page_num: Page number (1-indexed)
            
        Returns:
            tuple: (page_num, has_images)
        """
        image_list = await asyncio.to_thread(page.get_images)
        has_images = len(image_list) > 0
        return (page_num, has_images)

    async def insufficient_files(self) -> dict:
        """
        This function is designated to check the sufficiency of files
        Args:

        Returns:
            dict, dictionary with key - a file address of the project,
            value - ({page: (number of symbols, deficit state)}, memo)

        """
        
        tasks = [
            self.__process_insufficient_file(file_path)
            for file_path in self.files
        ]
      
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        result = {
            file_path: ({}, f"Error: {str(file_result)}") if isinstance(file_result, Exception) else file_result
            for file_path, file_result in zip(self.files, results)
        }
        
        return result

    async def __process_insufficient_file(self, file_path: str, min_symbols: int = 50) -> tuple:
        """
        Process a single PDF file asynchronously
        
        Args:
            file_path: Path to the file in GCS
            min_symbols: Minimum symbol threshold
            
        Returns:
            tuple: (page_info dict, memo string)
        """
        try:
            doc = await asyncio.to_thread(
                self.pdf_cache.get_or_load,
                file_path,
                None,
                self.bucket
            )
            
        except NotFound:
            logger.error(f"File not found in GCS: {file_path}")
            raise Exception("file not found")
        except Forbidden:
            logger.error(f"Access denied for file in GCS: {file_path}")
            raise Exception("access denied")
        except GoogleCloudError as e:
            logger.error(f"Google Cloud error loading file {file_path}: {str(e)}")
            raise Exception(f"error loading from GCS - {str(e)}")
        except fitz.FileDataError:
            logger.error(f"File is damaged or has invalid PDF format: {file_path}")
            raise Exception("file is damaged or has invalid PDF format")
        except Exception as e:
            logger.error(f"Failed to load PDF {file_path}: {str(e)}")
            raise Exception(f"failed to load PDF - {str(e)}")

        try:
            page_tasks = [
                ComplianceClass.__process_page(doc[page_num], page_num + 1, min_symbols)
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
            logger.error(f"Error processing document pages for file {file_path}: {str(e)}")
            raise Exception(f"error processing document pages - {str(e)}")

    @staticmethod
    async def __process_page(page, page_num: int, min_symbols: int) -> tuple:
        """
        Process a single page asynchronously
        
        Args:
            page: PyMuPDF page object
            page_num: Page number (1-indexed)
            min_symbols: Minimum symbol threshold
            
        Returns:
            tuple: (page_num, symbol_count, deficit, has_images)
        """
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

    async def page_number(self, max_concurrent=5) -> dict:
        """
        This function is designated to check number of pages
        Args:
            max_concurrent: int, maximum number of concurrent tasks
        Returns:
            dict, dictionary with key - a file address of the project,
            value - (number of pages, flag, memo)

        """
        files_page_info = {}
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_file(file_path: str):
            async with semaphore:
                num_pages = 0
                memo = ""
                flag = "нормальный"

                try:
                    pdf = await asyncio.to_thread(
                        self.pdf_cache.get_or_load,
                        unquote(file_path),
                        None,
                        self.bucket
                    )

                    def analyze_pdf():
                        # Use cached PDF (no 'with' statement needed)
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
                        elif num_pages > 100:
                            flag = "длинный"

                        return num_pages, flag, memo

                    num_pages, flag, memo = await asyncio.to_thread(analyze_pdf)

                except (google_exceptions.NotFound, fitz.FileDataError, ValueError) as e:
                    logger.error(f"Error opening file {file_path}: {e}")
                    num_pages, flag, memo = 0, None, "файл не открылся"

                files_page_info[file_path] = (num_pages, flag, memo)

        await asyncio.gather(*(process_file(f) for f in self.files))
        return files_page_info

    async def check_format(self) -> dict:
        """
        Check if files are in PDF format.
        
        Returns:
            dict: Dictionary with key - file path, value - 'pdf' or 'not pdf'
        """
        async def process_file_format(file_path: str) -> tuple[str, str]:
            """Process single file to check PDF format."""
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
        
        tasks = [process_file_format(file_path) for file_path in self.files]
        results = await asyncio.gather(*tasks)
        
        format_check = {file_path: format_status 
                       for file_path, format_status in results}
        return format_check
    
    async def classify_documents(self) -> dict:
        """
        This function is designated to classify files based on type_project
        Args:

        Returns:
            dict - {document_path: classification_result},
            value - classification result
        """
        tasks = [
            self.__classify_pipeline(file_path=file_path)
            for file_path in self.files
            ]

        results = await asyncio.gather(*tasks)

        return {file_path: result for file_path, result in results}

    async def __classify_pipeline(self, file_path: str, MIN_TEXT_LENGTH = 50) -> Tuple[str, tuple]:
        """
        This function categorizes a single document
        Args:
            file_path: str - path to the document
            MIN_TEXT_LENGTH: minimum text length threshold

        Returns:
            tuple - (document_path, classification_result)
                where classification_result is either:
                - (tome_number, title) if tome is present
                - (title, tag) otherwise
                - ("Error: <message>", None) if error occurs
        """
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

            is_scanned = len(text.strip()) < MIN_TEXT_LENGTH

            if not is_scanned:
                tome_result = ComplianceClass.__is_tome_present(text)
                if tome_result:
                    tome_number, title = tome_result
                    return (file_path, (tome_number, title))

            title = await ComplianceClass.__extract_title(text, doc, is_scanned)

            tag = await ComplianceClass.__get_closest_tag(text)

            return (file_path, (title, tag))

        except fitz.FileDataError:
            logger.error(f"Classify: Corrupted or invalid PDF: {file_path}")
            return (file_path, ("Error: corrupted PDF", None))
        except ValueError as e:
            logger.error(f"Classify: Value error processing {file_path}: {str(e)}")
            return (file_path, (f"Error: {str(e)}", None))
        except Exception as e:
            logger.error(f"Classify: Unexpected error processing {file_path}: {str(e)}", exc_info=True)
            return (file_path, (f"Error: {str(e)}", None))

    @staticmethod
    def __is_tome_present(text: str) -> Optional[Tuple[str, str]]:
        """
        This function checks if tome pattern is present in text
        Args:
            text: str - text from the document

        Returns:
            Optional[Tuple[str, str]] - (tome_number, title) if found, None otherwise
        """
        pattern = r'Том\s+(\d+(?:\.\d+)?)\s*\n\s*(.+?)(?:\n|$)'

        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            tome_number = match.group(1)
            title = match.group(2).strip()
            return (tome_number, title)

        return None

    @staticmethod
    async def __extract_title(text: str, doc, is_scanned: bool, dpi=150) -> str:
        """
        This function extracts the document title using LLM
        Args:
            text: str - text from the document
            doc: PyMuPDF document object
            is_scanned: bool - whether document is scanned (no text layer)

        Returns:
            str - extracted title
        """
        llm = ChatVertexAI(model_name="gemini-2.5-flash", temperature=0)

        if is_scanned:

            page = doc.load_page(0)
            pix = await asyncio.to_thread(page.get_pixmap,  dpi=dpi)
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

    @staticmethod
    async def __get_closest_tag(text: str, top_k=1, similarity_threshold=0.3) -> str:
        """
        This function gets the closest matching tag from the text using vector similarity search.

        Args:
            text: str - text from the document

        Returns:
            str - matching tag name or default tag if no match found
        """
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
                logger.info(f"Found matching tag: {tag_name} (similarity: {similarity:.3f})")
                return tag_name
            else:
                logger.error("No matching tag found, using default")
                return "NOT_FOUND"

        except Exception as e:
            logger.error(f"Error in __get_closest_tag: {e}")
            return "ERROR"

        finally:
            if db is not None:
                db.close()

    def get_cache_stats(self) -> dict:
        """
        Получить статистику PDF кэша.
        
        Returns:
            dict: Cache statistics with hits, misses, size, and hit_rate
        """
        return self.pdf_cache.get_stats()

    async def verify_document_dates(self, document_titles: Dict[str, str] = None) -> Dict[str, Dict]:
        """
        Verify document dates and determine expiration status with optimized processing.
        
        Args:
            document_titles: Optional mapping of file paths to document titles
            
        Returns:
            Dict: {file_path: {status, has_expiration_period, expiration_date, messages, ...}}
        """
        logger.info(f"Starting date verification for {len(self.files)} files")
        
        # Get dates from existing date_check method
        files_dates = await self.date_check()
        current_date = date.today()
        results = {}
        
        for file_path in self.files:
            try:
                dates = files_dates.get(file_path, [])
                title = (document_titles or {}).get(file_path, "")
                has_expiration = self._has_expiration_period(title)
                
                # Single call to verify dates (includes parsing optimization)
                status, exp_date, messages = self._verify_dates(dates, has_expiration, current_date)
                
                # Count valid dates efficiently (avoid re-parsing)
                valid_count = sum(1 for msg in messages if " -> " in msg and "не удалось распознать" not in msg)
                
                results[file_path] = {
                    "status": status,
                    "has_expiration_period": has_expiration,
                    "expiration_date": exp_date.isoformat() if exp_date else None,
                    "messages": messages,
                    "dates_found": len(dates),
                    "valid_dates_count": valid_count
                }
                
                logger.debug(f"File {file_path}: {status}")
                
            except Exception as e:
                logger.error(f"Error verifying {file_path}: {e}")
                results[file_path] = {
                    "status": "ошибка обработки",
                    "error": str(e),
                    "messages": [f"Ошибка при обработке: {e}"]
                }
        
        logger.info(f"Completed date verification for {len(results)} files")
        return results
    
    def _has_expiration_period(self, title: str) -> bool:
        """Check if document type has expiration period based on title keywords."""
        title_lower = title.lower()
        return any(keyword in title_lower for keyword in self._EXPIRABLE_KEYWORDS)
    
    def _parse_date(self, date_string: str) -> Optional[date]:
        """Parse date string using optimized patterns for DD.MM.YYYY, Kazakh and Russian months."""
        if not date_string:
            return None
        
        text = date_string.strip()
        
        # Try DD.MM.YYYY pattern first (most common)
        match = re.search(self._DATE_PATTERNS[0], text)
        if match:
            try:
                return date(int(match.group(3)), int(match.group(2)), int(match.group(1)))
            except ValueError:
                pass
        
        # Try YYYY-MM-DD pattern
        match = re.search(self._DATE_PATTERNS[1], text)
        if match:
            try:
                return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
            except ValueError:
                pass
        
        # Try month names (Kazakh/Russian) - case insensitive
        match = re.search(self._DATE_PATTERNS[2], text.lower())
        if match:
            try:
                day, month_name, year = int(match.group(1)), match.group(2), int(match.group(3))
                month = self._MONTH_NAMES.get(month_name)
                if month:
                    return date(year, month, day)
            except (ValueError, KeyError):
                pass
        
        return None
    
    def _verify_dates(self, dates: List[Tuple[int, str]], has_expiration: bool, current_date: date) -> Tuple[str, Optional[date], List[str]]:
        """Verify document dates and return status with optimized processing."""
        if not dates:
            return ("некорректная дата" if has_expiration else "без срока действия"), None, ["Даты не найдены"]
        
        # Parse and collect valid dates in one pass
        valid_dates, messages = [], []
        for page_num, date_string in dates:
            if parsed := self._parse_date(date_string):
                valid_dates.append((parsed, page_num, date_string))
                messages.append(f"Страница {page_num}: {date_string} -> {parsed}")
            else:
                messages.append(f"Страница {page_num}: не удалось распознать '{date_string}'")
        
        if not valid_dates:
            messages.append("Найденные даты не распознаны")
            return "некорректная дата", None, messages
        
        most_recent_date, page_num, _ = max(valid_dates)
        
        # Non-expirable documents
        if not has_expiration:
            messages.append("Документ не требует проверки срока действия")
            return "без срока действия", most_recent_date, messages
        
        # Expiration check with single comparison
        days_diff = (most_recent_date - current_date).days
        if days_diff >= 0:
            messages.append(f"Действителен до {most_recent_date} ({days_diff} дней)")
            return "действующий", most_recent_date, messages
        else:
            messages.append(f"Просрочен на {abs(days_diff)} дней (истек {most_recent_date})")
            return "просроченный", most_recent_date, messages