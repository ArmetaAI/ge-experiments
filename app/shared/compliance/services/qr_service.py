"""QR code detection service."""
import asyncio
import cv2
import fitz
import numpy
from urllib.parse import unquote
from pyzbar import pyzbar
from typing import List
from google.api_core import exceptions as google_exceptions

from .base_service import BaseComplianceService


class QRCodeService(BaseComplianceService):
    """Service for detecting QR codes in PDF documents."""
    
    async def process(self, files: List[str]) -> dict:
        """
        Count QR codes in each file.
        
        Args:
            files: List of file paths in GCS
            
        Returns:
            dict: {file_path: qr_count}
        """
        tasks = [self._process_file(fp) for fp in files]
        results = await asyncio.gather(*tasks)
        return dict(zip(files, results))
    
    async def _process_file(self, file_path: str) -> int:
        """Process single file for QR codes."""
        return await asyncio.to_thread(self._process_file_sync, file_path)
    
    def _process_file_sync(self, file_path: str) -> int:
        """Sync processing (runs in thread pool)."""
        try:
            doc = self.pdf_cache.get_or_load(
                file_path=unquote(file_path),
                bucket=self.bucket
            )
            return self._count_qr_in_doc(doc, file_path)
        except google_exceptions.NotFound:
            self.logger.warning(f"File not found: {file_path}")
            return 0
        except fitz.FileDataError:
            self.logger.warning(f"Corrupt PDF: {file_path}")
            return 0
        except Exception as e:
            self.logger.error(f"Error processing {file_path}: {e}")
            return 0
    
    def _count_qr_in_doc(self, doc: fitz.Document, file_path: str) -> int:
        """
        Count QR codes in entire document.

        Args:
            doc: Document object
            file_path: File path for logging context

        Returns:
            int: Total number of QR codes found
        """
        total_qr_codes = 0
        image_cache: dict[int, int] = {}

        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            image_list = page.get_images(full=True)
            if not image_list:
                continue

            total_qr_codes += self._process_images_on_page(
                doc, page_num, file_path, image_list, image_cache
            )

        return total_qr_codes

    def _process_images_on_page(
            self,
            doc: fitz.Document,
            page_num: int,
            file_path: str,
            image_list: list,
            image_cache: dict[int, int]
    ) -> int:
        """
        Process all images on a single page and return their QR code count.

        This method contains the single inner loop over images.
        It is a nested loop but its time complexity is O(pages + images),
        which is best we could come up with (30.10.2025).
        Now uses caching to avoid re-processing identical images.

        Args:
            doc: Document object to extract images
            page_num: Current page number for logging
            file_path: File path for logging
            image_list: List of images to process
            image_cache: Cache of already processed images (xref -> qr_count)

        Returns:
            int: Number of QR codes found on this page
        """
        page_qr_codes = 0

        for img_info in image_list:
            xref = img_info[0]
            if xref == 0:
                continue

            # Check cache first
            if xref in image_cache:
                page_qr_codes += image_cache[xref]
                continue

            try:
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                np_arr = numpy.frombuffer(image_bytes, numpy.uint8)
                img_cv = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                if img_cv is None:
                    self.logger.warning(
                        f"QR Check: OpenCV failed to decode image (xref={xref}) "
                        f"on page {page_num + 1} in {file_path}"
                    )
                    image_cache[xref] = 0
                    continue

                gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(
                    gray, 0, 255,
                    cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )
                decoded_objects = pyzbar.decode(
                    thresh,
                    symbols=[pyzbar.ZBarSymbol.QRCODE]
                )

                qr_count_for_current_image = len(decoded_objects)
                page_qr_codes += qr_count_for_current_image

                # Cache the result
                image_cache[xref] = qr_count_for_current_image

            except Exception as e:
                self.logger.warning(
                    f"QR Check: Failed to process one image (xref={xref}) "
                    f"on page {page_num + 1} in {file_path}. Error: {e}"
                )
                image_cache[xref] = 0
                continue

        return page_qr_codes

