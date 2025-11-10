"""
Script to extract document titles and keywords from PDFs using Vertex AI OCR.

This script:
1. Processes first 2 pages of each PDF with OCR
2. Extracts document title using Vertex AI vision
3. Extracts keywords from the content
4. Saves results to CSV file

Usage:
    python scripts/extract_tags_from_pdfs.py --input-dir tagged_documents
"""

import asyncio
import argparse
import csv
import base64
import logging
from datetime import datetime
from pathlib import Path
import sys
from typing import Optional, List, Dict

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

import fitz  # PyMuPDF
from langchain_google_vertexai import ChatVertexAI
from langchain_core.messages import HumanMessage

from app.utils.processing_utils import parse_json_response

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentOCRProcessor:
    """
    Processor for extracting titles and keywords from PDF documents using OCR.
    """

    def __init__(self):
        """Initialize the OCR processor."""
        self.llm = ChatVertexAI(model_name="gemini-2.0-flash-exp", temperature=0)
        logger.info("DocumentOCRProcessor initialized")

    async def extract_title_and_keywords(
        self,
        file_path: Path,
        max_pages: int = 2
    ) -> Optional[Dict[str, any]]:
        """
        Extract title and keywords from a PDF file using Vertex AI OCR.

        Args:
            file_path: Path to the local PDF file
            max_pages: Number of pages to process (default: 2)

        Returns:
            Dictionary with:
                - file_path: str
                - title: str
                - keywords: List[str]
                - error: str (if any)
        """
        logger.info(f"Processing: {file_path.name}")

        try:
            # Open local PDF file
            doc = await asyncio.to_thread(fitz.open, str(file_path))

            # Process first N pages
            num_pages = min(max_pages, doc.page_count)
            logger.info(f"  Processing {num_pages} page(s)...")

            # Extract images from first N pages
            page_images = []
            for page_num in range(num_pages):
                page = doc.load_page(page_num)
                pix = await asyncio.to_thread(page.get_pixmap, dpi=150)
                img_bytes = await asyncio.to_thread(pix.tobytes, "png")
                img_base64 = base64.b64encode(img_bytes).decode()
                page_images.append(img_base64)
                logger.info(f"  Extracted page {page_num + 1}")

            await asyncio.to_thread(doc.close)

            # Call Vertex AI with all page images
            result = await self._call_vertex_ai_ocr(page_images, file_path)

            if result:
                logger.info(f"  ✓ Title: {result.get('title', 'N/A')}")
                logger.info(f"  ✓ Keywords: {len(result.get('keywords', []))} found")
                return {
                    "file_path": str(file_path.name),
                    "title": result.get("title", ""),
                    "keywords": result.get("keywords", []),
                    "error": None
                }
            else:
                logger.warning(f"  ✗ Failed to extract data")
                return {
                    "file_path": str(file_path.name),
                    "title": "",
                    "keywords": [],
                    "error": "Failed to parse Vertex AI response"
                }

        except Exception as e:
            logger.error(f"  ✗ Error processing {file_path.name}: {e}")
            return {
                "file_path": str(file_path.name),
                "title": "",
                "keywords": [],
                "error": str(e)
            }

    async def _call_vertex_ai_ocr(
        self,
        page_images: List[str],
        file_path: Path
    ) -> Optional[Dict]:
        """
        Call Vertex AI with page images to extract title and keywords.

        Args:
            page_images: List of base64-encoded page images
            file_path: Original file path for context

        Returns:
            Dictionary with title and keywords
        """
        # Build message content with all pages
        content = []

        # Add all page images
        for idx, img_base64 in enumerate(page_images):
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{img_base64}"
                }
            })

        # Add the instruction text
        content.append({
            "type": "text",
            "text": """Ты эксперт по анализу проектной документации Казахстана.

Проанализируй эти изображения страниц документа и выполни следующие задачи:

1. **ИЗВЛЕКИ НАЗВАНИЕ/ЗАГОЛОВОК ДОКУМЕНТА**:
   - Найди основное название или заголовок документа (обычно на первой странице)
   - Это может быть тип документа (например: "Пояснительная записка", "Архитектурные решения")
   - Или конкретное название проекта
   - Если есть номер тома (например "Том 1"), включи его в название

2. **ИЗВЛЕКИ КЛЮЧЕВЫЕ СЛОВА**:
   - Определи от 5 до 15 ключевых слов/фраз, которые характеризуют этот документ
   - Включи технические термины, типы работ, разделы, названия систем
   - Включи организации, стандарты, нормативные документы (если упоминаются)
   - Включи географические локации (адрес, регион, город)
   - НЕ включай общие слова вроде "документ", "страница", "проект" без контекста

3. **ВЕРНИ РЕЗУЛЬТАТ В JSON ФОРМАТЕ**:
```json
{
    "title": "Название документа или тип документа",
    "keywords": ["ключевое слово 1", "ключевое слово 2", "ключевое слово 3", ...]
}
```

**ВАЖНО**:
- Если не можешь найти название, используй тип документа или описание содержимого
- Ключевые слова должны быть конкретными и релевантными содержимому
- Возвращай ТОЛЬКО JSON, без дополнительных объяснений"""
        })

        # Create message
        message = HumanMessage(content=content)

        try:
            # Call Vertex AI
            response = await asyncio.to_thread(self.llm.invoke, [message])

            # Parse JSON response
            result = parse_json_response(response.content)

            return result

        except Exception as e:
            logger.error(f"Vertex AI call failed: {e}")
            return None

    async def process_directory(
        self,
        input_dir: Path,
        max_files: Optional[int] = None
    ) -> List[Dict]:
        """
        Process all PDF files in a local directory.

        Args:
            input_dir: Path to directory containing PDF files
            max_files: Maximum number of files to process (None for all)

        Returns:
            List of results for each file
        """
        logger.info(f"Scanning directory: {input_dir}")

        # Find all PDF files in directory
        pdf_files = list(input_dir.glob('*.pdf'))

        logger.info(f"Found {len(pdf_files)} PDF files")

        # Limit if specified
        if max_files:
            pdf_files = pdf_files[:max_files]
            logger.info(f"Processing first {max_files} files")

        # Process each file
        results = []
        for idx, file_path in enumerate(pdf_files, 1):
            logger.info(f"\n[{idx}/{len(pdf_files)}]")
            result = await self.extract_title_and_keywords(file_path)
            if result:
                results.append(result)

            # Small delay to avoid rate limiting
            await asyncio.sleep(0.5)

        return results


def save_to_csv(results: List[Dict], output_path: str):
    """
    Save extraction results to CSV file.

    Args:
        results: List of extraction results
        output_path: Path to output CSV file
    """
    logger.info(f"\nSaving results to: {output_path}")

    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['file_path', 'title', 'keywords', 'error']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in results:
            # Convert keywords list to comma-separated string
            result_row = {
                'file_path': result['file_path'],
                'title': result['title'],
                'keywords': ', '.join(result['keywords']) if result['keywords'] else '',
                'error': result['error'] or ''
            }
            writer.writerow(result_row)

    logger.info(f"✓ Saved {len(results)} results to CSV")


async def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Extract titles and keywords from PDF documents using Vertex AI OCR'
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default='tagged_documents',
        help='Directory containing PDF files (default: tagged_documents)'
    )
    parser.add_argument(
        '--max-files',
        type=int,
        default=None,
        help='Maximum number of files to process (default: all)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output CSV file path (default: results/document_tags_TIMESTAMP.csv)'
    )

    args = parser.parse_args()

    # Convert input dir to Path
    input_dir = Path(__file__).parent.parent / args.input_dir
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return

    # Set default output path with timestamp
    if args.output is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path(__file__).parent.parent / 'results'
        output_dir.mkdir(exist_ok=True)
        args.output = str(output_dir / f'document_tags_{timestamp}.csv')

    logger.info("=" * 80)
    logger.info("PDF TITLE & KEYWORDS EXTRACTION")
    logger.info("=" * 80)
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Max files: {args.max_files or 'unlimited'}")
    logger.info(f"Output: {args.output}")
    logger.info("=" * 80)

    # Create processor
    processor = DocumentOCRProcessor()

    # Process files
    results = await processor.process_directory(
        input_dir=input_dir,
        max_files=args.max_files
    )

    # Save to CSV
    if results:
        save_to_csv(results, args.output)

        # Print summary
        logger.info("\n" + "=" * 80)
        logger.info("SUMMARY")
        logger.info("=" * 80)
        successful = sum(1 for r in results if not r['error'])
        failed = sum(1 for r in results if r['error'])
        logger.info(f"Total processed: {len(results)}")
        logger.info(f"Successful: {successful}")
        logger.info(f"Failed: {failed}")
        logger.info("=" * 80)
    else:
        logger.warning("No results to save")


if __name__ == "__main__":
    asyncio.run(main())