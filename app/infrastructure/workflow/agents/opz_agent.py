"""
OPZ Agent - LangGraph subgraph for extracting composition table from ОПЗ document.
"""

import fitz  # PyMuPDF
import json
import os
import re
import tempfile

from typing import List, Dict, Any
from pathlib import Path

from langgraph.graph import StateGraph, END
from lmnr import observe

import app.infrastructure.workflow.agents.table_extractor as table_extractor
from app.infrastructure.workflow.states.opz_state import OPZGraphState
from langchain_google_vertexai import ChatVertexAI
from langchain_core.messages import HumanMessage, SystemMessage




llm = ChatVertexAI(model_name="gemini-2.5-flash", temperature=0)


# ============================================================================
# LANGGRAPH NODES - Each node is a pure function that operates on state
# ============================================================================
#NEEDS: psd_files, logger
#FIXME: OPZ finding must be more robust. 
#TODO: Files maybe not explanatory enough, it could be required to check the file content
@observe(tags=['opz_agent'], name="find_opz_document")
def find_opz_document(state: OPZGraphState) -> Dict[str, Any]:
    """
    Node 1: Find the ОПЗ document in the file list.
    Searches for files matching OPZ patterns in the uploaded files.
    """
    logger = state.get("logger")
    if logger:
        logger.info("Searching for ОПЗ document in uploaded files", "find_opz")
    
    try:
        file_list = state["psd_files"]
        
        if not file_list:
            raise Exception("No files found in database")
        
        opz_patterns = ["опз", "opz", "общая", "пояснительная записка", "состав проекта"]
        
        variants = []
        for file_info in file_list:
            filename_lower = file_info["filename"].lower()
            
            for pattern in opz_patterns:
                if pattern in filename_lower:
                    variants.append(file_info)
                    break

        if not variants:
            raise Exception("ОПЗ document not found in uploaded files")
        
        # Take the first match (or use LLM to select best match if multiple)
        opz_file = variants[0]
        
        if logger:
            logger.success(f"Found ОПЗ document: {opz_file['filename']}", "find_opz")
        
        return {
            "opz_file": opz_file,
            "current_step": "opz_found"
        }
        
    except Exception as e:
        if logger:
            logger.error(f"Error finding ОПЗ document: {str(e)}", "find_opz")
        return {
            "errors": state.get("errors", []) + [f"find_opz_document: {str(e)}"],
            "current_step": "error"
        }

#NEEDS: opz_file, gcs_client, logger, ADDS: document
@observe(tags=['opz_agent'], name="download_opz_document")
def download_opz_document(state: OPZGraphState) -> Dict[str, Any]:
    """
    Node 2: Download the ОПЗ PDF from Google Cloud Storage to a temporary file.
    The temporary file path is stored in state for subsequent processing.
    """
    logger = state.get("logger")
    gcs_client = state.get("gcs_client")
    opz_file = state.get("opz_file")

    if logger:
        logger.info(f"Downloading ОПЗ document from GCS", "download")

    temp_file_path = None
    try:
        gcs_path = opz_file.get("gcs_path")

        if not gcs_path or not gcs_path.startswith("gs://"):
            raise ValueError(f"Invalid GCS path: {gcs_path}")

        # Parse GCS path
        path_parts = gcs_path[5:].split("/", 1)
        bucket_name = path_parts[0]
        blob_name = path_parts[1] if len(path_parts) > 1 else ""

        # Download from GCS
        bucket = gcs_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        # Create temporary file with PDF extension
        suffix = Path(blob_name).suffix or ".pdf"
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        temp_file_path = temp_file.name
        temp_file.close()

        blob.download_to_filename(temp_file_path)

        if logger:
            logger.success(f"✓ Downloaded to temporary file: {temp_file_path}", "download")

        return {
            "document": temp_file_path,
            "current_step": "document_downloaded"
        }

    except Exception as e:
        # Если это не вычищать то может быть hard memory leak - скачался но ошибка прошла и файл не удалился
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as cleanup_error:
                if logger:
                    logger.warning(f"Failed to clean up temp file: {cleanup_error}", "download")

        if logger:
            logger.error(f"Error downloading document: {str(e)}", "download")
        return {
            "errors": state.get("errors", []) + [f"download_opz_document: {str(e)}"],
            "current_step": "error"
        }

#NEEDS: logger, document; ADDS: image_pages, pdf_pages
#TODO: Full text extraction is too much for LLM. Consider using image processing.
@observe(tags=['opz_agent'], name="extract_pdf_pages")
def extract_pdf_pages(state: OPZGraphState) -> Dict[str, Any]:
    """
    Node 3: Extract text from all PDF pages for LLM processing.
    Uses PyMuPDF to extract text content page by page.
    """
    logger = state.get("logger")
    document_path = state.get("document")

    if logger:
        logger.info("Extracting text from PDF pages", "extract_pages")

    doc = None
    try:
        doc = fitz.open(document_path)

        # Extract text from each page with metadata
        pages_data = []
        image_pages = []
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            text = page.get_text()
            image = page.get_images(full=True)
            image_pages.append({"page_number": page_idx + 1, "image_variable": image})

            pages_data.append({
                "page_number": page_idx + 1,
                "text": text,
                "char_count": len(text)
            })

        if logger:
            logger.success(f"✓ Extracted text from {len(pages_data)} pages", "extract_pages")

        return {
            "pdf_pages": pages_data,
            "image_pages": image_pages,
            "current_step": "pages_extracted"
        }

    except Exception as e:
        if logger:
            logger.error(f"Error extracting PDF pages: {str(e)}", "extract_pages")
        return {
            "errors": state.get("errors", []) + [f"extract_pdf_pages: {str(e)}"],
            "current_step": "error"
        }
    finally:
        if doc is not None:
            doc.close()

#NEEDS: logger, pdf_pages; ADDS: composition_start_page
#TODO: Testing
#FIXME: Current search is not robust enough. Careful prompt engineering needed.
@observe(tags=['opz_agent'], name="find_composition_section")
def find_composition_section(state: OPZGraphState) -> Dict[str, Any]:
    """
    Node 4: Use LLM to find the "Состав проекта" (Project Composition) section.
    The LLM analyzes page texts to identify where the composition table begins.
    """
    logger = state.get("logger")
    pages_data = state.get("pdf_pages", [])
    
    if logger:
        logger.info("Using LLM to locate composition section", "find_section")
    pattern = re.compile(r"(состав\s+проекта|состав\s+проектной\s+документации)", re.IGNORECASE)
    try:
        # Create a summary of pages for the LLM
        start_page = ("0",0.0)
        for page in pages_data:  # Limit to first 20 pages
            preview = page["text"]
            if pattern.search(preview):
                system_prompt = SystemMessage(content="""Ты эксперт по анализу проектной документации Казахстана.
Твоя задача - найти страницу, которая содержит раздел-таблицу "Состав проекта" (или "Состав проектной документации").
Этот раздел обычно содержит таблицу, которая перечисляет все документы, входящие в состав проектной документации.

Верни JSON ответ в формате:
{
    "page_number": <номер страницы>,
    "confidence": <уверенность от 0 до 1>,
    "reasoning": "<краткое обоснование>"
}

Если не нашел, верни page_number = null.""")
                user_prompt = HumanMessage(content=f"""
                {json.dumps([preview], ensure_ascii=False, indent=2)}

                На какой странице находится раздел "Состав проекта"?""")
        
                # Call LLM
                response = llm.invoke([system_prompt, user_prompt])
                result = json.loads(response.content)

                page_number = result.get("page_number")
                confidence = float(result.get("confidence", 0))
                if confidence > start_page[1]:
                    start_page = (page_number, confidence)

        if start_page[0] is None or start_page[1] < 0.5:
            raise Exception(f"LLM and Regex search could not find composition section with confidence. Result: {result}")
        
        if logger:
            logger.success(f"✓ Found composition section on page {start_page[0]} (confidence: {start_page[1]})", "find_section")

        return {
            "composition_start_page": start_page[0],
            "current_step": "section_found"
        }
        
    except Exception as e:
        if logger:
            logger.error(f"❌ Error finding composition section: {str(e)}", "find_section")
        return {
            "errors": state.get("errors", []) + [f"find_composition_section: {str(e)}"],
            "current_step": "error"
        }

#NEEDS: logger, image_pages, composition_start_page; ADDS: composition_table
#OPTIMIZE: Currently uses full image data - I think maybe we can optimize for memory efficiency
@observe(tags=['opz_agent'], name="extract_composition_table")
def extract_composition_table(state: OPZGraphState) -> Dict[str, Any]:
    """
    Node 5: Use LLM to extract the composition table from identified pages.
    The LLM parses the table structure and extracts document numbers and names.
    """
    logger = state.get("logger")
    start_page = state.get("composition_start_page", 0)
    image_pages = state.get("image_pages", [])
    
    # Collect all tables from all pages
    all_standardized_rows = []
    all_concerns = []
    table_metadata = []
    
    if logger:
        logger.info(f"Processing {len(image_pages) - start_page} pages starting from page {start_page}", "extract_table")
    
    for i in range(start_page, len(image_pages)):
        page = image_pages[i]
        page_number = i + 1
        
        if logger:
            logger.info(f"Extracting tables from page {page_number}", "extract_table")
        
        # Invoke table extractor for this page
        table_state = table_extractor.invoke({
            "page_image": page.get("image_variable"), 
            "additional_context": """
                Extract document composition tables ONLY.
                Use these EXACT column names:
                - "doc_number": document number/code (e.g., "01", "ПЗ-1")
                - "doc_name": full document name
                - "doc_volume": volume/part info (OPTIONAL)
                - "doc_notes": additional notes (OPTIONAL)
                
                IGNORE: stamp tables, signature blocks, revision tables.
                One composition table per page."""
        })
        
        # Get extracted tables (list of table dicts)
        page_tables = table_state.get("extracted_tables", [])
        concerns = table_state.get("concerns", "")
        
        if concerns:
            all_concerns.append(f"Page {page_number}: {concerns}")
        
        # Add page context to each table
        for table in page_tables:
            table_id = table.get("table_id")
            rows = table.get("rows", [])
            
            # Validate and standardize rows
            standardized_rows = []
            for row in rows:
                # Ensure required fields exist
                if not row.get("doc_number") or not row.get("doc_name"):
                    if logger:
                        logger.warning(f"Skipping invalid row on page {page_number}: {row}", "extract_table")
                    continue
                
                # Create standardized row
                std_row = {
                    "doc_number": str(row.get("doc_number", "")).strip(),
                    "doc_name": str(row.get("doc_name", "")).strip(),
                    "doc_volume": str(row.get("doc_volume", "")).strip() if row.get("doc_volume") else None,
                    "doc_notes": str(row.get("doc_notes", "")).strip() if row.get("doc_notes") else None,
                    "_source_page": page_number,
                    "_source_table_id": table_id
                }
                standardized_rows.append(std_row)
            
            all_standardized_rows.extend(standardized_rows)
            
            # Store table metadata for validation
            table_metadata.append({
                "page": page_number,
                "table_id": table_id,
                "expected_rows": table.get("row_count", 0),
                "extracted_rows": len(standardized_rows),
                "columns": table.get("column_count", 0)
            })
    
    # Create final composition table
    composition_table = {
        "total_documents": len(all_standardized_rows),
        "total_tables_processed": len(table_metadata),
        "pages_processed": list(range(start_page + 1, len(image_pages) + 1)),
        "documents": all_standardized_rows,  # Now guaranteed to have doc_number/doc_name
        "table_metadata": table_metadata,     # For validation
        "concerns": all_concerns
    }
    
    # Aggregate all rows from all tables into a single unified list
    

    if logger:
        logger.success(f"Extracted {len(all_standardized_rows)} document entries from {len(table_metadata)} tables across {len(image_pages) - start_page} pages", "extract_table")
        if all_concerns:
            logger.warning(f"Encountered {len(all_concerns)} concerns during extraction", "extract_table")
    
    return {
        "extracted_composition_table": composition_table,
        "current_step": "table_extracted"
    }
    


@observe(tags=['opz_agent'], name="cleaning_up")
def cleanup_temporary_file(state: OPZGraphState) -> Dict[str, Any]:
    """
    Node 6: Clean up the temporary PDF file.
    This node always runs at the end to ensure resources are cleaned up.
    """
    logger = state.get("logger")
    document_path = state.get("document")
    
    try:
        if document_path and os.path.exists(document_path):
            os.unlink(document_path)
            if logger:
                logger.info("🧹 Cleaned up temporary file", "cleanup")
        
        return {
            "document": None,
            "current_step": "completed"
        }
        
    except Exception as e:
        if logger:
            logger.warning(f"ERROR:Could not clean up temporary file: {str(e)}", "cleanup")
        return {}


# ============================================================================
# SUBGRAPH CREATION
# ============================================================================

@observe(tags=['subgraph_creation'], name="graph_opz_creation")
def create_opz_subgraph() -> StateGraph:
    """
    Create the OPZ extraction subgraph.
    
    This subgraph handles the complete OPZ document processing:
    1. Find OPZ document in file list
    2. Download from GCS to temporary file
    3. Extract text from PDF pages
    4. Use LLM to locate composition section
    5. Use LLM to extract composition table
    6. Clean up temporary files
    
    Returns:
        Compiled StateGraph ready to be invoked
    """
    workflow = StateGraph(OPZGraphState)
    
    # Add nodes
    workflow.add_node("find_opz", find_opz_document)
    workflow.add_node("download", download_opz_document)
    workflow.add_node("extract_pages", extract_pdf_pages)
    workflow.add_node("find_section", find_composition_section)
    workflow.add_node("extract_table", extract_composition_table)
    workflow.add_node("cleanup", cleanup_temporary_file)
    
    # Define edges (linear flow with error handling)
    workflow.set_entry_point("find_opz")
    workflow.add_edge("find_opz", "download")
    workflow.add_edge("download", "extract_pages")
    workflow.add_edge("extract_pages", "find_section")
    workflow.add_edge("find_section", "extract_table")
    workflow.add_edge("extract_table", "cleanup")
    workflow.add_edge("cleanup", END)
    
    return workflow.compile()


# ============================================================================
# EXPORT - This subgraph can be imported and used by the main workflow
# ============================================================================

opz_subgraph = create_opz_subgraph()
