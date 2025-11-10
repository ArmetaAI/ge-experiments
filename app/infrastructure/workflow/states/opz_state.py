from typing import Any, Dict, List, Optional, TypedDict
from sqlalchemy.orm import Session

class OPZGraphState(TypedDict):
    """
    State object for the OPZ extraction subgraph.
    
    This state flows through the OPZ document processing workflow and contains:
    - Project identification and database access
    - GCS client for downloading files
    - Logger for real-time progress tracking
    - File list from database
    - ОПЗ document reference and temporary file path
    - Extracted PDF pages data
    - Composition section location
    - Extracted composition table
    - Error tracking and workflow status
    """
    
    
    # ОПЗ document information
    psd_files: Optional[List[Dict[str, Any]]]

    opz_file: Optional[Dict[str, str]]  # Contains: filename, gcs_path
    
    # Temporary file path for downloaded PDF
    document: Optional[str]  # Path to temporary PDF file
    
    # Extracted PDF pages with text content
    pdf_pages: Optional[List[Dict[str, Any]]]  # List of {page_number, text, char_count}

    image_pages: Optional[List[Dict[str, Any]]]  # List of {page_number, image_variable}
    
    # Page number where composition section starts
    composition_start_page: Optional[int]
    
    # Extracted composition table
    # List of dictionaries with keys: doc_number, doc_name
    extracted_composition_table: Optional[List[Dict[str, str]]]
    
    # Final report for OPZ processing
    report_opz: Optional[Dict[str, Any]]
    