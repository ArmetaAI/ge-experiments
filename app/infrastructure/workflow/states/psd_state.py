from typing import Any, Dict, List, Literal, Optional, TypedDict
from sqlalchemy.orm import Session
from .base import BaseState


class PSDGraphState(BaseState):
    """
    Main state object that flows through the LangGraph workflow.
    
    This state contains:
    - Project identification
    - Database session for data persistence
    - Logger for real-time progress tracking
    - PSD related file list from the database
    - ОПЗ document reference
    - Extracted composition table from ОПЗ
    - Final report for PSD
    - Error tracking
    """

    psd_files: Optional[List[Dict[str, Any]]]
    
    # Extracted table from ОПЗ document
    # List of dictionaries with keys: doc_number, doc_name, etc.
    extracted_composition_table: Optional[List[Dict[str, str]]]
    
    # Final report for PSD
    final_report_psd: Optional[Dict[str, Any]]
    
    # Package ID for event logging
    package_id: Optional[int]

    # Event logger for tracking workflow progress
    event_logger: Optional[Any] # PackageEventLogger instance 