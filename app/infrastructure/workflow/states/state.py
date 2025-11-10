"""
LangGraph State Management for Госэкспертиза completeness check workflow.

This module defines the GraphState TypedDict that will be passed between
nodes in the LangGraph workflow. The state includes all necessary information
for processing the completeness check.
"""

from typing import TypedDict, List, Optional, Dict, Any
from sqlalchemy.orm import Session
from .base import BaseState


class MainState(BaseState):
    """
    Main state object that flows through the LangGraph workflow.
    
    This state contains:
    - Project identification
    - Database session for data persistence
    - Logger for real-time progress tracking
    - File list from the database
    - ОПЗ document reference
    - Extracted composition table from ОПЗ
    - Comparison report results
    - Error tracking
    """
    
    
    # ОПЗ document information
    opz_file: Optional[Dict[str, str]]  # Contains: filename, gcs_path
    
    # Extracted table from ОПЗ document
    # List of dictionaries with keys: doc_number, doc_name, etc.
    extracted_composition_table: Optional[List[Dict[str, str]]]
    
    # Comparison results
    final_report: Optional[Dict[str, Any]]


# Alias for backward compatibility
GraphState = MainState
    
    


def create_initial_state(project_id: str, db_session: Session, logger=None, gcs_client=None) -> MainState:
    """
    Create the initial state for a new completeness check workflow.
    
    Args:
        project_id: Unique project identifier
        db_session: SQLAlchemy database session
        logger: ProjectLogger instance for real-time logging
        
    Returns:
        GraphState: Initialized state object
    """
    return MainState(
        project_id=project_id,
        db_session=db_session,
        logger=logger,
        file_list_from_db=None,
        gcs_client=gcs_client,
        opz_file=None,
        extracted_composition_table=None,
        comparison_report=None,
        errors=[],
        current_step="initialized"
    )
