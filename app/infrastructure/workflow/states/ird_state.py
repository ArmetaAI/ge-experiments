from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional, TypedDict, Literal
from sqlalchemy.orm import Session
from .base import BaseState


#TO BE DISCUSSED: How the psd_workflow will work
Discipline = Literal[
    "ARCHITECTURAL",
    "STRUCTURAL",
    "ELECTRICAL",
    "HVAC",  # Heating, Ventilation, and Air Conditioning
    "PLUMBING_AND_WATER",
    "FIRE_SAFETY",
    "GENERAL_COORDINATION", # For tasks that require comparing documents
    "UNKNOWN" # For documents that can't be classified
]

class WorkPackage(BaseModel):
    """
    Represents a single, self-contained task for a specialized agent
    to perform on a set of construction documents.
    """
    task_id: int = Field(
        description="A unique sequential integer ID for this task, e.g., 1, 2, 3.",
    )
    discipline: Discipline = Field(
        description="The specific engineering discipline this task relates to. This will determine which agent is assigned.",
    )
    task_description: str = Field(
        description="A clear, concise, and specific instruction for the assigned agent. What should it accomplish with the source documents?",
    )
    source_documents: List[str] = Field(
        description="A list of document filenames or IDs that are required to complete this task.",
    )
    dependencies: List[int] = Field(
        default=[],
        description="A list of `task_id`s that must be completed before this task can start. Leave empty if there are no dependencies.",
    )


class ProjectPlan(BaseModel):
    """
    Defines the complete, sequenced plan for processing a package of
    construction project documents. It is composed of a list of work packages.
    """
    work_packages: List[WorkPackage] = Field(
        description="The comprehensive list of all tasks to be executed to process the document package."
    )


class IRDGraphState(BaseState):
    """
    Main state object that flows through the LangGraph workflow.
    
    This state contains:
    - Project identification
    - Database session for data persistence
    - Logger for real-time progress tracking
    - PSD related file list from the database
    - ОПЗ document reference
    - Extracted composition table from ОПЗ
    - Final report for IRD
    - Error tracking
    """
    

    ird_files: Optional[List[Dict[str, Any]]]
    
    validation_results: Optional[Dict[str, Any]]
    
    compliance_results: Optional[Dict[str, Any]]
    # Final report for IRD
    final_report_ird: Optional[Dict[str, Any]]

    # Package ID for event logging
    package_id: Optional[int]

    # Event logger for tracking workflow progress
    event_logger: Optional[Any] # PackageEventLogger instance


class PSDWorkerState(TypedDict):
    project_id: str
    db_session: Session
    logger: Any  # Replace with actual logger type if available

    psd_files: Optional[List[Dict[str, Any]]]
    
    extracted_composition_table: Optional[List[Dict[str, str]]]
    
    worker_report: str

    errors: List[str]

class IRDComplianceState(TypedDict):
    check_format_result: dict
    empty_lists_result: dict
    insufficient_files_result: dict
    page_number_result: dict
    classify_result: dict