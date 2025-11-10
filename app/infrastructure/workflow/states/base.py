from typing import TypedDict, Any, Dict, List, Optional


class BaseState(TypedDict):
    """
    Base state object that flows through the LangGraph workflow.
    
    This state contains:
    - Project identification
    - Database session for data persistence
    - Logger for real-time progress tracking
    - File list from the database
    - Error tracking
    """
    
    # Project identification
    project_id: str
    
    # Database session (passed through workflow)
    db_session: Any  # SQLAlchemy Session
    
    # Logger for real-time progress tracking
    logger: Optional[Any]  # ProjectLogger instance

    gcs_client: Any  # Google Cloud Storage client
    
    # Error handling
    errors: List[str]
    
    # Workflow status
    current_step: str