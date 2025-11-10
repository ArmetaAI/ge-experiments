"""
Project Logger Utility for real-time log streaming to frontend.

This module provides a logging system that captures logs and stores them
in the database for real-time monitoring by the frontend.
"""

import json
from datetime import datetime as dt
import datetime
from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session
from app.infrastructure.persistence.database.models import SessionLocal, Project


class ProjectLogger:
    """
    Logger that captures print statements and stores them in the database
    for real-time frontend monitoring.
    """
    
    def __init__(self, project_id: str, db_session: Optional[Session] = None):
        """
        Initialize project logger.
        
        Args:
            project_id: Project unique identifier
            db_session: Optional database session, creates new one if None
        """
        self.project_id = project_id
        self.db_session = db_session
        self._should_close_session = db_session is None
        
        if self.db_session is None:
            self.db_session = SessionLocal()
    
    def log(self, message: str, level: str = "info", step: Optional[str] = None) -> None:
        """
        Add a log entry to the project logs.
        
        Args:
            message: Log message
            level: Log level (info, success, error, warning)
            step: Optional workflow step identifier
        """
        timestamp = dt.now(datetime.timezone.utc).isoformat() + "Z"

        log_entry = {
            "timestamp": timestamp,
            "level": level,
            "message": message,
            "step": step
        }
        
        # Also print to console for backward compatibility
        print(f"[{timestamp}] [{level.upper()}] {message}")
        
        try:
            # Get current project
            project = self.db_session.query(Project).filter(
                Project.id == self.project_id
            ).first()
            
            if project:
                # Initialize logs if None
                if project.logs is None:
                    project.logs = []
                
                # Add new log entry
                current_logs = project.logs.copy() if isinstance(project.logs, list) else []
                current_logs.append(log_entry)
                
                # Keep only the latest 100 log entries to prevent database bloat
                if len(current_logs) > 100:
                    current_logs = current_logs[-100:]
                
                project.logs = current_logs
                project.updated_at = dt.now(datetime.timezone.utc)
                
                self.db_session.commit()
                
        except Exception as e:
            print(f"[Logger Error] Failed to save log to database: {str(e)}")
            # Don't raise exception to avoid breaking workflow
    
    def info(self, message: str, step: Optional[str] = None) -> None:
        """Log info message."""
        self.log(message, "info", step)
    
    def success(self, message: str, step: Optional[str] = None) -> None:
        """Log success message."""
        self.log(message, "success", step)
    
    def error(self, message: str, step: Optional[str] = None) -> None:
        """Log error message."""
        self.log(message, "error", step)
    
    def warning(self, message: str, step: Optional[str] = None) -> None:
        """Log warning message."""
        self.log(message, "warning", step)
    
    def close(self) -> None:
        """Close database session if we created it."""
        if self._should_close_session and self.db_session:
            self.db_session.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def get_project_logs(project_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    """
    Retrieve logs for a project.
    
    Args:
        project_id: Project unique identifier
        limit: Maximum number of logs to return (default 50)
        
    Returns:
        List of log entries with timestamps, levels, and messages
    """
    db_session = SessionLocal()
    
    try:
        project = db_session.query(Project).filter(
            Project.id == project_id
        ).first()
        
        if not project or not project.logs:
            return []
        
        # Return the latest logs
        logs = project.logs if isinstance(project.logs, list) else []
        return logs[-limit:] if len(logs) > limit else logs
        
    except Exception as e:
        print(f"[Logger Error] Failed to retrieve logs: {str(e)}")
        return []
    
    finally:
        db_session.close()
