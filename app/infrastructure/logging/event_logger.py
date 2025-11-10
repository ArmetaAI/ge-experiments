"""
Event logger utility for tracking package processing progress.

This module provides functionality to log events during package processing
for real-time monitoring on the frontend.
"""

from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from sqlalchemy.orm import Session
from contextlib import contextmanager

from app.infrastructure.persistence.database.models import ProjectPackage

class PackageEventLogger:
    """
    Logger for package processing events.

    Generates and stores events in the package's logs field for real-time
    monitoring of processing progress.

    Usage:
        # Basic usage
        logger = PackageEventLogger(db, package_id)
        logger.node_started("classify_docs")
        # ... do work ...
        logger.node_completed("classify_docs", {"files_classified": 10})

        # Context manager usage
        with logger.track_node("extract_table"):
            # ... do work ...
            pass
    """

    def __init__(self, db: Session, package_id: int):
        """
        Initialize the event logger.

        Args:
            db: Database session
            package_id: ID of the package to log events for
        """
        self.db = db
        self.package_id = package_id

    def log_event(
        self,
        event: str,
        node: str,
        status: str = "started",
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log an event for the package.

        Args:
            event: Event type (e.g., "node_started", "node_completed", "node_failed")
            node: Name of the workflow node (e.g., "classify_docs", "extract_table")
            status: Event status ("started", "completed", "failed", "skipped")
            details: Optional additional details about the event
        """

        try:
            # Get the package
            package = self.db.query(ProjectPackage).filter(
                ProjectPackage.id == self.package_id
            ).first()
            
            if not package:
                print(f"[EventLogger Error] Package {self.package_id} not found")
                return

            # Create event entry - EXACTLY as specified in requirements
            event_entry = {
                "event": event,
                "node": node,
                "status": status,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            # Add details if provided
            if details:
                event_entry["details"] = details

            # Initialize logs if None
            if package.logs is None:
                package.logs = []

            # Append event to logs
            current_logs = list(package.logs) if isinstance(package.logs, list) else []
            current_logs.append(event_entry)

            # Limit to last 500 events to prevent database bloat
            max_logs = 500
            if len(current_logs) > max_logs:
                current_logs = current_logs[-max_logs:]

            package.logs = current_logs

            # Update package timestamp
            package.updated_at = datetime.now(timezone.utc)

            # Commit to database
            self.db.commit()

            print(f"[EventLogger] Package {self.package_id}: {event} - {node} ({status})")

        except Exception as e:
            print(f"[EventLogger Error] Failed to log event for package {self.package_id}: {str(e)}")
            self.db.rollback()

    def node_started(self, node: str, details: Optional[Dict[str, Any]] = None) -> None:
        """
        Log that a node has started processing.

        Args:
            node: Node name (e.g., "classify_docs", "extract_table")
            details: Optional details (e.g., {"input_files": 5})
        """
        self.log_event("node_started", node, "started", details)
    
    def node_completed(self, node: str, details: Optional[Dict[str, Any]] = None) -> None:
        """
        Log that a node has completed processing.

        Args:
            node: Node name
            details: Optional details (e.g., {"output_count": 10, "duration_ms": 1500})
        """
        self.log_event("node_completed", node, "completed", details)

    def node_failed(self, node: str, error: str, details: Optional[Dict[str, Any]] = None) -> None:
        """
        Log that a node has failed.

        Args:
            node: Node name
            error: Error message
            details: Optional additional details
        """
        failure_details = {"error": error}
        if details:
            failure_details.update(details)
        self.log_event("node_failed", node, "failed", failure_details)
    
    def node_skipped(self, node: str, reason: str, details: Optional[Dict[str, Any]] = None) -> None:
        """
        Log that a node was skipped.

        Args:
            node: Node name
            reason: Why the node was skipped
            details: Optional additional details
        """
        skip_details = {"reason": reason}
        if details:
            skip_details.update(details)
        self.log_event("node_skipped", node, "skipped", skip_details)

    @contextmanager
    def track_node(self, node: str, details: Optional[Dict[str, Any]] = None):
        """
        Context manager to track a node's processing.

        Automatically logs start and completion/failure of the node.

        Args:
            node: Node name
            details: Optional details to log at start

        Yields:
            self: Event logger instance
        
        Example:
            with event_logger.track_node("extract_table", {"files": 5}):
                # ... do work ...
                pass
        """
        start_time = datetime.now(timezone.utc)
        self.node_started(node, details)

        try:
            yield self

            # Calculate duration
            duration_ms = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)
            completion_details = {"duration_ms": duration_ms}
            if details:
                completion_details.update(details)
            
            self.node_completed(node, completion_details)

        except Exception as e:
            # Calculate duration even on failure
            duration_ms = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)

            self.node_failed(node, str(e), {"duration_ms": duration_ms})
            raise # Re-raise the exception

    def update_package_status(self, status: str) -> None:
        """
        Update the package status.

        Args:
            status: New status (uploaded, processing, completed, failed)
        """
        try:
            package = self.db.query(ProjectPackage).filter(
                ProjectPackage.id == self.package_id
            ).first()

            if package:
                package.status = status
                package.updated_at = datetime.now(timezone.utc)
                self.db.commit()
                print(f"[EventLogger] Package {self.package_id} status updated to: {status}")
            else:
                print(f"[EventLogger Error] Package {self.package_id} not found")

        except Exception as e:
            print(f"[EventLogger Error] Failed to update package status: {str(e)}")
            self.db.rollback()

    def get_logs(self) -> List[Dict[str, Any]]:
        """
        Get all logs for the package.

        Returns:
            List of log events in chronological order
        """
        try:
            package = self.db.query(ProjectPackage).filter(
                ProjectPackage.id == self.package_id
            ).first()

            if not package:
                print(f"[EventLogger Error] Package {self.package_id} not found")
                return []

            return package.logs or []

        except Exception as e:
            print(f"[EventLogger Error] Failed to get logs for package {self.package_id}: {str(e)}")
            return []
