"""
Reporter Agent - Agent for saving comparison report to database.

FIX APPLIED: Accept and use ProjectLogger for proper log streaming.
"""

from typing import Dict, Any
from datetime import datetime as dt
import datetime
from sqlalchemy.orm import Session

from app.infrastructure.persistence.database.models import Project


class ReporterAgent:
    """
    Agent for  saving the final comparison report to the database.
    """
    
    def __init__(self, logger=None):
        """
        Initialize the ReporterAgent.
        
        Args:
            logger: Optional ProjectLogger instance for real-time log streaming
        """
        self.name = "ReporterAgent"
        self.logger = logger
    
    def _log(self, message: str, level: str = "info"):
        """Helper to log with fallback to print."""
        if self.logger:
            self.logger.log(message, level, "report_generation")
        else:
            print(f"[{self.name}] {message}")
    
    def save_report_to_db(
        self,
        db_session: Session,
        project_id: str,
        comparison_report: Dict[str, Any],
        status: str = "completed"
    ) -> None:
        """
        Save the comparison report to the database.
        
        Args:
            db_session: SQLAlchemy database session
            project_id: Project unique identifier
            comparison_report: Dictionary containing comparison results
            status: Final project status (default: "completed")
        """
        self._log(f"Saving report for project {project_id}")
        
        try:
            # Query project
            project = db_session.query(Project).filter(
                Project.id == project_id
            ).first()
            
            if not project:
                raise Exception(f"Project {project_id} not found in database")
            
            # Format the report with additional metadata
            final_report = {
                "timestamp": dt.now(datetime.timezone.utc).isoformat(),
                "project_id": project_id,
                "status": status,
                **comparison_report
            }
            
            # Update project record
            project.results_json = final_report
            project.status = status
            project.updated_at = dt.now(datetime.timezone.utc)

            # Commit changes
            db_session.commit()
            
            self._log(f"Report saved successfully with status: {status}", "success")
            self._log(f"Completion rate: {comparison_report.get('completion_rate', 0)}%")
            
        except Exception as e:
            db_session.rollback()
            self._log(f"Error saving report: {str(e)}", "error")
            raise
    
    def format_comparison_results(
        self,
        matched_files: int,
        not_found_files: int,
        total_files: int,
        missing_documents: list,
        extracted_table: list = None
    ) -> Dict[str, Any]:
        """
        Format comparison results into a structured report.
        
        Args:
            matched_files: Number of files successfully matched
            not_found_files: Number of files not found in ОПЗ table
            total_files: Total number of uploaded files
            missing_documents: List of documents from ОПЗ not found in uploads
            extracted_table: The composition table extracted from ОПЗ document
            
        Returns:
            Dict containing formatted comparison report
        """
        completion_rate = (matched_files / total_files * 100) if total_files > 0 else 0
        
        self._log(f"Formatting results: {matched_files}/{total_files} matched ({completion_rate:.1f}%)")
        
        report = {
            "total_files": total_files,
            "matched_files": matched_files,
            "not_found_files": not_found_files,
            "completion_rate": round(completion_rate, 2),
            "missing_documents": missing_documents,
            "summary": self._generate_summary(
                matched_files,
                not_found_files,
                total_files,
                missing_documents
            )
        }
        
        # Add extracted composition table if available
        if extracted_table:
            report["extracted_composition_table"] = extracted_table
            report["extracted_documents_count"] = len(extracted_table)
            self._log(f"Included {len(extracted_table)} documents from ОПЗ composition table")
        
        return report
    
    def _generate_summary(
        self,
        matched_files: int,
        not_found_files: int,
        total_files: int,
        missing_documents: list
    ) -> str:
        """Generate a human-readable summary of the comparison results."""
        completion_rate = (matched_files / total_files * 100) if total_files > 0 else 0
        
        summary_parts = [
            f"Проверено файлов: {total_files}",
            f"Соответствует документам из ОПЗ: {matched_files}",
            f"Не найдено в ОПЗ: {not_found_files}",
            f"Процент комплектности: {completion_rate:.1f}%"
        ]
        
        if missing_documents:
            summary_parts.append(
                f"Отсутствующие документы из ОПЗ: {len(missing_documents)}"
            )
        
        return " | ".join(summary_parts)
