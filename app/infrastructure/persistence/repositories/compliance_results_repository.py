"""
Compliance Results Repository - Data access layer for compliance check results.

This repository handles persistence of compliance check results to ProjectFile records.
It provides clean separation between workflow orchestration and database operations.
"""

import os
from typing import Dict, Any, List
from sqlalchemy.orm import Session

from app.infrastructure.persistence.database.models import ProjectFile


class ComplianceResultsRepository:
    """
    Repository for managing compliance check results in the database.

    This repository encapsulates all database operations related to saving
    compliance check results to ProjectFile records, ensuring proper separation
    of concerns between workflow orchestration and data persistence.
    """

    def __init__(self, db_session: Session):
        """
        Initialize repository with database session.

        Args:
            db_session: SQLAlchemy database session
        """
        self.db_session = db_session

    def save_result_to_files(
        self,
        package_id: int,
        result_key: str,
        result_data: Dict[str, Any]
    ) -> int:
        """
        Save a single compliance result type to ProjectFile records.

        This method updates the compliance_results JSON field for each file
        by extracting the filename from GCS paths and matching against database records.

        Args:
            package_id: Package ID to filter files
            result_key: Key name for the result (e.g., "check_format_result", "page_number_result")
            result_data: Dictionary with GCS file paths as keys and results as values
                        Example: {"projects/.../file.pdf": "pdf"}

        Returns:
            int: Number of files updated

        Example:
            >>> repo = ComplianceResultsRepository(db_session)
            >>> updated = repo.save_result_to_files(
            ...     package_id=123,
            ...     result_key="check_format_result",
            ...     result_data={"projects/abc/Лицензия.pdf": "pdf"}
            ... )
            >>> print(f"Updated {updated} files")
        """
        try:
            # Get all files in this package
            files = self.db_session.query(ProjectFile).filter(
                ProjectFile.package_id == package_id
            ).all()

            # Create mapping: filename -> file record
            file_map = {f.original_filename: f for f in files}

            # Update each file's compliance_results
            updated_count = 0
            for gcs_path, result_value in result_data.items():
                # Extract filename from GCS path
                # "projects/.../Лицензия TAIMAS-S.pdf" → "Лицензия TAIMAS-S.pdf"
                filename = os.path.basename(gcs_path)

                if filename in file_map:
                    file_record = file_map[filename]

                    # Initialize compliance_results if None
                    if file_record.compliance_results is None:
                        file_record.compliance_results = {}

                    # Update with new result
                    file_record.compliance_results[result_key] = result_value
                    updated_count += 1

            self.db_session.commit()
            print(f"[ComplianceResultsRepository] Saved {result_key} for {updated_count}/{len(result_data)} files")
            return updated_count

        except Exception as e:
            print(f"[ComplianceResultsRepository] Error saving {result_key}: {str(e)}")
            self.db_session.rollback()
            raise

    def save_all_results(
        self,
        package_id: int,
        state: Dict[str, Any],
        result_keys: List[str] = None
    ) -> Dict[str, int]:
        """
        Save all compliance check results to database at once.

        This is a convenience method that saves multiple compliance result types
        in a single transaction. Recommended for use at the end of a workflow.

        Args:
            package_id: Package ID to filter files
            state: Workflow state containing all results
            result_keys: List of result keys to save. If None, saves all standard IRD/PSD results.

        Returns:
            Dict[str, int]: Dictionary mapping result_key to number of files updated

        Example:
            >>> repo = ComplianceResultsRepository(db_session)
            >>> stats = repo.save_all_results(
            ...     package_id=123,
            ...     state=final_state,
            ...     result_keys=["check_format_result", "page_number_result"]
            ... )
            >>> print(f"Total updates: {sum(stats.values())}")
        """
        # Default result keys for IRD workflow
        if result_keys is None:
            result_keys = [
                "check_format_result",
                "page_number_result",
                "empty_lists_result",
                "insufficient_files_result",
                "classify_result"
            ]

        try:
            # Get all files in this package
            files = self.db_session.query(ProjectFile).filter(
                ProjectFile.package_id == package_id
            ).all()

            # Create mapping: filename -> file record
            file_map = {f.original_filename: f for f in files}
            print(f"[ComplianceResultsRepository] Processing {len(files)} files in package {package_id}")
            print(f"[ComplianceResultsRepository] File map keys: {list(file_map.keys())}")

            update_stats = {}
            total_updates = 0

            # Process each result type
            for result_key in result_keys:
                result_data = state.get(result_key, {})
                print(f"[ComplianceResultsRepository] Processing {result_key}: {len(result_data) if result_data else 0} entries")

                if not result_data:
                    print(f"[ComplianceResultsRepository] No data for {result_key}, skipping")
                    update_stats[result_key] = 0
                    continue

                # Update each file
                files_updated = 0
                for gcs_path, result_value in result_data.items():
                    # Extract filename from GCS path
                    filename = os.path.basename(gcs_path)
                    print(f"[ComplianceResultsRepository] Processing {result_key} for {filename} (from {gcs_path})")

                    if filename in file_map:
                        file_record = file_map[filename]

                        # Initialize compliance_results if None
                        if file_record.compliance_results is None:
                            file_record.compliance_results = {}

                        # Update with new result
                        file_record.compliance_results[result_key] = result_value
                        files_updated += 1
                        total_updates += 1
                        print(f"[ComplianceResultsRepository] Updated {filename} with {result_key}")
                    else:
                        print(f"[ComplianceResultsRepository] WARNING: {filename} not found in file_map")

                update_stats[result_key] = files_updated

            self.db_session.commit()
            print(f"[ComplianceResultsRepository] Saved all compliance results: {total_updates} total updates for {len(file_map)} files")
            return update_stats

        except Exception as e:
            print(f"[ComplianceResultsRepository] Error saving compliance results: {str(e)}")
            import traceback
            traceback.print_exc()
            self.db_session.rollback()
            raise
