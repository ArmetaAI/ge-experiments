"""
Project Repository - Service layer for project and package operations.

This module provides a repository pattern implementation to reduce code duplication
and centralize project-related database operations.
"""
import traceback
import uuid
import logging
from typing import List, Tuple
from datetime import datetime, timezone

from fastapi import UploadFile, HTTPException
from sqlalchemy.orm import Session

from app.infrastructure.persistence.database.models import Project, ProjectFile, ProjectPackage
from app.infrastructure.storage.gcs_storage_service import StorageService


class ProjectRepository:
    """
    Repository for project-related database operations.
    
    Provides centralized methods for:
    - Project creation and validation
    - Package creation and file uploads
    - Common upload logic for different package types
    """
    
    def __init__(self, db: Session):
        """
        Initialize repository with database session.
        
        Args:
            db: SQLAlchemy database session
        """
        self.db = db
        self.storage_service = StorageService()
        self.logger = logging.getLogger("ProjectRepository")

    def create_project(
        self,
        project_name: str | None = None,
        category: str | None = None,
        complexity_level: str | None = None,
        responsibility_class: str | None = None,
        source: str | None = None,
        region: str | None = None,
    ) -> str:
        """Create a new project with optional metadata.

        Args:
            project_name: Optional project name
            category: Optional project category
            complexity_level: Optional complexity level
            responsibility_class: Optional responsibility class
            source: Optional data source
            region: Optional region

        Returns:
            str: The unique project ID
        """
        try:
            project_id = str(uuid.uuid4())
            now = datetime.now(timezone.utc)

            new_project = Project(
                id=project_id,
                status="initialized",
                project_name=project_name,
                category=category,
                complexity_level=complexity_level,
                responsibility_class=responsibility_class,
                source=source,
                region=region,
                created_at=now,
                updated_at=now,
            )

            self.db.add(new_project)
            self.db.commit()
            return project_id
            
        except Exception as e:
            self.db.rollback()
            self.logger.error(f"Error creating project: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail="Failed to create new project"
            )

    def validate_project_exists(self, project_id: str) -> Project:
        """
        Validate that a project exists and return it.
        
        Args:
            project_id: Project identifier to validate
            
        Returns:
            Project: The project record
            
        Raises:
            HTTPException: If project not found
        """
        project = self.db.query(Project).filter(Project.id == project_id).first()
        if not project:
            raise HTTPException(
                status_code=404,
                detail=f"Project with ID {project_id} not found"
            )
        return project
    
    def create_package_and_upload_files(
        self, 
        project_id: str, 
        package_type: str, 
        files: List[UploadFile]
    ) -> Tuple[ProjectPackage, List[str]]:
        """
        Create a package and upload files for a project.
        
        This method centralizes the common upload logic for both PSD and IRD packages:
        1. Creates a ProjectPackage record
        2. Uploads files to Google Cloud Storage
        3. Creates ProjectFile records for each uploaded file
        
        Args:
            project_id: Project identifier
            package_type: Type of package ('PSD' or 'IRD')
            files: List of files to upload
            
        Returns:
            Tuple[ProjectPackage, List[str]]: The created package and list of uploaded filenames
            
        Raises:
            Exception: If upload or database operations fail
        """
        # Create new package for the project
        now = datetime.now(timezone.utc)
        package = ProjectPackage(
            project_id=project_id,
            package_type=package_type,
            status="uploaded",
            created_at=now,
            updated_at=now
        )
        self.db.add(package)
        self.db.flush()  # Get the package ID
        
        # Upload files and create records
        uploaded_files = []
        for file in files:
            try:
                # Sanitize filename to handle encoding issues
                original_filename = file.filename
                if not original_filename:
                    original_filename = "unnamed_file"
                
                self.logger.info(f"Uploading file: {original_filename} for {package_type} package")
                
                # Upload to GCS using StorageService
                gcs_path = self.storage_service.upload_file(
                    project_id=project_id,
                    file=file,
                    package_type=package_type  # Pass package_type for folder structure
                )
                
                self.logger.info(f"File uploaded to GCS: {gcs_path}")
                
                # Create ProjectFile record with proper encoding
                project_file = ProjectFile(
                    package_id=package.id,
                    original_filename=original_filename,
                    gcs_path=gcs_path,
                    validation_status="pending",
                    created_at=now
                )
                self.db.add(project_file)
                uploaded_files.append(original_filename)
                
                self.logger.info(f"File record created in database: {original_filename}")
                
            except Exception as e:
                # Log detailed error information
                error_trace = traceback.format_exc()
                self.logger.error(f"Error uploading {file.filename}: {str(e)}\n{error_trace}")
                
                # For now, let's not skip files - raise the error to see what's happening
                raise Exception(f"Failed to upload {file.filename}: {str(e)}")
        
        return package, uploaded_files
    
    def upload_files_to_existing_project(
        self, 
        project_id: str, 
        package_type: str, 
        files: List[UploadFile]
    ) -> Tuple[str, List[str]]:
        """
        Upload files to an existing project with error handling.
        
        This method combines validation, package creation, and file upload
        with comprehensive error handling and database transaction management.
        
        Args:
            project_id: Existing project identifier
            package_type: Type of package ('PSD' or 'IRD')
            files: List of files to upload
            
        Returns:
            Tuple[str, List[str]]: Project ID and list of uploaded filenames
            
        Raises:
            HTTPException: If validation or upload fails
        """
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")
        
        # Validate project exists
        self.validate_project_exists(project_id)
        
        try:
            # Create package and upload files
            package, uploaded_files = self.create_package_and_upload_files(
                project_id, package_type, files
            )
            
            self.db.commit()
            return project_id, uploaded_files
            
        except Exception as e:
            self.db.rollback()
            # Log the full error for debugging
            error_trace = traceback.format_exc()
            self.logger.error(f"[API] {package_type} upload error: {error_trace}")
            
            # Return a safe error message
            error_msg = str(e)
            try:
                error_msg = error_msg.encode('utf-8', errors='replace').decode('utf-8')
            except UnicodeError:
                error_msg = f"{package_type} upload processing error"
            
            raise HTTPException(
                status_code=500,
                detail=f"Failed to process {package_type} upload: {error_msg}"
            )