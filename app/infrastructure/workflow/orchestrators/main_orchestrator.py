"""
Main Orchestrator - Entry point for the completeness check workflow.

This module provides the dispatcher that routes workflows based on package type:
- PSD (Проектно-сметная Документация) → psd_workflow
- IRD (Исходно-разрешительная Документация) → ird_workflow
"""

from google.cloud import storage
from lmnr import Laminar, observe
from typing import Literal

from app.infrastructure.persistence.database.models import SessionLocal, Project, ProjectPackage, ProjectFile
from app.infrastructure.persistence.repositories.compliance_results_repository import ComplianceResultsRepository
from app.infrastructure.workflow.states.psd_state import PSDGraphState
from app.infrastructure.workflow.states.ird_state import IRDGraphState
from app.infrastructure.workflow.orchestrators.psd_workflow import psd_workflow_app
from app.infrastructure.workflow.orchestrators.ird_workflow import ird_workflow_app
from app.infrastructure.logging.logger import ProjectLogger
from app.infrastructure.logging.event_logger import PackageEventLogger
from app.shared.config.settings import settings
from app.shared.compliance.ComplianceClass import ComplianceClass

# TODO: Move to a more appropriate place
PROJECT_ID = settings.GCS_PROJECT_ID if settings.GCS_PROJECT_ID else None


gcs_client = storage.Client(project=PROJECT_ID)

@observe(tags=['main_orchestrator'], name="full_workflow", session_id="completeness_check")
async def run_package_workflow(project_id: str, package_type: Literal["PSD", "IRD"]) -> None:
    """
    Dispatcher function to run the appropriate workflow based on package type.
    
    This function:
    1. Creates a database session
    2. Updates package status to 'processing'
    3. Initializes the workflow state
    4. Routes to PSD or IRD workflow based on package_type
    5. Handles any errors
    
    Args:
        project_id: Unique project identifier
        package_type: Type of package to process ("PSD" or "IRD")
        
    Note:
        This function is designed to be called as a background task
        from the FastAPI endpoint. It does not return a value but
        updates the database directly.
    """
    # Create database session
    db_session = SessionLocal()

    # Create project logger for real-time monitoring
    with ProjectLogger(project_id, db_session) as logger:
        logger.info(f"Starting {package_type} workflow for project {project_id}", "start")
        
        try:
            # Get the specific package
            package = db_session.query(ProjectPackage).filter(
                ProjectPackage.project_id == project_id,
                ProjectPackage.package_type == package_type
            ).first()

            if not package:
                logger.info(f"{package_type} package not found for project {project_id} - skipping workflow")
                return  # Gracefully skip this workflow if package doesn't exist
            
            # Create event logger for package progress tracking
            event_logger = PackageEventLogger(db_session, package.id)

            # Update package status to processing
            package.status = "processing"
            db_session.commit()
            
            logger.success(f"{package_type} package status updated to 'processing'", "processing")
            
            # Create initial state based on package type
            logger.info("Initializing workflow state", "init")
            
            if package_type == "PSD":
                # Create PSD state
                initial_state = PSDGraphState(
                    project_id=project_id,
                    db_session=db_session,
                    logger=logger,
                    gcs_client=gcs_client,
                    package_id=package.id,
                    event_logger=event_logger,
                    file_list_from_db=None,
                    psd_files=None,
                    extracted_composition_table=None,
                    final_report_psd=None,
                    errors=[],
                    current_step="initialized"
                )
                
                # Run PSD workflow
                logger.info("Invoking PSD LangGraph workflow", "psd_workflow")
                final_state = psd_workflow_app.invoke(initial_state)
                
            elif package_type == "IRD":
                # Get IRD files from database
                logger.info("Retrieving IRD files from database", "get_ird_files")
                ird_files = db_session.query(ProjectFile).join(ProjectPackage).filter(
                    ProjectPackage.project_id == project_id,
                    ProjectPackage.package_type == "IRD",
                    ProjectFile.package_id == package.id
                ).all()

                if not ird_files:
                    logger.error("No IRD files found for project")
                    package.status = "failed"
                    package.results_json = {"error": "No IRD files found"}
                    db_session.commit()
                    return

                # Extract GCS file paths
                file_paths = [file.gcs_path for file in ird_files]
                logger.info(f"Found {len(file_paths)} IRD files", "get_ird_files")

                # Create ComplianceClass object
                logger.info("Creating ComplianceClass object", "create_compliance")
                compliance_object = ComplianceClass(
                    files=file_paths,
                    bucket_name=settings.GCS_BUCKET_NAME,
                    type_project="IRD"
                )

                # Create IRD state
                initial_state = IRDGraphState(
                    project_id=project_id,
                    db_session=db_session,
                    logger=logger,
                    gcs_client=gcs_client,
                    package_id=package.id,
                    event_logger=event_logger,
                    file_list_from_db=None,
                    ird_files=None,
                    extracted_composition_table=None,
                    final_report_ird=None,
                    errors=[],
                    current_step="initialized"
                )

                # Run IRD workflow with compliance_object in config
                logger.info("Invoking IRD LangGraph workflow", "ird_workflow")
                final_state = await ird_workflow_app.ainvoke(
                    initial_state,
                    config={
                        "configurable": {
                            "compliance_object": compliance_object
                        }
                    }
                )

                # Save compliance results to individual files
                logger.info("Saving compliance results to database", "save_compliance")
                try:
                    # Create repository and save results
                    compliance_repo = ComplianceResultsRepository(db_session)
                    update_stats = compliance_repo.save_all_results(
                        package_id=package.id,
                        state=final_state,
                        result_keys=[
                            "check_format_result",
                            "page_number_result",
                            "empty_lists_result",
                            "insufficient_files_result",
                            "classify_result"
                        ]
                    )
                    logger.success(f"Compliance results saved successfully: {sum(update_stats.values())} total updates", "save_compliance")
                except Exception as save_error:
                    logger.error(f"Failed to save compliance results: {str(save_error)}", "save_compliance")
                
            else:
                raise ValueError(f"Unknown package type: {package_type}")
            
            # Check for errors
            errors = final_state.get("errors", [])
            if errors:
                logger.error(f"{package_type} workflow completed with errors:")
                for error in errors:
                    logger.error(f"  - {error}")
                package.status = "failed"
            else:
                logger.success(f"{package_type} workflow completed successfully", "completed")
                package.status = "completed"
            
            logger.info(f"Final step: {final_state.get('current_step')}")
            
            # Update package with final status
            db_session.commit()
            
        except Exception as e:
            logger.error(f"Fatal error in {package_type} workflow: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Update package status to failed
            try:
                package = db_session.query(ProjectPackage).filter(
                    ProjectPackage.project_id == project_id,
                    ProjectPackage.package_type == package_type
                ).first()
                if package:
                    package.status = "failed"
                    package.results_json = {
                        "error": str(e),
                        "message": f"{package_type} workflow execution failed"
                    }
                    db_session.commit()
                    logger.error(f"{package_type} package status updated to 'failed'")
            except Exception as db_error:
                logger.error(f"Failed to update package status: {str(db_error)}")
                db_session.rollback()
        
        finally:
            # Close database session
            db_session.close()
        
        logger.success(f"{package_type} workflow finished", "finished")


async def run_completeness_check(project_id: str) -> None:
    """
    Legacy function for backward compatibility.

    Runs both PSD and IRD workflows for a project.
    This is called when both package types are uploaded.

    Args:
        project_id: Unique project identifier

    Note:
        This function is kept for backward compatibility.
        New code should use run_package_workflow() instead.
    """
    # Run both workflows
    await run_package_workflow(project_id, "PSD")
    await run_package_workflow(project_id, "IRD")

    # Update project status based on package results
    db_session = SessionLocal()
    try:
        project = db_session.query(Project).filter(Project.id == project_id).first()
        if project:
            # Get all packages for this project
            packages = db_session.query(ProjectPackage).filter(
                ProjectPackage.project_id == project_id
            ).all()

            if packages:
                # Determine overall project status based on package statuses
                package_statuses = [pkg.status for pkg in packages]

                if any(status == "failed" for status in package_statuses):
                    project.status = "failed"
                elif all(status == "completed" for status in package_statuses):
                    project.status = "completed"
                elif any(status == "processing" for status in package_statuses):
                    project.status = "processing"
                else:
                    project.status = "uploaded"

                db_session.commit()
    except Exception as e:
        print(f"Error updating project status: {e}")
        db_session.rollback()
    finally:
        db_session.close()
