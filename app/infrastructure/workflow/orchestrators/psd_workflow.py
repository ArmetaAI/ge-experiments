"""
PSD Workflow Definition for Госэкспертиза completeness check.

This workflow handles PSD (Проектно-сметная Документация) files:
1. Retrieves PSD files from database
2. Extracts composition table from ОПЗ PDF
3. Compares uploaded files against the table
4. Generates completeness report

FIXES APPLIED:
1. Re-query database in generate_and_save_report to get fresh validation statuses
2. Ensure logger is always used (never fall back to print)
3. Pass logger to OPZAgent for proper log streaming
4. Filter by package_type = 'PSD'
"""

from thefuzz import fuzz
from typing import Dict, Any

from langgraph.graph import StateGraph, END
from lmnr import Laminar, observe

from app.infrastructure.workflow.agents.opz_agent import opz_subgraph
from app.infrastructure.workflow.agents.reporter_agent import ReporterAgent
from app.infrastructure.persistence.database.models import ProjectFile, ProjectPackage
from app.infrastructure.workflow.states.psd_state import PSDGraphState

PSD_NODE_TAG = "psd_graph_node"
PSD_SESSION_ID = "psd_workflow"

@observe(session_id=PSD_SESSION_ID, tags=[PSD_NODE_TAG], name="get_psd_files")
def get_psd_files(state: PSDGraphState) -> Dict[str, Any]:
    """
    Node 1: Retrieve PSD files from the database.
    
    Filters files by package_type = 'PSD' to process only PSD documentation.
    """
    logger = state.get("logger")
    event_logger = state.get("event_logger")
    if logger:
        logger.info(f"Retrieving PSD files for project {state['project_id']}", "get_psd_files")
    
    if event_logger:
        event_logger.node_started("get_psd_files")

    try:
        db_session = state["db_session"]
        project_id = state["project_id"]
        
        # Get PSD files only through ProjectPackage relationship
        files = db_session.query(ProjectFile).join(ProjectPackage).filter(
            ProjectPackage.project_id == project_id,
            ProjectPackage.package_type == "PSD"  # Filter by PSD package type
        ).all()
        
        file_list = [
            {
                "id": f.id,
                "filename": f.original_filename,
                "gcs_path": f.gcs_path,
                "validation_status": f.validation_status
            }
            for f in files
        ]
        
        if logger:
            logger.success(f"Found {len(file_list)} PSD files", "get_psd_files")
        
        if event_logger:
            event_logger.node_completed("get_psd_files", {"files_count": len(file_list)})

        return {
            "psd_files": file_list,
            "current_step": "psd_files_retrieved"
        }
        
    except Exception as e:
        if logger:
            logger.error(f"Error retrieving PSD files: {str(e)}", "get_psd_files")
        
        if event_logger:
            event_logger.node_failed("get_psd_files", str(e))

        return {
            "errors": state.get("errors", []) + [f"get_psd_files: {str(e)}"],
            "current_step": "error"
        }

@observe(session_id=PSD_SESSION_ID, tags=[PSD_NODE_TAG], name="psd_extract_table")
def psd_extract_table(state: PSDGraphState) -> Dict[str, Any]:
    """
    Node 2: Extract the composition table from the ОПЗ PDF.
    """
    logger = state.get("logger")
    event_logger = state.get("event_logger")

    if logger:
        logger.info("Extracting composition table from ОПЗ", "psd_extract_table")
    
    if event_logger:
        event_logger.node_started("psd_extract_table")

    try:
        # Create agent and pass logger for proper log streaming
        opz_state = opz_subgraph.invoke({
            "project_id": state["project_id"],
            "db_session": state["db_session"],
            "logger": logger,
            "gcs_client": state["gcs_client"],
            "psd_files": state["psd_files"],
        })
        composition_table = opz_state.get("extracted_composition_table", [])
        opz_file = opz_state.get("opz_file")
        
        if event_logger:
            event_logger.node_completed("psd_extract_table", {"table_rows": len(composition_table)})

        return {
            "extracted_composition_table": composition_table,
            "opz_file": opz_file,
            "current_step": "psd_table_extracted"
        }
        
    except Exception as e:
        if logger:
            logger.error(f"Error extracting table: {str(e)}", "psd_extract_table")
        if event_logger:
            event_logger.node_failed("psd_extract_table", str(e))
        return {
            "errors": state.get("errors", []) + [f"psd_extract_table: {str(e)}"],
            "current_step": "error"
        }

@observe(session_id=PSD_SESSION_ID, tags=[PSD_NODE_TAG], name="psd_compare_files")
def psd_compare_files(state: PSDGraphState) -> Dict[str, Any]:
    """
    Node 3: Compare uploaded PSD files against the composition table.
    """
    logger = state.get("logger")
    event_logger = state.get("event_logger")

    if logger:
        logger.info("Comparing PSD files against ОПЗ table", "psd_compare")

    if event_logger:
        event_logger.node_started("psd_compare_files")
    
    try:
        db_session = state["db_session"]
        file_list = state["psd_files"]
        composition_table = state["extracted_composition_table"]
        documents = composition_table.get("documents", [])
        matched_count = 0
        not_found_count = 0
        opz_file = state.get("opz_file")
        
        for file_info in file_list:
            filename = file_info["filename"]
            file_id = file_info["id"]
            
            # Skip ОПЗ file itself
            if opz_file and file_info.get("gcs_path") == opz_file.get("gcs_path"):
                continue
            
            # Find best match using fuzzy string matching
            best_match = None
            best_score = 0
            
            for doc_entry in documents:
                doc_number = doc_entry.get("doc_number", "")
                doc_name = doc_entry.get("doc_name", "")
                
                score_number = fuzz.partial_ratio(filename, doc_number)
                score_name = fuzz.partial_ratio(filename, doc_name)
                score_combined = max(score_number, score_name)
                
                if score_combined > best_score:
                    best_score = score_combined
                    best_match = doc_entry
            
            # Update database with match result
            project_file = db_session.query(ProjectFile).filter(
                ProjectFile.id == file_id
            ).first()
            
            if project_file:
                MATCH_THRESHOLD = 60
                
                if best_score >= MATCH_THRESHOLD and best_match:
                    project_file.validation_status = "matched"
                    project_file.matched_doc_number = best_match.get("doc_number")
                    project_file.matched_doc_name = best_match.get("doc_name")
                    project_file.match_score = best_score
                    matched_count += 1
                    
                    if logger:
                        logger.info(f"Matched '{filename}' → {best_match.get('doc_number')} ({best_score}%)", "psd_compare")
                else:
                    project_file.validation_status = "not_found"
                    project_file.match_score = best_score
                    not_found_count += 1
                    
                    if logger:
                        logger.warning(f"No match for '{filename}' (best score: {best_score}%)", "psd_compare")
        
        # Commit all updates
        db_session.commit()
        
        if logger:
            logger.success(f"PSD comparison complete - Matched: {matched_count}, Not found: {not_found_count}", "psd_compare")
        
        if event_logger:
            event_logger.node_completed("psd_compare_files", {
                "matched_count": matched_count,
                "not_found_count": not_found_count
            })

        return {
            "current_step": "psd_files_compared"
        }
        
    except Exception as e:
        db_session.rollback()
        if logger:
            logger.error(f"Error comparing PSD files: {str(e)}", "psd_compare")
        if event_logger:
            event_logger.node_failed("psd_compare_files", str(e))
        return {
            "errors": state.get("errors", []) + [f"psd_compare_files: {str(e)}"],
            "current_step": "error"
        }

@observe(session_id=PSD_SESSION_ID, tags=[PSD_NODE_TAG], name="psd_generate_report")
def psd_generate_report(state: PSDGraphState) -> Dict[str, Any]:
    """
    Node 4: Generate final PSD report and save to database.
    
    FIX: Re-query the database to get FRESH validation statuses instead of
    using stale data from psd_files state variable.
    """
    logger = state.get("logger")
    event_logger = state.get("event_logger")

    if logger:
        logger.info("Generating final PSD report", "psd_report")
    
    if event_logger:
        event_logger.node_started("psd_generate_report")
    
    try:
        db_session = state["db_session"]
        project_id = state["project_id"]
        composition_table = state["extracted_composition_table"]
        opz_file = state.get("opz_file")
        
        # FIX: Re-query database to get UPDATED validation statuses for PSD files only
        all_files = db_session.query(ProjectFile).join(ProjectPackage).filter(
            ProjectPackage.project_id == project_id,
            ProjectPackage.package_type == "PSD"  # Only PSD files
        ).all()
        
        if logger:
            logger.info(f"Re-queried database: found {len(all_files)} PSD files", "psd_report")
        
        # Filter out OPZ file and count statuses from FRESH data
        project_files = [f for f in all_files if f.gcs_path != (opz_file.get("gcs_path") if opz_file else None)]
        
        total_files = len(project_files)
        matched_files = len([f for f in project_files if f.validation_status == "matched"])
        not_found_files = len([f for f in project_files if f.validation_status == "not_found"])
        
        if logger:
            logger.info(f"PSD Stats - Total: {total_files}, Matched: {matched_files}, Not found: {not_found_files}", "psd_report")
        
        # Find missing documents (in ОПЗ but not in uploads)
        uploaded_doc_numbers = set()
        for file in project_files:
            if file.matched_doc_number:
                uploaded_doc_numbers.add(file.matched_doc_number)
        
        # Handle composition_table safely
        missing_documents = []
        if composition_table:
            documents = composition_table.get("documents", [])
            missing_documents = [
                {
                    "doc_number": entry.get("doc_number", ""),
                    "doc_name": entry.get("doc_name", "")
                }
                for entry in documents
                if entry.get("doc_number") not in uploaded_doc_numbers
            ]
        
        if logger and missing_documents:
            logger.warning(f"Missing {len(missing_documents)} required PSD documents from ОПЗ", "psd_report")
        
        # Create reporter agent and format report
        reporter = ReporterAgent(logger=logger)
        comparison_report = reporter.format_comparison_results(
            matched_files=matched_files,
            not_found_files=not_found_files,
            total_files=total_files,
            missing_documents=missing_documents,
            extracted_table=composition_table
        )
        
        # Save to database
        status = "completed" if len(state.get("errors", [])) == 0 else "failed"
        reporter.save_report_to_db(
            db_session=db_session,
            project_id=project_id,
            comparison_report=comparison_report,
            status=status
        )
        
        if logger:
            logger.success(f"PSD report saved - Status: {status}, Completion: {comparison_report.get('completion_rate')}%", "psd_report")
        
        if event_logger:
            event_logger.node_completed("psd_generate_report", {
                "completion_rate": comparison_report.get('completion_rate'),
                "total_files": total_files,
                "matched_files": matched_files
            })
        
        return {
            "comparison_report": comparison_report,
            "final_report_psd": comparison_report,
            "current_step": "psd_completed"
        }
        
    except Exception as e:
        if logger:
            logger.error(f"Error generating PSD report: {str(e)}", "psd_report")
        
        if event_logger:
            event_logger.node_failed("psd_generate_report", str(e))
            
        return {
            "errors": state.get("errors", []) + [f"psd_generate_report: {str(e)}"],
            "current_step": "error"
        }

@observe(session_id=PSD_SESSION_ID, tags=["workflow_creation"], name="graph_psd_workflow")
def create_psd_workflow() -> StateGraph:
    """
    Create the LangGraph workflow for PSD completeness checking.
    """
    print("[PSD Workflow] Creating PSD completeness check workflow...")
    
    workflow = StateGraph(PSDGraphState)
    
    workflow.add_node("get_psd_files", get_psd_files)
    workflow.add_node("extract_table", psd_extract_table)
    workflow.add_node("compare", psd_compare_files)
    workflow.add_node("report", psd_generate_report)
    
    workflow.set_entry_point("get_psd_files")
    workflow.add_edge("get_psd_files", "extract_table")
    workflow.add_edge("extract_table", "compare")
    workflow.add_edge("compare", "report")
    workflow.add_edge("report", END)
    
    app = workflow.compile()
    
    print("[PSD Workflow] PSD workflow compiled successfully")
    
    return app


# Export compiled workflow
psd_workflow_app = create_psd_workflow()