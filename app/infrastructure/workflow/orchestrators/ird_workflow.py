"""
IRD Workflow Definition for Госэкспертиза completeness check.

This workflow handles IRD (Исходно-разрешительная Документация) files:
1. Retrieves IRD files from database
2. Validates document structure
3. Checks compliance with regulations
4. Generates IRD completeness report
"""
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
from lmnr import observe

from app.infrastructure.workflow.states.ird_state import IRDComplianceState

IRD_SESSION_ID = "ird_workflow"
IRD_NODE_TAG = "ird_graph_node"

@observe(session_id=IRD_SESSION_ID, tags=[IRD_NODE_TAG], name="check_format")
async def check_format(state: IRDComplianceState, config: RunnableConfig) -> dict:
    """
    Check file format compliance for IRD documents.

    Args:
        state: Current workflow state
        config: Runnable configuration containing compliance_object

    Returns:
        dict: Format check results
    """
    compliance_object = config.get("configurable", {}).get("compliance_object")
    check_format_result = await compliance_object.check_format()
    return {"check_format_result": check_format_result}

@observe(session_id=IRD_SESSION_ID, tags=[IRD_NODE_TAG], name="page_number")
async def page_number(state: IRDComplianceState, config: RunnableConfig) -> dict:
    """
    Validate page numbering in IRD documents.

    Args:
        state: Current workflow state
        config: Runnable configuration containing compliance_object

    Returns:
        dict: Page number validation results
    """
    compliance_object = config.get("configurable", {}).get("compliance_object")
    page_number_result = await compliance_object.page_number()
    return {"page_number_result": page_number_result}

@observe(session_id=IRD_SESSION_ID, tags=[IRD_NODE_TAG], name="empty_lists")
async def empty_lists(state: IRDComplianceState, config: RunnableConfig) -> dict:
    """
    Check for empty lists in IRD documents.

    Args:
        state: Current workflow state
        config: Runnable configuration containing compliance_object

    Returns:
        dict: Empty list check results
    """
    compliance_object = config.get("configurable", {}).get("compliance_object")
    empty_lists_result = await compliance_object.empty_lists()
    return {"empty_lists_result": empty_lists_result}

@observe(session_id=IRD_SESSION_ID, tags=[IRD_NODE_TAG], name="insufficient_files")
async def insufficient_files(state: IRDComplianceState, config: RunnableConfig) -> dict:
    """
    Check for missing or insufficient IRD files.

    Args:
        state: Current workflow state
        config: Runnable configuration containing compliance_object

    Returns:
        dict: Insufficient files check results
    """
    compliance_object = config.get("configurable", {}).get("compliance_object")
    insufficient_files_result = await compliance_object.insufficient_files()
    return {"insufficient_files_result": insufficient_files_result}

@observe(session_id=IRD_SESSION_ID, tags=[IRD_NODE_TAG], name="classify_documents")
async def classify_documents(state: IRDComplianceState, config: RunnableConfig) -> dict:
    """
    Classify IRD documents by type and extract titles.

    Args:
        state: Current workflow state
        config: Runnable configuration containing compliance_object

    Returns:
        dict: Document classification results
    """
    compliance_object = config.get("configurable", {}).get("compliance_object")
    classify_result = await compliance_object.classify_documents()
    return {"classify_result": classify_result}

@observe(session_id=IRD_SESSION_ID, tags=["workflow_creation"], name="graph_ird_full")
def create_ird_workflow() -> StateGraph:
    """
    Create the LangGraph workflow for IRD completeness checking.
    Args:

    Returns:
        StateGraph, compiled langgraph
    """
    print("[IRD Workflow] Creating IRD completeness check workflow...")

    workflow = StateGraph(IRDComplianceState)

    workflow.add_node("check_format", check_format)
    workflow.add_node("page_number", page_number)
    workflow.add_node("empty_lists", empty_lists)
    workflow.add_node("insufficient_files", insufficient_files)
    workflow.add_node("classify_documents", classify_documents)

    workflow.set_entry_point("check_format")
    workflow.add_edge("check_format", "page_number")
    workflow.add_edge("page_number", "empty_lists")
    workflow.add_edge("empty_lists", "insufficient_files")
    workflow.add_edge("insufficient_files", "classify_documents")
    workflow.add_edge("classify_documents", END)
    
    app = workflow.compile()
    
    print("[IRD Workflow] IRD workflow compiled successfully")
    
    return app


ird_workflow_app = create_ird_workflow()
