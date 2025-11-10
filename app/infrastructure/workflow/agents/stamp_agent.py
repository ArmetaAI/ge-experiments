"""
PDF Stamp Validation Agent using LangGraph
Handles large PDFs by processing pages incrementally
"""

from typing import TypedDict, List, Optional, Annotated
import operator
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from langchain_google_vertexai import ChatVertexAI
from utils.processing_utils import parse_json_response, image_llm_call
import base64
from io import BytesIO

STAMP_DETECTION_PROMPT = """Analyze this PDF page image. 
                    Does it contain an official stamp or seal?
                    
                    Respond in JSON format:
                    {
                        "has_stamp": true/false,
                        "stamp_location": "description of where stamp is located",
                        "confidence": 0.0-1.0
                    }"""

STAMP_VALIDATION_PROMPT = """Analyze the stamp/seal in detail:
                    
                    1. Is it digitally signed? (look for signature indicators)
                    2. Extract any text/numbers from the stamp
                    3. Check if it appears authentic (not photocopied/forged)
                    4. Verify any dates are legible
                    5. Check for any integrity indicators
                    
                    Respond in JSON:
                    {
                        "is_signed": true/false,
                        "extracted_text": "text from stamp",
                        "appears_authentic": true/false,
                        "date_visible": "date if present",
                        "validation_notes": "any concerns or observations"
                    }"""

# State definition
class StampValidationState(TypedDict):
    page_image: Optional[bytes]
    page_results: Annotated[List[dict], operator.add]
    final_report: Optional[dict]

#TODO: Test the validator
#FIXME: Memory issues with large PDFs - consider using temporary files or cloud storage for page images
class PDFStampValidator:
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        self.llm = ChatVertexAI(model_name=model_name)
        self.graph = self._build_graph()
        
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(StampValidationState)
        
        # Add nodes
        workflow.add_node("detect_stamp", self.detect_stamp_node)
        workflow.add_node("validate_stamp", self.validate_stamp_node)
        workflow.add_node("aggregate_results", self.aggregate_results_node)
        
        # Define edges
        workflow.set_entry_point("detect_stamp")
        workflow.add_edge("detect_stamp", "validate_stamp")
        workflow.add_conditional_edges(
            "detect_stamp",
            self.should_validate_stamp,
            {
                "validate": "validate_stamp",
                "skip": "aggregate_results"
            }
        )
        workflow.add_edge("validate_stamp", "aggregate_results")
        workflow.add_conditional_edges(
            "aggregate_results",
            self.should_continue,
            {
                "continue": "extract_page",
                "end": END
            }
        )
        
        return workflow.compile()
    
    def detect_stamp_node(self, state: StampValidationState) -> dict:
        """Use vision model to detect if stamp exists on page"""
        image = state["page_image"]

        image_info = image_llm_call(image, self.llm, STAMP_DETECTION_PROMPT)
        
        # Parse response (add proper JSON parsing)
        detection_result = parse_json_response(image_info.content)
        
        return {
            "page_results": [{
                "page": state["current_page"],
                "detection": detection_result
            }]
        }
    
    def validate_stamp_node(self, state: StampValidationState) -> dict:
        """Validate stamp details if detected"""
        image = state["page_image"]
        current_result = state["page_results"][-1]
        
        if not current_result["detection"]["has_stamp"]:
            return {}
        
        validation_info = image_llm_call(image, self.llm, STAMP_VALIDATION_PROMPT)
        validation_result = parse_json_response(validation_info.content)

        # Update the last page result with validation
        current_result["validation"] = validation_result
        
        return {}
    
    def should_validate_stamp(self, state: StampValidationState) -> str:
        """Decide if stamp validation is needed"""
        current_result = state["page_results"][-1]
        if current_result["detection"]["has_stamp"]:
            return "validate"
        return "skip"
    
    def _generate_report(self, page_results: List[dict]) -> dict:
        """Generate final validation report"""
        total_pages = len(page_results)
        pages_with_stamps = sum(1 for r in page_results if r["detection"]["has_stamp"])
        
        validated_stamps = [
            r for r in page_results 
            if "validation" in r
        ]
        
        return {
            "total_pages_processed": total_pages,
            "pages_with_stamps": pages_with_stamps,
            "validated_stamps": len(validated_stamps),
            "detailed_results": page_results,
            "summary": {
                "all_stamps_signed": all(
                    v.get("validation", {}).get("is_signed", False) 
                    for v in validated_stamps
                ),
                "any_authenticity_concerns": any(
                    not v.get("validation", {}).get("appears_authentic", True)
                    for v in validated_stamps
                )
            }
        }