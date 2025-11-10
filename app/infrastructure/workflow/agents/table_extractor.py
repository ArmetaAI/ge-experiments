from langchain_google_vertexai import ChatVertexAI
from langgraph.graph import StateGraph, END, START
from lmnr import observe
from typing import TypedDict, List, Optional, Any

from app.shared.utils.processing_utils import parse_json_response, image_llm_call

llm = ChatVertexAI(model_name="gemini-2.5-flash", temperature=0)

TABLE_TEXTRACTION_PROMPT = """Extract all tables from this PDF page image.:
                Respond in JSON:
                {
                    "tables": [
                        {
                            "table_id": 1,
                            "rows":[
                                {"column1": "value1", "column2": "value2", ...},
                                ...
                            ],
                            "row_count": 5,
                            "column_count": 3
                        },
                        {
                            "table_id": 2,
                            "rows":[
                                {"column1": "value1", "column2": "value2", ...},
                                ...
                            ],
                            "row_count": 10,
                            "column_count": 4
                        }
                    ]
                }
                    "validation_notes": "any concerns or observations"
                }
            """

class TableExtractionState(TypedDict):
    page_image: Any
    additional_context: str
    extracted_tables: Optional[List[dict]]  # [{'doc_number': str, 'doc_name': str, ...}]
    concerns: str

@observe(tags=['table_extractor'], name="table_extracting")
def table_extracting_node(state: TableExtractionState) -> dict:
    """Detect and extract tables from the current PDF page image using LLM"""
    image = state["page_image"]
    
    
    # Convert image to base64
    response = image_llm_call(image, llm, TABLE_TEXTRACTION_PROMPT, state["additional_context"])

    extracted_tables = parse_json_response(response.content).get("tables", [])
    concerns = parse_json_response(response.content).get("validation_notes", "")

    return {"extracted_tables": extracted_tables, "concerns": concerns}

table_graph_builder = StateGraph(TableExtractionState)
table_graph_builder.add_node("table_extracting", table_extracting_node)
table_graph_builder.add_edge(START, "table_extracting")
table_graph_builder.add_edge("table_extracting", END)

table_extractor = table_graph_builder.compile()