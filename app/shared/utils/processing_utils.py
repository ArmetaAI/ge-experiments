import base64
import json
import re

from io import BytesIO
from langchain_core.messages import HumanMessage
from lmnr import observe
from typing import Optional, Any


def parse_json_response(content: str) -> dict:
        """Parse JSON from LLM response"""
        
        # Extract JSON from markdown code blocks if present
        json_match = re.search(r'```json\n(.*?)\n```', content, re.DOTALL)
        if json_match:
            content = json_match.group(1)
        
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Fallback parsing
            return {"error": "Failed to parse response", "raw": content}
        
@observe(tags=['image_call', 'processing_utils'])
def image_llm_call(image:Optional[bytes], llm, prompt:str, additional_context:Optional[str]=None) -> Any:
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    # Vision prompt for stamp detection
    message = HumanMessage(
        content=[
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{img_str}"
                },
            },
            {
                "type": "text",
                "text": f"{prompt}"
            },
            {
                "type": "text",
                "text": additional_context
            }
        ]
    )
    
    response = llm.invoke([message])
    return response