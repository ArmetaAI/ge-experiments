from PIL import Image
import io

from app.infrastructure.workflow.orchestrators.ird_workflow import ird_workflow_app
from app.infrastructure.workflow.orchestrators.psd_workflow import psd_workflow_app
Image.open(io.BytesIO(ird_workflow_app.get_graph().draw_mermaid_png())).save("ird_workflow.png")
Image.open(io.BytesIO(psd_workflow_app.get_graph().draw_mermaid_png())).save("psd_workflow.png")