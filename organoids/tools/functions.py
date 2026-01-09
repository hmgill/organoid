from typing import Optional, Dict, Any, List
from google.adk.tools.tool_context import ToolContext
from services import processing
from schemas.models import DetectedOrganoid


def remove_background_tool(
    image_path: str,
    tool_context: ToolContext = None
) -> Dict[str, Any]:
    """
    Removes the background from a microscopy image using BRIA RMBG-2.0.

    Args:
        image_path: Local path to the input image.
        tool_context: Automatically injected by ADK.

    Returns:
        dict: Contains 'status' and the 'clean_image_path'.
    """
    print(f"\n[Tool] remove_background_tool called with: {image_path}")
    
    try:
        # Access session state if needed, e.g.: user_id = tool_context.session.user_id
        clean_path = processing.execute_background_removal(image_path)
        
        result = {
            "status": "success",
            "clean_image_path": clean_path,
            "message": f"Background removed successfully. Clean image saved to: {clean_path}"
        }
        print(f"[Tool] remove_background_tool result: {result}")
        return result
        
    except Exception as e:
        error_result = {
            "status": "error", 
            "message": f"Background removal failed: {str(e)}"
        }
        print(f"[Tool] remove_background_tool error: {error_result}")
        return error_result

def annotate_image_tool(
    image_path: str,
    organoids: List[DetectedOrganoid],
    tool_context: ToolContext = None
) -> Dict[str, Any]:
    """
    Draws identified points on the image based on DetectedOrganoid models.

    Args:
        image_path: Path to the image to annotate (usually the clean one).
        organoids: List of DetectedOrganoid Pydantic models containing points to mark.
        tool_context: Automatically injected by ADK.

    Returns:
        dict: Contains 'status' and 'annotated_image_path'.
    """
    print(f"\n[Tool] annotate_image_tool called")
    print(f"  Image: {image_path}")
    print(f"  Number of organoids: {len(organoids)}")
    
    try:
        if not organoids:
            print("[Tool] WARNING: No organoids provided")
            return {
                "status": "error",
                "message": "No organoids found in analysis data"
            }
        
        final_path = processing.execute_annotation(image_path, organoids)
        
        result = {
            "status": "success",
            "annotated_image_path": final_path,
            "message": f"Annotation complete. {len(organoids)} organoids marked. Image saved to: {final_path}"
        }
        print(f"[Tool] annotate_image_tool result: {result}")
        return result
        
    except Exception as e:
        error_result = {
            "status": "error", 
            "message": f"Annotation failed: {str(e)}"
        }
        print(f"[Tool] annotate_image_tool error: {error_result}")
        return error_result
