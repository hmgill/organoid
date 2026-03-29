from typing import Dict, Any
from google.adk.tools.tool_context import ToolContext
from services import processing
from PIL import Image
import base64
import os
from pathlib import Path
from io import BytesIO
import asyncio


def remove_background_tool(
    image_path: str,
    tool_context: ToolContext = None
) -> Dict[str, Any]:
    """
    Removes the background from a microscopy image using BRIA RMBG-2.0 via HF Inference API.

    Args:
        image_path: Local path to the input image.
        tool_context: Automatically injected by ADK.

    Returns:
        dict: Contains 'status' and the 'clean_image_path'.
    """
    print(f"\n[Tool] remove_background_tool called with: {image_path}")
    
    try:
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


def segment_organoid_sam3_tool(
    image_path: str,
    center_x: int,
    center_y: int,
    tool_context: ToolContext = None
) -> Dict[str, Any]:
    """
    Segments an organoid from a background-removed image using SAM3 with a positive point prompt.
    
    This tool uses the Segment Anything Model 3 (SAM3) to create a precise segmentation mask
    of the organoid based on a center point coordinate. The point acts as a positive prompt,
    telling SAM3 "this is inside the object I want to segment."

    Args:
        image_path: Path to the background-removed image to segment.
        center_x: X coordinate of the organoid center (positive point prompt).
        center_y: Y coordinate of the organoid center (positive point prompt).
        tool_context: Automatically injected by ADK.

    Returns:
        dict: Contains 'status', 'mask_path', and relevant information about the segmentation.
    """
    print(f"\n[Tool] segment_organoid_sam3_tool called")
    print(f"  Image: {image_path}")
    print(f"  Center point: ({center_x}, {center_y})")
    
    try:
        mask_path = processing.execute_sam3_segmentation(
            image_path=image_path,
            point_x=center_x,
            point_y=center_y
        )
        
        result = {
            "status": "success",
            "mask_path": mask_path,
            "center_point": {"x": center_x, "y": center_y},
            "message": f"SAM3 segmentation complete. Mask saved to: {mask_path}"
        }
        print(f"[Tool] segment_organoid_sam3_tool result: {result}")
        return result
        
    except Exception as e:
        error_result = {
            "status": "error",
            "message": f"SAM3 segmentation failed: {str(e)}",
            "note": "Check model access and dependencies"
        }
        print(f"[Tool] segment_organoid_sam3_tool error: {error_result}")
        return error_result


def analyze_with_vision_tool(
    image_path: str,
    tool_context: ToolContext = None
) -> Dict[str, Any]:
    """
    Loads an image and calls the vision analyst agent with multimodal content (text + image).
    
    This is the KEY tool that enables the analyst to actually SEE the image.
    
    Workflow:
    1. Loads the background-removed image
    2. Resizes it for efficient context usage (max 1024px on longest side)
    3. Creates a text prompt with ORIGINAL and display dimensions
    4. Creates a multimodal message with both text AND image
    5. Calls the vision analyst agent directly (bypassing AgentTool limitation)
    6. The analyst visually examines the image, estimates centers, calls SAM3
    7. Returns the analyst's complete response
    
    Args:
        image_path: Path to the background-removed image to analyze
        tool_context: Automatically injected by ADK
    
    Returns:
        dict: Contains status, analyst's response, and analysis results
    """
    print(f"\n[Tool] analyze_with_vision_tool called with: {image_path}")
    
    try:
        # Import here to avoid circular dependency
        from agents.analyst import analyst_agent
        from google.adk.runners import Runner
        from google.adk.sessions import InMemorySessionService
        from google.genai import types
        
        if not os.path.exists(image_path):
            return {
                "status": "error",
                "message": f"Image file not found: {image_path}"
            }
        
        # Load image and get dimensions
        img = Image.open(image_path).convert("RGB")
        original_width, original_height = img.size
        
        print(f"  [Vision Tool] Original image: {original_width}x{original_height}")
        
        # Resize for efficiency (max 1024px on longest side)
        MAX_SIZE = 1024
        if max(original_width, original_height) > MAX_SIZE:
            ratio = MAX_SIZE / max(original_width, original_height)
            new_width = int(original_width * ratio)
            new_height = int(original_height * ratio)
            img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            print(f"  [Vision Tool] Resized to: {new_width}x{new_height} for efficiency")
        else:
            img_resized = img
            new_width, new_height = original_width, original_height
            print(f"  [Vision Tool] No resize needed")
        
        # Convert resized image to base64
        buffer = BytesIO()
        img_resized.save(buffer, format='JPEG', quality=85)
        image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # Determine MIME type
        suffix = Path(image_path).suffix.lower()
        mime_type_map = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.webp': 'image/webp'
        }
        mime_type = mime_type_map.get(suffix, 'image/jpeg')
        
        # Create the text prompt with dimension info
        text_prompt = f"""Please examine this organoid microscopy image and identify center coordinates.

ORIGINAL image: {original_width} x {original_height} pixels
Preview: {new_width} x {new_height} pixels
Path for SAM3: {image_path}

The image shows organoids (dark spherical structures) on a white background.

Your task:
1. Visually identify each distinct organoid (ignore budding regions, small fragments)
2. For each organoid, estimate the center of its MAIN BODY
3. IMPORTANT: Coordinates must be for the ORIGINAL {original_width}x{original_height} image
   - Observe relative position in the preview (centered? upper-left? etc.)
   - Convert to original coordinates using the formulas in your instructions
   - Example: If centered in preview → ({original_width//2}, {original_height//2})
4. Call segment_organoid_sam3_tool for each organoid with:
   - image_path: "{image_path}"
   - center_x: [your estimated x for ORIGINAL size]
   - center_y: [your estimated y for ORIGINAL size]

Coordinate system: (0,0) is top-left, x: 0-{original_width}, y: 0-{original_height}

After segmenting all organoids, report your findings with coordinates and output paths."""
        
        print(f"  [Vision Tool] Creating multimodal message (text + image)...")
        
        # Create multimodal message - THIS IS THE KEY
        # The analyst will receive BOTH text instructions AND the actual image
        multimodal_message = types.Content(
            role="user",
            parts=[
                types.Part(text=text_prompt),
                types.Part(
                    inline_data=types.Blob(
                        mime_type=mime_type,
                        data=image_data
                    )
                )
            ]
        )
        
        print(f"  [Vision Tool] Calling vision analyst with image...")
        
        # Define the async function to run the analysis
        async def run_analysis():
            # Create a temporary session for this analysis
            session_service = InMemorySessionService()
            
            session = await session_service.create_session(
                app_name="organoid_vision_analysis",
                user_id="vision-user",
                session_id=f"vision-{abs(hash(image_path)) % 100000}"
            )
            
            runner = Runner(
                app_name="organoid_vision_analysis",
                agent=analyst_agent,
                session_service=session_service
            )
            
            response_parts = []
            print(f"\n  [Vision Tool] --- Vision Analyst Response ---")
            async for event in runner.run_async(
                user_id=session.user_id,
                session_id=session.id,
                new_message=multimodal_message
            ):
                if hasattr(event, 'content') and event.content:
                    for part in event.content.parts:
                        if hasattr(part, 'text') and part.text:
                            response_parts.append(part.text)
                            # Print in real-time so we can see progress
                            print(part.text, end="", flush=True)
            
            print(f"\n  [Vision Tool] --- End Vision Analyst Response ---\n")
            return "".join(response_parts)
        
        # Run the async analysis
        # Check if we're already in an async context
        try:
            loop = asyncio.get_running_loop()
            print(f"  [Vision Tool] Already in async context, using asyncio.ensure_future()")
            # We're inside an async context, need to run in a new event loop in a thread
            import concurrent.futures
            import threading
            
            def run_in_new_loop():
                """Run the coroutine in a new event loop in a separate thread"""
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    return new_loop.run_until_complete(run_analysis())
                finally:
                    new_loop.close()
            
            # Run in a thread to avoid nested event loop issues
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_in_new_loop)
                response_text = future.result()
                
        except RuntimeError:
            # No event loop running, safe to use asyncio.run()
            print(f"  [Vision Tool] No event loop, using asyncio.run()")
            response_text = asyncio.run(run_analysis())
        
        print(f"  [Vision Tool] Vision analyst completed analysis ({len(response_text)} chars)")
        
        result = {
            "status": "success",
            "image_path": image_path,
            "original_dimensions": {"width": original_width, "height": original_height},
            "analyst_response": response_text,
            "message": "Visual analysis complete. See analyst_response for detailed findings."
        }
        
        print(f"[Tool] analyze_with_vision_tool SUCCESS\n")
        return result
        
    except Exception as e:
        import traceback
        error_result = {
            "status": "error",
            "message": f"Vision analysis failed: {str(e)}",
            "traceback": traceback.format_exc()
        }
        print(f"[Tool] analyze_with_vision_tool ERROR:")
        print(error_result["traceback"])
        return error_result
