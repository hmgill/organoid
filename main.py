import os
import sys
import asyncio
import traceback
import base64
import argparse
from pathlib import Path

# Ensure the project root is in the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from agents.manager import manager_agent


def image_to_base64(image_path: str) -> str:
    """Convert image file to base64 string."""
    with open(image_path, 'rb') as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')


async def run_workflow(input_path: str):
    
    # Check if image exists
    if not os.path.exists(input_path):
        print(f"[!] ERROR: Image file '{input_path}' not found!")
        return
    
    # Determine image MIME type
    suffix = Path(input_path).suffix.lower()
    mime_type_map = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.webp': 'image/webp'
    }
    mime_type = mime_type_map.get(suffix, 'image/jpeg')
    
    # Convert image to base64
    print(f"[System] Loading image: {input_path}")
    image_data = image_to_base64(input_path)
    print(f"[System] Image loaded ({len(image_data)} bytes)")
    
    user_prompt = (
        f"Please analyze this microscopy image of organoids. "
        f"Remove the background and then identify the center coordinates of any organoids you can see. "
        f"The image file is: {input_path}"
    )

    print(f"[System] Creating session...")
    session_service = InMemorySessionService()
    session = await session_service.create_session(
        app_name="organoids",
        user_id="user-1",
        session_id="session-1"
    )
    
    print(f"[System] Session created: {session.id}")
    
    runner = Runner(
        app_name = "organoids",
        agent = manager_agent,
        session_service = session_service
    )

    # Execute the Workflow
    try:
        print("--- System Response ---")
        
        # Create message with BOTH text and image
        user_message = types.Content(
            role="user",
            parts=[
                types.Part(text=user_prompt),
                types.Part(
                    inline_data=types.Blob(
                        mime_type=mime_type,
                        data=image_data
                    )
                )
            ]
        )
        
        response_text = []
        async for event in runner.run_async(
            user_id=session.user_id,
            session_id=session.id,
            new_message=user_message
        ):
            # Extract text from the event properly
            if hasattr(event, 'content') and event.content:
                for part in event.content.parts:
                    if hasattr(part, 'text') and part.text:
                        print(part.text, end="", flush=True)
                        response_text.append(part.text)
                        
        print("\n\n[Done]")
        
        if not response_text:
            print("[!] WARNING: No response text received from the agent")
        else:
            print(f"\n[Success] Received {len(response_text)} response parts")
        
        # Check for output files
        if os.path.exists("outputs"):
            output_files = os.listdir("outputs")
            if output_files:
                print(f"[Success] Output files created: {output_files}")
            else:
                print("[!] WARNING: No files in outputs directory")
        else:
            print("[!] WARNING: outputs directory not created")
        
    except Exception as e:
        print(f"\n[!] Runtime Error: {e}")
        print("\n[!] Full traceback:")
        traceback.print_exc()

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Remove background from organoid microscopy images")
    parser.add_argument(
        "image_path",
        nargs="?",
        default="demo_image3.jpg",
        help="Path to the input image (default: demo.jpg)"
    )
    
    args = parser.parse_args()
    # /N/slate/hungill/organoids/data/budding/train/budding/f6 10x.jpg
    # Run the async main function with the provided image path
    asyncio.run(run_workflow(args.image_path))
