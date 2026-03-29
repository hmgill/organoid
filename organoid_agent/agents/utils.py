"""
Helper utilities for the manager agent.
"""
import base64
import os
from pathlib import Path


def load_image_as_base64(image_path: str) -> tuple[str, str]:
    """
    Load an image file and convert it to base64 encoding.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        tuple: (base64_data, mime_type)
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Determine MIME type
    suffix = Path(image_path).suffix.lower()
    mime_type_map = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.webp': 'image/webp'
    }
    mime_type = mime_type_map.get(suffix, 'image/jpeg')
    
    # Read and encode
    with open(image_path, 'rb') as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')
    
    return image_data, mime_type
