import os
import torch
import numpy as np
from PIL import Image, ImageDraw
from torchvision import transforms

# Singleton Model Cache
_RMBG_MODEL = None

def get_rmbg_model():
    global _RMBG_MODEL
    if _RMBG_MODEL is None:
        print("  [System] Loading briaai/RMBG-2.0...")
        
        # Load the model directly using torch.hub or custom loading
        # RMBG-2.0 uses BiRefNet which requires trust_remote_code
        # We need to load it with the proper approach
        
        from transformers import AutoModelForImageSegmentation
        import transformers
        
        # Patch the PreTrainedModel to add missing attribute if needed
        if hasattr(transformers, 'PreTrainedModel'):
            if not hasattr(transformers.PreTrainedModel, 'all_tied_weights_keys'):
                transformers.PreTrainedModel.all_tied_weights_keys = property(lambda self: [])
                print("  [System] Applied compatibility patch")
        
        # Load with trust_remote_code - this loads the custom BiRefNet model
        _RMBG_MODEL = AutoModelForImageSegmentation.from_pretrained(
            "briaai/RMBG-2.0",
            trust_remote_code=True
        )
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  [System] Using device: {device}")
        _RMBG_MODEL.to(device)
        _RMBG_MODEL.eval()
        print("  [System] Model loaded successfully")
            
    return _RMBG_MODEL

def execute_background_removal(image_path: str) -> str:
    """Core logic to remove background."""
    print(f"  [BG Removal] Starting background removal for: {image_path}")
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    try:
        model = get_rmbg_model()
        device = next(model.parameters()).device
        
        # Prepare Image
        print(f"  [BG Removal] Loading and preprocessing image...")
        orig_image = Image.open(image_path).convert("RGB")
        print(f"  [BG Removal] Original image size: {orig_image.size}")
        
        image_size = (1024, 1024)
        transform_image = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        input_tensor = transform_image(orig_image).unsqueeze(0).to(device)
        
        # Inference
        print(f"  [BG Removal] Running inference...")
        with torch.no_grad():
            # BiRefNet returns a list of predictions, we want the last one
            preds = model(input_tensor)
            
            # Handle output format - should be list or tuple
            if isinstance(preds, (list, tuple)):
                preds = preds[-1]
            
            preds = preds.sigmoid().cpu()
        
        # Post-Process
        print(f"  [BG Removal] Post-processing mask...")
        pred = preds[0].squeeze()
        mask = transforms.ToPILImage()(pred).resize(orig_image.size)
        
        clean_image = Image.new("RGBA", orig_image.size, (0, 0, 0, 0))
        clean_image.paste(orig_image, (0, 0), mask)
        
        # Save with proper naming
        base, ext = os.path.splitext(image_path)
        output_path = f"{base}_clean.png"
        clean_image.save(output_path)
        
        print(f"  [BG Removal] ✓ Success! Saved to: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"  [BG Removal] ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"Background removal service failed: {e}")

def execute_annotation(image_path: str, organoids_data: list) -> str:
    """Core logic to draw points."""
    print(f"  [Annotation] Starting annotation for: {image_path}")
    print(f"  [Annotation] Number of organoids to annotate: {len(organoids_data)}")
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    try:
        img = Image.open(image_path).convert("RGBA")
        print(f"  [Annotation] Image size: {img.size}")
        draw = ImageDraw.Draw(img)
        
        total_points = 0
        for i, organoid in enumerate(organoids_data):
            # Handle dictionary access safely
            if isinstance(organoid, dict):
                org_id = organoid.get('organoid_id', f'org_{i}')
                points = organoid.get('points', [])
            else:
                org_id = organoid.organoid_id if hasattr(organoid, 'organoid_id') else f'org_{i}'
                points = organoid.points if hasattr(organoid, 'points') else []
            
            print(f"  [Annotation] Processing {org_id}: {len(points)} points")
            
            for j, point in enumerate(points):
                # Handle Pydantic object or dict
                if isinstance(point, dict):
                    x, y = point['x'], point['y']
                    label = point.get('label', 'unknown')
                else:
                    x, y = point.x, point.y
                    label = point.label if hasattr(point, 'label') else 'unknown'
                
                # Color code by label
                if label == 'budding_region':
                    color = "red"
                elif label == 'center':
                    color = "blue"
                elif label == 'edge':
                    color = "green"
                else:
                    color = "yellow"
                    
                r = 5  # Slightly larger for visibility
                draw.ellipse((x-r, y-r, x+r, y+r), fill=color, outline="white", width=2)
                total_points += 1
        
        print(f"  [Annotation] Total points drawn: {total_points}")
        
        # Create outputs directory if needed
        os.makedirs("outputs", exist_ok=True)
        
        output_filename = f"annotated_{os.path.basename(image_path)}"
        output_path = os.path.join("outputs", output_filename)
        img.save(output_path)
        
        print(f"  [Annotation] ✓ Success! Saved to: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"  [Annotation] ✗ Error: {e}")
        raise RuntimeError(f"Annotation service failed: {e}")
