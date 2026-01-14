import os
from PIL import Image, ImageDraw
from huggingface_hub import InferenceClient
from config.settings import HF_TOKEN
import numpy as np
import cv2


def create_filled_mask(binary_mask: np.ndarray, original_image: Image.Image, smooth: bool = True) -> Image.Image:
    """
    Create a color image showing the organoid region from the original image
    with everything outside the outer contour set to white.
    
    Takes the SAM3 binary mask, finds its outer contour, optionally smooths it,
    extracts all pixels within that contour from the original image, and sets 
    the background to white.
    
    Args:
        binary_mask: Binary mask array (uint8) where 255 = organoid pixels, 0 = background
        original_image: Original background-removed PIL Image (RGB)
        smooth: Whether to apply contour smoothing (default: True)
    
    Returns:
        PIL Image (RGB) where:
        - Inside outer contour: original pixel values from the image
        - Outside outer contour: white [255, 255, 255] background
    """
    # Ensure binary mask is uint8
    mask_uint8 = binary_mask.astype('uint8')
    
    # Optional: Apply morphological operations to smooth the mask before contouring
    if smooth:
        # Close small holes and smooth edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel, iterations=2)
        # Smooth with Gaussian blur then re-threshold
        mask_uint8 = cv2.GaussianBlur(mask_uint8, (5, 5), 0)
        _, mask_uint8 = cv2.threshold(mask_uint8, 127, 255, cv2.THRESH_BINARY)
    
    # Find external contours only
    contours, _ = cv2.findContours(
        mask_uint8, 
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    if not contours:
        # No contours found, return white image
        return Image.new('RGB', original_image.size, (255, 255, 255))
    
    # Find the largest contour (should be the main organoid)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Smooth the contour using polygon approximation
    if smooth:
        # Approximate the contour to reduce vertices and smooth edges
        epsilon = 0.001 * cv2.arcLength(largest_contour, True)
        largest_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    # Create a mask for the filled contour region
    contour_mask = np.zeros(mask_uint8.shape, dtype=np.uint8)
    cv2.drawContours(
        contour_mask,
        [largest_contour],
        contourIdx=-1,
        color=255,    # White fill
        thickness=-1  # Negative = filled
    )
    
    # Optional: Apply Gaussian blur to the contour mask for smooth edges
    if smooth:
        contour_mask = cv2.GaussianBlur(contour_mask, (3, 3), 0)
        _, contour_mask = cv2.threshold(contour_mask, 127, 255, cv2.THRESH_BINARY)
    
    # Convert original image to numpy array (RGB)
    original_array = np.array(original_image.convert('RGB'))
    
    # Start with white background
    result = np.ones_like(original_array) * 255
    
    # Copy pixels from original image where contour mask is filled
    result[contour_mask > 0] = original_array[contour_mask > 0]
    
    return Image.fromarray(result.astype('uint8'))

    return Image.fromarray(result)


def execute_background_removal(image_path: str) -> str:
    """Remove background using HuggingFace Inference API."""
    print(f"  [BG Removal] Starting background removal for: {image_path}")
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    try:
        # Initialize HF Inference Client
        print(f"  [BG Removal] Initializing HuggingFace client...")
        client = InferenceClient(token=HF_TOKEN)
        
        # Remove background using API
        print(f"  [BG Removal] Calling RMBG-2.0 via API...")
        result = client.image_segmentation(
            image=image_path,
            model="briaai/RMBG-2.0"
        )
        
        bg_removed = result[0]['mask']
        print(f"  [BG Removal] Received mask, mode: {bg_removed.mode}")
        
        # Convert RGBA to RGB with white background
        if bg_removed.mode == 'RGBA':
            print(f"  [BG Removal] Converting RGBA to RGB with white background...")
            rgb_img = Image.new('RGB', bg_removed.size, (255, 255, 255))
            rgb_img.paste(bg_removed, mask=bg_removed.split()[3])
        else:
            rgb_img = bg_removed.convert('RGB')
        
        # Create outputs directory if needed
        os.makedirs("outputs", exist_ok=True)
        
        # Save with proper naming
        output_filename = f"bg_removed_{os.path.basename(image_path)}"
        output_path = os.path.join("outputs", output_filename)
        rgb_img.save(output_path)
        
        print(f"  [BG Removal] ✓ Success! Saved to: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"  [BG Removal] ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"Background removal service failed: {e}")


def execute_sam3_segmentation(image_path: str, point_x: int, point_y: int) -> str:
    """
    Segment organoid using SAM3 Tracker with a positive point prompt (local model).
    
    Args:
        image_path: Path to the background-removed image
        point_x: X coordinate of the center point
        point_y: Y coordinate of the center point
    
    Returns:
        Path to the segmentation mask image
    """
    print(f"  [SAM3] Starting SAM3 segmentation for: {image_path}")
    print(f"  [SAM3] Using positive point: ({point_x}, {point_y})")
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    try:
        import torch
        from transformers import Sam3TrackerProcessor, Sam3TrackerModel
        import numpy as np
        from PIL import ImageDraw
        
        # Check for GPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  [SAM3] Using device: {device}")
        
        # Load model and processor
        print(f"  [SAM3] Loading SAM3 model...")
        model = Sam3TrackerModel.from_pretrained("facebook/sam3").to(device)
        processor = Sam3TrackerProcessor.from_pretrained("facebook/sam3")
        print(f"  [SAM3] Model loaded successfully")
        
        # Load the image
        print(f"  [SAM3] Loading image...")
        image = Image.open(image_path).convert("RGB")
        img_width, img_height = image.size
        print(f"  [SAM3] Image size: {image.size}")
        
        # Prepare point prompt in SAM3 format
        # Format: [[[[x, y]]]] - 4 dimensions (image, object, point_per_object, coordinates)
        input_points = [[[[point_x, point_y]]]]
        
        # Labels: 1 for positive click, 0 for negative click
        # Format: [[[1]]] - 3 dimensions (image, object, point_label)
        input_labels = [[[1]]]
        
        print(f"  [SAM3] Processing with point: {input_points}")
        
        # Process inputs
        inputs = processor(
            images=image,
            input_points=input_points,
            input_labels=input_labels,
            return_tensors="pt"
        ).to(device)
        
        # Generate mask
        print(f"  [SAM3] Running inference...")
        with torch.no_grad():
            outputs = model(**inputs)
        
        print(f"  [SAM3] Raw output shape: {outputs.pred_masks.shape}")
        
        # Post-process masks
        # Sam3Tracker returns masks in shape [batch, num_objects, num_masks, H, W]
        # The masks are already at the correct resolution
        pred_masks = outputs.pred_masks.cpu()
        
        # Get the best mask for the first object
        # Shape: [batch, objects, masks, H, W] -> take [0, 0, 0] = first batch, first object, first mask
        mask_logits = pred_masks[0, 0, 0]  # [H, W]
        
        print(f"  [SAM3] Mask shape after selection: {mask_logits.shape}")
        
        # Apply sigmoid to convert logits to probabilities
        mask_probs = torch.sigmoid(mask_logits)
        
        # Resize to original image size if needed
        if mask_probs.shape[0] != img_height or mask_probs.shape[1] != img_width:
            print(f"  [SAM3] Resizing mask from {mask_probs.shape} to ({img_height}, {img_width})")
            import torch.nn.functional as F
            mask_probs = F.interpolate(
                mask_probs.unsqueeze(0).unsqueeze(0),  # Add batch and channel dims
                size=(img_height, img_width),
                mode='bilinear',
                align_corners=False
            )[0, 0]  # Remove batch and channel dims
        
        # Convert to binary mask (threshold at 0.5)
        binary_mask = (mask_probs > 0.25).numpy().astype('uint8') * 255
        
        print(f"  [SAM3] Final mask shape: {binary_mask.shape}")
        
        # Create outputs directory
        os.makedirs("outputs", exist_ok=True)
        
        # Save binary mask
        output_filename = f"sam3_mask_{os.path.basename(image_path)}"
        output_path = os.path.join("outputs", output_filename)
        Image.fromarray(binary_mask).save(output_path)
        print(f"  [SAM3] Binary mask saved to: {output_path}")
        
        # Create visualization with colored overlay and point
        print(f"  [SAM3] Creating visualization...")
        
        # Generate a random color for the mask overlay
        np.random.seed(hash(image_path) % (2**32))  # Deterministic but unique per image
        overlay_color = tuple(np.random.randint(0, 255, 3).tolist())
        print(f"  [SAM3] Using overlay color: RGB{overlay_color}")
        
        # Create RGBA overlay
        overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        
        # Create colored mask with transparency
        mask_bool = binary_mask > 127
        colored_mask = np.zeros((img_height, img_width, 4), dtype=np.uint8)
        colored_mask[mask_bool] = [*overlay_color, 128]  # 50% transparency
        
        # Convert to PIL Image
        mask_overlay = Image.fromarray(colored_mask, mode='RGBA')
        
        # Composite: base image + colored mask overlay
        vis_image = image.convert('RGBA')
        vis_image = Image.alpha_composite(vis_image, mask_overlay)
        
        # Draw the selected point
        draw = ImageDraw.Draw(vis_image)
        
        # Draw crosshair at the point location
        point_size = 10
        point_color = (255, 255, 0, 255)  # Yellow
        outline_color = (0, 0, 0, 255)  # Black outline
        
        # Draw point marker (circle with crosshair)
        # Outer circle (outline)
        draw.ellipse(
            [point_x - point_size, point_y - point_size,
             point_x + point_size, point_y + point_size],
            fill=None,
            outline=outline_color,
            width=3
        )
        
        # Inner circle
        draw.ellipse(
            [point_x - point_size, point_y - point_size,
             point_x + point_size, point_y + point_size],
            fill=None,
            outline=point_color,
            width=2
        )
        
        # Crosshair lines
        crosshair_length = 20
        # Horizontal line
        draw.line(
            [point_x - crosshair_length, point_y, point_x + crosshair_length, point_y],
            fill=outline_color,
            width=3
        )
        draw.line(
            [point_x - crosshair_length, point_y, point_x + crosshair_length, point_y],
            fill=point_color,
            width=2
        )
        
        # Vertical line
        draw.line(
            [point_x, point_y - crosshair_length, point_x, point_y + crosshair_length],
            fill=outline_color,
            width=3
        )
        draw.line(
            [point_x, point_y - crosshair_length, point_x, point_y + crosshair_length],
            fill=point_color,
            width=2
        )
        
        # Add text label
        label_text = f"Point: ({point_x}, {point_y})"
        # Position label above the point
        label_x = point_x + 15
        label_y = point_y - 25
        
        # Draw text with outline for visibility
        for offset_x in [-1, 0, 1]:
            for offset_y in [-1, 0, 1]:
                if offset_x != 0 or offset_y != 0:
                    draw.text(
                        (label_x + offset_x, label_y + offset_y),
                        label_text,
                        fill=outline_color
                    )
        draw.text((label_x, label_y), label_text, fill=point_color)
        
        # Convert back to RGB for saving
        vis_image = vis_image.convert('RGB')
        
        # Save visualization
        vis_filename = f"sam3_visualization_{os.path.basename(image_path)}"
        vis_output_path = os.path.join("outputs", vis_filename)
        vis_image.save(vis_output_path)
        
        print(f"  [SAM3] ✓ Visualization saved to: {vis_output_path}")
        print(f"  [SAM3] ✓ Binary mask saved to: {output_path}")
        
        # Create filled mask with white background
        print(f"  [SAM3] Creating filled mask with white background...")
        filled_mask = create_filled_mask(binary_mask, image)
        
        filled_filename = f"sam3_filled_{os.path.basename(image_path)}"
        filled_output_path = os.path.join("outputs", filled_filename)
        filled_mask.save(filled_output_path)
        
        print(f"  [SAM3] ✓ Filled mask saved to: {filled_output_path}")
        print(f"  [SAM3] ✓ Success! Generated 3 outputs:")
        print(f"         - Binary mask: {output_path}")
        print(f"         - Visualization: {vis_output_path}")
        print(f"         - Filled mask: {filled_output_path}")
        
        return output_path
        
    except ImportError as e:
        error_msg = (
            f"Required libraries not installed: {e}\n"
            "Install with: pip install torch transformers accelerate"
        )
        print(f"  [SAM3] ✗ Error: {error_msg}")
        raise RuntimeError(error_msg)
        
    except Exception as e:
        print(f"  [SAM3] ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        
        # Provide helpful error message
        error_msg = (
            f"SAM3 segmentation failed: {e}\n\n"
            "Possible issues:\n"
            "1. Model access: Request access at https://huggingface.co/facebook/sam3\n"
            "2. Authentication: Run 'huggingface-cli login' with your token\n"
            "3. GPU memory: Model requires ~3GB VRAM (use CPU if needed)\n"
            "4. Dependencies: Ensure transformers>=4.48.0 and torch>=2.0"
        )
        raise RuntimeError(error_msg)
