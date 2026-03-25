import os
from pathlib import Path

from PIL import Image, ImageDraw
from huggingface_hub import InferenceClient
from config.settings import HF_TOKEN
import numpy as np
import cv2


def get_image_output_dir(image_path: str | Path, base_output_dir: str | Path = "outputs") -> Path:
    """
    Return (and create) a per-image subdirectory under base_output_dir.

    The subdirectory is named after the image stem, e.g.:
        outputs/org64_B2A-2_d19_LabA/

    Args:
        image_path:       Path to the source image.
        base_output_dir:  Root output directory (default: "outputs").

    Returns:
        Path to the per-image output directory (guaranteed to exist).
    """
    stem = Path(image_path).stem
    output_dir = Path(base_output_dir) / stem
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def create_filled_mask(binary_mask: np.ndarray, original_image: Image.Image, smooth: bool = True) -> Image.Image:
    """
    Create a color image showing the organoid region from the background-removed image
    with everything outside the outer contour set to white.

    Takes the SAM3 binary mask, finds its outer contour, fills it completely,
    and extracts all pixels within that filled contour from the background-removed image.

    Args:
        binary_mask:    Binary mask array (uint8) where 255 = organoid pixels, 0 = background.
        original_image: Background-removed PIL Image (RGB).
        smooth:         Whether to apply contour smoothing (default: True).

    Returns:
        PIL Image (RGB) where:
        - Inside filled outer contour: original pixel values from bg-removed image
        - Outside outer contour: white [255, 255, 255] background
    """
    mask_uint8 = binary_mask.astype('uint8')

    if smooth:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask_uint8 = cv2.GaussianBlur(mask_uint8, (5, 5), 0)
        _, mask_uint8 = cv2.threshold(mask_uint8, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        mask_uint8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return Image.new('RGB', original_image.size, (255, 255, 255))

    largest_contour = max(contours, key=cv2.contourArea)

    if smooth:
        epsilon = 0.001 * cv2.arcLength(largest_contour, True)
        largest_contour = cv2.approxPolyDP(largest_contour, epsilon, True)

    contour_mask = np.zeros(mask_uint8.shape, dtype=np.uint8)
    cv2.drawContours(
        contour_mask,
        [largest_contour],
        contourIdx=-1,
        color=255,
        thickness=-1,
    )

    if smooth:
        contour_mask = cv2.GaussianBlur(contour_mask, (3, 3), 0)
        _, contour_mask = cv2.threshold(contour_mask, 127, 255, cv2.THRESH_BINARY)

    bg_removed_array = np.array(original_image.convert('RGB'))
    result = np.ones_like(bg_removed_array) * 255
    result[contour_mask > 0] = bg_removed_array[contour_mask > 0]

    return Image.fromarray(result.astype('uint8'))


def execute_background_removal(image_path: str) -> str:
    """
    Remove background using HuggingFace Inference API.

    Saves the result to outputs/<image_stem>/bg_removed_<filename>.
    Also copies the original input image into the same subdirectory for reference.

    Args:
        image_path: Path to the source image.

    Returns:
        Path to the background-removed image (str).
    """
    image_path = Path(image_path)
    print(f"  [BG Removal] Starting background removal for: {image_path}")

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Per-image output directory
    output_dir = get_image_output_dir(image_path)

    try:
        print(f"  [BG Removal] Initializing HuggingFace client...")
        client = InferenceClient(token=HF_TOKEN)

        print(f"  [BG Removal] Calling RMBG-2.0 via API...")
        result = client.image_segmentation(
            image=str(image_path),
            model="briaai/RMBG-2.0",
        )

        bg_removed = result[0]['mask']
        print(f"  [BG Removal] Received mask, mode: {bg_removed.mode}")

        if bg_removed.mode == 'RGBA':
            print(f"  [BG Removal] Converting RGBA to RGB with white background...")
            rgb_img = Image.new('RGB', bg_removed.size, (255, 255, 255))
            rgb_img.paste(bg_removed, mask=bg_removed.split()[3])
        else:
            rgb_img = bg_removed.convert('RGB')

        # Copy original input into the subdirectory for reference
        import shutil
        shutil.copy2(image_path, output_dir / image_path.name)
        print(f"  [BG Removal] Copied original to: {output_dir / image_path.name}")

        # Save background-removed image
        output_path = output_dir / f"bg_removed_{image_path.name}"
        rgb_img.save(output_path)

        print(f"  [BG Removal] ✓ Success! Saved to: {output_path}")
        return str(output_path)

    except Exception as e:
        print(f"  [BG Removal] ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"Background removal service failed: {e}")


def execute_sam3_segmentation(image_path: str, point_x: int, point_y: int) -> str:
    """
    Segment organoid using SAM3 Tracker with a positive point prompt (local model).

    All outputs (binary mask, visualization, filled mask) are saved to the same
    per-image subdirectory as the background-removed image.

    Args:
        image_path: Path to the background-removed image.
        point_x:    X coordinate of the center point.
        point_y:    Y coordinate of the center point.

    Returns:
        Path to the binary segmentation mask image (str).
    """
    image_path = Path(image_path)
    print(f"  [SAM3] Starting SAM3 segmentation for: {image_path}")
    print(f"  [SAM3] Using positive point: ({point_x}, {point_y})")

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Resolve output directory from the image's parent (already a per-image subdir)
    # If the bg-removed image is already inside outputs/<stem>/, reuse that dir.
    # Otherwise fall back to creating one from the stem.
    if image_path.parent.name == image_path.stem.removeprefix("bg_removed_"):
        output_dir = image_path.parent
    else:
        output_dir = get_image_output_dir(image_path)

    try:
        import torch
        from transformers import Sam3TrackerProcessor, Sam3TrackerModel
        import numpy as np
        from PIL import ImageDraw

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  [SAM3] Using device: {device}")

        print(f"  [SAM3] Loading SAM3 model...")
        model = Sam3TrackerModel.from_pretrained("facebook/sam3").to(device)
        processor = Sam3TrackerProcessor.from_pretrained("facebook/sam3")
        print(f"  [SAM3] Model loaded successfully")

        print(f"  [SAM3] Loading image...")
        image = Image.open(image_path).convert("RGB")
        img_width, img_height = image.size
        print(f"  [SAM3] Image size: {image.size}")

        input_points = [[[[point_x, point_y]]]]
        input_labels = [[[1]]]

        print(f"  [SAM3] Processing with point: {input_points}")

        inputs = processor(
            images=image,
            input_points=input_points,
            input_labels=input_labels,
            return_tensors="pt",
        ).to(device)

        print(f"  [SAM3] Running inference...")
        with torch.no_grad():
            # multimask_output=False tells SAM3 to return its single best mask
            # rather than 3 candidates. With multimask_output=True (the default),
            # pred_masks has shape [batch, objects, 3, H, W] and index [0] is the
            # most conservative (smallest) mask — which under-segments diffuse
            # organoid edges. Setting False collapses that dimension to 1 and
            # returns the highest-confidence mask, matching the Gradio app behaviour.
            outputs = model(**inputs, multimask_output=False)

        print(f"  [SAM3] Raw output shape: {outputs.pred_masks.shape}")

        # post_process_masks handles resize to original dimensions + binarisation,
        # replacing the manual sigmoid + threshold + F.interpolate pipeline.
        # With multimask_output=False the shape is [batch, objects, 1, H, W];
        # [0] selects batch 0 → [objects, 1, H, W], then [0, 0] → [H, W].
        masks = processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"],
            binarize=True,
        )[0]  # [objects, 1, H, W]

        binary_mask = masks[0, 0].numpy().astype("uint8") * 255
        print(f"  [SAM3] Final mask shape: {binary_mask.shape}")

        # --- Save outputs into the per-image subdirectory ---

        # Binary mask
        output_path = output_dir / f"sam3_mask_{image_path.name}"
        Image.fromarray(binary_mask).save(output_path)
        print(f"  [SAM3] Binary mask saved to: {output_path}")

        # Visualization
        print(f"  [SAM3] Creating visualization...")
        np.random.seed(hash(str(image_path)) % (2**32))
        overlay_color = tuple(np.random.randint(0, 255, 3).tolist())
        print(f"  [SAM3] Using overlay color: RGB{overlay_color}")

        overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))

        mask_bool = binary_mask > 127
        colored_mask = np.zeros((img_height, img_width, 4), dtype=np.uint8)
        colored_mask[mask_bool] = [*overlay_color, 128]

        mask_overlay = Image.fromarray(colored_mask, mode='RGBA')
        vis_image = Image.alpha_composite(image.convert('RGBA'), mask_overlay)

        draw = ImageDraw.Draw(vis_image)
        point_size = 10
        point_color = (255, 255, 0, 255)
        outline_color = (0, 0, 0, 255)

        draw.ellipse(
            [point_x - point_size, point_y - point_size,
             point_x + point_size, point_y + point_size],
            fill=None, outline=outline_color, width=3,
        )
        draw.ellipse(
            [point_x - point_size, point_y - point_size,
             point_x + point_size, point_y + point_size],
            fill=None, outline=point_color, width=2,
        )

        crosshair_length = 20
        for fill, width in [(outline_color, 3), (point_color, 2)]:
            draw.line([point_x - crosshair_length, point_y, point_x + crosshair_length, point_y], fill=fill, width=width)
            draw.line([point_x, point_y - crosshair_length, point_x, point_y + crosshair_length], fill=fill, width=width)

        label_text = f"Point: ({point_x}, {point_y})"
        label_x, label_y = point_x + 15, point_y - 25
        for ox in [-1, 0, 1]:
            for oy in [-1, 0, 1]:
                if ox != 0 or oy != 0:
                    draw.text((label_x + ox, label_y + oy), label_text, fill=outline_color)
        draw.text((label_x, label_y), label_text, fill=point_color)

        vis_image = vis_image.convert('RGB')
        vis_output_path = output_dir / f"sam3_visualization_{image_path.name}"
        vis_image.save(vis_output_path)
        print(f"  [SAM3] Visualization saved to: {vis_output_path}")

        # Filled mask
        print(f"  [SAM3] Creating filled mask with white background...")
        filled_mask = create_filled_mask(binary_mask, image)
        filled_output_path = output_dir / f"sam3_filled_{image_path.name}"
        filled_mask.save(filled_output_path)
        print(f"  [SAM3] Filled mask saved to: {filled_output_path}")

        print(f"  [SAM3] ✓ Success! All outputs saved to: {output_dir}/")
        print(f"         - Binary mask:   {output_path.name}")
        print(f"         - Visualization: {vis_output_path.name}")
        print(f"         - Filled mask:   {filled_output_path.name}")

        return str(output_path)

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
        raise RuntimeError(
            f"SAM3 segmentation failed: {e}\n\n"
            "Possible issues:\n"
            "1. Model access: Request access at https://huggingface.co/facebook/sam3\n"
            "2. Authentication: Run 'huggingface-cli login' with your token\n"
            "3. GPU memory: Model requires ~3GB VRAM (use CPU if needed)\n"
            "4. Dependencies: Ensure transformers>=4.48.0 and torch>=2.0"
        )
