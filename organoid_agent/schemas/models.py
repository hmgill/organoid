from pydantic import BaseModel, Field
from typing import List, Optional

class Point(BaseModel):
    """Represents a 2D coordinate point."""
    x: int = Field(..., description="X coordinate in pixels")
    y: int = Field(..., description="Y coordinate in pixels")

class DetectedOrganoid(BaseModel):
    """Represents a detected organoid with its center point and segmentation."""
    organoid_id: str = Field(..., description="Unique identifier (e.g., 'org_1', 'org_2')")
    center: Point = Field(..., description="Approximate center point of the organoid")
    confidence: str = Field(default="medium", description="Confidence level: 'high', 'medium', or 'low'")
    mask_path: Optional[str] = Field(default=None, description="Path to SAM3 segmentation mask (if generated)")

class AnalysisResult(BaseModel):
    """Complete analysis result with all detected organoids."""
    background_removed_image: str = Field(..., description="Path to the background-removed image")
    organoid_count: int = Field(..., description="Total number of organoids detected")
    organoids: List[DetectedOrganoid] = Field(..., description="List of detected organoids with centers and masks")
    image_dimensions: dict = Field(..., description="Image width and height in pixels")
