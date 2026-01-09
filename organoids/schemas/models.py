from pydantic import BaseModel, Field
from typing import List

class Point(BaseModel):
    x: int = Field(..., description="X coordinate pixels")
    y: int = Field(..., description="Y coordinate pixels")
    label: str = Field(..., description="Label: 'budding_region', 'center', or 'edge'")

class DetectedOrganoid(BaseModel):
    organoid_id: str = Field(..., description="Unique ID (e.g., 'org_1')")
    # FIX: Lower min_items to 1 so the app doesn't crash on small objects
    points: List[Point] = Field(..., min_items=1, max_items=10, description="Key points found on the organoid.")

class AnalystOutput(BaseModel):
    annotated_image_path: str = Field(..., description="Path to the final image with annotations drawn.")
    organoid_count: int = Field(..., description="Total number of organoids found.")
    organoids: List[DetectedOrganoid] = Field(..., description="Structured data of findings.")
