from .functions import (
    remove_background_tool,
    segment_organoid_sam3_tool,
    analyze_with_vision_tool
)

# Analyst tools: SAM3 segmentation (called by analyst after visual examination)
analyst_tools = [
    segment_organoid_sam3_tool,
]

# Manager tools: Background removal and vision analysis orchestration
manager_tools = [
    remove_background_tool,
    analyze_with_vision_tool,
]


__all__ = [
    "analyst_tools",
    "manager_tools",
    "remove_background_tool",
    "segment_organoid_sam3_tool",
    "analyze_with_vision_tool"
]
