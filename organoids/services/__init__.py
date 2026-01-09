# services/__init__.py

from .processing import (
    execute_background_removal,
    execute_annotation
)

__all__ = [
    "execute_background_removal",
    "execute_annotation"
]
