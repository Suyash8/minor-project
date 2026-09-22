"""
Model registry and factory for Oriented Bounding Box Detectors.
"""

from typing import Optional
from src.models.base import BaseOBBDetector
from src.models.yolo_obb import YoloOBBDetector
from src.models.custom_obb import CustomOBBDetector

def get_model(model_name: str, device: str = "cpu", num_classes: int = 10) -> BaseOBBDetector:
    """
    Factory function returning the appropriate detector wrapper.
    """
    m_lower = model_name.lower()
    if "yolo" in m_lower:
        return YoloOBBDetector(model_name=m_lower, device=device)
    elif "custom" in m_lower:
        return CustomOBBDetector(model_name=m_lower, device=device, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model name: '{model_name}'. Supported: yolov8n-obb, yolov8s-obb, yolo11n-obb, custom-obb")

__all__ = [
    "BaseOBBDetector",
    "YoloOBBDetector",
    "CustomOBBDetector",
    "get_model",
]
