"""
Training module registry and orchestrator for Aerial OBB models.
"""

from __future__ import annotations

from src.training.train_yolo import train_yolo_obb
from src.training.train_custom import train_custom_detector

__all__ = [
    "train_yolo_obb",
    "train_custom_detector",
]
