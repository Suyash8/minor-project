"""
Training module registry and orchestrator for Aerial OBB models.
"""

from __future__ import annotations

from src.training.train_yolo import train_yolo_obb
from src.training.train_custom import train_custom_detector
from src.training.train_dense import train_dense_model
from src.training.train_detr import train_detr_model
from src.training.dispatcher import train_model, get_model_family

__all__ = [
    "train_model",
    "train_dense_model",
    "train_detr_model",
    "train_yolo_obb",
    "train_custom_detector",
    "get_model_family",
]
