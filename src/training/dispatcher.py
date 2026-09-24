"""
Master Dispatcher and Routing Suite for Aerial OBB Model Training.
Routes any requested model to its dedicated training engine:
  - Ultralytics YOLO Engine -> train_yolo_obb
  - Hungarian DETR Engine   -> train_detr_model
  - Dense Grid OBB Engine   -> train_dense_model
"""

from __future__ import annotations

from typing import Dict, Any, Optional
from src.training.train_yolo import train_yolo_obb
from src.training.train_dense import train_dense_model
from src.training.train_detr import train_detr_model

DETR_MODELS = {
    "ars-detr",
    "ars_detr",
    "ao2-detr",
    "ao2_detr",
    "rio-detr",
    "rio_detr",
    "rhino",
    "rhino-detector",
    "oriented-former",
    "orientedformer",
}

DENSE_MODELS = {
    "std",
    "std-detector",
    "rvsa",
    "rvsa-detector",
    "lsknet",
    "lsk-net",
    "swin-obb",
    "swin",
    "custom-obb",
}

YOLO_MODELS = {
    "yolov8n-obb",
    "yolov8s-obb",
    "yolov8m-obb",
    "yolov8l-obb",
    "yolo11n-obb",
    "yolo11s-obb",
    "yolo11m-obb",
    "yolo11l-obb",
}


def get_model_family(model_name: str) -> str:
    """
    Determine the training paradigm family for a given model architecture.
    Returns: 'yolo', 'detr', or 'dense'.
    """
    m_lower = model_name.lower().strip()
    if m_lower in DETR_MODELS:
        return "detr"
    elif m_lower in YOLO_MODELS:
        return "yolo"
    elif m_lower in DENSE_MODELS:
        return "dense"
    elif "yolo" in m_lower:
        # Default yolo variants to Ultralytics if not pure custom
        return "yolo"
    else:
        return "dense"


def train_model(
    model_name: str,
    dataset_name: str = "visdrone",
    epochs: int = 15,
    batch_size: int = 16,
    img_size: int = 640,
    lr: Optional[float] = None,
    weight_decay: float = 1e-4,
    device: str = "auto",
    workers: int = 4,
    resume: bool = True,
    project_dir: Optional[str] = None,
    data_dir: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Universal training dispatcher for all 10+ aerial OBB architectures.
    Automatically identifies model family, configures optimal learning rate and loss criterion,
    and runs full GPU training with checkpointing.
    """
    family = get_model_family(model_name)

    if family == "yolo":
        return train_yolo_obb(
            model_name=model_name,
            dataset_name=dataset_name,
            epochs=epochs,
            batch_size=batch_size,
            imgsz=img_size,
            device=device,
            workers=workers,
            resume=resume,
            project_dir=project_dir,
            data_dir=data_dir,
            **kwargs,
        )

    elif family == "detr":
        detr_lr = lr if lr is not None else 1e-4
        return train_detr_model(
            model_name=model_name,
            dataset_name=dataset_name,
            epochs=epochs,
            batch_size=min(batch_size, 8),  # DETR queries are memory-intensive
            img_size=img_size,
            lr=detr_lr,
            weight_decay=weight_decay,
            device=device,
            workers=workers,
            resume=resume,
            project_dir=project_dir,
            data_dir=data_dir,
            **kwargs,
        )

    else:
        # Dense Grid models (lsknet, swin-obb, std, rvsa, custom-obb)
        dense_lr = lr if lr is not None else 1e-3
        return train_dense_model(
            model_name=model_name,
            dataset_name=dataset_name,
            epochs=epochs,
            batch_size=batch_size,
            img_size=img_size,
            lr=dense_lr,
            weight_decay=weight_decay,
            device=device,
            workers=workers,
            resume=resume,
            project_dir=project_dir,
            data_dir=data_dir,
            **kwargs,
        )
