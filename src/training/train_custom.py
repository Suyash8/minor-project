"""
Custom-OBB (Rotated FCOS / CenterNet-OBB baseline) Training Entry Point
========================================================================
v4: this module is a thin compatibility delegate. The custom-obb detector is
a dense anchor-free grid model, so all training behavior now lives in the
unified dense engine (`src/training/train_dense.py`) which provides AMP,
gradient accumulation, OOM recovery, EMA, Drive mirroring and the §4.4
run directory contract. This file is kept so older imports and scripts that
call `train_custom_model(...)` keep working.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.training.train_dense import train_dense_model


def train_custom_detector(
    model_name: str = "custom-obb",
    dataset_name: str = "visdrone",
    epochs: int = 15,
    batch_size: int = 16,
    img_size: int = 640,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "auto",
    workers: int = 4,
    resume: bool = True,
    project_dir: Optional[str] = None,
    data_dir: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Legacy entry point name kept for older imports (delegates to the
    unified dense engine, same as train_custom_model)."""
    return train_dense_model(
        model_name=model_name,
        dataset_name=dataset_name,
        epochs=epochs,
        batch_size=batch_size,
        img_size=img_size,
        lr=lr,
        weight_decay=weight_decay,
        device=device,
        workers=workers,
        resume=resume,
        project_dir=project_dir,
        data_dir=data_dir,
        **kwargs,
    )


def train_custom_model(
    model_name: str = "custom-obb",
    dataset_name: str = "visdrone",
    epochs: int = 15,
    batch_size: int = 16,
    img_size: int = 640,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "auto",
    workers: int = 4,
    resume: bool = True,
    project_dir: Optional[str] = None,
    data_dir: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Backward-compatible wrapper around the unified dense training engine."""
    return train_dense_model(
        model_name=model_name,
        dataset_name=dataset_name,
        epochs=epochs,
        batch_size=batch_size,
        img_size=img_size,
        lr=lr,
        weight_decay=weight_decay,
        device=device,
        workers=workers,
        resume=resume,
        project_dir=project_dir,
        data_dir=data_dir,
        **kwargs,
    )
