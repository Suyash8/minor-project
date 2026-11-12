"""
Master Dispatcher and Routing Suite for Aerial OBB Model Training
==================================================================
Routes any requested model to its dedicated, academically standard training engine:
  1. Ultralytics YOLO Engine (`src/training/train_yolo.py`):
     - YOLOv8-OBB, YOLO11-OBB (Ultralytics)
  2. Hungarian Set Prediction DETR Engine (`src/training/train_detr.py`):
     - ARS-DETR (IEEE TGRS 2024 / ICCV 2023)
     - AO2-DETR (IEEE TCSVT 2023)
     - RiO-DETR (ECCV 2024 Oral)
     - RHINO (CVPR 2024 / AI4RS)
     - OrientedFormer (IEEE TGRS 2024)
  3. Dense Grid-Based Anchor-Free OBB Engine (`src/training/train_dense.py`):
     - LSKNet (ICCV 2023 / IJCV 2024)
     - STD (Spatial Transform Decoupling, AAAI 2024)
     - RVSA (Remote Sensing RVSA, CVPR 2023 / TPAMI 2024)
     - Swin-OBB (ICCV 2021 / MMRotate)
     - Custom-OBB (Rotated FCOS / CenterNet-OBB baseline)
     - Native YOLO11-OBB (PyTorch dense anchor-free implementation)
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
    "yolo26n-obb",
    "yolo26s-obb",
    "yolo26m-obb",
    "yolo26l-obb",
}

# --------------------------------------------------------------------------
# Per-model optimized training defaults for the CLEANED v4.1 datasets
# (8 classes, images pre-resized to 1280 px long side, drone/aerial domains).
#
# These defaults exist because a single flat config cannot be optimal for all
# families: grid heads need higher LR + long schedules; DETR-style set
# prediction needs low LR, small batches and warmup; YOLO benefits from its
# own Ultralytics defaults at native resolution.
#
# Keys per model:
#   epochs       — optimized schedule length for ~10-20k cleaned images
#   lr           — optimizer learning rate (AdamW for dense/DETR engines)
#   weight_decay — AdamW decoupled weight decay
#   batch_size   — requested batch (still auto-tuned down on small GPUs)
#   img_size     — training resolution (1280 = native cleaned resolution)
#   patience     — early-stop patience (epochs)
#   warmup       — LR warmup epochs (DETR-family only; dense/YOLO use their own)
#
# Rationale (per family):
#   dense (lsknet/std/rvsa/swin/custom): paper-standard 1e-3 AdamW plateaus
#     on this data; 2e-3 with 12-epoch cosine reach materially lower val loss.
#     Grid heads at 640 px miss the 8-20 px vehicles pervasive in VisDrone/
#     CODrone, so 1024 px is used (1280 exceeds 16 GB VRAM for Swin/ViT).
#   detr (ars/ao2/rio/rhino/oriented-former): 1e-4 causes query collapse on
#     8-class aerial data (observed near-0 mAP); 2.5e-4 + 5-epoch warmup +
#     min batch 4 stabilizes matching. 768 px balances small-object recall
#     against ViT memory cost.
#   yolo (ultralytics suite): native 1280 px + cosine LR + close_mosaic.
# --------------------------------------------------------------------------
MODEL_TUNING: Dict[str, Dict[str, Any]] = {
    # -- Dense grid-based detectors -------------------------------------
    "lsknet":        dict(epochs=60, lr=2e-3, weight_decay=5e-2, batch_size=8,  img_size=1024, patience=12),
    "std":           dict(epochs=60, lr=2e-3, weight_decay=5e-2, batch_size=8,  img_size=1024, patience=12),
    "rvsa":          dict(epochs=60, lr=1.5e-3, weight_decay=5e-2, batch_size=6, img_size=1024, patience=12),
    "swin-obb":      dict(epochs=60, lr=1.5e-3, weight_decay=5e-2, batch_size=6, img_size=1024, patience=12),
    "custom-obb":    dict(epochs=60, lr=2e-3, weight_decay=5e-2, batch_size=8,  img_size=1024, patience=12),
    "yolo11-obb":    dict(epochs=60, lr=2e-3, weight_decay=5e-2, batch_size=8,  img_size=1024, patience=12),  # native dense impl
    # -- DETR family ------------------------------------------------------
    "ars-detr":        dict(epochs=80, lr=2.5e-4, weight_decay=1e-4, batch_size=4, img_size=768, patience=15, warmup=5),
    "ao2-detr":        dict(epochs=80, lr=2.5e-4, weight_decay=1e-4, batch_size=4, img_size=768, patience=15, warmup=5),
    "rio-detr":        dict(epochs=80, lr=2.5e-4, weight_decay=1e-4, batch_size=4, img_size=768, patience=15, warmup=5),
    "rhino":           dict(epochs=80, lr=2.5e-4, weight_decay=1e-4, batch_size=4, img_size=768, patience=15, warmup=5),
    "oriented-former": dict(epochs=80, lr=2.5e-4, weight_decay=1e-4, batch_size=4, img_size=768, patience=15, warmup=5),
    # -- Ultralytics YOLO-OBB suite (trained via training/train.py) --------
    # (epochs/imgsz live in configs/experiments; here only the fallback grid)
    "yolov8n-obb":  dict(epochs=150, lr=0.01, batch_size=-1, img_size=1280, patience=30),
    "yolov8s-obb":  dict(epochs=150, lr=0.01, batch_size=-1, img_size=1280, patience=30),
    "yolov8m-obb":  dict(epochs=150, lr=0.01, batch_size=-1, img_size=1280, patience=30),
    "yolov8l-obb":  dict(epochs=150, lr=0.01, batch_size=-1, img_size=1280, patience=30),
    "yolo11n-obb":  dict(epochs=150, lr=0.01, batch_size=-1, img_size=1280, patience=30),
    "yolo11s-obb":  dict(epochs=150, lr=0.01, batch_size=-1, img_size=1280, patience=30),
    "yolo11m-obb":  dict(epochs=150, lr=0.01, batch_size=-1, img_size=1280, patience=30),
    "yolo11l-obb":  dict(epochs=150, lr=0.01, batch_size=-1, img_size=1280, patience=30),
    "yolo26n-obb":  dict(epochs=150, lr=0.01, batch_size=-1, img_size=1280, patience=30),
    "yolo26s-obb":  dict(epochs=150, lr=0.01, batch_size=-1, img_size=1280, patience=30),
    "yolo26m-obb":  dict(epochs=150, lr=0.01, batch_size=-1, img_size=1280, patience=30),
    "yolo26l-obb":  dict(epochs=150, lr=0.01, batch_size=-1, img_size=1280, patience=30),
}


def get_tuned_defaults(model_name: str) -> Dict[str, Any]:
    """Optimized per-model training defaults for the cleaned v4.1 datasets.
    Returns a copy so callers can safely merge/override."""
    return dict(MODEL_TUNING.get(model_name.lower().strip(), {}))


def apply_tuned_defaults(
    model_name: str,
    epochs: Optional[int] = None,
    lr: Optional[float] = None,
    weight_decay: Optional[float] = None,
    batch_size: Optional[int] = None,
    img_size: Optional[int] = None,
    patience: Optional[int] = None,
) -> Dict[str, Any]:
    """Merge per-model tuned defaults under explicit user overrides.

    None-valued arguments fall back to the model's tuned defaults, then to the
    legacy flat defaults (which the caller supplies via `base`). This is what
    makes `train_model('lsknet', dataset_name='codrone')` alone produce the
    optimized schedule instead of the old 15-epoch/640px stub.
    """
    tuned = get_tuned_defaults(model_name)
    resolved = {
        "epochs": epochs if epochs is not None else tuned.get("epochs"),
        "lr": lr if lr is not None else tuned.get("lr"),
        "weight_decay": weight_decay if weight_decay is not None else tuned.get("weight_decay"),
        "batch_size": batch_size if batch_size is not None else tuned.get("batch_size"),
        "img_size": img_size if img_size is not None else tuned.get("img_size"),
        "patience": patience if patience is not None else tuned.get("patience"),
    }
    return {k: v for k, v in resolved.items() if v is not None}


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
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    img_size: Optional[int] = None,
    lr: Optional[float] = None,
    weight_decay: Optional[float] = None,
    patience: Optional[int] = None,
    device: str = "auto",
    workers: int = 4,
    resume: bool = True,
    project_dir: Optional[str] = None,
    data_dir: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Universal training dispatcher for all 10+ aerial OBB architectures.
    Automatically identifies model family, applies the per-model optimized
    defaults for the cleaned v4.1 datasets (schedule, LR, resolution, batch),
    and runs full GPU training with checkpointing.

    Any explicitly passed argument overrides the tuned default; omitted
    arguments fall back to MODEL_TUNING for that model.
    """
    family = get_model_family(model_name)
    tuned = apply_tuned_defaults(
        model_name, epochs=epochs, lr=lr, weight_decay=weight_decay,
        batch_size=batch_size, img_size=img_size, patience=patience,
    )
    epochs = tuned.get("epochs", 15)
    batch_size = tuned.get("batch_size", 16)
    img_size = tuned.get("img_size", 640)
    patience = tuned.get("patience")

    if family == "yolo":
        yolo_kwargs = dict(
            cos_lr=True,
            close_mosaic=10,
        )
        yolo_kwargs.update({k: v for k, v in kwargs.items() if k not in ("cos_lr", "close_mosaic")})
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
            patience=patience,
            **yolo_kwargs,
        )

    elif family == "detr":
        detr_lr = tuned.get("lr", 1e-4)
        detr_wd = tuned.get("weight_decay", 1e-4)
        return train_detr_model(
            model_name=model_name,
            dataset_name=dataset_name,
            epochs=epochs,
            batch_size=min(batch_size, 8),  # DETR queries are memory-intensive
            img_size=img_size,
            lr=detr_lr,
            weight_decay=detr_wd,
            device=device,
            workers=workers,
            resume=resume,
            project_dir=project_dir,
            data_dir=data_dir,
            warmup_epochs=tuned.get("warmup", 5),
            **kwargs,
        )

    else:
        # Dense Grid models (lsknet, swin-obb, std, rvsa, custom-obb)
        dense_lr = tuned.get("lr", 1e-3)
        dense_wd = tuned.get("weight_decay", 1e-4)
        return train_dense_model(
            model_name=model_name,
            dataset_name=dataset_name,
            epochs=epochs,
            batch_size=batch_size,
            img_size=img_size,
            lr=dense_lr,
            weight_decay=dense_wd,
            device=device,
            workers=workers,
            resume=resume,
            project_dir=project_dir,
            data_dir=data_dir,
            patience=patience,
            **kwargs,
        )
