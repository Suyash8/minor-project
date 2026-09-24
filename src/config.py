"""
Central Configuration for Datasets, Models, and Evaluation Metrics.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Any, Optional

# Base directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_WEIGHTS_DIR = PROJECT_ROOT / "weights"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"

def resolve_pipeline_paths(
    data_dir: Optional[str] = None,
    weights_dir: Optional[str] = None,
    results_dir: Optional[str] = None,
    run_id: Optional[str] = None,
) -> Dict[str, Path]:
    """
    Resolve and construct storage paths for datasets, pretrained weights, and benchmark results.
    Incorporates seamless Google Drive integration for Google Colab runtimes:
      - Persistent Drive root: `/content/drive/MyDrive/object-detection`
      - Persistent data: `object-detection/data/`
      - Persistent weights: `object-detection/weights/`
      - Persistent outputs: `object-detection/results/`
    When running locally or when Drive is not mounted, falls back to local workspace paths.
    """
    from src.utils.env import is_colab, is_drive_mounted, get_drive_root

    in_colab = is_colab()
    drive_mounted = is_drive_mounted()
    drive_dir = get_drive_root()

    # 1. Resolve DATA_DIR
    if data_dir:
        resolved_data = Path(data_dir).resolve()
    elif drive_mounted:
        candidate_drive_dirs = [
            drive_dir / "data",
            Path("/content/drive/MyDrive/data"),
            Path("/content/drive/MyDrive/minor-project/data"),
            Path("/content/drive/MyDrive/Datasets"),
        ]
        found_drive_data = None
        for cand in candidate_drive_dirs:
            if cand.exists() and any(cand.iterdir()):
                found_drive_data = cand.resolve()
                break
        if found_drive_data:
            resolved_data = found_drive_data
        elif (PROJECT_ROOT / "data").exists() and any((PROJECT_ROOT / "data").iterdir()):
            resolved_data = DEFAULT_DATA_DIR.resolve()
        else:
            resolved_data = (drive_dir / "data").resolve()
    else:
        resolved_data = DEFAULT_DATA_DIR.resolve()

    # 2. Resolve WEIGHTS_DIR
    if weights_dir:
        resolved_weights = Path(weights_dir).resolve()
    elif drive_mounted:
        resolved_weights = (drive_dir / "weights").resolve()
    else:
        resolved_weights = DEFAULT_WEIGHTS_DIR.resolve()

    # 3. Resolve RESULTS_DIR
    if results_dir:
        resolved_results = Path(results_dir).resolve()
    elif drive_mounted:
        resolved_results = (drive_dir / "results").resolve()
    else:
        resolved_results = DEFAULT_RESULTS_DIR.resolve()

    # Create directories if they do not exist
    resolved_data.mkdir(parents=True, exist_ok=True)
    resolved_weights.mkdir(parents=True, exist_ok=True)
    resolved_results.mkdir(parents=True, exist_ok=True)

    resolved_output = None
    if run_id:
        resolved_output = (resolved_results / run_id).resolve()
        resolved_output.mkdir(parents=True, exist_ok=True)

    return {
        "data_dir": resolved_data,
        "weights_dir": resolved_weights,
        "results_dir": resolved_results,
        "output_dir": resolved_output or resolved_results,
        "in_colab": in_colab,
        "drive_mounted": drive_mounted,
    }


# Standard dataset class definitions
DATASET_CLASSES: Dict[str, List[str]] = {
    "codrone": [
        "pedestrian",
        "cyclist",
        "car",
        "van",
        "truck",
        "bus",
        "tricycle",
        "motorcyclist",
        "traffic_light",
        "traffic_sign",
        "bridge",
        "boat",
    ],
    "visdrone": [
        "pedestrian",
        "people",
        "bicycle",
        "car",
        "van",
        "truck",
        "tricycle",
        "awning-tricycle",
        "bus",
        "motor",
    ],
    "dota": [
        "plane",
        "ship",
        "storage-tank",
        "baseball-diamond",
        "tennis-court",
        "basketball-court",
        "ground-track-field",
        "harbor",
        "bridge",
        "large-vehicle",
        "small-vehicle",
        "helicopter",
        "roundabout",
        "soccer-ball-field",
        "swimming-pool",
    ],
}

# Unified road user & vehicle classes for cross-dataset comparison
UNIFIED_CLASSES: List[str] = [
    "car",
    "van",
    "truck",
    "bus",
    "pedestrian",
    "cyclist",
    "motorcyclist",
]

# Supported model architectures
SUPPORTED_MODELS: List[str] = [
    # Nano models (edge, lightweight, T4 friendly)
    "yolov8n-obb",
    "yolo11n-obb",
    # Small models (balanced speed/accuracy)
    "yolov8s-obb",
    "yolo11s-obb",
    # Medium models (high capacity for A100/L4/V100)
    "yolov8m-obb",
    "yolo11m-obb",
    # Large models (maximum representation capacity)
    "yolov8l-obb",
    "yolo11l-obb",
    # Custom PyTorch Anchor-Free detector
    "custom-obb",
    # 10 Key Survey Models (Post-2022, Transformer-Focused)
    "std",
    "rvsa",
    "ars-detr",
    "oriented-former",
    "rio-detr",
    "rhino",
    "ao2-detr",
    "swin-obb",
    "lsknet",
    "yolo11-obb",
]

# Supported datasets
SUPPORTED_DATASETS: List[str] = [
    "codrone",
    "visdrone",
    "dota",
]

# Evaluation defaults
DEFAULT_IOU_THRESHOLDS = [0.50, 0.75]
DEFAULT_IOU_MAP_RANGE = [round(x, 2) for x in list(float(i) / 100.0 for i in range(50, 100, 5))]
DEFAULT_CONF_THRESH = 0.25
DEFAULT_NMS_IOU_THRESH = 0.45

# Object scale definitions (COCO aerial standards in square pixels)
SCALE_THRESHOLDS = {
    "small": (0, 32 ** 2),         # < 1024 px^2
    "medium": (32 ** 2, 96 ** 2),   # 1024 to 9216 px^2
    "large": (96 ** 2, float("inf")), # > 9216 px^2
}

# Static deterministic dataset folder paths relative to each dataset directory
DATASET_STATIC_PATHS: Dict[str, Dict[str, Dict[str, Optional[str]]]] = {
    "codrone": {
        "val": {
            "images": "val/images",
            "labels": "val/annfile",
        },
        "train": {
            "images": "train/images",
            "labels": "train/annfile",
        },
        "test": {
            "images": "test/images",
            "labels": "test/annfile",
        },
    },
    "visdrone": {
        "val": {
            "images": "VisDrone2019-DET-val/images",
            "labels": "VisDrone2019-DET-val/annotations",
        },
        "train": {
            "images": "VisDrone2019-DET-train/images",
            "labels": "VisDrone2019-DET-train/annotations",
        },
        "test": {
            "images": "VisDrone2019-DET-test-dev/images",
            "labels": "VisDrone2019-DET-test-dev/annotations",
        },
    },
    "dota": {
        "val": {
            "images": "images/val",
            "labels": "labels/val",
        },
        "train": {
            "images": "images/train",
            "labels": "labels/train",
        },
        "test": {
            "images": "images/test",
            "labels": None,
        },
    },
}

