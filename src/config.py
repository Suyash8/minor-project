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
    elif drive_mounted and (drive_dir / "data").exists():
        resolved_data = (drive_dir / "data").resolve()
    elif drive_mounted and not (PROJECT_ROOT / "data").exists():
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
    "yolov8n-obb",
    "yolov8s-obb",
    "yolo11n-obb",
    "custom-obb",
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
