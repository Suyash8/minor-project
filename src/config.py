"""
Central Configuration for Datasets, Models, and Evaluation Metrics.
"""

from pathlib import Path
from typing import Dict, List, Any

# Base directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_WEIGHTS_DIR = PROJECT_ROOT / "weights"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"

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
