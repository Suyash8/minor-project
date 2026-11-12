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
    else:
        # Check high-speed local disk locations first (especially in Colab, where Drive is slow FUSE)
        local_fast_candidates = [
            Path("/content/data"),
            Path("/tmp/data"),
            PROJECT_ROOT / "data",
        ]
        found_local_data = None
        for cand in local_fast_candidates:
            if cand.exists() and any(cand.iterdir()):
                found_local_data = cand.resolve()
                break

        if found_local_data:
            resolved_data = found_local_data
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

# Unified 8-class space produced by data_cleaning/clean_build_v4.py (v4.1 splits).
# All three cleaned datasets (and the merged 'all' set) share exactly this list,
# so every model trains and cross-evaluates in one label space.
CLEANED_DATASET_CLASSES: Dict[str, List[str]] = {
    "codrone": [
        "pedestrian",
        "bicycle",
        "car",
        "van",
        "truck",
        "bus",
        "tricycle",
        "motorcyclist",
    ],
    "visdrone": [
        "pedestrian",
        "bicycle",
        "car",
        "van",
        "truck",
        "bus",
        "tricycle",
        "motorcyclist",
    ],
    "dota": [
        "pedestrian",
        "bicycle",
        "car",
        "van",
        "truck",
        "bus",
        "tricycle",
        "motorcyclist",
    ],
    "all": [
        "pedestrian",
        "bicycle",
        "car",
        "van",
        "truck",
        "bus",
        "tricycle",
        "motorcyclist",
    ],
}

# Static deterministic dataset folder paths relative to each dataset directory.
# Each split lists layouts in priority order; the first layout whose images
# directory exists wins. Layout "cleaned" (v4.1 output:
# {split}/images + {split}/labels in YOLO-OBB 8-class format) is preferred so
# the dense/DETR engines train on exactly the same cleaned data as the
# Ultralytics suite; the raw Drive layouts remain as fallbacks.
DATASET_STATIC_PATHS: Dict[str, Dict[str, List[Dict[str, Optional[str]]]]] = {
    "codrone": {
        "val": [
            {"images": "val/images", "labels": "val/labels", "layout": "cleaned"},
            {"images": "val/images", "labels": "val/annfile", "layout": "raw"},
        ],
        "train": [
            {"images": "train/images", "labels": "train/labels", "layout": "cleaned"},
            {"images": "train/images", "labels": "train/annfile", "layout": "raw"},
        ],
        "test": [
            {"images": "test/images", "labels": "test/labels", "layout": "cleaned"},
            {"images": "test/images", "labels": "test/annfile", "layout": "raw"},
        ],
    },
    "visdrone": {
        "val": [
            {"images": "val/images", "labels": "val/labels", "layout": "cleaned"},
            {"images": "VisDrone2019-DET-val/images", "labels": "VisDrone2019-DET-val/annotations", "layout": "raw"},
        ],
        "train": [
            {"images": "train/images", "labels": "train/labels", "layout": "cleaned"},
            {"images": "VisDrone2019-DET-train/images", "labels": "VisDrone2019-DET-train/annotations", "layout": "raw"},
        ],
        "test": [
            {"images": "test/images", "labels": "test/labels", "layout": "cleaned"},
            {"images": "VisDrone2019-DET-test-dev/images", "labels": "VisDrone2019-DET-test-dev/annotations", "layout": "raw"},
        ],
    },
    "dota": {
        "val": [
            {"images": "val/images", "labels": "val/labels", "layout": "cleaned"},
            {"images": "images/val", "labels": "labels/val", "layout": "raw"},
        ],
        "train": [
            {"images": "train/images", "labels": "train/labels", "layout": "cleaned"},
            {"images": "images/train", "labels": "labels/train", "layout": "raw"},
        ],
        "test": [
            {"images": "test/images", "labels": "test/labels", "layout": "cleaned"},
            {"images": "images/test", "labels": None, "layout": "raw"},
        ],
    },
}

# Root that contains the cleaned datasets ({codrone,visdrone,dota}/<split>/{images,labels}).
# Priority: explicit data_dir > $VEHICLE_OBB_DATA > Colab Drive > repo-local data/.
def cleaned_data_root(explicit: Optional[str] = None) -> Path:
    import os
    if explicit:
        return Path(explicit).expanduser().resolve()
    env = os.environ.get("VEHICLE_OBB_DATA")
    if env:
        return Path(env).expanduser().resolve()
    drive = Path("/content/drive/MyDrive/object-detection/cleaned_data")
    if drive.is_dir():
        return drive
    return PROJECT_ROOT / "data" / "cleaned_data"

