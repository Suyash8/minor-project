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

# ---------------------------------------------------------------------------
# CANONICAL MODEL REGISTRY (v4) — the single source of truth for all 23 models.
#
# Family A — Ultralytics YOLO-OBB suite (12): trained through the Ultralytics
#   engine (training/run.py -> training/train.py).
# Family B — Dense grid detectors (6): trained through src/training/train_dense.py.
# Family C — DETR-family query detectors (5): trained through src/training/train_detr.py.
#
# Every entry point, CLI choice list, config validator and docs table must be
# derived from CANONICAL_MODELS / MODEL_FAMILY below. No model list may be
# hardcoded anywhere else (v4 work item W1 / debt row D5).
# ---------------------------------------------------------------------------
YOLO_SUITE_MODELS: List[str] = [
    "yolov8n-obb", "yolov8s-obb", "yolov8m-obb", "yolov8l-obb",
    "yolo11n-obb", "yolo11s-obb", "yolo11m-obb", "yolo11l-obb",
    "yolo26n-obb", "yolo26s-obb", "yolo26m-obb", "yolo26l-obb",
]
DENSE_SUITE_MODELS: List[str] = [
    "lsknet", "std", "rvsa", "swin-obb", "custom-obb", "yolo11-obb",
]
DETR_SUITE_MODELS: List[str] = [
    "ars-detr", "ao2-detr", "rio-detr", "rhino", "oriented-former",
]

CANONICAL_MODELS: List[str] = YOLO_SUITE_MODELS + DENSE_SUITE_MODELS + DETR_SUITE_MODELS
assert len(CANONICAL_MODELS) == 23, "registry must expose exactly 23 models"

MODEL_FAMILY: Dict[str, str] = (
    {m: "yolo" for m in YOLO_SUITE_MODELS}
    | {m: "dense" for m in DENSE_SUITE_MODELS}
    | {m: "detr" for m in DETR_SUITE_MODELS}
)

# Approximate relative training cost rank, used by the unified pipeline to
# order jobs small/fast-first (so a time-limited Colab session finishes the
# cheap models before it dies). Lower = cheaper.
MODEL_SIZE_RANK: Dict[str, int] = {
    **{m: 0 for m in CANONICAL_MODELS if m.endswith("n-obb")},
    **{m: 1 for m in CANONICAL_MODELS if m.endswith("s-obb")},
    **{m: 2 for m in CANONICAL_MODELS if m.endswith("m-obb")},
    **{m: 3 for m in CANONICAL_MODELS if m.endswith("l-obb")},
    "custom-obb": 0,
    "yolo11-obb": 1,   # native dense re-implementation
    "std": 1,
    "lsknet": 1,
    "swin-obb": 2,
    "rvsa": 2,
    "ars-detr": 1,
    "ao2-detr": 1,
    "rio-detr": 2,
    "rhino": 2,
    "oriented-former": 2,
}

# Selection aliases accepted by every CLI that takes --models. A bare model
# name always refers to itself; these aliases expand to groups.
MODEL_ALIASES: Dict[str, List[str]] = {
    "all": list(CANONICAL_MODELS),
    "yolo": list(YOLO_SUITE_MODELS),
    "ultralytics": list(YOLO_SUITE_MODELS),
    "dense": list(DENSE_SUITE_MODELS),
    "detr": list(DETR_SUITE_MODELS),
    "yolov8": [m for m in YOLO_SUITE_MODELS if m.startswith("yolov8")],
    "yolo11": [m for m in YOLO_SUITE_MODELS if m.startswith("yolo11")],  # Ultralytics yolo11 family (native impl is 'yolo11-obb')
    "yolo26": [m for m in YOLO_SUITE_MODELS if m.startswith("yolo26")],
    "nano": [m for m in YOLO_SUITE_MODELS if m.endswith("n-obb")],
    "small": [m for m in YOLO_SUITE_MODELS if m.endswith("s-obb")],
    "medium": [m for m in YOLO_SUITE_MODELS if m.endswith("m-obb")],
    "large": [m for m in YOLO_SUITE_MODELS if m.endswith("l-obb")],
}


def resolve_model_names(names: List[str]) -> List[str]:
    """Expand aliases and validate model names against the canonical registry.

    Accepts any mix of bare model names and MODEL_ALIASES keys; returns a
    de-duplicated list in canonical registry order.
    """
    if not names:
        raise ValueError("no models selected")
    requested: List[str] = []
    unknown: List[str] = []
    for raw in names:
        key = str(raw).lower().strip()
        if key in MODEL_ALIASES:
            requested.extend(MODEL_ALIASES[key])
        elif key in CANONICAL_MODELS:
            requested.append(key)
        else:
            unknown.append(str(raw))
    if unknown:
        raise ValueError(
            f"unknown model(s): {unknown}. Choose from {CANONICAL_MODELS} "
            f"or aliases {sorted(MODEL_ALIASES)}"
        )
    return [m for m in CANONICAL_MODELS if m in set(requested)]


# Supported model architectures (v4: derived from the canonical registry;
# kept under the old name for backward compatibility with existing scripts).
SUPPORTED_MODELS: List[str] = list(CANONICAL_MODELS)

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
    local_nvme = Path("/content/data")
    if local_nvme.is_dir() and any(local_nvme.iterdir()):
        return local_nvme
    drive = Path("/content/drive/MyDrive/object-detection/cleaned_data")
    if drive.is_dir():
        return drive
    return PROJECT_ROOT / "data" / "cleaned_data"

