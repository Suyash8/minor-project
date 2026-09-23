"""
Comprehensive evaluation metrics suite for Aerial OBB Detection.
"""

from __future__ import annotations

from .geometry import (
    obb_to_corners,
    polygon_to_obb_params,
    polygon_iou,
    compute_obb_iou_matrix,
    gaussian_wasserstein_distance,
)
from .detection_map import compute_map_metrics, evaluate_class_detection
from .confusion_matrix import compute_detection_confusion_matrix
from .classification import compute_classification_metrics
from .regression import (
    compute_regression_metrics,
    compute_r2_score,
    compute_pearson_correlation,
    compute_spearman_correlation,
)

__all__ = [
    "obb_to_corners",
    "polygon_to_obb_params",
    "polygon_iou",
    "compute_obb_iou_matrix",
    "gaussian_wasserstein_distance",
    "compute_map_metrics",
    "evaluate_class_detection",
    "compute_detection_confusion_matrix",
    "compute_classification_metrics",
    "compute_regression_metrics",
    "compute_r2_score",
    "compute_pearson_correlation",
    "compute_spearman_correlation",
]
