"""
Shared rotated-NMS for every OBB engine (v4, W5.4).

One implementation, used by the dense, DETR and custom inference paths and
by the unified evaluator. IoU comes from src/metrics/geometry.py (shapely
polygon intersection), so training-time validation and final evaluation use
exactly the same geometry as the metric protocol.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from src.metrics.geometry import compute_obb_iou_matrix, obb_to_corners, polygon_iou


def rotated_nms(boxes: np.ndarray, scores: np.ndarray,
                iou_thresh: float = 0.45,
                max_det: int = 300) -> np.ndarray:
    """Greedy rotated-NMS over (N, 5) cx,cy,w,h,angle_deg boxes.

    Returns the kept indices, highest confidence first. Deterministic for a
    fixed input ordering (ties broken by original index order).
    Uses vectorized AABB filtering and strict geometric upper-bounding so that
    pairwise Shapely polygon intersections are only evaluated on actual overlaps.
    """
    boxes, scores = np.asarray(boxes, np.float32), np.asarray(scores, np.float32)
    order = np.argsort(-scores, kind="stable")
    if not len(order) or max_det <= 0:
        return np.zeros(0, dtype=np.int64)
    overlaps = compute_obb_iou_matrix(boxes[order], boxes[order], device="cpu") > iou_thresh
    suppressed = np.zeros(len(order), dtype=bool)
    keep = []
    for i in range(len(order)):
        if not suppressed[i]:
            keep.append(order[i])
            if len(keep) == max_det:
                break
            suppressed[i+1:] |= overlaps[i, i+1:]
    return np.asarray(keep, dtype=np.int64)


def nms_per_class(boxes: np.ndarray, scores: np.ndarray, labels: np.ndarray,
                  iou_thresh: float = 0.45,
                  max_det: int = 300) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """rotated_nms applied independently per class; returns filtered arrays."""
    if len(boxes) == 0:
        return (np.zeros((0, 5), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.int64))
    keep_all = []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        keep_all.append(idx[rotated_nms(boxes[idx], scores[idx],
                                        iou_thresh=iou_thresh, max_det=max_det)])
    keep = np.concatenate(keep_all)
    keep = keep[np.argsort(-scores[keep], kind="stable")]
    keep = keep[:max_det]
    return boxes[keep], scores[keep], labels[keep]
