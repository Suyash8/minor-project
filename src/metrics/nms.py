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

from src.metrics.geometry import compute_obb_iou_matrix


def rotated_nms(boxes: np.ndarray, scores: np.ndarray,
                iou_thresh: float = 0.45,
                max_det: int = 300) -> np.ndarray:
    """Greedy rotated-NMS over (N, 5) cx,cy,w,h,angle_deg boxes.

    Returns the kept indices, highest confidence first. Deterministic for a
    fixed input ordering (ties broken by original index order).
    """
    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    if len(boxes) == 0:
        return np.zeros((0,), dtype=np.int64)

    order = np.argsort(-scores, kind="stable")
    keep = []
    suppressed = np.zeros(len(boxes), dtype=bool)

    for oi in order:
        if suppressed[oi]:
            continue
        keep.append(int(oi))
        if len(keep) >= max_det:
            break
        rest = order[~suppressed[order]]
        rest = rest[rest != oi]
        if len(rest) == 0:
            break
        ious = compute_obb_iou_matrix(boxes[oi:oi + 1], boxes[rest])[0]
        suppressed[rest[ious > iou_thresh]] = True

    return np.array(keep, dtype=np.int64)


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
    return boxes[keep], scores[keep], labels[keep]
