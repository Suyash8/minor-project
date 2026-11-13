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
    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    N = len(boxes)
    if N == 0:
        return np.zeros((0,), dtype=np.int64)

    # Precompute corners and bounding geometry once for all candidate boxes
    corners = [obb_to_corners(b[0], b[1], b[2], b[3], b[4]) for b in boxes]
    areas = boxes[:, 2] * boxes[:, 3]
    mins = np.array([[c[:, 0].min(), c[:, 1].min()] for c in corners], dtype=np.float32)
    maxs = np.array([[c[:, 0].max(), c[:, 1].max()] for c in corners], dtype=np.float32)

    order = np.argsort(-scores, kind="stable")
    keep = []
    suppressed = np.zeros(N, dtype=bool)

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

        # Fast vectorized AABB intersection with all remaining candidates
        inter_x = np.maximum(0.0, np.minimum(maxs[oi, 0], maxs[rest, 0]) - np.maximum(mins[oi, 0], mins[rest, 0]))
        inter_y = np.maximum(0.0, np.minimum(maxs[oi, 1], maxs[rest, 1]) - np.maximum(mins[oi, 1], mins[rest, 1]))
        inter_aabb = inter_x * inter_y

        # Strict mathematical upper bound: IoU_rot <= inter_AABB / (area_oi + area_rest - inter_AABB)
        union_bound = areas[oi] + areas[rest] - inter_aabb
        upper_bound_iou = np.where(union_bound > 0, inter_aabb / np.maximum(union_bound, 1e-6), 0.0)

        # Only evaluate polygon IoU on candidates that can geometrically meet iou_thresh
        cand_mask = upper_bound_iou >= iou_thresh
        cand_rest = rest[cand_mask]

        for r_idx in cand_rest:
            iou = polygon_iou(corners[oi], corners[r_idx])
            if iou > iou_thresh:
                suppressed[r_idx] = True

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
