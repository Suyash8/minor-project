"""
Multi-class Detection Confusion Matrix with Background (False Positive / False Negative) handling.
"""

from __future__ import annotations

from typing import List, Dict, Any, Tuple
import numpy as np
from src.metrics.geometry import compute_obb_iou_matrix

def update_confusion_matrix_image(
    matrix: np.ndarray,
    gt: List[Dict[str, Any]],
    pred: List[Dict[str, Any]],
    iou_threshold: float = 0.50,
) -> None:
    """Update a (K+1) x (K+1) confusion matrix for one image using unmatched-GT greedy search (F12).

    Prefers matching an unmatched same-class GT with highest IoU >= iou_threshold so
    diagonal TP counts stay consistent with AP matching in crowded scenes; falls back
    to the highest-IoU unmatched GT of any class to record cross-class confusion.
    """
    k = matrix.shape[0] - 1
    gt_boxes, gt_classes = [], []
    for c in range(k):
        for b in gt[c].get("boxes", np.zeros((0, 5))):
            gt_boxes.append(b)
            gt_classes.append(c)

    pred_boxes, pred_scores, pred_classes = [], [], []
    for c in range(k):
        boxes = pred[c].get("boxes", np.zeros((0, 5)))
        scores = pred[c].get("scores", np.zeros(0))
        for b, s in zip(boxes, scores):
            pred_boxes.append(b)
            pred_scores.append(float(s))
            pred_classes.append(c)

    n_gt, n_pred = len(gt_boxes), len(pred_boxes)
    if n_gt == 0 and n_pred == 0:
        return
    if n_gt == 0:
        for pc in pred_classes:
            matrix[k, pc] += 1
        return
    if n_pred == 0:
        for gc in gt_classes:
            matrix[gc, k] += 1
        return

    gt_arr = np.asarray(gt_boxes, dtype=np.float32).reshape(-1, 5)
    pred_arr = np.asarray(pred_boxes, dtype=np.float32).reshape(-1, 5)
    iou_mat = compute_obb_iou_matrix(gt_arr, pred_arr)
    order = np.argsort(-np.asarray(pred_scores, dtype=np.float64))
    gt_matched = np.zeros(n_gt, dtype=bool)
    gt_cls_arr = np.asarray(gt_classes, dtype=np.int64)

    for p_idx in order:
        p_cls = int(pred_classes[p_idx])
        ious = iou_mat[:, p_idx]
        unmatched_mask = ~gt_matched
        # 1) Prefer best UNMATCHED same-class GT (aligns TP/FP/FN with AP matching)
        same_avail = np.where(unmatched_mask & (gt_cls_arr == p_cls), ious, -1.0)
        best_same = int(same_avail.argmax()) if len(same_avail) else -1
        if best_same >= 0 and same_avail[best_same] >= iou_threshold:
            matrix[p_cls, p_cls] += 1
            gt_matched[best_same] = True
            continue
        # 2) Otherwise check best UNMATCHED cross-class GT for genuine class confusion
        any_avail = np.where(unmatched_mask, ious, -1.0)
        best_any = int(any_avail.argmax()) if len(any_avail) else -1
        if best_any >= 0 and any_avail[best_any] >= iou_threshold:
            matrix[int(gt_cls_arr[best_any]), p_cls] += 1
            gt_matched[best_any] = True
        else:
            matrix[k, p_cls] += 1

    for g_idx in range(n_gt):
        if not gt_matched[g_idx]:
            matrix[gt_classes[g_idx], k] += 1


def compute_detection_confusion_matrix(
    all_gt: List[List[Dict[str, Any]]],
    all_preds: List[List[Dict[str, Any]]],
    class_names: List[str],
    iou_threshold: float = 0.50,
) -> Tuple[np.ndarray, List[str]]:
    """
    Construct a (K+1) x (K+1) confusion matrix where class K is 'Background'.
    Row: Ground Truth category
    Col: Predicted category
    """
    k = len(class_names)
    matrix = np.zeros((k + 1, k + 1), dtype=np.int64)
    labels = list(class_names) + ["background"]

    for img_idx in range(len(all_gt)):
        update_confusion_matrix_image(matrix, all_gt[img_idx], all_preds[img_idx], iou_threshold=iou_threshold)

    return matrix, labels
