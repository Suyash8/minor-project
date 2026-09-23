"""
Multi-class Detection Confusion Matrix with Background (False Positive / False Negative) handling.
"""

from __future__ import annotations

from typing import List, Dict, Any, Tuple
import numpy as np
from src.metrics.geometry import compute_obb_iou_matrix

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

    num_images = len(all_gt)

    for img_idx in range(num_images):
        # Flatten all GT for this image
        gt_boxes = []
        gt_classes = []
        for c in range(k):
            boxes = all_gt[img_idx][c].get("boxes", np.zeros((0, 5)))
            for b in boxes:
                gt_boxes.append(b)
                gt_classes.append(c)

        # Flatten all predictions for this image
        pred_boxes = []
        pred_scores = []
        pred_classes = []
        for c in range(k):
            boxes = all_preds[img_idx][c].get("boxes", np.zeros((0, 5)))
            scores = all_preds[img_idx][c].get("scores", np.zeros(0))
            for b, s in zip(boxes, scores):
                pred_boxes.append(b)
                pred_scores.append(s)
                pred_classes.append(c)

        n_gt = len(gt_boxes)
        n_pred = len(pred_boxes)

        if n_gt == 0 and n_pred == 0:
            continue
        elif n_gt == 0:
            # All predictions are False Positives from background
            for pc in pred_classes:
                matrix[k, pc] += 1
            continue
        elif n_pred == 0:
            # All ground truths missed (False Negatives -> background)
            for gc in gt_classes:
                matrix[gc, k] += 1
            continue

        gt_boxes_arr = np.array(gt_boxes)
        pred_boxes_arr = np.array(pred_boxes)
        pred_scores_arr = np.array(pred_scores)

        # Sort predictions by score descending
        sort_order = np.argsort(-pred_scores_arr)
        pred_boxes_arr = pred_boxes_arr[sort_order]
        pred_classes = [pred_classes[i] for i in sort_order]

        iou_mat = compute_obb_iou_matrix(gt_boxes_arr, pred_boxes_arr)

        gt_matched = np.zeros(n_gt, dtype=bool)
        pred_matched = np.zeros(n_pred, dtype=bool)

        for p_idx in range(n_pred):
            p_cls = pred_classes[p_idx]
            ious = iou_mat[:, p_idx]
            best_gt_idx = int(np.argmax(ious))
            best_iou = float(ious[best_gt_idx])

            if best_iou >= iou_threshold and not gt_matched[best_gt_idx]:
                gt_cls = gt_classes[best_gt_idx]
                matrix[gt_cls, p_cls] += 1
                gt_matched[best_gt_idx] = True
                pred_matched[p_idx] = True
            elif best_iou >= iou_threshold and gt_matched[best_gt_idx]:
                # Duplicate prediction on same GT -> background FP
                matrix[k, p_cls] += 1
                pred_matched[p_idx] = True
            else:
                # No overlapping GT -> background FP
                matrix[k, p_cls] += 1

        # Any unmatched GT is a False Negative (missed -> background)
        for g_idx in range(n_gt):
            if not gt_matched[g_idx]:
                matrix[gt_classes[g_idx], k] += 1

    return matrix, labels
