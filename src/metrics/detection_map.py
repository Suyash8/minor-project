"""
Mean Average Precision (mAP) computation for Oriented Bounding Box detection.
Computes mAP@0.50, mAP@0.75, mAP@0.50:0.95, and scale-specific AP (AP_s, AP_m, AP_l).
"""

from __future__ import annotations

from typing import List, Dict, Any, Tuple
import numpy as np
from src.metrics.geometry import compute_obb_iou_matrix

def compute_ap_from_pr(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """
    Compute 101-point interpolated Average Precision (COCO standard).
    """
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    # Ensure precision is monotonically decreasing
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    # 101-point interpolation
    recall_thresholds = np.linspace(0.0, 1.0, 101)
    inds = np.searchsorted(mrec, recall_thresholds, side="left")
    inds = np.clip(inds, 0, len(mpre) - 1)
    ap = float(np.mean(mpre[inds]))
    return ap

def evaluate_class_multi_iou(
    gt_list: List[Dict[str, Any]],
    pred_list: List[Dict[str, Any]],
    iou_thresholds: List[float],
    scale_filter: Tuple[float, float] = (0.0, float("inf")),
) -> Dict[float, Dict[str, Any]]:
    """
    Evaluate detection for a single class across multiple IoU thresholds in a
    single geometry pass. Pairwise box IoUs are computed exactly once per prediction,
    avoiding redundant O(N*M) Shapely polygon intersections.
    """
    n_gt = 0
    filtered_gt_per_img = []

    for img_gt in gt_list:
        boxes = img_gt.get("boxes", np.zeros((0, 5)))
        areas = boxes[:, 2] * boxes[:, 3] if len(boxes) > 0 else np.zeros(0)
        mask = (areas >= scale_filter[0]) & (areas < scale_filter[1])
        valid_boxes = boxes[mask]
        n_gt += len(valid_boxes)
        filtered_gt_per_img.append(valid_boxes)

    all_preds = []
    for img_idx, img_pred in enumerate(pred_list):
        boxes = img_pred.get("boxes", np.zeros((0, 5)))
        scores = img_pred.get("scores", np.zeros(0))
        areas = boxes[:, 2] * boxes[:, 3] if len(boxes) > 0 else np.zeros(0)
        mask = (areas >= scale_filter[0]) & (areas < scale_filter[1])

        for b, s in zip(boxes[mask], scores[mask]):
            all_preds.append((float(s), img_idx, b))

    results: Dict[float, Dict[str, Any]] = {}
    if n_gt == 0:
        empty = {"ap": 0.0, "precision": 0.0, "recall": 0.0, "tp": 0, "fp": len(all_preds), "fn": 0}
        return {th: dict(empty) for th in iou_thresholds}
    if len(all_preds) == 0:
        empty = {"ap": 0.0, "precision": 0.0, "recall": 0.0, "tp": 0, "fp": 0, "fn": n_gt}
        return {th: dict(empty) for th in iou_thresholds}

    all_preds.sort(key=lambda x: x[0], reverse=True)

    # Match each prediction to best ground-truth box once
    pred_matches = []
    for score, img_idx, p_box in all_preds:
        gt_boxes = filtered_gt_per_img[img_idx]
        if len(gt_boxes) == 0:
            pred_matches.append((img_idx, -1, 0.0))
        else:
            ious = compute_obb_iou_matrix(gt_boxes, np.expand_dims(p_box, 0))[:, 0]
            best_gt_idx = int(np.argmax(ious))
            best_iou = float(ious[best_gt_idx])
            pred_matches.append((img_idx, best_gt_idx, best_iou))

    for th in iou_thresholds:
        gt_matched_per_img = [np.zeros(len(b), dtype=bool) for b in filtered_gt_per_img]
        tp = np.zeros(len(all_preds))
        fp = np.zeros(len(all_preds))

        for p_idx, (img_idx, best_gt_idx, best_iou) in enumerate(pred_matches):
            if best_gt_idx >= 0 and best_iou >= th and not gt_matched_per_img[img_idx][best_gt_idx]:
                tp[p_idx] = 1.0
                gt_matched_per_img[img_idx][best_gt_idx] = True
            else:
                fp[p_idx] = 1.0

        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        recalls = cum_tp / float(n_gt)
        precisions = cum_tp / (cum_tp + cum_fp + 1e-12)

        ap = compute_ap_from_pr(recalls, precisions)
        total_tp = int(cum_tp[-1])
        total_fp = int(cum_fp[-1])
        total_fn = n_gt - total_tp

        results[th] = {
            "ap": ap,
            "precision": float(precisions[-1]) if len(precisions) > 0 else 0.0,
            "recall": float(recalls[-1]) if len(recalls) > 0 else 0.0,
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
        }

    return results


def evaluate_class_detection(
    gt_list: List[Dict[str, Any]],
    pred_list: List[Dict[str, Any]],
    iou_threshold: float = 0.50,
    scale_filter: Tuple[float, float] = (0.0, float("inf")),
) -> Dict[str, Any]:
    """
    Evaluate detection for a single class at a specific IoU threshold.
    Filters GT and predictions by box area if scale_filter is provided.
    """
    return evaluate_class_multi_iou(gt_list, pred_list, [iou_threshold], scale_filter)[iou_threshold]


def compute_map_metrics(
    all_gt: List[List[Dict[str, Any]]],
    all_preds: List[List[Dict[str, Any]]],
    class_names: List[str],
) -> Dict[str, Any]:
    """
    Compute comprehensive mAP suite across classes and IoU thresholds.
    all_gt[img][cls_idx] = {'boxes': ndarray of [cx, cy, w, h, angle_deg]}
    all_preds[img][cls_idx] = {'boxes': ndarray, 'scores': ndarray}
    """
    num_classes = len(class_names)
    num_images = len(all_gt)

    range_ious = [round(x, 2) for x in list(float(i) / 100.0 for i in range(50, 100, 5))]

    ap50_per_class = []
    ap75_per_class = []
    range_ap_per_class = []

    ap_small_list = []
    ap_med_list = []
    ap_large_list = []
    classes_with_gt = []

    for c in range(num_classes):
        gt_cls = [all_gt[img][c] for img in range(num_images)]
        pred_cls = [all_preds[img][c] for img in range(num_images)]

        # Single-pass multi-threshold evaluation across range_ious (0.50 .. 0.95)
        multi_res = evaluate_class_multi_iou(gt_cls, pred_cls, range_ious)
        ap50_per_class.append(multi_res[0.50]["ap"])
        ap75_per_class.append(multi_res[0.75]["ap"])

        range_aps = [multi_res[iou_th]["ap"] for iou_th in range_ious]
        range_ap_per_class.append(float(np.mean(range_aps)))

        # Scale breakdowns at IoU 0.50
        ap_s = evaluate_class_detection(gt_cls, pred_cls, iou_threshold=0.50, scale_filter=(0, 32 ** 2))["ap"]
        ap_m = evaluate_class_detection(gt_cls, pred_cls, iou_threshold=0.50, scale_filter=(32 ** 2, 96 ** 2))["ap"]
        ap_l = evaluate_class_detection(gt_cls, pred_cls, iou_threshold=0.50, scale_filter=(96 ** 2, float("inf")))["ap"]

        ap_small_list.append(ap_s)
        ap_med_list.append(ap_m)
        ap_large_list.append(ap_l)

        n_gt_c = sum(len(g.get("boxes", [])) for g in gt_cls)
        if n_gt_c > 0:
            classes_with_gt.append(c)

    eval_indices = classes_with_gt if len(classes_with_gt) > 0 else list(range(num_classes))
    return {
        "map50": float(np.mean([ap50_per_class[i] for i in eval_indices])),
        "map75": float(np.mean([ap75_per_class[i] for i in eval_indices])),
        "map50_95": float(np.mean([range_ap_per_class[i] for i in eval_indices])),
        "ap_small": float(np.mean(ap_small_list)),
        "ap_medium": float(np.mean(ap_med_list)),
        "ap_large": float(np.mean(ap_large_list)),
        "per_class_ap50": {class_names[i]: ap50_per_class[i] for i in range(num_classes)},
        "per_class_ap75": {class_names[i]: ap75_per_class[i] for i in range(num_classes)},
    }
