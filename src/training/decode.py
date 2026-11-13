"""
Shared output decoding for the dense, DETR and custom engines (v4, W5.3/5.4).

Every engine decodes to ONE OBB convention — (cx, cy, w, h, angle_deg) with
angle in degrees in [0, 180), OpenCV-style — in the coordinate frame of the
network input image (img_size x img_size). The metric protocol
(src/metrics/) consumes exactly this convention.

Preserves two hard-won v3 fixes:
- sin(2theta)/cos(2theta) heads are halved after atan2 (commit 4b514c0),
- the dense loss uses a pos-weighted objectness BCE (commit a447499); the
  decode side mirrors the same scale priors (stride*3.0 / stride*6.0) used by
  src/training/train_dense.py::encode_dense_targets.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from src.metrics.nms import nms_per_class

# scale priors shared with encode_dense_targets (train_dense.py)
W_PRIOR = 3.0
H_PRIOR = 6.0
# Minimum box/prior ratio before log-space saturation (review per-model
# coverage): 0.1 floored coarse-grid models (swin-obb at stride 64 has a
# 384 px height prior, so any object under ~38 px saturated and decoded to
# the wrong size). 0.02 keeps tiny aerial objects representable while the
# log stays finite. The encoder clamp and the decoder clip MUST use the
# same bound or the rectangle changes between train and inference.
MIN_PRIOR_FRAC = 0.02


def decode_dense_grid(
    cls_logits: torch.Tensor,
    reg_preds: torch.Tensor,
    img_size: int,
    conf_thresh: float = 0.05,
    topk_per_class: int = 200,
    nms_iou: float = 0.45,
    max_det: int = 1000,  # 300 capped recall on dense scenes (review B)
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Decode dense-grid outputs into per-image (boxes, scores, labels).

    Args:
        cls_logits: (B, C, H, W) raw logits (sigmoid applied internally).
        reg_preds:  (B, 7, H, W) [dx, dy, log_w, log_h, sin2a, cos2a, obj_logit].
        img_size:   network input size (square) the grid maps back to.

    Returns:
        list over the batch of (boxes (N,5) cx,cy,w,h,angle_deg in pixels of
        the img_size input, scores (N,), labels (N,)) after per-class
        rotated NMS.
    """
    B, C, H, W = cls_logits.shape
    stride_x = img_size / float(W)
    stride_y = img_size / float(H)
    cls_probs = torch.sigmoid(cls_logits)
    obj_probs = torch.sigmoid(reg_preds[:, 6])

    out = []
    for b in range(B):
        score_map = cls_probs[b] * obj_probs[b].unsqueeze(0)  # (C, H, W)
        boxes_all, scores_all, labels_all = [], [], []
        for c in range(C):
            mask = score_map[c] > conf_thresh
            if not bool(mask.any()):
                continue
            ys, xs = torch.where(mask)
            sc = score_map[c][ys, xs]
            if sc.numel() > topk_per_class:
                sc, topi = torch.topk(sc, topk_per_class)
                ys, xs = ys[topi], xs[topi]
            reg = reg_preds[b, :, ys, xs].float().cpu().numpy()  # (7, K)
            sc_np = sc.float().cpu().numpy()
            xs_np = xs.float().cpu().numpy()
            ys_np = ys.float().cpu().numpy()

            cx = (xs_np + 0.5 + reg[0]) * stride_x
            cy = (ys_np + 0.5 + reg[1]) * stride_y
            bw = np.exp(np.clip(reg[2], math.log(MIN_PRIOR_FRAC), 3.0)) * stride_x * W_PRIOR
            bh = np.exp(np.clip(reg[3], math.log(MIN_PRIOR_FRAC), 3.0)) * stride_y * H_PRIOR
            # atan2-halving fix (4b514c0): heads emit sin(2a)/cos(2a).
            ang = np.degrees(np.arctan2(reg[4], reg[5]) / 2.0) % 180.0

            boxes = np.stack([cx, cy, bw, bh, ang], axis=1).astype(np.float32)
            boxes_all.append(boxes)
            scores_all.append(sc_np.astype(np.float32))
            labels_all.append(np.full(len(sc_np), c, dtype=np.int64))

        if boxes_all:
            boxes = np.concatenate(boxes_all)
            scores = np.concatenate(scores_all)
            labels = np.concatenate(labels_all)
            boxes, scores, labels = nms_per_class(boxes, scores, labels,
                                                  iou_thresh=nms_iou,
                                                  max_det=max_det)
        else:
            boxes = np.zeros((0, 5), dtype=np.float32)
            scores = np.zeros((0,), dtype=np.float32)
            labels = np.zeros((0,), dtype=np.int64)
        out.append((boxes, scores, labels))
    return out


def decode_detr(
    cls_logits: torch.Tensor,
    boxes_norm: torch.Tensor,
    angle_rad: torch.Tensor,
    img_size: int,
    num_classes: int,
    conf_thresh: float = 0.05,
    nms_iou: float = 0.45,
    max_det: int = 1000,  # 300 capped recall on dense scenes (review B)
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Decode DETR query outputs (normalized cx,cy,w,h + angle in radians)
    into the shared pixel convention. The angle is used DIRECTLY
    (degrees(rad) % 180): the DETR loss supervises sin(2a)/cos(2a) of this
    angle; sin(2a)/cos(2a) constrain it modulo 180, which is exactly the
    suite's angle range, so no halving is applied here (docstring corrected
    per review C)."""
    B, Q, _ = cls_logits.shape
    probs = torch.softmax(cls_logits, dim=-1)[..., :num_classes]  # drop background
    out = []
    for b in range(B):
        pb = probs[b].float().cpu().numpy()          # (Q, C)
        bn = boxes_norm[b].float().cpu().numpy()     # (Q, 4)
        ar = angle_rad[b].float().cpu().numpy().reshape(-1)  # (Q,)
        sc = pb.max(axis=1)
        lb = pb.argmax(axis=1)
        keep = sc > conf_thresh
        if not bool(keep.any()):
            out.append((np.zeros((0, 5), dtype=np.float32),
                        np.zeros((0,), dtype=np.float32),
                        np.zeros((0,), dtype=np.int64)))
            continue
        bn_k = bn[keep]
        ang = np.degrees(ar[keep]) % 180.0
        boxes = np.stack([bn_k[:, 0] * img_size, bn_k[:, 1] * img_size,
                          bn_k[:, 2] * img_size, bn_k[:, 3] * img_size,
                          ang], axis=1).astype(np.float32)
        boxes, scores, labels = nms_per_class(
            boxes, sc[keep].astype(np.float32), lb[keep].astype(np.int64),
            iou_thresh=nms_iou, max_det=max_det)
        out.append((boxes, scores, labels))
    return out


def to_metric_format(
    decoded: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    num_classes: int,
) -> List[List[Dict[str, np.ndarray]]]:
    """Convert decode output to the src/metrics per-class contract:
    out[img][cls] = {'boxes': (N, 5), 'scores': (N,)}."""
    out = []
    for boxes, scores, labels in decoded:
        img = []
        for c in range(num_classes):
            m = labels == c
            img.append({
                "boxes": boxes[m] if m.any() else np.zeros((0, 5), dtype=np.float32),
                "scores": scores[m] if m.any() else np.zeros((0,), dtype=np.float32),
            })
        out.append(img)
    return out


# ---------------------------------------------------------------------------
# Round-trip support for the W5.3 unit test: encode random boxes through the
# training encoder, decode, and require IoU >= 0.99.
# ---------------------------------------------------------------------------
def encode_dense_sample(
    boxes: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    feat_h: int,
    feat_w: int,
    img_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Single-image twin of train_dense.encode_dense_targets (for tests and
    for the custom engine's grid targets)."""
    stride_x = img_size / float(feat_w)
    stride_y = img_size / float(feat_h)
    cls_t = torch.zeros((num_classes, feat_h, feat_w))
    reg_t = torch.zeros((7, feat_h, feat_w))
    for box, lbl in zip(boxes, labels):
        cx, cy, w, h, ang = [float(v) for v in box[:5]]
        gx = int(cx / stride_x)
        gy = int(cy / stride_y)
        if not (0 <= gx < feat_w and 0 <= gy < feat_h and 0 <= int(lbl) < num_classes):
            continue
        cls_t[int(lbl), gy, gx] = 1.0
        reg_t[:, gy, gx] = torch.tensor([
            (cx / stride_x) - (gx + 0.5),
            (cy / stride_y) - (gy + 0.5),
            math.log(max(w / (stride_x * W_PRIOR), MIN_PRIOR_FRAC)),
            math.log(max(h / (stride_y * H_PRIOR), MIN_PRIOR_FRAC)),
            math.sin(2.0 * math.radians(ang)),
            math.cos(2.0 * math.radians(ang)),
            1.0,
        ])
    return cls_t, reg_t
