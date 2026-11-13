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
    from src.metrics.tensor_obb import nms_per_class_tensor
    B, C, H, W = cls_logits.shape
    sx, sy = img_size/float(W), img_size/float(H)
    out = []
    for b in range(B):
        boxes_all, scores_all, labels_all = [], [], []
        score_map = cls_logits[b].sigmoid()*reg_preds[b, 6].sigmoid()[None]
        for c in range(C):
            ys, xs = torch.where(score_map[c] > conf_thresh)
            sc = score_map[c, ys, xs]
            if not sc.numel():
                continue
            if sc.numel() > topk_per_class:
                sc, ids = sc.topk(topk_per_class)
                ys, xs = ys[ids], xs[ids]
            reg = reg_preds[b, :, ys, xs].float()
            boxes_all.append(torch.stack(((xs+.5+reg[0])*sx,
                (ys+.5+reg[1])*sy,
                reg[2].clamp(math.log(MIN_PRIOR_FRAC), 3).exp()*sx*W_PRIOR,
                reg[3].clamp(math.log(MIN_PRIOR_FRAC), 3).exp()*sy*H_PRIOR,
                torch.rad2deg(torch.atan2(reg[4], reg[5])*.5) % 180), 1))
            scores_all.append(sc.float())
            labels_all.append(torch.full_like(xs, c))
        if boxes_all:
            boxes, scores, labels = torch.cat(boxes_all), torch.cat(scores_all), torch.cat(labels_all)
            if boxes.is_cuda:
                boxes, scores, labels = nms_per_class_tensor(boxes, scores, labels, nms_iou, max_det)
                out.append(tuple(v.detach().cpu().numpy() for v in (boxes, scores, labels)))
            else:
                out.append(nms_per_class(*(v.detach().numpy() for v in (boxes, scores, labels)), nms_iou, max_det))
        else:
            out.append((np.zeros((0,5), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int64)))
    return out


def decode_detr(
    cls_logits: torch.Tensor,
    boxes_norm: torch.Tensor,
    angle_rad: torch.Tensor,
    img_size: int,
    num_classes: int,
    conf_thresh: float = 0.01,
    nms_iou: float = 0.45,
    max_det: int = 1000,  # 300 capped recall on dense scenes (review B)
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Decode DETR query outputs (normalized cx,cy,w,h + angle in radians)
    into the shared pixel convention. The angle is used DIRECTLY
    (degrees(rad) % 180): the DETR loss supervises sin(2a)/cos(2a) of this
    angle; sin(2a)/cos(2a) constrain it modulo 180, which is exactly the
    suite's angle range, so no halving is applied here (docstring corrected
    per review C)."""
    from src.metrics.tensor_obb import nms_per_class_tensor
    probs = cls_logits[..., :num_classes].sigmoid()
    out = []
    for b in range(len(cls_logits)):
        # Independent foreground scores; one query can support several classes.
        qi, labels = torch.where(probs[b] > conf_thresh)
        scores = probs[b, qi, labels].float()
        if len(scores) > max_det*4:
            scores, ids = scores.topk(max_det*4)
            qi, labels = qi[ids], labels[ids]
        bn = boxes_norm[b, qi].float()*img_size
        boxes = torch.cat((bn, torch.rad2deg(angle_rad[b, qi].float().reshape(-1, 1)) % 180), -1)
        if boxes.is_cuda:
            vals = nms_per_class_tensor(boxes, scores, labels, nms_iou, max_det)
            out.append(tuple(v.detach().cpu().numpy() for v in vals))
        else:
            out.append(nms_per_class(*(v.detach().numpy() for v in (boxes, scores, labels)), nms_iou, max_det))
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
