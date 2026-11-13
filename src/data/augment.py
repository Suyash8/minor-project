"""
OBB-aware image/label transforms for the aerial detection data layer (v4, W3).

Boxes travel through every transform as corner polygons (N, 4, 2) in pixel
coordinates and are re-derived to (cx, cy, w, h, angle_deg) only at the end,
so angles are never distorted by resampling math:

- letterbox (aspect-ratio-correct resize; replaces the old square stretch,
  which distorted OBB angles)
- random horizontal / vertical flip
- random 90/180/270 rotation (aerial imagery is rotation-symmetric)
- HSV jitter (OpenCV)
- random scale / translate (with box clipping + >=4 px min-size filter)

All functions are pure numpy/OpenCV and deterministic under a passed
numpy Generator, which keeps unit-test round-trips exact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:  # pragma: no cover - cv2 is a hard dep in production
    HAS_CV2 = False

from src.metrics.geometry import obb_to_corners, polygon_to_obb_params

MIN_BOX_PX = 4.0  # mirrors the cleaner's box filter (spec §2.2)


# ---------------------------------------------------------------------------
# corner <-> parameter conversion helpers
# ---------------------------------------------------------------------------
def boxes_to_corners(boxes: np.ndarray) -> np.ndarray:
    """(N, 5) cx,cy,w,h,angle_deg -> (N, 4, 2) corner polygons."""
    if len(boxes) == 0:
        return np.zeros((0, 4, 2), dtype=np.float32)
    return np.stack([obb_to_corners(*map(float, b)) for b in boxes]).astype(np.float32)


def corners_to_boxes(corners: np.ndarray, img_w: float, img_h: float,
                     clip: bool = True,
                     return_keep: bool = False):
    """(N, 4, 2) corners -> (M, 5) cx,cy,w,h,angle_deg, clipped to the image.

    Boxes whose fitted side is shorter than MIN_BOX_PX or whose centre falls
    outside the frame are dropped. With return_keep=True also returns the
    indices of surviving input polygons (for label alignment).
    """
    if len(corners) == 0:
        empty = np.zeros((0, 5), dtype=np.float32)
        return (empty, np.zeros((0,), dtype=np.int64)) if return_keep else empty
    pts = corners.astype(np.float64).copy()
    if clip:
        pts[..., 0] = np.clip(pts[..., 0], 0.0, max(img_w - 1e-3, 0.0))
        pts[..., 1] = np.clip(pts[..., 1], 0.0, max(img_h - 1e-3, 0.0))
    out = []
    keep = []
    for i, poly in enumerate(pts):
        cx, cy, w, h, ang = polygon_to_obb_params(poly)
        if w < MIN_BOX_PX or h < MIN_BOX_PX:
            continue
        if not (0.0 <= cx <= img_w and 0.0 <= cy <= img_h):
            continue
        out.append([cx, cy, w, h, ang])
        keep.append(i)
    keep_arr = np.array(keep, dtype=np.int64)
    if not out:
        empty = np.zeros((0, 5), dtype=np.float32)
        return (empty, keep_arr) if return_keep else empty
    boxes = np.array(out, dtype=np.float32)
    return (boxes, keep_arr) if return_keep else boxes


# ---------------------------------------------------------------------------
# geometric primitives (operate on corners)
# ---------------------------------------------------------------------------
def letterbox(img: np.ndarray, corners: np.ndarray, target: int,
              fill: int = 114) -> Tuple[np.ndarray, np.ndarray]:
    """Resize long side to `target`, pad to a square with neutral fill.

    Boxes are scaled and offset identically, preserving angles (the old
    square stretch did not).
    """
    h, w = img.shape[:2]
    scale = target / float(max(h, w))
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    if HAS_CV2:
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    else:  # pragma: no cover
        from PIL import Image
        resized = np.array(Image.fromarray(img).resize((new_w, new_h)))
    canvas = np.full((target, target, 3), fill, dtype=np.uint8)
    pad_x = (target - new_w) // 2
    pad_y = (target - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    if len(corners):
        corners = corners * scale + np.array([pad_x, pad_y], dtype=np.float32)
    return canvas, corners.astype(np.float32)


def hflip(img: np.ndarray, corners: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    w = img.shape[1]
    out = np.ascontiguousarray(img[:, ::-1])
    if len(corners):
        corners = corners.copy()
        corners[..., 0] = w - corners[..., 0]
    return out, corners


def vflip(img: np.ndarray, corners: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    h = img.shape[0]
    out = np.ascontiguousarray(img[::-1, :])
    if len(corners):
        corners = corners.copy()
        corners[..., 1] = h - corners[..., 1]
    return out, corners


def rot90(img: np.ndarray, corners: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Rotate by k*90 degrees counter-clockwise (k in {1, 2, 3})."""
    k = k % 4
    if k == 0:
        return img, corners
    h, w = img.shape[:2]
    out = np.ascontiguousarray(np.rot90(img, k))
    if len(corners):
        corners = corners.copy()
        if k == 1:
            corners[..., 0], corners[..., 1] = corners[..., 1], w - corners[..., 0]
        elif k == 2:
            corners[..., 0] = w - corners[..., 0]
            corners[..., 1] = h - corners[..., 1]
        else:
            corners[..., 0], corners[..., 1] = h - corners[..., 1], corners[..., 0]
    return out, corners


def hsv_jitter(img: np.ndarray, h_delta: float, s_gain: float, v_gain: float,
               rng: np.random.Generator) -> np.ndarray:
    """Random HSV jitter; no-op gains/deltas near zero are skipped."""
    if not HAS_CV2 or (h_delta <= 0 and s_gain <= 0 and v_gain <= 0):
        return img
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-h_delta, h_delta) * 180.0) % 180.0
    hsv[..., 1] *= rng.uniform(1 - s_gain, 1 + s_gain)
    hsv[..., 2] *= rng.uniform(1 - v_gain, 1 + v_gain)
    hsv[..., 1:] = np.clip(hsv[..., 1:], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def scale_translate(img: np.ndarray, corners: np.ndarray,
                    scale_range: float, translate_frac: float,
                    rng: np.random.Generator,
                    fill: int = 114) -> Tuple[np.ndarray, np.ndarray]:
    """Random zoom (1 +/- scale_range) and shift (+/- translate_frac of size)."""
    h, w = img.shape[:2]
    s = float(rng.uniform(1 - scale_range, 1 + scale_range))
    dx = float(rng.uniform(-translate_frac, translate_frac) * w)
    dy = float(rng.uniform(-translate_frac, translate_frac) * h)
    if HAS_CV2:
        mat = np.array([[s, 0, dx], [0, s, dy]], dtype=np.float32)
        out = cv2.warpAffine(img, mat, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(fill, fill, fill))
    else:  # pragma: no cover
        out = img
    if len(corners):
        corners = corners * s + np.array([dx, dy], dtype=np.float32)
    return out, corners.astype(np.float32)


# ---------------------------------------------------------------------------
# configurable pipeline
# ---------------------------------------------------------------------------
@dataclass
class AugmentConfig:
    """Probabilistic OBB-aware augmentation schedule (probabilities per image)."""
    hflip_p: float = 0.5
    vflip_p: float = 0.1
    rot90_p: float = 0.3          # probability of any 90/180/270 rotation
    hsv_h: float = 0.015
    hsv_s: float = 0.5
    hsv_v: float = 0.4
    scale_range: float = 0.2
    translate_frac: float = 0.1
    fill: int = 114

    @classmethod
    def dense_default(cls) -> "AugmentConfig":
        return cls()

    @classmethod
    def detr_light(cls) -> "AugmentConfig":
        """Lighter schedule for query-based models (matching is sensitive to
        heavy geometric noise)."""
        return cls(vflip_p=0.0, rot90_p=0.15, hsv_s=0.3, hsv_v=0.3,
                   scale_range=0.1, translate_frac=0.05)


def apply_augmentations(img: np.ndarray, corners: np.ndarray,
                        cfg: AugmentConfig,
                        rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Apply the configured random transforms to image + corner polygons."""
    if rng.random() < cfg.hflip_p:
        img, corners = hflip(img, corners)
    if rng.random() < cfg.vflip_p:
        img, corners = vflip(img, corners)
    if rng.random() < cfg.rot90_p:
        img, corners = rot90(img, corners, int(rng.integers(1, 4)))
    img = hsv_jitter(img, cfg.hsv_h, cfg.hsv_s, cfg.hsv_v, rng)
    if cfg.scale_range > 0 or cfg.translate_frac > 0:
        img, corners = scale_translate(img, corners, cfg.scale_range,
                                       cfg.translate_frac, rng, fill=cfg.fill)
    return img, corners
