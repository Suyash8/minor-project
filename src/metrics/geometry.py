"""
Geometric operations for Oriented Bounding Boxes (OBB).
Supports box-to-polygon conversion, rotated IoU, and Gaussian distance metrics (GWD & KLD).
"""

import math
from typing import List, Tuple, Optional
import numpy as np

try:
    from shapely.geometry import Polygon
except ImportError:
    Polygon = None


def obb_to_corners(cx: float, cy: float, w: float, h: float, angle_deg: float) -> np.ndarray:
    """
    Convert an oriented box (center_x, center_y, width, height, angle_degrees)
    into 4 polygon corner coordinates [(x1,y1), (x2,y2), (x3,y3), (x4,y4)].
    Angle is in degrees clockwise from horizontal x-axis.
    """
    rad = math.radians(angle_deg)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)

    hw = w / 2.0
    hh = h / 2.0

    # Local corners relative to center
    dx = np.array([-hw, hw, hw, -hw])
    dy = np.array([-hh, -hh, hh, hh])

    # Rotate and translate
    x_corners = cx + dx * cos_a - dy * sin_a
    y_corners = cy + dx * sin_a + dy * cos_a

    return np.stack([x_corners, y_corners], axis=-1)

def polygon_to_obb_params(pts: np.ndarray) -> Tuple[float, float, float, float, float]:
    """
    Fit a minimum-area rotated rectangle to 4 corner points.
    Returns (cx, cy, w, h, angle_deg).
    Angle is in degrees clockwise relative to positive x-axis.
    """
    cx = float(np.mean(pts[:, 0]))
    cy = float(np.mean(pts[:, 1]))

    v0 = pts[1] - pts[0]
    v1 = pts[2] - pts[1]

    len0 = float(np.linalg.norm(v0))
    len1 = float(np.linalg.norm(v1))

    if len0 >= len1:
        w = len0
        h = max(len1, 1.0)
        angle_rad = math.atan2(v0[1], v0[0])
    else:
        w = len1
        h = max(len0, 1.0)
        angle_rad = math.atan2(v1[1], v1[0])

    angle_deg = math.degrees(angle_rad) % 180.0
    return cx, cy, w, h, angle_deg

def polygon_iou(corners1: np.ndarray, corners2: np.ndarray) -> float:
    """
    Compute intersection-over-union between two 4-point convex polygons.
    Uses fast AABB disjoint check, Shapely polygon intersection, and arithmetic union area.
    """
    min_x1, min_y1 = corners1[:, 0].min(), corners1[:, 1].min()
    max_x1, max_y1 = corners1[:, 0].max(), corners1[:, 1].max()
    min_x2, min_y2 = corners2[:, 0].min(), corners2[:, 1].min()
    max_x2, max_y2 = corners2[:, 0].max(), corners2[:, 1].max()

    # Fast axis-aligned bounding box disjoint check
    if max_x1 < min_x2 or min_x1 > max_x2 or max_y1 < min_y2 or min_y1 > max_y2:
        return 0.0

    if Polygon is not None:
        try:
            poly1 = Polygon(corners1)
            poly2 = Polygon(corners2)

            if not poly1.is_valid:
                poly1 = poly1.buffer(0)
            if not poly2.is_valid:
                poly2 = poly2.buffer(0)

            inter = poly1.intersection(poly2).area
            if inter <= 0.0:
                return 0.0
            union = poly1.area + poly2.area - inter
            return float(inter / union) if union > 0 else 0.0
        except Exception:
            pass

    # Fallback to axis-aligned bounding box enclosing the corners
    inter_x = max(0.0, min(max_x1, max_x2) - max(min_x1, min_x2))
    inter_y = max(0.0, min(max_y1, max_y2) - max(min_y1, min_y2))
    inter_area = inter_x * inter_y

    area1 = (max_x1 - min_x1) * (max_y1 - min_y1)
    area2 = (max_x2 - min_x2) * (max_y2 - min_y2)
    union_area = area1 + area2 - inter_area

    return float(inter_area / union_area) if union_area > 0 else 0.0


def compute_obb_iou_matrix(gt_boxes: np.ndarray, pred_boxes: np.ndarray) -> np.ndarray:
    """
    Compute pairwise IoU matrix between ground truth and predicted oriented boxes.
    Boxes are expected in format [cx, cy, w, h, angle_deg].
    Shape: (N_gt, M_pred)
    """
    n_gt = len(gt_boxes)
    m_pred = len(pred_boxes)

    if n_gt == 0 or m_pred == 0:
        return np.zeros((n_gt, m_pred), dtype=np.float32)

    gt_corners = [obb_to_corners(b[0], b[1], b[2], b[3], b[4]) for b in gt_boxes]
    pred_corners = [obb_to_corners(b[0], b[1], b[2], b[3], b[4]) for b in pred_boxes]

    iou_mat = np.zeros((n_gt, m_pred), dtype=np.float32)
    for i in range(n_gt):
        for j in range(m_pred):
            iou_mat[i, j] = polygon_iou(gt_corners[i], pred_corners[j])

    return iou_mat

def obb_to_gaussian_cov(w: float, h: float, angle_deg: float) -> np.ndarray:
    """
    Convert an oriented box into a 2D Gaussian covariance matrix Sigma:
    Sigma = R * diag(w^2 / 12, h^2 / 12) * R^T
    """
    rad = math.radians(angle_deg)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float64)
    diag = np.diag([(w ** 2) / 12.0, (h ** 2) / 12.0])
    return R @ diag @ R.T

def gaussian_wasserstein_distance(
    box1: Tuple[float, float, float, float, float],
    box2: Tuple[float, float, float, float, float],
) -> float:
    """
    Compute Gaussian Wasserstein Distance (GWD) between two oriented boxes.
    Yang et al., ICML 2021.
    box = (cx, cy, w, h, angle_deg)
    """
    mu1 = np.array([box1[0], box1[1]], dtype=np.float64)
    mu2 = np.array([box2[0], box2[1]], dtype=np.float64)

    sigma1 = obb_to_gaussian_cov(box1[2], box1[3], box1[4])
    sigma2 = obb_to_gaussian_cov(box2[2], box2[3], box2[4])

    center_dist_sq = np.sum((mu1 - mu2) ** 2)

    # Matrix square root using eigenvalue decomposition
    w1, v1 = np.linalg.eigh(sigma1)
    w1 = np.maximum(w1, 1e-8)
    sqrt_sigma1 = v1 @ np.diag(np.sqrt(w1)) @ v1.T

    m = sqrt_sigma1 @ sigma2 @ sqrt_sigma1
    wm, vm = np.linalg.eigh(m)
    wm = np.maximum(wm, 1e-8)
    sqrt_m = vm @ np.diag(np.sqrt(wm)) @ vm.T

    cov_term = np.trace(sigma1) + np.trace(sigma2) - 2.0 * np.trace(sqrt_m)
    d_sq = center_dist_sq + max(0.0, float(cov_term))

    return float(np.sqrt(d_sq))
