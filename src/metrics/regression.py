"""
Continuous Regression & Correlation Metrics for Oriented Bounding Boxes:
Angle MAE, RMSE, Pearson Correlation (r), Spearman Rank Correlation (rho),
R² Score, Center Offset Error, and Aspect Ratio Fit.
"""

from __future__ import annotations

from typing import Dict, Any, Tuple
import numpy as np
from scipy import stats

def compute_r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Coefficient of Determination (R² Score).
    R² = 1 - SS_res / SS_tot
    """
    if len(y_true) < 2:
        return 0.0
    residuals = y_true - y_pred
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot < 1e-12:
        return 1.0 if ss_res < 1e-12 else 0.0
    return float(1.0 - (ss_res / ss_tot))

def compute_pearson_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Pearson correlation coefficient (r).
    """
    if len(y_true) < 2 or np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
        return 0.0
    r, _ = stats.pearsonr(y_true, y_pred)
    return float(r) if not np.isnan(r) else 0.0

def compute_spearman_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Spearman rank correlation (rho).
    """
    if len(y_true) < 2 or np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
        return 0.0
    rho, _ = stats.spearmanr(y_true, y_pred)
    return float(rho) if not np.isnan(rho) else 0.0

def compute_angular_differences(angles_true: np.ndarray, angles_pred: np.ndarray, period: float = 180.0) -> np.ndarray:
    """
    Compute periodic angular difference accounting for OBB rotational symmetry.
    diff = min(|t - p| % period, period - (|t - p| % period))
    """
    diff = np.abs(angles_true - angles_pred) % period
    diff = np.minimum(diff, period - diff)
    return diff

def compute_regression_metrics(
    matched_gt_boxes: np.ndarray,
    matched_pred_boxes: np.ndarray,
) -> Dict[str, Any]:
    """
    Compute regression and correlation metrics on matched (TP) detection pairs.
    Boxes format: [cx, cy, w, h, angle_deg]
    """
    if len(matched_gt_boxes) == 0:
        return {
            "num_matched": 0,
            "angle_mae": float("nan"),
            "angle_rmse": float("nan"),
            "angle_pearson_r": float("nan"),
            "angle_spearman_rho": float("nan"),
            "angle_r2": float("nan"),
            "center_offset_mae": float("nan"),
            "center_offset_rmse": float("nan"),
            "aspect_ratio_r2": float("nan"),
            "aspect_ratio_pearson_r": float("nan"),
        }

    gt_angles = matched_gt_boxes[:, 4]
    pred_angles = matched_pred_boxes[:, 4]

    # Periodic angular errors
    angle_diffs = compute_angular_differences(gt_angles, pred_angles, period=180.0)
    angle_mae = float(np.mean(angle_diffs))
    angle_rmse = float(np.sqrt(np.mean(angle_diffs ** 2)))

    # Direct correlation & R²
    angle_r = compute_pearson_correlation(gt_angles, pred_angles)
    angle_rho = compute_spearman_correlation(gt_angles, pred_angles)
    angle_r2 = compute_r2_score(gt_angles, pred_angles)

    # Center position errors (pixels)
    dx = matched_gt_boxes[:, 0] - matched_pred_boxes[:, 0]
    dy = matched_gt_boxes[:, 1] - matched_pred_boxes[:, 1]
    center_dists = np.sqrt(dx ** 2 + dy ** 2)
    center_mae = float(np.mean(center_dists))
    center_rmse = float(np.sqrt(np.mean(center_dists ** 2)))

    # Aspect ratio regression
    gt_ar = matched_gt_boxes[:, 2] / np.maximum(matched_gt_boxes[:, 3], 1e-4)
    pred_ar = matched_pred_boxes[:, 2] / np.maximum(matched_pred_boxes[:, 3], 1e-4)
    ar_r2 = compute_r2_score(gt_ar, pred_ar)
    ar_r = compute_pearson_correlation(gt_ar, pred_ar)

    return {
        "num_matched": len(matched_gt_boxes),
        "angle_mae": angle_mae,
        "angle_rmse": angle_rmse,
        "angle_pearson_r": angle_r,
        "angle_spearman_rho": angle_rho,
        "angle_r2": angle_r2,
        "center_offset_mae": center_mae,
        "center_offset_rmse": center_rmse,
        "aspect_ratio_r2": ar_r2,
        "aspect_ratio_pearson_r": ar_r,
    }
