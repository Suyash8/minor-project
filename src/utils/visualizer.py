"""
Plotting and visual artifact generation for OBB detection evaluation.
Uses headless matplotlib (Agg backend) for reliable execution in scripts and Colab.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    output_path: Path,
    title: str = "Detection Confusion Matrix (Including Background)",
    normalize: bool = True,
) -> None:
    """
    Render and save a high-contrast confusion matrix heatmap.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        plot_data = np.divide(
            cm.astype(float), row_sums,
            out=np.zeros_like(cm, dtype=float),
            where=(row_sums > 0)
        )
        fmt = ".2f"
    else:
        plot_data = cm
        fmt = "d"

    fig, ax = plt.subplots(figsize=(max(8, len(class_names) * 0.8), max(7, len(class_names) * 0.7)), dpi=150)
    cax = ax.imshow(plot_data, interpolation="nearest", cmap="Blues")
    fig.colorbar(cax, fraction=0.046, pad=0.04)

    ax.set_title(title, fontsize=14, pad=12, fontweight="bold")
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)

    ax.set_ylabel("True Category", fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted Category", fontsize=11, fontweight="bold")

    # Annotate cells
    thresh = plot_data.max() / 2.0 if plot_data.max() > 0 else 0.5
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            val = plot_data[i, j]
            text = f"{val:.2f}" if normalize else f"{int(val)}"
            color = "white" if val > thresh else "black"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_angle_correlation(
    true_angles: np.ndarray,
    pred_angles: np.ndarray,
    output_path: Path,
    title: str = "Oriented Bounding Box Angle Correlation",
    r2_score: Optional[float] = None,
    pearson_r: Optional[float] = None,
    mae: Optional[float] = None,
) -> None:
    """
    Scatter plot comparing Ground Truth vs Predicted vehicle orientation angles,
    with an identity line, linear regression trendline, and summary stats.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 7), dpi=150)

    # Filter any NaNs
    valid_mask = ~np.isnan(true_angles) & ~np.isnan(pred_angles)
    x = true_angles[valid_mask]
    y = pred_angles[valid_mask]

    if len(x) == 0:
        x = np.array([0.0])
        y = np.array([0.0])

    ax.scatter(x, y, alpha=0.55, edgecolors="none", c="#1D4ED8", s=30, label="Matched Detections")

    # Diagonal ideal line (y = x)
    min_val = min(float(x.min()), float(y.min()))
    max_val = max(float(x.max()), float(y.max()))
    pad = (max_val - min_val) * 0.05 if max_val > min_val else 1.0
    line_range = np.linspace(min_val - pad, max_val + pad, 100)
    ax.plot(line_range, line_range, "k--", alpha=0.6, label="Ideal Perfect Fit (y = x)")

    # Linear trendline
    if len(x) > 1 and np.std(x) > 1e-4:
        slope, intercept = np.polyfit(x, y, 1)
        ax.plot(line_range, slope * line_range + intercept, color="#DC2626", linewidth=1.8, label=f"Trendline (slope={slope:.2f})")

    stats_text = []
    if pearson_r is not None:
        stats_text.append(f"Pearson r: {pearson_r:.4f}")
    if r2_score is not None:
        stats_text.append(f"R² Score: {r2_score:.4f}")
    if mae is not None:
        stats_text.append(f"Angle MAE: {mae:.2f}°")

    if stats_text:
        ax.text(
            0.05, 0.93, "\n".join(stats_text),
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="square,pad=0.5", facecolor="#F8FAFC", edgecolor="#CBD5E1", alpha=0.9),
        )

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Ground Truth Orientation Angle (°)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Predicted Orientation Angle (°)", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="lower right", framealpha=0.85)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_benchmark_comparison(
    benchmark_data: List[Dict[str, Any]],
    output_path: Path,
    metric_keys: Optional[List[str]] = None,
) -> None:
    """
    Bar chart comparing models across key metrics (mAP50, mAP75, Precision, Recall).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not benchmark_data:
        return

    if metric_keys is None:
        metric_keys = ["map50", "map75", "precision", "recall"]

    models = [item["model"] for item in benchmark_data]
    n_models = len(models)
    n_metrics = len(metric_keys)

    fig, ax = plt.subplots(figsize=(max(8, n_models * 2), 6), dpi=150)
    x = np.arange(n_models)
    bar_width = 0.8 / n_metrics
    colors = ["#2563EB", "#10B981", "#F59E0B", "#8B5CF6", "#EC4899"]

    for i, metric in enumerate(metric_keys):
        values = [item.get(metric, 0.0) * 100 if item.get(metric, 0.0) <= 1.0 else item.get(metric, 0.0) for item in benchmark_data]
        offset = (i - n_metrics / 2 + 0.5) * bar_width
        bars = ax.bar(x + offset, values, bar_width, label=metric.upper(), color=colors[i % len(colors)], alpha=0.88)
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f"{height:.1f}",
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=8)

    ax.set_title("Model Benchmark Comparison on Aerial Object Detection", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Score (%)", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
