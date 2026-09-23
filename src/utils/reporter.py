"""
Markdown and terminal reporting utilities for benchmark analysis.
"""

from __future__ import annotations

from typing import List, Dict, Any

def format_metrics_table(results: List[Dict[str, Any]]) -> str:
    """
    Generate an ASCII table summarizing all model and dataset benchmark results.
    """
    headers = [
        "Model", "Dataset", "mAP50", "mAP75", "mAP50-95", 
        "Prec", "Rec", "F1", "Acc", "Angle MAE", "Pearson r", "R²", "FPS"
    ]

    col_widths = [max(len(h), 8) for h in headers]
    col_widths[0] = max(col_widths[0], 14)

    lines = []
    # Header
    header_str = " | ".join(f"{h:^{w}}" for h, w in zip(headers, col_widths))
    sep_str = "-+-".join("-" * w for w in col_widths)
    lines.append(header_str)
    lines.append(sep_str)

    for row in results:
        vals = [
            f"{row.get('model', 'N/A'):<{col_widths[0]}}",
            f"{row.get('dataset', 'N/A'):<{col_widths[1]}}",
            f"{row.get('map50', 0.0):.3f}",
            f"{row.get('map75', 0.0):.3f}",
            f"{row.get('map50_95', 0.0):.3f}",
            f"{row.get('precision', 0.0):.3f}",
            f"{row.get('recall', 0.0):.3f}",
            f"{row.get('f1', 0.0):.3f}",
            f"{row.get('accuracy', 0.0):.3f}",
            f"{row.get('angle_mae', 0.0):.2f}°",
            f"{row.get('pearson_r', 0.0):.3f}",
            f"{row.get('r2_score', 0.0):.3f}",
            f"{row.get('fps', 0.0):.1f}",
        ]
        row_str = " | ".join(f"{v:^{w}}" for v, w in zip(vals, col_widths))
        lines.append(row_str)

    return "\n".join(lines)


def format_delta_table(pre_results: List[Dict[str, Any]], post_results: List[Dict[str, Any]]) -> str:
    """
    Generate an ASCII table comparing Pre-Train Baseline vs. Post-Train Fine-Tuned metrics.
    """
    post_map = {(r["model"], r["dataset"]): r for r in post_results}
    headers = [
        "Model", "Dataset", "Pre mAP50", "Post mAP50", "Gain (Δ)", 
        "Pre F1", "Post F1", "Pre Angle", "Post Angle"
    ]
    col_widths = [14, 10, 10, 11, 10, 8, 8, 10, 10]

    lines = []
    header_str = " | ".join(f"{h:^{w}}" for h, w in zip(headers, col_widths))
    sep_str = "-+-".join("-" * w for w in col_widths)
    lines.append(header_str)
    lines.append(sep_str)

    for pre in pre_results:
        key = (pre["model"], pre["dataset"])
        if key in post_map:
            post = post_map[key]
            pre_m = pre.get("map50", 0.0)
            post_m = post.get("map50", 0.0)
            diff_m = (post_m - pre_m) * 100.0
            diff_str = f"{'+' if diff_m >= 0 else ''}{diff_m:.1f}%"

            vals = [
                f"{pre.get('model', 'N/A'):<{col_widths[0]}}",
                f"{pre.get('dataset', 'N/A'):<{col_widths[1]}}",
                f"{pre_m:.3f}",
                f"{post_m:.3f}",
                f"{diff_str}",
                f"{pre.get('f1', 0.0):.3f}",
                f"{post.get('f1', 0.0):.3f}",
                f"{pre.get('angle_mae', 0.0):.1f}°",
                f"{post.get('angle_mae', 0.0):.1f}°",
            ]
            lines.append(" | ".join(f"{v:^{w}}" for v, w in zip(vals, col_widths)))

    return "\n".join(lines)


def generate_markdown_report(
    results: List[Dict[str, Any]],
    best_model: str,
    best_dataset: str,
    pre_results: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Create a comprehensive GitHub-flavored Markdown report of the experiment run,
    including training progression deltas when pre-train baseline metrics exist.
    """
    md = [
        "# Aerial OBB Object Detection Benchmark & Comparative Analysis Report",
        "",
        "## Executive Summary",
        f"- **Best Performing Model**: `{best_model}`",
        f"- **Best Evaluated Dataset**: `{best_dataset}`",
        "",
    ]

    if pre_results:
        md.extend([
            "## Pre-Training vs Post-Training Performance Progression",
            "This table illustrates empirical model gains from zero-shot / base pretraining to full fine-tuning on the aerial datasets.",
            "",
            "| Model | Dataset | Pre-Train mAP50 | Post-Train mAP50 | mAP Gain (Δ) | Pre F1 | Post F1 | Pre Angle MAE | Post Angle MAE |",
            "|---|---|---|---|---|---|---|---|---|",
        ])
        post_map = {(r["model"], r["dataset"]): r for r in results}
        for pre in pre_results:
            key = (pre["model"], pre["dataset"])
            if key in post_map:
                post = post_map[key]
                pre_m = pre.get("map50", 0.0)
                post_m = post.get("map50", 0.0)
                diff_m = (post_m - pre_m) * 100.0
                sign = "+" if diff_m >= 0 else ""
                md.append(
                    f"| {pre['model']} | {pre['dataset']} | {pre_m:.4f} | {post_m:.4f} | **{sign}{diff_m:.1f}%** | "
                    f"{pre.get('f1', 0):.4f} | {post.get('f1', 0):.4f} | {pre.get('angle_mae', 0):.2f}° | {post.get('angle_mae', 0):.2f}° |"
                )
        md.append("")

    md.extend([
        "## Complete Performance Metrics Across Models & Datasets",
        "",
        "| Model | Dataset | mAP@0.50 | mAP@0.75 | mAP@0.50:0.95 | Precision | Recall | F1-Score | Accuracy | Angle MAE | Pearson r | R² Score | FPS |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ])

    for r in results:
        md.append(
            f"| {r.get('model')} | {r.get('dataset')} | {r.get('map50', 0):.4f} | {r.get('map75', 0):.4f} | "
            f"{r.get('map50_95', 0):.4f} | {r.get('precision', 0):.4f} | {r.get('recall', 0):.4f} | "
            f"{r.get('f1', 0):.4f} | {r.get('accuracy', 0):.4f} | {r.get('angle_mae', 0):.2f}° | "
            f"{r.get('pearson_r', 0):.4f} | {r.get('r2_score', 0):.4f} | {r.get('fps', 0):.1f} |"
        )

    md.extend([
        "",
        "## Metrics Definitions & Interpretation",
        "1. **Detection Metrics (mAP50, mAP75, mAP50-95)**: Area under precision-recall curve. mAP75 measures strict spatial and angular alignment critical for aerial vehicles.",
        "2. **Classification Metrics (Precision, Recall, F1, Accuracy)**: Evaluates whether matched oriented detections correctly distinguish vehicle types and road users without background false alarms.",
        "3. **Orientation Regression Metrics (Angle MAE, Pearson r, R²)**: Continuous goodness-of-fit for heading angles. Pearson r close to 1.0 indicates strong directional fidelity.",
        "4. **Operational Metrics (FPS, Latency)**: Real-time suitability for UAV edge deployment (target: >30 FPS).",
        "",
    ])

    return "\n".join(md)

