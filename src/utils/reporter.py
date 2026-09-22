"""
Markdown and terminal reporting utilities for benchmark analysis.
"""

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

def generate_markdown_report(results: List[Dict[str, Any]], best_model: str, best_dataset: str) -> str:
    """
    Create a comprehensive GitHub-flavored Markdown report of the experiment run.
    """
    md = [
        "# Aerial OBB Object Detection Benchmark & Comparative Analysis Report",
        "",
        "## Executive Summary",
        f"- **Best Performing Model**: `{best_model}`",
        f"- **Best Evaluated Dataset**: `{best_dataset}`",
        "",
        "## Complete Performance Metrics Across Models & Datasets",
        "",
        "| Model | Dataset | mAP@0.50 | mAP@0.75 | mAP@0.50:0.95 | Precision | Recall | F1-Score | Accuracy | Angle MAE | Pearson r | R² Score | FPS |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]

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
