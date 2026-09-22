#!/usr/bin/env python3
"""
Standalone report and visualization generator from saved benchmark results.
"""

import sys
import json
import argparse
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.reporter import format_metrics_table, generate_markdown_report
from src.utils.visualizer import plot_benchmark_comparison

def main():
    parser = argparse.ArgumentParser(description="Generate markdown reports and plots from saved benchmark results.")
    parser.add_argument("--run-dir", type=str, required=True, help="Path to run output directory containing benchmark_metrics.json.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    json_path = run_dir / "benchmark_metrics.json"

    if not json_path.exists():
        print(f"[!] File not found: {json_path}", file=sys.stderr)
        sys.exit(1)

    with open(json_path, "r") as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    best_row = df.loc[df["map50"].idxmax()]
    best_model = best_row["model"]
    best_dataset = best_row["dataset"]

    print("\n" + "=" * 80)
    print(f" BENCHMARK REPORT: {run_dir.name}")
    print("=" * 80)
    print(format_metrics_table(data))
    print("=" * 80)

    # Markdown report
    md_content = generate_markdown_report(data, best_model=best_model, best_dataset=best_dataset)
    report_path = run_dir / "benchmark_report.md"
    with open(report_path, "w") as f:
        f.write(md_content)
    print(f"[✓] Written markdown report to: {report_path}")

    # Comparison chart
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    chart_path = plots_dir / "model_benchmark_comparison.png"
    plot_benchmark_comparison(data, chart_path)
    print(f"[✓] Saved comparison plot to: {chart_path}")

if __name__ == "__main__":
    main()
