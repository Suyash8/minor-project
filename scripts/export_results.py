#!/usr/bin/env python3
# =============================================================================
# EXPORT RESULTS -> EXCEL-READY MASTER TABLES
# =============================================================================
# Scans every run directory under a results root (default: saves/results) and
# consolidates ALL metrics that were ever produced into a small set of tidy,
# de-duplicated CSVs under <output-dir>/ (default: saves/excel_export/).
#
# After the Colab cross-dataset run, copy the produced run folder (e.g.
# results/cross_eval_20ep/) into this repo and point --extra-dirs at it, or
# pass --results-root pointing at the Drive results directory. Everything the
# xlsx needs is then in one folder.
#
# Outputs
#   summary_all_runs.csv        long table: one row per (run, phase, model,
#                               dataset) with every metric column.
#   summary_latest.csv          same columns, but only the LATEST run per
#                               (phase, model, dataset) - use this for the
#                               final report numbers.
#   summary_matrix_post_train.csv   models as COLUMNS, metrics as ROWS
#                               (mirrors the 'Summary' sheet layout of the
#                               results xlsx).
#   class_wise_all.csv          one row per (run, phase, model, dataset,
#                               class): AP50, AP75, TP, FP, FN, Precision,
#                               Recall, F1 - feeds the class_wise_results sheet.
#   cross_dataset_pairs.csv     merged cross_dataset_summary.csv files (all
#                               found cross-eval runs).
#   cross_dataset_global.csv    merged global_summary.csv files.
#   EXPORT_README.md            column dictionary + which file feeds which
#                               xlsx sheet.
#
# Usage:
#   python scripts/export_results.py                      # repo defaults
#   python scripts/export_results.py --extra-dirs /content/drive/MyDrive/object-detection/results/cross_eval_20ep
#   python scripts/export_results.py --results-root /content/drive/MyDrive/object-detection/results
# =============================================================================

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Every metric the benchmark suite can report, in reporting order.
METRIC_COLS = [
    "map50", "map75", "map50_95",
    "ap_small", "ap_medium", "ap_large",
    "precision", "recall", "f1", "accuracy", "specificity",
    "angle_mae", "angle_rmse", "pearson_r", "spearman_rho", "r2_score",
    "center_offset_mae", "center_offset_rmse", "aspect_ratio_r2",
    "fps", "mean_latency_ms",
]

ID_COLS_BASE = ["run", "run_dir", "run_timestamp", "phase", "model", "dataset",
                "source_dataset", "eval_dataset", "total_gt_boxes"]


def _clean(v):
    """JSON/CSV-safe scalar."""
    if v is None:
        return None
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
        return None
    return v


def _find_run_dirs(roots):
    """Yield run directories: any folder under roots containing checkpoints/,
    run_manifest.json, or *_summary.csv. Explicit roots qualify too."""
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        candidates = [root] + sorted(root.rglob("*"))
        for d in candidates:
            if not d.is_dir():
                continue
            has_ckpt = (d / "checkpoints").is_dir()
            has_manifest = (d / "run_manifest.json").exists()
            has_summary = any(d.glob("*summary*.csv"))
            has_cross = (d / "cross_dataset_summary.csv").exists()
            if has_ckpt or has_manifest or has_summary or has_cross:
                yield d


def _run_timestamp(d: Path) -> str:
    mtime = max(
        (f.stat().st_mtime for f in d.rglob("*") if f.is_file()),
        default=d.stat().st_mtime,
    )
    return datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()


def _load_eval_checkpoints(run_dir: Path, run_name: str):
    """Rows from checkpoints/<dataset>_<model>_<phase>_eval.json."""
    rows = []
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.is_dir():
        return rows
    for f in sorted(ckpt_dir.glob("*_eval.json")):
        name = f.stem  # e.g. visdrone_yolo11-obb_post_train_eval
        if name.endswith("_eval"):
            name = name[: -len("_eval")]
        parts = name.split("_")
        if len(parts) < 3:
            continue
        phase = parts[-1]              # pre_train | post_train | eval
        model = parts[-2]              # model names contain '-' but not '_'
        dataset = "_".join(parts[:-2])  # dataset names contain '_'
        try:
            data = json.load(open(f))
        except Exception:
            continue
        summary = data.get("summary", data)
        row = {c: _clean(summary.get(c)) for c in METRIC_COLS}
        row.update({
            "run": run_name,
            "run_dir": str(run_dir),
            "run_timestamp": _run_timestamp(run_dir),
            "phase": phase,
            "model": summary.get("model", model),
            "dataset": summary.get("dataset", dataset),
            "total_gt_boxes": summary.get("total_gt_boxes"),
        })
        rows.append(row)
    return rows


def _load_manifest_rows(run_dir: Path, run_name: str):
    """Rows embedded in run_manifest.json summaries (any phase)."""
    mf = run_dir / "run_manifest.json"
    if not mf.exists():
        return []
    try:
        data = json.load(open(mf))
    except Exception:
        return []
    rows = []
    for s in data.get("summaries", []):
        if not isinstance(s, dict) or "model" not in s:
            continue
        row = {c: _clean(s.get(c)) for c in METRIC_COLS}
        row.update({
            "run": run_name,
            "run_dir": str(run_dir),
            "run_timestamp": data.get("updated_at") or _run_timestamp(run_dir),
            "phase": s.get("phase", "unknown"),
            "model": s.get("model"),
            "dataset": s.get("dataset"),
            "total_gt_boxes": s.get("total_gt_boxes"),
        })
        rows.append(row)
    return rows


def _load_benchmark_csvs(run_dir: Path, run_name: str):
    """Rows from benchmark_summary.csv / pre_train_summary.csv /
    post_train_summary.csv (top-level run outputs)."""
    rows = []
    for fname, phase_hint in [
        ("benchmark_summary.csv", None),
        ("pre_train_summary.csv", "pre_train"),
        ("post_train_summary.csv", "post_train"),
    ]:
        f = run_dir / fname
        if not f.exists():
            continue
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        for _, r in df.iterrows():
            row = {c: _clean(r.get(c)) for c in METRIC_COLS if c in df.columns}
            row.update({
                "run": run_name,
                "run_dir": str(run_dir),
                "run_timestamp": _run_timestamp(run_dir),
                "phase": r.get("phase", phase_hint) if "phase" in df.columns else phase_hint,
                "model": r.get("model"),
                "dataset": r.get("dataset"),
                "total_gt_boxes": r.get("total_gt_boxes"),
            })
            rows.append(row)
    return rows


def _load_class_wise(run_dir: Path, run_name: str):
    """Per-class rows: repo class_wise_results CSVs + cross-eval per_class/."""
    rows = []
    # 1. repo-style class_wise_results_*.csv (inside the run dir OR its parent root)
    candidates = sorted(run_dir.rglob("class_wise_results_*.csv"))
    parent = run_dir.parent
    if parent.exists():
        candidates += sorted(parent.glob("class_wise_results_*.csv"))
    seen = set()
    for f in candidates:
        if f in seen:
            continue
        seen.add(f)
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        # infer model/dataset from filename: class_wise_results_<model>_<dataset>.csv
        stem = f.stem[len("class_wise_results_"):]
        model, dataset = stem, stem
        for ds in ("visdrone", "codrone", "dota"):
            if stem.lower().endswith(ds):
                dataset = ds
                model = stem[: -(len(ds) + 1)]
                break
        for _, r in df.iterrows():
            rows.append({
                "run": run_name, "run_dir": str(f),
                "phase": "post_train", "model": model, "dataset": dataset,
                "source_dataset": None, "eval_dataset": dataset,
                "class": r.get("class") or r.get("Classes") or r.get("Class"),
                "AP50": _clean(r.get("AP50")),
                "AP75": _clean(r.get("AP75", r.get("AP50-95"))),
                "TP": _clean(r.get("TP")), "FP": _clean(r.get("FP")),
                "FN": _clean(r.get("FN")),
                "Precision": _clean(r.get("Precision")),
                "Recall": _clean(r.get("Recall")),
                "F1": _clean(r.get("F1")),
                "total_instances": _clean(r.get("Total_Instances")),
            })
    # 2. cross-eval per_class/<source>__<model>__<eval>.csv
    per_class_dirs = [run_dir / "per_class"]
    parent_pc = run_dir.parent / "per_class"
    if parent_pc.is_dir():
        per_class_dirs.append(parent_pc)
    for per_class_dir in per_class_dirs:
        if not per_class_dir.is_dir():
            continue
        for f in sorted(per_class_dir.glob("*.csv")):
            tag = f.stem  # source__model__eval
            parts = tag.split("__")
            if len(parts) != 3:
                continue
            src, model, ev = parts
            try:
                df = pd.read_csv(f)
            except Exception:
                continue
            for _, r in df.iterrows():
                rows.append({
                    "run": run_name, "run_dir": str(run_dir),
                    "phase": "cross_eval", "model": model,
                    "dataset": ev, "source_dataset": src, "eval_dataset": ev,
                    "class": r.get("class"),
                    "AP50": _clean(r.get("AP50")), "AP75": _clean(r.get("AP75")),
                    "TP": _clean(r.get("TP")), "FP": _clean(r.get("FP")),
                    "FN": _clean(r.get("FN")),
                    "Precision": _clean(r.get("Precision")),
                    "Recall": _clean(r.get("Recall")),
                    "F1": _clean(r.get("F1")),
                })
    return rows


def _top_level_class_wise(results_root: Path):
    """Scan results_root, its parent and its children for loose
    class_wise_results_*.csv files and per_class/ dirs (repo-root layout)."""
    csv_files, pc_dirs = [], []
    for base in {results_root, results_root.parent, *results_root.glob("*")}:
        try:
            if base.is_dir():
                csv_files += sorted(base.glob("class_wise_results_*.csv"))
                pc_dirs += [d for d in [base / "per_class"] if d.is_dir()]
            elif base.is_file():
                pass
        except Exception:
            continue
    csv_files = sorted(set(csv_files))
    pc_dirs = sorted(set(pc_dirs))
    return csv_files, pc_dirs


def _parse_class_wise_csv(f: Path, phase: str, run_label: str, run_dir_label: str):
    try:
        df = pd.read_csv(f)
    except Exception:
        return []
    stem = f.stem[len("class_wise_results_"):]
    model, dataset = stem, stem
    for ds in ("visdrone", "codrone", "dota"):
        if stem.lower().endswith(ds):
            dataset = ds
            model = stem[: -(len(ds) + 1)]
            break
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "run": run_label, "run_dir": run_dir_label,
            "phase": phase, "model": model, "dataset": dataset,
            "source_dataset": None, "eval_dataset": dataset,
            "class": r.get("class") or r.get("Classes") or r.get("Class"),
            "AP50": _clean(r.get("AP50")),
            "AP75": _clean(r.get("AP75", r.get("AP50-95"))),
            "TP": _clean(r.get("TP")), "FP": _clean(r.get("FP")),
            "FN": _clean(r.get("FN")),
            "Precision": _clean(r.get("Precision")),
            "Recall": _clean(r.get("Recall")),
            "F1": _clean(r.get("F1")),
            "total_instances": _clean(r.get("Total_Instances")),
        })
    return rows


def _load_cross_dataset(run_dir: Path, run_name: str):
    """cross_dataset_summary.csv + global_summary.csv if present."""
    pairs, glob = None, None
    f1 = run_dir / "cross_dataset_summary.csv"
    if f1.exists():
        try:
            pairs = pd.read_csv(f1)
            pairs.insert(0, "run", run_name)
        except Exception:
            pass
    f2 = run_dir / "global_summary.csv"
    if f2.exists():
        try:
            glob = pd.read_csv(f2)
            glob.insert(0, "run", run_name)
        except Exception:
            pass
    return pairs, glob


def main():
    ap = argparse.ArgumentParser(description="Export all benchmark results to excel-ready CSVs")
    ap.add_argument("--results-root", default=str(PROJECT_ROOT / "saves" / "results"),
                    help="Root directory containing run folders.")
    ap.add_argument("--extra-dirs", nargs="*", default=[],
                    help="Additional run directories (e.g. a Drive cross-eval run folder).")
    ap.add_argument("--output-dir", default=str(PROJECT_ROOT / "saves" / "excel_export"),
                    help="Where the exported CSVs are written.")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    roots = [args.results_root] + list(args.extra_dirs)
    run_dirs = sorted({d for d in _find_run_dirs(roots)})
    print(f"[*] Discovered {len(run_dirs)} run folder(s):")
    for d in run_dirs:
        print(f"      {d}")

    # ---- summary rows ------------------------------------------------------
    all_rows = []
    class_rows = []
    pair_frames, glob_frames = [], []
    for d in run_dirs:
        run_name = d.name
        all_rows += _load_eval_checkpoints(d, run_name)
        all_rows += _load_manifest_rows(d, run_name)
        all_rows += _load_benchmark_csvs(d, run_name)
        class_rows += _load_class_wise(d, run_name)
        pairs, glob = _load_cross_dataset(d, run_name)
        if pairs is not None:
            pair_frames.append(pairs)
        if glob is not None:
            glob_frames.append(glob)

    # loose repo-level class_wise CSVs / per_class dirs (e.g. saves/class_wise_results_*.csv)
    loose_csvs, loose_pc = _top_level_class_wise(Path(args.results_root))
    for f in loose_csvs:
        class_rows += _parse_class_wise_csv(f, phase="post_train",
                                            run_label="repo_class_wise",
                                            run_dir_label=str(f))
    extra_dirs = list(args.extra_dirs)
    for ed in extra_dirs:
        edp = Path(ed)
        if (edp / "per_class").is_dir():
            class_rows += _load_class_wise(edp, edp.name)

    df = pd.DataFrame(all_rows)
    for c in ID_COLS_BASE:
        if c not in df.columns:
            df[c] = None
    df = df[ID_COLS_BASE + METRIC_COLS]

    # de-duplicate identical (run, phase, model, dataset) rows (manifests and
    # checkpoint files often carry the same pair)
    dedup_keys = [c for c in ["run", "phase", "model", "dataset",
                              "source_dataset", "eval_dataset"] if c in df.columns]
    before = len(df)
    df = df.drop_duplicates(subset=dedup_keys, keep="last").reset_index(drop=True)
    print(f"[*] Summary rows: {before} -> {len(df)} after de-duplication")

    # ---- write summary tables ----------------------------------------------
    df.to_csv(out_dir / "summary_all_runs.csv", index=False)

    # latest run per (phase, model, dataset)
    if len(df):
        df_sorted = df.sort_values("run_timestamp")
        latest = df_sorted.drop_duplicates(
            subset=[c for c in ["phase", "model", "dataset"] if c in df.columns],
            keep="last")
        latest.to_csv(out_dir / "summary_latest.csv", index=False)

        # matrix: models as columns (post_train / eval phase), metrics as rows
        mat_df = latest[latest["phase"].isin(["post_train", "eval", "cross_eval"])]
        if len(mat_df):
            records = []
            for metric in METRIC_COLS:
                rec = {"metric": metric}
                for _, r in mat_df.iterrows():
                    col = f"{r['model']} ({r['dataset']})"
                    rec[col] = r[metric]
                records.append(rec)
            pd.DataFrame(records).to_csv(
                out_dir / "summary_matrix_post_train.csv", index=False)
    else:
        pd.DataFrame(columns=ID_COLS_BASE + METRIC_COLS).to_csv(
            out_dir / "summary_latest.csv", index=False)

    # ---- class-wise ----------------------------------------------------------
    df_cls = pd.DataFrame(class_rows)
    if len(df_cls):
        df_cls = df_cls.drop_duplicates(
            subset=["run_dir", "phase", "model", "dataset", "source_dataset",
                    "eval_dataset", "class"], keep="first")
    df_cls.to_csv(out_dir / "class_wise_all.csv", index=False)
    print(f"[*] Class-wise rows: {len(class_rows)} -> {len(df_cls)} after de-duplication")

    # ---- cross-dataset --------------------------------------------------------
    if pair_frames:
        pd.concat(pair_frames, ignore_index=True).to_csv(
            out_dir / "cross_dataset_pairs.csv", index=False)
    else:
        pd.DataFrame().to_csv(out_dir / "cross_dataset_pairs.csv")
    if glob_frames:
        pd.concat(glob_frames, ignore_index=True).to_csv(
            out_dir / "cross_dataset_global.csv", index=False)
    else:
        pd.DataFrame().to_csv(out_dir / "cross_dataset_global.csv")

    # ---- README ---------------------------------------------------------------
    readme = f"""# Excel Export - generated {datetime.now().isoformat(timespec='seconds')}

Source run folders scanned: {len(run_dirs)}

| File | Feeds | Notes |
|---|---|---|
| `summary_all_runs.csv` | appendix / audit | every (run, phase, model, dataset) with all metrics |
| `summary_latest.csv` | xlsx **Summary** sheet | latest run per (phase, model, dataset) |
| `summary_matrix_post_train.csv` | xlsx **Summary** sheet | models as columns, metrics as rows - direct paste |
| `class_wise_all.csv` | xlsx **class_wise_results** sheet | per-class AP50/AP75/TP/FP/FN/P/R/F1 |
| `cross_dataset_pairs.csv` | cross-dataset section | one row per (model, source, eval) pair |
| `cross_dataset_global.csv` | cross-dataset global table | macro-average per (model, source) |

## Column dictionary (summary tables)
- IDs: `run`, `run_dir`, `run_timestamp` (UTC ISO), `phase` (pre_train / post_train / eval / cross_eval),
  `model`, `dataset` (eval dataset; for cross_eval this is the target dataset),
  `source_dataset` (training dataset for cross_eval rows), `total_gt_boxes`
- Detection: `map50`, `map75`, `map50_95`, `ap_small`, `ap_medium`, `ap_large`
- Classification: `precision`, `recall`, `f1`, `accuracy`, `specificity` (macro-averaged)
- Angle regression (matched boxes): `angle_mae` (deg), `angle_rmse` (deg), `pearson_r`,
  `spearman_rho`, `r2_score`, `center_offset_mae`/`rmse` (px), `aspect_ratio_r2`
- Efficiency: `fps`, `mean_latency_ms`

## Reproduce
```
python scripts/export_results.py \\
    --extra-dirs /path/to/cross_eval_run_folder
```
"""
    (out_dir / "EXPORT_README.md").write_text(readme, encoding="utf-8")

    print(f"\n[✓] Exported to {out_dir}:")
    for f in sorted(out_dir.glob("*.csv")):
        try:
            n = sum(1 for _ in open(f, encoding="utf-8")) - 1
        except Exception:
            n = "?"
        print(f"      {f.name:36s} {n} rows")
    print(f"      EXPORT_README.md")


if __name__ == "__main__":
    main()
