#!/usr/bin/env python3
"""
summarize_runs.py — aggregate every training run under the runs dir into one
table: model, dataset, best epoch metrics (from Ultralytics results.csv),
test-split evaluation (from eval_*.json), parameters, git commit.

Usage:
  python training/summarize_runs.py                # markdown + CSV to runs dir
  python training/summarize_runs.py --runs-dir /content/drive/MyDrive/object-detection/runs
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402


def read_results_csv(run_dir: str) -> dict | None:
    """Best epoch (max mAP50-95) from Ultralytics results.csv."""
    rc = os.path.join(run_dir, "results.csv")
    if not os.path.isfile(rc):
        return None
    with open(rc) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    key = None
    for cand in ("metrics/mAP50-95(B)", "metrics/mAP50-95"):
        if cand in rows[0]:
            key = cand
            break
    if key is None:
        return None
    best = max(rows, key=lambda r: float(r.get(key) or 0))
    return {k.strip(): v for k, v in best.items()}


def summarize_run(run_dir: str) -> dict | None:
    man_p = os.path.join(run_dir, "run_manifest.json")
    man = {}
    if os.path.isfile(man_p):
        with open(man_p) as f:
            man = json.load(f)
    try:
        common.find_weights(run_dir)
    except FileNotFoundError:
        if not man:
            return None
    # v4 manifests store resolved_tuning; v3 stored train_kwargs
    tk = man.get("resolved_tuning") or man.get("train_kwargs", {})
    row = dict(
        run=os.path.basename(run_dir),
        model=man.get("model", "?"),
        dataset=man.get("dataset", "?"),
        git_commit=man.get("git_commit", "?"),
        epochs=tk.get("epochs", "?"),
        imgsz=tk.get("img_size", tk.get("imgsz", "?")),
        best_epoch="", map50_val="", map5095_val="",
        map50_test="", map5095_test="", params_m="",
        latency_ms="", imgs_per_sec="", run_dir=run_dir,
    )
    best = read_results_csv(run_dir)
    if best:
        row["best_epoch"] = best.get("epoch", "")
        row["map50_val"] = round(float(best.get("metrics/mAP50(B)", 0) or 0), 4)
        row["map5095_val"] = round(float(best.get("metrics/mAP50-95(B)", 0) or 0), 4)
    for split in ("test", "val"):
        ev = os.path.join(run_dir, f"eval_{row['dataset']}_{split}.json")
        if os.path.isfile(ev):
            with open(ev) as f:
                e = json.load(f)
            if e.get("comparable", True) is False:
                # Ultralytics-native fallback metrics are a different,
                # non-comparable protocol; never mix them into the summary.
                print(f"[skip] {os.path.basename(run_dir)}: {os.path.basename(ev)} "
                      f"is protocol '{e.get('protocol')}' (not comparable)")
                continue
            if not common.protocol_current(e):
                # B-round note 3: older-protocol eval JSONs (different
                # coord_space/GT handling) must not be summarized with
                # current results.
                print(f"[skip] {os.path.basename(run_dir)}: "
                      f"{os.path.basename(ev)} has coord_space "
                      f"'{e.get('coord_space')}' from an older, incompatible "
                      f"eval protocol - re-run evaluation")
                continue
            row["map50_test"] = e.get("map50", "")
            row["map5095_test"] = e.get("map50_95", "")
            row["params_m"] = round(e.get("parameters", 0) / 1e6, 2)
            row["latency_ms"] = (e.get("speed_ms") or {}).get("inference", "")
            row["imgs_per_sec"] = e.get("imgs_per_sec", "")
            break
    return row


def main(argv=None):
    p = argparse.ArgumentParser(description="Summarize training runs")
    p.add_argument("--runs-dir", default=None)
    p.add_argument("--out", default=None,
                   help="master summary CSV (dataset x model -> mAP/params/"
                        "latency), e.g. results/master_summary.csv (W6.4)")
    a = p.parse_args(argv)
    runs_dir = common.default_runs_dir(a.runs_dir)

    rows = []
    for p in common.discover_run_dirs(runs_dir):
        r = summarize_run(p)
        if r:
            rows.append(r)
    if not rows:
        sys.exit(f"no runs found under {runs_dir}")

    csv_path = os.path.join(runs_dir, "summary.csv")
    cols = ["run", "model", "dataset", "epochs", "imgsz", "best_epoch",
            "map50_val", "map5095_val", "map50_test", "map5095_test",
            "params_m", "latency_ms", "imgs_per_sec", "git_commit", "run_dir"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    md_path = os.path.join(runs_dir, "summary.md")
    with open(md_path, "w") as f:
        f.write("# Training runs summary\n\n")
        f.write("| " + " | ".join(cols[:-1]) + " |\n")
        f.write("|" + "---|" * (len(cols) - 1) + "\n")
        for r in rows:
            f.write("| " + " | ".join(str(r.get(c, "")) for c in cols[:-1]) + " |\n")

    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)

    print(f"{len(rows)} runs summarized ->\n  {csv_path}\n  {md_path}\n"
          + (f"  {a.out}\n" if a.out else ""))
    with open(md_path) as f:
        print(f.read())


if __name__ == "__main__":
    main()
