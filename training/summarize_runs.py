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


def read_results_csv(run_dir: str, selection_metric: str = "mAP50") -> dict | None:
    """Best epoch from results.csv matching the trainer's checkpoint selection metric (F10).

    Native training selects checkpoints by validation mAP50 (`metrics/mAP50(B)`),
    NOT mAP50-95. Records secondary `best_map5095_epoch` separately so callers
    never mistake a curve peak on another metric for the selected checkpoint.
    """
    rc = os.path.join(run_dir, "results.csv")
    if not os.path.isfile(rc):
        return None
    with open(rc) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    rows = [{k.strip(): v for k, v in r.items()} for r in rows]

    def _find_key(cands):
        return next((c for c in cands if c in rows[0]), None)

    map50_key = _find_key(("metrics/mAP50(B)", "metrics/mAP50"))
    map95_key = _find_key(("metrics/mAP50-95(B)", "metrics/mAP50-95"))
    primary_key = map95_key if selection_metric.lower() in ("map50-95", "map50_95") else (map50_key or map95_key)
    if primary_key is None:
        return None

    def _f(r, key):
        try:
            v = float(r.get(key) or 0)
            return v if v == v else float("-inf")
        except (TypeError, ValueError):
            return float("-inf")

    best = max(rows, key=lambda r: _f(r, primary_key))
    out = dict(best)
    out["_rows"] = len(rows)
    out["_selection_metric"] = primary_key
    if map95_key is not None:
        best95 = max(rows, key=lambda r: _f(r, map95_key))
        out["best_map5095_epoch"] = best95.get("epoch", "")
    return out


def summarize_run(run_dir: str) -> dict | None:
    man_p = os.path.join(run_dir, "run_manifest.json")
    man = {}
    if os.path.isfile(man_p):
        try:
            with open(man_p) as f:
                man = json.load(f)
        except (OSError, ValueError) as ex:
            print(f"[warn] {os.path.basename(run_dir)}: unreadable "
                  f"run_manifest.json ({ex}); using weights/results only")
    selected_weights = None
    weights_selection = "missing"
    try:
        selected_weights = common.find_weights(run_dir)
        weights_selection = "best" if os.path.basename(selected_weights) == "best.pt" else "last"
    except FileNotFoundError:
        if not man:
            return None

    bundle = {}
    bp = os.path.join(run_dir, "bundle_manifest.json")
    if os.path.isfile(bp):
        try:
            with open(bp) as f:
                bundle = json.load(f)
        except (OSError, ValueError):
            bundle = {}

    # v4 manifests store resolved_tuning; v3 stored train_kwargs
    tk = man.get("resolved_tuning") or man.get("train_kwargs", {})
    model_name = man.get("model", "?")
    dataset_name = man.get("dataset", "?")
    sel_metric = bundle.get("selection_metric") or tk.get("selection_metric") or "mAP50"
    from src.config import model_implementation_kind
    row = dict(
        run=os.path.basename(run_dir),
        model=model_name,
        implementation=man.get("implementation") or model_implementation_kind(model_name),
        dataset=dataset_name,
        git_commit=man.get("git_commit", "?"),
        epochs=tk.get("epochs", "?"),
        imgsz=tk.get("img_size", tk.get("imgsz", "?")),
        epochs_done="", status="", weights_selection=weights_selection,
        eval_status="missing",
        best_epoch="", best_map5095_epoch="", map50_val="", map5095_val="",
        map50_test="", map5095_test="", params_m="",
        latency_ms="", imgs_per_sec="", run_dir=run_dir,
    )
    best = read_results_csv(run_dir, selection_metric=sel_metric)
    done = (man.get("final_metrics") or {}).get("epochs_trained")
    if done in (None, "") and best:
        done = best.get("_rows")
    row["epochs_done"] = done if done not in (None, "") else ""
    try:
        row["status"] = ("complete" if int(done) >= int(row["epochs"])
                         else f"partial ({int(done)}/{int(row['epochs'])})")
    except (TypeError, ValueError):
        row["status"] = "unknown"
    fm_best_epoch = (man.get("final_metrics") or {}).get("best_epoch") or bundle.get("best_epoch")
    if best:
        row["best_epoch"] = fm_best_epoch if fm_best_epoch not in (None, "") else best.get("epoch", "")
        row["best_map5095_epoch"] = best.get("best_map5095_epoch", "")
        row["map50_val"] = round(float(best.get("metrics/mAP50(B)", 0) or 0), 4)
        row["map5095_val"] = round(float(best.get("metrics/mAP50-95(B)", 0) or 0), 4)
    elif fm_best_epoch not in (None, ""):
        row["best_epoch"] = fm_best_epoch

    # F11: ONLY accept verified test-split evaluations bound to the selected weights;
    # never publish validation scores in test columns or accept stale/mismatched weights.
    ev = os.path.join(run_dir, f"eval_{dataset_name}_test.json")
    if os.path.isfile(ev):
        try:
            with open(ev) as f:
                e = json.load(f)
        except (OSError, ValueError) as ex:
            print(f"[skip] {os.path.basename(run_dir)}: {os.path.basename(ev)} unreadable ({ex})")
            row["eval_status"] = "unreadable"
            return row
        if e.get("comparable", True) is False:
            print(f"[skip] {os.path.basename(run_dir)}: {os.path.basename(ev)} "
                  f"is protocol '{e.get('protocol')}' (not comparable)")
            row["eval_status"] = "non_comparable"
        elif not common.protocol_current(e):
            print(f"[skip] {os.path.basename(run_dir)}: {os.path.basename(ev)} has coord_space "
                  f"'{e.get('coord_space')}' from an older, incompatible eval protocol - re-run evaluation")
            row["eval_status"] = "incompatible_protocol"
        elif e.get("split") not in (None, "test") or e.get("dataset") not in (None, dataset_name):
            print(f"[skip] {os.path.basename(run_dir)}: {os.path.basename(ev)} "
                  f"dataset/split mismatch (got {e.get('dataset')}/{e.get('split')})")
            row["eval_status"] = "split_mismatch"
        elif selected_weights is not None and not common.eval_weights_match(e, selected_weights):
            print(f"[skip] {os.path.basename(run_dir)}: {os.path.basename(ev)} "
                  f"weights_fingerprint does not match selected {os.path.basename(selected_weights)} (stale)")
            row["eval_status"] = "stale_weights"
        elif not (isinstance(e.get("map50"), (int, float))
                  and isinstance(e.get("map50_95"), (int, float))
                  and e["map50"] == e["map50"]
                  and e["map50_95"] == e["map50_95"]):
            print(f"[skip] {os.path.basename(run_dir)}: {os.path.basename(ev)} has non-finite mAP")
            row["eval_status"] = "non_finite"
        else:
            row["eval_status"] = "verified"
            row["map50_test"] = e.get("map50", "")
            row["map5095_test"] = e.get("map50_95", "")
            row["params_m"] = round(e.get("parameters", 0) / 1e6, 2)
            row["latency_ms"] = (e.get("speed_ms") or {}).get("inference", "")
            row["imgs_per_sec"] = e.get("imgs_per_sec", "")
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
    cols = ["run", "model", "implementation", "dataset", "epochs", "epochs_done", "status",
            "weights_selection", "eval_status", "imgsz", "best_epoch", "best_map5095_epoch",
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
