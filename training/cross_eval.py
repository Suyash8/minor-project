#!/usr/bin/env python3
"""
cross_eval.py — cross-dataset evaluation matrix for trained runs.

For every finished run (best.pt present) under the runs dir, evaluates on the
TEST split of every cleaned dataset and writes:
  <runs_dir>/cross_eval.csv      rows: run, model, trained_on, eval_on, mAP50, mAP50-95
  <runs_dir>/cross_eval.md       the same as a matrix (mAP50 / mAP50-95)
  <runs_dir>/cross_eval.json     full detail incl. per-class

Usage:
  python training/cross_eval.py                       # all runs x all datasets
  python training/cross_eval.py --runs r1 r2          # specific run dirs
  python training/cross_eval.py --datasets visdrone dota --split test
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402


def discover_runs(runs_dir: str) -> list[str]:
    """Run dirs with trained weights, both layouts."""
    return common.discover_run_dirs(runs_dir)


def run_info(run_dir: str) -> dict:
    man = os.path.join(run_dir, "run_manifest.json")
    info = dict(run=os.path.basename(run_dir), model="?", trained_on="?")
    if os.path.isfile(man):
        with open(man) as f:
            m = json.load(f)
        info.update(model=m.get("model", "?"), trained_on=m.get("dataset", "?"))
    return info


def main(argv=None):
    p = argparse.ArgumentParser(description="Cross-dataset evaluation matrix")
    p.add_argument("--runs", nargs="+", help="specific run dirs (default: all "
                   "runs with best.pt under --runs-dir)")
    p.add_argument("--runs-dir", default=None)
    p.add_argument("--datasets", nargs="+", default=common.REAL_DATASETS,
                   choices=common.REAL_DATASETS)
    p.add_argument("--split", default="test", choices=common.SPLITS)
    p.add_argument("--imgsz", type=int, default=None,
                   help="default: each run's trained img_size from its manifest")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default=None)
    p.add_argument("--data-root", default=None)
    p.add_argument("--stage", action="store_true",
                   help="stage every requested dataset to fast local disk ONCE "
                        "before the matrix runs (never re-staged per cell, W6.2)")
    a = p.parse_args(argv)

    runs_dir = common.default_runs_dir(a.runs_dir)
    runs = a.runs or discover_runs(runs_dir)
    if not runs:
        sys.exit(f"no finished runs (best.pt) found under {runs_dir}")
    data_root = common.resolve_data_root(a.data_root)

    if a.stage:
        # Stage once for the whole matrix; cells read the staged copy.
        data_root, _yaml = common.stage_local("all", data_root)

    import evaluate
    rows = []
    for rd in runs:
        info = run_info(rd)
        try:
            weights = common.find_weights(rd)
        except FileNotFoundError:
            continue
        print(f"\n##### {info['run']}  (model={info['model']}, "
              f"trained_on={info['trained_on']}) #####")
        for ds in a.datasets:
            try:
                # v4 (W6): one src/metrics protocol for every family
                # (Ultralytics runs YOLO inference only).
                r = evaluate.evaluate_run(
                    rd, ds, split=a.split, data_root=data_root,
                    imgsz=a.imgsz, batch=a.batch, device=a.device,
                    model_name=None if info["model"] == "?" else info["model"],
                    save=True)
                if r.get("comparable", True) is False:
                    # Ultralytics-native fallback metrics are a different,
                    # non-comparable protocol; never mix them into the matrix.
                    print(f"[skip] {ds}: protocol "
                          f"'{r.get('protocol')}' is not comparable")
                    continue
                if not common.protocol_current(r):
                    # B-round note 3: an eval from an older protocol
                    # (coord_space 'letterbox(imgsz)' or missing) mixed
                    # different GT sets per family; never aggregate it with
                    # current results.
                    print(f"[skip] {ds}: coord_space "
                          f"'{r.get('coord_space')}' is from an older, "
                          f"incompatible eval protocol - re-run evaluation")
                    continue
                rows.append(dict(**info, eval_on=ds,
                                 map50=r["map50"], map50_95=r["map50_95"],
                                 per_class=r["per_class"],
                                 json_path=r.get("json_path")))
            except FileNotFoundError as e:
                print(f"[skip] {ds}: {e}")
            except Exception as e:
                print(f"[error] {ds}: {e}")

    if not rows:
        sys.exit("no evaluations completed")

    # ---- csv ----
    csv_path = os.path.join(runs_dir, "cross_eval.csv")
    with open(csv_path, "w") as f:
        f.write("run,model,trained_on,eval_on,map50,map50_95\n")
        for r in rows:
            f.write(f"{r['run']},{r['model']},{r['trained_on']},"
                    f"{r['eval_on']},{r['map50']:.5f},{r['map50_95']:.5f}\n")

    # ---- markdown matrix ----
    md_path = os.path.join(runs_dir, "cross_eval.md")
    with open(md_path, "w") as f:
        f.write(f"# Cross-dataset evaluation ({a.split} split, imgsz={a.imgsz})\n\n")
        head = "| run (model / trained on) | " + \
               " | ".join(f"{d} mAP50 / mAP50-95" for d in a.datasets) + " |"
        f.write(head + "\n|" + "---|" * (len(a.datasets) + 1) + "\n")
        for rd in runs:
            info = run_info(rd)
            cells = []
            for ds in a.datasets:
                rr = next((x for x in rows if x["run"] == info["run"]
                           and x["eval_on"] == ds), None)
                cells.append(f"{rr['map50']:.3f} / {rr['map50_95']:.3f}"
                             if rr else "—")
            f.write(f"| {info['run']}<br>({info['model']} / {info['trained_on']}) | "
                    + " | ".join(cells) + " |\n")

    # ---- json ----
    json_path = os.path.join(runs_dir, "cross_eval.json")
    with open(json_path, "w") as f:
        json.dump(dict(split=a.split, imgsz=(a.imgsz or "per-run trained"), rows=rows), f, indent=2)

    # ---- heatmap (mAP50, runs x datasets) ----
    heat_path = os.path.join(runs_dir, "cross_eval_heatmap.png")
    try:
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        run_names = [run_info(rd)["run"] for rd in runs]
        mat = np.full((len(run_names), len(a.datasets)), np.nan)
        for i, rn in enumerate(run_names):
            for j, ds in enumerate(a.datasets):
                rr = next((x for x in rows if x["run"] == rn
                           and x["eval_on"] == ds), None)
                if rr:
                    mat[i, j] = rr["map50"]
        fig_h = max(3.0, 0.45 * len(run_names) + 1.5)
        fig, ax = plt.subplots(figsize=(max(6.0, 1.8 * len(a.datasets)), fig_h))
        vmax = float(np.nanmax(mat)) if len(rows) else 0.6
        im = ax.imshow(np.ma.masked_invalid(mat), cmap="viridis", vmin=0.0,
                       vmax=max(0.6, vmax))
        ax.set_xticks(range(len(a.datasets)), a.datasets)
        ax.set_yticks(range(len(run_names)), run_names)
        ax.set_title(f"Cross-dataset mAP50 ({a.split} split)")
        for i in range(len(run_names)):
            for j in range(len(a.datasets)):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:.3f}", ha="center",
                            va="center", fontsize=8,
                            color="white" if mat[i, j] < 0.3 else "black")
        fig.colorbar(im, ax=ax, label="mAP50")
        fig.tight_layout()
        fig.savefig(heat_path, dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"[warn] heatmap not written: {e}")
        heat_path = None

    print(f"\nwrote:\n  {csv_path}\n  {md_path}\n  {json_path}")
    if heat_path:
        print(f"  {heat_path}")
    print("\nMatrix (mAP50 / mAP50-95):")
    with open(md_path) as f:
        print(f.read())


if __name__ == "__main__":
    main()
