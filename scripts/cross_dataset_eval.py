#!/usr/bin/env python3
"""
[DEPRECATED SHIM] scripts/cross_dataset_eval.py -> training/cross_eval.py (v4)

Cross-dataset evaluation is unified in `training/cross_eval.py` (W6.2): one
protocol for every family, staged data reused across cells, one tidy CSV +
markdown matrix + heatmap. The old class-remap logic is obsolete: all cleaned
v4.1 datasets share the same frozen 8-class space, so no remapping is needed.

New canonical usage:
    python training/cross_eval.py --runs-dir runs/train --stage
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

NOTICE = (
    "[deprecation] scripts/cross_dataset_eval.py is a shim in v4; use "
    "`python training/cross_eval.py --runs-dir runs/train`."
)


def main() -> None:
    print(NOTICE, file=sys.stderr)
    p = argparse.ArgumentParser(description="(shim) cross-dataset eval matrix")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--weights-dir", default=None,
                   help="old name for the runs directory")
    p.add_argument("--runs-dir", default=None)
    p.add_argument("--results-dir", default=None, help="ignored in v4; the "
                   "matrix CSV/MD/PNG land in the runs dir")
    p.add_argument("--datasets", nargs="+", default=None)
    p.add_argument("--split", default="test")
    p.add_argument("--device", default=None)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--stage", action="store_true",
                   help="stage all datasets to fast disk once first")
    # accepted for backward compatibility, intentionally unused in v4:
    p.add_argument("--run-name", default=None)
    p.add_argument("--models", nargs="+", default=None)
    p.add_argument("--source-datasets", nargs="+", default=None)
    p.add_argument("--conf-thresh", type=float, default=None)
    p.add_argument("--iou-thresh", type=float, default=None)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--match-mode", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--force", action="store_true")
    a = p.parse_args()

    from training import cross_eval

    argv = []
    runs_dir = a.runs_dir or a.weights_dir
    if runs_dir:
        argv += ["--runs-dir", runs_dir]
    if a.data_dir:
        argv += ["--data-root", a.data_dir]
    if a.datasets:
        argv += ["--datasets", *a.datasets]
    argv += ["--split", a.split, "--batch", str(a.batch_size)]
    if a.device:
        argv += ["--device", a.device]
    if a.stage:
        argv += ["--stage"]
    cross_eval.main(argv)


if __name__ == "__main__":
    main()
