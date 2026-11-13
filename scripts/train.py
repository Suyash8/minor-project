#!/usr/bin/env python3
"""
[DEPRECATED SHIM] scripts/train.py -> training/run.py (v4)

Training is unified in `training/run.py` (W1): 23 models, family aliases,
per-model tuned defaults (MODEL_TUNING), background staging, failure
isolation, auto-resume. This shim maps the old flags so existing commands
keep working.

New canonical usage:
    python training/run.py --models lsknet --datasets codrone
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CANONICAL_MODELS  # noqa: E402
from training.common import DATASETS as SUPPORTED_DATASETS  # noqa: E402

NOTICE = (
    "[deprecation] scripts/train.py is a shim in v4; use "
    "`python training/run.py --models ... --datasets ...` "
    "(tuned per-model defaults apply automatically)."
)
SUPPORTED_MODELS = CANONICAL_MODELS


def main() -> None:
    print(NOTICE, file=sys.stderr)
    p = argparse.ArgumentParser(description="(shim) train OBB models")
    p.add_argument("--models", nargs="+", default=["yolov8n-obb"])
    p.add_argument("--datasets", nargs="+", default=["visdrone"],
                   help=f"choices: {SUPPORTED_DATASETS} or 'all'")
    p.add_argument("--epochs", type=int, default=None,
                   help="default: the model's tuned schedule in MODEL_TUNING")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--cache", type=str, default=None, choices=[None, "ram", "disk"],
                   help="ignored in v4 (fingerprinted label cache is automatic)")
    p.add_argument("--resume", type=lambda x: str(x).lower() in ("true", "1", "yes"),
                   default=True)
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--weights-dir", type=str, default=None,
                   help="ignored in v4; runs go under runs/train/")
    p.add_argument("--runs-dir", type=str, default="runs")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    if a.cache:
        print("[shim] --cache ignored: v4 uses a fingerprinted label cache "
              "automatically", file=sys.stderr)
    if a.weights_dir:
        print("[shim] --weights-dir ignored: v4 writes runs/train/<ds>_<model>/",
              file=sys.stderr)

    from training import run

    argv = ["--models", *a.models, "--datasets", *a.datasets,
            "--mode", "train", "--device", a.device,
            "--workers", str(a.workers), "--seed", str(a.seed),
            "--runs-dir", a.runs_dir]
    if a.epochs is not None:
        argv += ["--epochs", str(a.epochs)]
    if a.batch_size is not None:
        argv += ["--batch", str(a.batch_size)]
    if a.imgsz is not None:
        argv += ["--img-size", str(a.imgsz)]
    if a.data_dir:
        argv += ["--data-dir", a.data_dir]
    if not a.resume:
        argv += ["--no-auto-resume"]
    raise SystemExit(run.main(argv))


if __name__ == "__main__":
    main()
