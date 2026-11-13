#!/usr/bin/env python3
"""
[COMPATIBILITY ALIAS] training/train.py -> training/run.py (v4)

Kept working so existing notebooks and muscle memory don't break. The
canonical entry point is now:

    python training/run.py --datasets visdrone --models yolo26s-obb
    python training/run.py --models all --datasets all --mode full

Flag mapping: --datasets/--models/--epochs/--batch/--device/--workers/
--patience/--runs-dir pass through; --imgsz -> --img-size;
--data-root -> --data-dir; --no-stage-local drops the stage step;
--no-eval-test selects --mode train (default here: stage,train,eval);
--dry-run prints the resolved plan without training.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training import common  # noqa: E402

NOTICE = (
    "[deprecation] training/train.py is a compatibility alias in v4; the "
    "canonical entry point is `python training/run.py`."
)


def main() -> int:
    print(NOTICE, file=sys.stderr)
    p = argparse.ArgumentParser(description="(alias) train OBB models via "
                                            "training/run.py")
    p.add_argument("--datasets", nargs="+", choices=common.DATASETS,
                   default=["visdrone"])
    p.add_argument("--models", nargs="+", default=["yolo26s-obb"])
    p.add_argument("--cfg", help="experiment yaml — NOT supported by the v4 "
                   "alias; pass the equivalent flags to training/run.py")
    p.add_argument("--epochs", type=int)
    p.add_argument("--imgsz", type=int)
    p.add_argument("--batch", type=int)
    p.add_argument("--device", help="e.g. 0, 0,1, cpu (default: auto)")
    p.add_argument("--workers", type=int)
    p.add_argument("--patience", type=int)
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--data-root")
    p.add_argument("--name", help="ignored in v4 (deterministic "
                   "runs/train/<ds>_<model> layout)")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                   help="ignored in v4; use explicit training/run.py flags")
    p.add_argument("--no-stage-local", action="store_true")
    p.add_argument("--no-eval-test", action="store_true")
    p.add_argument("--resume", help="checkpoint paths are auto-resumed in v4; "
                   "this flag only forces resume on")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    if a.cfg:
        p.error("--cfg experiment yamls were retired in v4; pass the same "
                "values as flags to training/run.py")
    if a.name:
        print("[alias] --name ignored: v4 uses runs/train/<dataset>_<model>",
              file=sys.stderr)
    if a.set:
        print(f"[alias] --set overrides ignored: {a.set}", file=sys.stderr)

    argv = ["--models", *a.models, "--datasets", *a.datasets,
            "--runs-dir", a.runs_dir]
    steps = [] if a.no_stage_local else ["stage"]
    steps.append("train")
    if not a.no_eval_test:
        steps.append("eval")
    argv += ["--steps", ",".join(steps)]
    if a.epochs is not None:
        argv += ["--epochs", str(a.epochs)]
    if a.imgsz is not None:
        argv += ["--img-size", str(a.imgsz)]
    if a.batch is not None:
        argv += ["--batch", str(a.batch)]
    if a.device:
        argv += ["--device", a.device]
    if a.workers is not None:
        argv += ["--workers", str(a.workers)]
    if a.patience is not None:
        argv += ["--patience", str(a.patience)]
    if a.data_root:
        argv += ["--data-dir", a.data_root]

    if a.dry_run:
        from src.config import resolve_model_names
        from src.training.dispatcher import apply_tuned_defaults
        models = resolve_model_names(a.models)
        print("resolved plan (dry run):")
        for ds in a.datasets:
            for m in models:
                print(f"  {ds:<10} {m:<18} steps={','.join(steps)} "
                      f"tuning={apply_tuned_defaults(m, epochs=a.epochs, img_size=a.imgsz, batch_size=a.batch, patience=a.patience)}")
        return 0

    from training import run
    return run.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
