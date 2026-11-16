#!/usr/bin/env python3
"""
[DEPRECATED SHIM] scripts/run_evaluation.py -> training/evaluate.py (v4)

Evaluation is unified in `training/evaluate.py` (W6): one protocol for all 23
models (Ultralytics val for YOLO; shared decode + rotated NMS + src/metrics
for dense/DETR), with sm/md/lg breakdown, confusion matrix and speed. This
shim maps the old flags so existing commands keep working and produce
byte-identical metrics to the canonical CLI.

New canonical usage:
    python training/evaluate.py --run runs/train/visdrone_lsknet --datasets visdrone
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

NOTICE = (
    "[deprecation] scripts/run_evaluation.py is a shim in v4; use "
    "`python training/evaluate.py --run <run_dir> --datasets ...`."
)


def main() -> None:
    print(NOTICE, file=sys.stderr)
    p = argparse.ArgumentParser(description="(shim) evaluate one checkpoint")
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--weights", type=str, default=None)
    p.add_argument("--run", type=str, default=None,
                   help="run directory (v4 style; preferred)")
    p.add_argument("--dataset", type=str, default=None)
    p.add_argument("--datasets", nargs="+", default=None)
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default=None)
    p.add_argument("--imgsz", type=int, default=None,
                   help="default: read run's trained resolution from manifest (F17)")
    p.add_argument("--iou-thresh", type=float, default=0.45)
    p.add_argument("--conf-thresh", type=float, default=0.05)
    p.add_argument("--max-samples", type=int, default=None,
                   help="forwarded as max_batches cap to the unified evaluator")
    p.add_argument("--allow-native-fallback", action="store_true")
    a = p.parse_args()

    from training import evaluate

    datasets = a.datasets or ([a.dataset] if a.dataset else None)
    if a.run:
        run_dir, weights = a.run, None
    elif a.weights:
        run_dir = str(Path(a.weights).resolve().parent.parent) \
            if Path(a.weights).parent.name == "weights" \
            else str(Path(a.weights).resolve().parent)
        weights = a.weights
    else:
        p.error("need --run or --weights")

    if weights and not a.model:
        p.error("--weights requires --model in v4 (or use --run so the model "
                "is read from run_manifest.json)")

    max_batches = max(1, (a.max_samples + a.batch_size - 1) // a.batch_size) if a.max_samples else None

    if weights:
        from src.training.dispatcher import get_model_family, get_tuned_defaults
        eff_imgsz = a.imgsz or int(get_tuned_defaults(a.model).get("img_size") or 1280)
        if get_model_family(a.model) == "yolo":
            for ds in (datasets or ["visdrone"]):
                evaluate.evaluate_yolo_unified(
                    weights, ds, split=a.split, data_root=a.data_dir,
                    imgsz=eff_imgsz, batch=a.batch_size, device=a.device,
                    conf_thresh=a.conf_thresh, iou_thresh=a.iou_thresh,
                    max_batches=max_batches, model_name=a.model,
                )
        else:
            for ds in (datasets or ["visdrone"]):
                evaluate.evaluate_generic(
                    a.model, weights, ds, split=a.split, data_root=a.data_dir,
                    imgsz=eff_imgsz, batch=a.batch_size, device=a.device,
                    conf_thresh=a.conf_thresh, iou_thresh=a.iou_thresh,
                    max_batches=max_batches,
                )
    else:
        for ds in (datasets or []):
            evaluate.evaluate_run(
                run_dir, ds, split=a.split, imgsz=a.imgsz,
                batch=a.batch_size, device=a.device,
                data_root=a.data_dir, model_name=a.model,
                allow_native_fallback=a.allow_native_fallback,
            )


if __name__ == "__main__":
    main()
