#!/usr/bin/env python3
"""
evaluate.py — evaluate a trained run on one or more cleaned datasets.

Produces: overall mAP50 / mAP50-95 / precision / recall, per-class table
(with ground-truth instance counts), inference speed, parameter count.
Results are saved next to the run as eval_<dataset>_<split>.json and printed.

Usage:
  python training/evaluate.py --run runs/train/visdrone_yolo26s_xxx
  python training/evaluate.py --weights runs/.../best.pt --datasets codrone visdrone dota
  python training/evaluate.py --run ... --split val --imgsz 1280
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402


def evaluate_weights(weights: str, dataset: str, split: str = "test",
                     data_root: str | None = None, imgsz: int = 1280,
                     batch: int = 16, device: str | None = None,
                     out_dir: str | None = None, stage_local: bool = False,
                     save: bool = True, verbose: bool = True) -> dict:
    """Run Ultralytics val() on one dataset/split; return (and optionally
    save + print) a structured result dict."""
    from ultralytics import YOLO

    data_root = common.resolve_data_root(data_root)
    if stage_local:
        data_root, ds_yaml = common.stage_local(dataset, data_root)
    else:
        ds_yaml = common.ensure_dataset_yaml(dataset, data_root)

    model = YOLO(weights)
    val_name = f"eval_{dataset}_{split}"
    m = model.val(data=ds_yaml, split=split, imgsz=imgsz, batch=batch,
                  device=device, workers=0 if device == "cpu" else 8,
                  plots=True, verbose=False,
                  project=out_dir or os.path.dirname(os.path.abspath(weights)),
                  name=val_name, exist_ok=True)

    box = m.box
    per_class = []
    for i, cidx in enumerate(box.ap_class_index):
        p, r, ap50, ap = box.class_result(i)
        c = int(cidx)
        per_class.append(dict(class_id=c,
                              name=common.CLASSES[c] if c < len(common.CLASSES) else f"cls{c}",
                              precision=round(float(p), 4),
                              recall=round(float(r), 4),
                              ap50=round(float(ap50), 4),
                              ap50_95=round(float(ap), 4)))
    per_class.sort(key=lambda d: d["class_id"])

    n_params = sum(p.numel() for p in model.model.parameters())
    res = dict(
        weights=os.path.abspath(weights),
        dataset=dataset, split=split, imgsz=imgsz,
        overall={k: round(float(v), 5) for k, v in m.results_dict.items()
                 if isinstance(v, (int, float))},
        map50=round(float(box.map50), 5),
        map50_95=round(float(box.map), 5),
        map75=round(float(box.map75), 5),
        speed_ms={k: round(float(v), 2) for k, v in (m.speed or {}).items()},
        parameters=int(n_params),
        per_class=per_class,
        gt_instances=common.count_instances(dataset, data_root, split),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    if save:
        out_dir = out_dir or os.path.dirname(os.path.abspath(weights))
        out = os.path.join(out_dir, f"eval_{dataset}_{split}.json")
        os.makedirs(out_dir, exist_ok=True)
        with open(out, "w") as f:
            json.dump(res, f, indent=2)
        res["json_path"] = out

    if verbose:
        print(f"\n=== {os.path.basename(weights)} on {dataset}/{split} "
              f"(imgsz={imgsz}) ===")
        print(f"mAP50: {res['map50']:.4f}   mAP50-95: {res['map50_95']:.4f}   "
              f"mAP75: {res['map75']:.4f}   params: {n_params/1e6:.2f}M")
        if res["speed_ms"]:
            sp = res["speed_ms"]
            print("speed(ms): " + "  ".join(f"{k}={v}" for k, v in sp.items()))
        print(f"{'class':<14}{'GT':>7}{'P':>8}{'R':>8}{'AP50':>8}{'AP50-95':>9}")
        for d in per_class:
            print(f"{d['name']:<14}{res['gt_instances'].get(d['name'], 0):>7}"
                  f"{d['precision']:>8.3f}{d['recall']:>8.3f}"
                  f"{d['ap50']:>8.3f}{d['ap50_95']:>9.3f}")
        missing = [c for c in common.CLASSES
                   if c not in {d["name"] for d in per_class}]
        if missing:
            print("classes with no predictions/GT in this split:", missing)
        if save:
            print("saved ->", res.get("json_path"))
    return res


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Evaluate a trained OBB run")
    p.add_argument("--run", help="run directory containing best.pt")
    p.add_argument("--weights", help="explicit weights path (or use --run)")
    p.add_argument("--datasets", nargs="+", choices=common.DATASETS,
                   help="default: the run's training dataset (from manifest), "
                        "else all three")
    p.add_argument("--split", default="test", choices=common.SPLITS)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default=None)
    p.add_argument("--data-root", default=None)
    p.add_argument("--no-save", action="store_true")
    aa = p.parse_args(argv)
    if not aa.run and not aa.weights:
        p.error("need --run or --weights")
    return aa


def main(argv=None):
    a = parse_args(argv)
    weights = common.find_weights(a.run or a.weights)
    run_dir = a.run or os.path.dirname(os.path.abspath(weights))

    datasets = a.datasets
    if not datasets:
        man = os.path.join(run_dir, "run_manifest.json")
        if os.path.isfile(man):
            with open(man) as f:
                td = json.load(f).get("dataset")
            if td in common.DATASETS:
                datasets = [td]
        if not datasets:
            datasets = common.REAL_DATASETS

    data_root = common.resolve_data_root(a.data_root)
    for ds in datasets:
        evaluate_weights(weights, ds, split=a.split, data_root=data_root,
                         imgsz=a.imgsz, batch=a.batch, device=a.device,
                         out_dir=run_dir, save=not a.no_save)


if __name__ == "__main__":
    main()
