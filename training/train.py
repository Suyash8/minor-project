#!/usr/bin/env python3
"""
train.py — train YOLO-OBB models (YOLO26 by default) on the cleaned datasets.

Optimized defaults for THIS data (drone/aerial, small vehicles, 8 classes,
images pre-resized to 1280 px long side):
  imgsz=1280 (native cleaned resolution), epochs=150 + patience=30 early stop,
  cosine LR, close_mosaic=10 (mosaic off for the last 10 epochs so small-box
  regression settles), auto batch size (fits the GPU you get), AMP on.

Every run writes a run_manifest.json (model, dataset, git commit, args) and,
after training, evaluates best.pt on the test split (eval_<ds>_test.json).

Quick starts (Colab):
  python training/train.py --datasets visdrone --models yolo26s-obb
  python training/train.py --cfg configs/experiments/visdrone_yolo26s.yaml
  python training/train.py --datasets all --models yolo26m-obb --epochs 100
  python training/train.py --resume runs/train/visdrone_yolo26s_xxx/last.pt

Dry run (prints the resolved config, no training):
  python training/train.py --datasets dota --models yolo26n-obb --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

DEFAULTS = dict(
    epochs=150,
    imgsz=1280,
    batch=-1,           # -1 = Ultralytics autobatch (GPU); CPU fallback 8
    workers=8,
    device=None,        # None = auto
    patience=30,
    cos_lr=True,
    close_mosaic=10,
    seed=42,
    plots=True,
    val=True,
    amp=True,
)

VALID_MODELS_RE = re.compile(r"^yolo(v8|11|12|26)?[nslmx]?-obb$", re.I)


# --------------------------------------------------------------------------
def parse_value(v: str):
    """'150' -> 150, 'true' -> True, '0.01' -> 0.01, 'none' -> None."""
    vl = v.strip().lower()
    if vl in ("true", "yes"):
        return True
    if vl in ("false", "no"):
        return False
    if vl in ("none", "null"):
        return None
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v


def load_cfg(path: str) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f) or {}


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Train YOLO26-OBB on the cleaned vehicle datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--datasets", nargs="+", choices=common.DATASETS,
                   help="which cleaned dataset(s) to train on")
    p.add_argument("--models", nargs="+",
                   help="Ultralytics OBB weights, e.g. yolo26s-obb yolo26m-obb")
    p.add_argument("--cfg", help="experiment yaml (configs/experiments/*.yaml); "
                   "CLI args override it")
    p.add_argument("--epochs", type=int)
    p.add_argument("--imgsz", type=int)
    p.add_argument("--batch", type=int)
    p.add_argument("--device", help="e.g. 0, 0,1, cpu (default: auto)")
    p.add_argument("--workers", type=int)
    p.add_argument("--patience", type=int)
    p.add_argument("--runs-dir", help="where runs go (default: Drive in Colab, ./runs locally)")
    p.add_argument("--data-root", help="cleaned_data root (default: auto)")
    p.add_argument("--name", help="run name (default: <ds>_<model>_<timestamp>)")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                   help="extra Ultralytics train kwargs, e.g. --set lr0=0.002 degrees=10")
    p.add_argument("--no-stage-local", action="store_true",
                   help="train straight from the data root (skip the fast local copy)")
    p.add_argument("--no-eval-test", action="store_true",
                   help="skip the automatic best.pt evaluation on the test split")
    p.add_argument("--resume", help="path to a run's last.pt to resume from")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def resolve_jobs(a) -> list[dict]:
    """Merge cfg file + CLI into a list of (dataset, model, train_kwargs) jobs."""
    cfg = load_cfg(a.cfg) if a.cfg else {}
    if a.cfg and not os.path.isfile(a.cfg):
        sys.exit(f"error: --cfg file not found: {a.cfg}")

    datasets = a.datasets or ([cfg["dataset"]] if cfg.get("dataset") else None)
    models = a.models or ([cfg["model"]] if cfg.get("model") else None)
    if not datasets:
        sys.exit("error: no dataset selected (use --datasets, or dataset: in --cfg)")
    if not models and not a.resume:
        models = ["yolo26s-obb"]

    kw = dict(DEFAULTS)
    kw.update(cfg.get("train_kwargs") or {})
    for k in ("epochs", "imgsz", "batch", "workers", "patience", "device"):
        v = getattr(a, k)
        if v is not None:
            kw[k] = v
        elif k in cfg:
            kw[k] = cfg[k]
    for kv in a.set:
        if "=" not in kv:
            sys.exit(f"error: --set expects KEY=VALUE, got '{kv}'")
        k, v = kv.split("=", 1)
        kw[k.strip()] = parse_value(v)

    jobs = []
    for ds in datasets:
        for m in (models or [None]):
            if m is not None and not VALID_MODELS_RE.match(m):
                sys.exit(f"error: model '{m}' does not look like a YOLO OBB "
                         "model (expected e.g. yolo26s-obb, yolo11m-obb)")
            jobs.append(dict(dataset=ds, model=m, train_kwargs=dict(kw)))
    return jobs


def run_one(job: dict, a, data_root: str, runs_dir: str) -> str | None:
    from ultralytics import YOLO

    ds, model, kw = job["dataset"], job["model"], job["train_kwargs"]
    device = kw.pop("device", None)
    kw["batch"] = kw.get("batch", -1)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = a.name or f"{ds}_{(model or 'resume').replace('-obb', '')}_{ts}"

    train_kwargs = dict(
        data=f"<dataset.yaml for {ds}>",
        project=runs_dir,
        name=name,
        device=device,
        **kw,
    )
    print("\n" + "=" * 70)
    print(f"TRAIN  dataset={ds}  model={model or '(resume)'}  run={name}")
    print("=" * 70)
    print(json.dumps(train_kwargs, indent=1, default=str))

    if a.dry_run:
        print("[dry-run] no training started")
        return None

    # fast local staging (Colab: Drive FUSE reads would dominate epoch time)
    if a.resume:
        pass          # resume reuses the yaml stored in the run's args.yaml
    elif not a.no_stage_local:
        data_root, ds_yaml = common.stage_local(ds, data_root)
    else:
        ds_yaml = common.ensure_dataset_yaml(ds, data_root)
    train_kwargs["data"] = ds_yaml

    if a.resume:
        yolo = YOLO(a.resume)
        yolo.train(resume=True)
        run_dir = str(getattr(yolo.trainer, "save_dir", os.path.dirname(os.path.dirname(a.resume))))
    else:
        yolo = YOLO(model)             # downloads DOTAv1-pretrained weights
        yolo.train(**train_kwargs)
        run_dir = str(getattr(yolo.trainer, "save_dir", ""))

    # manifest + final test evaluation
    common.write_manifest(run_dir, dict(
        kind="train", dataset=ds, model=model or "(resume)",
        resume=bool(a.resume), data_root=data_root,
        train_kwargs={k: str(v) for k, v in train_kwargs.items()}))

    try:
        best = common.find_weights(run_dir)
    except FileNotFoundError:
        best = None
    if not a.no_eval_test and best:
        try:
            import evaluate
            evaluate.evaluate_weights(
                best, ds, split="test", data_root=data_root,
                imgsz=kw.get("imgsz", 1280), device=device,
                out_dir=run_dir, stage_local=not a.no_stage_local)
        except Exception as e:                      # never lose a finished run
            print(f"[warn] post-train test eval failed: {e}")
    return run_dir


def main(argv=None):
    a = parse_args(argv)
    data_root = common.resolve_data_root(a.data_root)
    runs_dir = common.default_runs_dir(a.runs_dir)
    os.makedirs(runs_dir, exist_ok=True)
    print(f"data_root = {data_root}\nruns_dir  = {runs_dir}")

    if a.resume:
        a.datasets = a.datasets or ["_resume"]
        jobs = [dict(dataset="_resume", model=None,
                     train_kwargs=dict(DEFAULTS))]
        run_one(jobs[0], a, data_root, runs_dir)
        return

    jobs = resolve_jobs(a)
    done = []
    for job in jobs:
        try:
            rd = run_one(job, a, data_root, runs_dir)
            if rd:
                done.append((job["dataset"], job["model"], rd))
        except KeyboardInterrupt:
            print("\n[interrupt] stopping; finished runs so far:")
            for d, m, r in done:
                print(f"  {d}/{m}: {r}")
            raise
    print("\n" + "=" * 70)
    for d, m, r in done:
        print(f"DONE {d} {m} -> {r}")
    if len(done) > 1:
        print("\nnext: python training/cross_eval.py   # cross-dataset matrix")


if __name__ == "__main__":
    main()
