#!/usr/bin/env python3
"""
training/run.py — the ONE unified pipeline for the aerial OBB benchmark (v4, W1).

Runs any subset of the phases in a single process:
    stage → train → eval → cross-eval → summarize → export → report

Everything else (`training/train.py`, the `scripts/*.py` CLIs, the notebooks)
is a thin wrapper around this file.

Examples:
    # Full smoke run of every model on the mock dataset (CI / no GPU / no data)
    python training/run.py --models all --datasets mock \
        --steps stage,train,eval,summary --test

    # Real run: tuned defaults from MODEL_TUNING apply automatically (§2.6)
    python training/run.py --models lsknet --datasets codrone

    # Family aliases and subsets
    python training/run.py --models yolo26 nano --datasets visdrone dota

    # Diagnostics for bug reports
    python training/run.py --version
    python training/run.py --env
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.config import (  # noqa: E402
    CANONICAL_MODELS,
    MODEL_ALIASES,
    MODEL_SIZE_RANK,
    resolve_model_names,
)
from src.training.dispatcher import (  # noqa: E402
    apply_tuned_defaults,
    get_model_family,
    train_model,
)
from training import common  # noqa: E402
from training.staging import StageHandle, check_disk_guard, start_background_stage  # noqa: E402
from src.training.train_dense import GracefulStop  # noqa: E402

PIPELINE_VERSION = "4.0.0"
STATE_FILENAME = "pipeline_state.json"
STEPS = ["stage", "train", "eval", "cross", "summary", "export", "report"]
MODE_PRESETS = {
    "full": ["stage", "train", "eval", "cross", "summary", "report"],
    "train": ["stage", "train"],
    "eval": ["eval", "summary"],
}
DEFAULT_MIRROR_DIR = "/content/drive/MyDrive/object-detection/runs_v4"


# ---------------------------------------------------------------------------
# --version / --env (W8.5)
# ---------------------------------------------------------------------------
def print_version() -> None:
    print(f"vehicle-obb pipeline v{PIPELINE_VERSION}")
    print(f"git commit: {common.git_commit()}")
    print(f"models: {len(CANONICAL_MODELS)} registered "
          f"({', '.join(sorted(MODEL_ALIASES))} aliases)")


def print_env(data_root: Optional[str] = None) -> None:
    import torch

    print(f"pipeline version : {PIPELINE_VERSION}")
    print(f"git commit       : {common.git_commit()}")
    print(f"python           : {platform.python_version()} ({sys.executable})")
    print(f"platform         : {platform.platform()}")
    print(f"torch            : {torch.__version__}")
    print(f"torch CUDA build : {torch.version.cuda or 'cpu-only'}")
    print(f"cuda available   : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            print(f"gpu[{i}]           : {p.name} "
                  f"({p.total_memory / 1024**3:.1f} GB)")
    try:
        import ultralytics
        print(f"ultralytics      : {ultralytics.__version__}")
    except Exception:
        print("ultralytics      : NOT INSTALLED")
    try:
        import cv2
        print(f"opencv           : {cv2.__version__}")
    except Exception:
        print("opencv           : NOT INSTALLED")

    root = common.resolve_data_root(data_root)
    print(f"data root        : {root}")
    for ds in common.REAL_DATASETS + ["mock"]:
        ds_dir = os.path.join(root, ds)
        if os.path.isdir(ds_dir):
            n = sum(len(fs) for _r, _d, fs in os.walk(ds_dir))
            print(f"  {ds:<10}       : present ({n} files)")
        else:
            print(f"  {ds:<10}       : MISSING")
    print(f"registered models: {len(CANONICAL_MODELS)}")
    print(f"  yolo : {sum(1 for m in CANONICAL_MODELS if get_model_family(m) == 'yolo')}")
    print(f"  dense: {sum(1 for m in CANONICAL_MODELS if get_model_family(m) == 'dense')}")
    print(f"  detr : {sum(1 for m in CANONICAL_MODELS if get_model_family(m) == 'detr')}")


# ---------------------------------------------------------------------------
# Pipeline state (W7.1 job-queue resume)
# ---------------------------------------------------------------------------
class PipelineState:
    """Persistent job-queue state so a killed pipeline replays only what is
    left (`--auto-resume`, default on)."""

    def __init__(self, path: Path):
        self.path = path
        self.data: Dict[str, Any] = {
            "version": PIPELINE_VERSION,
            "created": datetime.now(timezone.utc).isoformat(),
            "jobs": [],
            "suite_steps": {},
        }

    @staticmethod
    def job_id(dataset: str, model: str) -> str:
        return f"{dataset}:{model}"

    def load(self) -> bool:
        if not self.path.exists():
            return False
        try:
            self.data = json.loads(self.path.read_text())
            return True
        except Exception as e:
            print(f"[pipeline][warn] could not read state file {self.path}: {e}")
            return False

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2))
        os.replace(tmp, self.path)

    def ensure_jobs(self, datasets: List[str], models: List[str]) -> None:
        known = {j["id"] for j in self.data["jobs"]}
        for ds in datasets:
            for m in models:
                jid = self.job_id(ds, m)
                if jid not in known:
                    self.data["jobs"].append({
                        "id": jid, "dataset": ds, "model": m,
                        "status": "pending", "error": None,
                        "traceback": None, "run_dir": None,
                        "updated": None,
                    })

    def job(self, dataset: str, model: str) -> Dict[str, Any]:
        jid = self.job_id(dataset, model)
        for j in self.data["jobs"]:
            if j["id"] == jid:
                return j
        raise KeyError(jid)

    def set_job(self, dataset: str, model: str, **fields: Any) -> None:
        j = self.job(dataset, model)
        j.update(fields)
        j["updated"] = datetime.now(timezone.utc).isoformat()
        self.save()

    def set_suite_step(self, step: str, status: str,
                       error: Optional[str] = None) -> None:
        self.data["suite_steps"][step] = {"status": status, "error": error}
        self.save()

    def summary_lines(self) -> List[str]:
        lines = []
        for j in self.data["jobs"]:
            line = f"  {j['dataset']:<10} {j['model']:<18} {j['status']}"
            if j.get("error"):
                line += f"  ({j['error']})"
            lines.append(line)
        return lines


# ---------------------------------------------------------------------------
# Job ordering (W1: small/fast first so a dying Colab session banks results)
# ---------------------------------------------------------------------------
def order_jobs(models: List[str], order: str) -> List[str]:
    key = lambda m: (MODEL_SIZE_RANK.get(m, 2), m)  # noqa: E731
    if order == "bigfirst":
        return sorted(models, key=lambda m: (-MODEL_SIZE_RANK.get(m, 2), m))
    return sorted(models, key=key)


# ---------------------------------------------------------------------------
# Test / smoke mode
# ---------------------------------------------------------------------------
def apply_test_overrides(args: argparse.Namespace) -> None:
    """--test: tiny settings that exercise the full code path end-to-end."""
    args.epochs = 1
    args.batch = 2
    args.micro_batch = 1
    args.img_size = 160
    args.workers = 0
    args.max_batches = 2
    args.order = args.order or "smallfirst"


def ensure_mock_data(data_root: str) -> None:
    mock_dir = Path(data_root) / "mock"
    if (mock_dir / "images" / "train").is_dir():
        return
    print("[pipeline] generating mock dataset (no GPU / no data path, W8.6)")
    from src.data.mock_data import create_mock_dataset
    create_mock_dataset("mock", Path(data_root), num_train=6, num_val=4,
                        img_size=(320, 320))


# ---------------------------------------------------------------------------
# Step implementations
# ---------------------------------------------------------------------------
def run_stage_step(datasets: List[str], args: argparse.Namespace,
                   handles: Dict[str, StageHandle]) -> None:
    """Stage each real dataset ONCE (background, overlapping training, W2)."""
    real = [d for d in datasets if d != "mock"]
    if not real:
        print("[stage] mock-only run — nothing to stage")
        return
    src_sets: List[str] = []
    for d in real:
        src_sets.extend(common.REAL_DATASETS if d == "all" else [d])
    src_sets = sorted(set(src_sets))
    data_root = common.resolve_data_root(args.data_dir)
    # W2 fix (review A1): the default stage root comes from staging.py's
    # platform-aware resolver (/content/data on Colab, source-adjacent
    # elsewhere), never a hardcoded /content path on plain Linux/Windows.
    from training.staging import stage_root as _stage_root
    stage_root = _stage_root(args.stage_dir)

    # W7.4 disk guard: raises a clear RuntimeError when free space is short
    check_disk_guard(src_sets, src=data_root, dst=stage_root)

    # W2.7 (review E): --prefetch val,test front-loads those splits so
    # evaluation reads never touch the source mount.
    prefetch_order: List[str] = []
    if args.prefetch:
        from training.staging import list_dataset_files
        want = [s.strip() for s in str(args.prefetch).split(",") if s.strip()]
        for d in src_sets:
            try:
                for rel, _sz in list_dataset_files(
                        os.path.join(data_root, d)):
                    parts = Path(rel).parts
                    if len(parts) >= 2 and parts[1] in want:
                        prefetch_order.append(rel)
            except OSError:
                pass
        if prefetch_order:
            print(f"[stage] prefetch: front-loading {len(prefetch_order)} "
                  f"files for splits {want}")

    handle = start_background_stage(
        src_sets, src_root=data_root, dst_root=stage_root,
        workers=args.stage_workers,
        order=prefetch_order or None)
    for d in real:
        handles[d] = handle
    print(f"[stage] background staging started for {src_sets} -> {stage_root}")


def _resolve_run_dir(args: argparse.Namespace, dataset: str, model: str) -> str:
    return str(Path(args.runs_dir) / "train" / f"{dataset}_{model}")


def run_train_job(dataset: str, model: str, args: argparse.Namespace,
                  handles: Dict[str, StageHandle],
                  state: PipelineState) -> Dict[str, Any]:
    """Train one (dataset, model) job with failure isolation (W1)."""
    run_dir = _resolve_run_dir(args, dataset, model)
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    state.set_job(dataset, model, status="running", run_dir=run_dir)

    tuned = apply_tuned_defaults(
        model, epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
        batch_size=args.batch, img_size=args.img_size, patience=args.patience)

    # W7.1: the manifest marks the run unfinished until final_metrics land;
    # --auto-resume scans for exactly this.
    manifest_path = Path(run_dir) / "run_manifest.json"
    manifest: Dict[str, Any] = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except Exception:
            manifest = {}
    manifest.update({
        "model": model, "dataset": dataset,
        "git_commit": common.git_commit(),
        "pipeline_version": PIPELINE_VERSION,
        "resolved_tuning": tuned,
        "status": "running",
        "test_mode": bool(args.test),
        "updated": datetime.now(timezone.utc).isoformat(),
    })
    manifest.setdefault("created", datetime.now(timezone.utc).isoformat())
    manifest_path.write_text(json.dumps(manifest, indent=2))

    t0 = time.time()
    try:
        handle = handles.get(dataset)
        result = train_model(
            model_name=model,
            dataset_name=dataset,
            epochs=tuned.get("epochs"),
            batch_size=tuned.get("batch_size"),
            img_size=tuned.get("img_size"),
            lr=tuned.get("lr"),
            weight_decay=tuned.get("weight_decay"),
            patience=tuned.get("patience"),
            device=args.device,
            workers=args.workers,
            resume=args.auto_resume,
            # W2 fix (review A1): while staging is active the dataset is
            # built from the SOURCE root (complete, stable inventory) and
            # reads are redirected to staged copies as they land. Without a
            # handle, the ordinary data root applies.
            data_dir=(handle.src_root if handle is not None
                      else args.data_dir),
            micro_batch=args.micro_batch,
            ema=args.ema,
            augment=not args.no_augment,
            resize_mode=args.resize_mode,
            compile_model=args.compile,
            deterministic=args.deterministic,
            seed=args.seed,
            mirror_dir=args.mirror_dir,
            mirror_every=args.mirror_every,
            staging=handle,
            run_dir=run_dir,
            max_batches=args.max_batches,
            test_mode=args.test,
            cache_mode=args.cache,
        )
        state.set_job(dataset, model, status="done", error=None, traceback=None)
        result["wall_time_s"] = round(time.time() - t0, 1)
        manifest["status"] = "done"
        manifest["final_metrics"] = {
            "best_map50": result.get("best_map50"),
            "best_val_loss": result.get("best_val_loss"),
            "epochs_trained": result.get("epochs_trained"),
            "oom_events": result.get("oom_events"),
            "wall_time_s": result["wall_time_s"],
        }
        manifest["updated"] = datetime.now(timezone.utc).isoformat()
        manifest_path.write_text(json.dumps(manifest, indent=2))
        return result
    except GracefulStop:
        # Signal during training: checkpoint + manifest already written by the
        # engine; keep the job pending so --auto-resume replays it.
        state.set_job(dataset, model, status="pending",
                      error="interrupted (signal)")
        manifest["status"] = "interrupted"
        manifest["updated"] = datetime.now(timezone.utc).isoformat()
        manifest_path.write_text(json.dumps(manifest, indent=2))
        raise KeyboardInterrupt
    except KeyboardInterrupt:
        state.set_job(dataset, model, status="pending",
                      error="interrupted (SIGINT)")
        raise
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[pipeline][FAIL] {dataset}/{model}: {e}", flush=True)
        state.set_job(dataset, model, status="failed", error=str(e),
                      traceback=tb)
        manifest_path = Path(run_dir) / "run_manifest.json"
        try:
            manifest = {}
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text())
            manifest.update({
                "status": "failed", "error": str(e), "traceback": tb,
                "model": model, "dataset": dataset,
                "git_commit": common.git_commit(),
                "pipeline_version": PIPELINE_VERSION,
            })
            manifest_path.write_text(json.dumps(manifest, indent=2))
        except Exception:
            pass
        return {"model_name": model, "dataset_name": dataset, "status": "failed",
                "error": str(e)}


def eval_data_root(datasets: List[str], args: argparse.Namespace,
                   handles: Dict[str, StageHandle]) -> Optional[str]:
    """Data root for eval/cross (review R5c): NEVER a partially-staged dir.

    resolve_data_root() prefers /content/data once it is non-empty, so an
    eval launched while background staging is still running would read a
    partial listing. With a live handle, read the SOURCE root until staging
    is complete (reads still overlap staging inside the families that
    support it; eval correctness beats speed). Without a handle, use the
    staged root only when every requested real dataset verifies complete
    against the true source.
    """
    from training.staging import stage_root as _stage_root, stage_status, \
        source_root as _source_root
    live = [h for h in (handles or {}).values() if h is not None]
    if live:
        complete = all(h.status().complete for h in live)
        root = live[0].dst_root if complete else live[0].src_root
        if not complete:
            print("[eval] staging INCOMPLETE - evaluating from the SOURCE "
                  f"root ({root}); staged copy is not trusted yet (R5c)",
                  flush=True)
        return root
    if args.data_dir:
        return args.data_dir
    src = _source_root(None)
    dst = _stage_root(args.stage_dir)
    real = [d for d in datasets if d in common.REAL_DATASETS]
    if real and os.path.abspath(src) != os.path.abspath(dst) \
            and os.path.isdir(dst):
        if all(stage_status(d, src_root=src, dst_root=dst).complete
               for d in real):
            return dst
        print("[eval] staged root is incomplete for at least one dataset; "
              f"evaluating from the source root ({src}) (R5c)", flush=True)
        return src
    return None  # resolve_data_root default (no staging in play)


def run_eval_step(datasets: List[str], models: List[str],
                  args: argparse.Namespace, state: PipelineState,
                  handles: Optional[Dict[str, StageHandle]] = None) -> None:
    from training.evaluate import evaluate_run

    eval_root = eval_data_root(datasets, args, handles or {})

    for ds in datasets:
        for m in models:
            job = state.job(ds, m)
            if job["status"] != "done" and not job.get("run_dir"):
                print(f"[eval] skipping {ds}/{m}: no completed training run")
                continue
            run_dir = job.get("run_dir") or _resolve_run_dir(args, ds, m)
            try:
                evaluate_run(
                    run_dir=run_dir, dataset=ds, split="test",
                    imgsz=(args.img_size or None),  # None -> trained img_size from manifest (review A5)
                    batch=args.batch or 8,
                    device=args.device, data_root=eval_root,
                    test_mode=args.test)
            except Exception as e:
                print(f"[eval][FAIL] {ds}/{m}: {e}", flush=True)
                state.set_job(ds, m, error=f"eval: {e}")


def run_suite_step(step: str, datasets: List[str], models: List[str],
                   args: argparse.Namespace, state: PipelineState,
                   handles: Optional[Dict[str, StageHandle]] = None) -> None:
    """cross / summary / export / report — suite-level, run once."""
    try:
        if step == "cross":
            from training import cross_eval
            argv = ["--runs-dir", args.runs_dir]
            cross_root = eval_data_root(datasets, args,
                                        handles or {})
            if cross_root or args.data_dir:
                argv += ["--data-root", cross_root or args.data_dir]
            cross_eval.main(argv)
        elif step == "summary":
            from training import summarize_runs
            argv = ["--runs-dir", args.runs_dir, "--out",
                    str(Path("results") / "master_summary.csv")]
            summarize_runs.main(argv)
        elif step == "export":
            # check=True (review D): a failed export must not be recorded as
            # "done" in the pipeline state.
            subprocess.run([sys.executable, "scripts/export_results.py"],
                           cwd=str(REPO_ROOT), check=True)
        elif step == "report":
            subprocess.run([sys.executable, "scripts/generate_report.py"],
                           cwd=str(REPO_ROOT), check=True)
        state.set_suite_step(step, "done")
        if step == "summary" and args.mirror_dir:
            # W6.4 (review D): the master summary rides the Drive mirror too.
            from src.training.engine_common import DriveMirror
            csv_path = Path("results") / "master_summary.csv"
            if csv_path.exists():
                m = DriveMirror(args.mirror_dir, run_subdir="_suite")
                m.submit(csv_path)
                m.flush()
    except Exception as e:
        print(f"[{step}][FAIL] {e}", flush=True)
        state.set_suite_step(step, "failed", error=str(e))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Unified aerial OBB benchmark pipeline (v4)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--models", nargs="+", default=["all"],
                   help="model names and/or aliases "
                        f"({', '.join(sorted(MODEL_ALIASES))})")
    p.add_argument("--datasets", nargs="+", default=["all"],
                   help="codrone | visdrone | dota | all | mock (multi ok)")
    p.add_argument("--steps", type=str, default=None,
                   help="comma-separated subset of " + ",".join(STEPS))
    p.add_argument("--mode", choices=sorted(MODE_PRESETS), default="full",
                   help="step preset (ignored when --steps is given)")
    p.add_argument("--order", choices=["smallfirst", "bigfirst"],
                   default="smallfirst",
                   help="job ordering: small/fast first banks results before a "
                        "time-limited session dies")
    p.add_argument("--test", action="store_true",
                   help="smoke mode: 1 epoch, 2 batches, imgsz 160, workers 0 "
                        "(use --datasets mock for the no-GPU/no-data path)")

    # override knobs (§2.6: explicit flags beat MODEL_TUNING)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch", type=int, default=None,
                   help="EFFECTIVE batch size (accumulation splits it into "
                        "micro-batches when VRAM requires)")
    p.add_argument("--micro-batch", type=int, default=None,
                   help="per-step micro batch override")
    p.add_argument("--img-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--max-batches", type=int, default=None,
                   help="cap train/val batches per epoch (smoke/debug)")

    # engine knobs
    p.add_argument("--device", default="auto")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--compile", action="store_true",
                   help="torch.compile the model (Linux, torch>=2.1)")
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--resize-mode", choices=["letterbox", "stretch"],
                   default="letterbox")
    p.add_argument("--ema", dest="ema", action="store_true", default=True)
    p.add_argument("--no-ema", dest="ema", action="store_false")

    # resilience (W7)
    p.add_argument("--auto-resume", dest="auto_resume", action="store_true",
                   default=True,
                   help="resume unfinished runs + replay the job queue")
    p.add_argument("--no-auto-resume", dest="auto_resume",
                   action="store_false")
    p.add_argument("--mirror-dir", default=None,
                   help=f"Drive mirror directory (default on Colab: "
                        f"{DEFAULT_MIRROR_DIR})")
    p.add_argument("--mirror-every", type=int, default=5)

    # paths
    p.add_argument("--data-dir", default=None)
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--stage-dir", default=None,
                   help="fast local staging root (default: "
                        "$VEHICLE_OBB_STAGE, else /content/data on Colab, "
                        "else alongside the source data)")
    p.add_argument("--stage-workers", type=int, default=16)
    # W3.7 / W2.7 (review E): real cache + prefetch controls
    p.add_argument("--cache", choices=["none", "annotations", "ram", "disk"],
                   default="annotations",
                   help="cache strategy: none = no caches; annotations = "
                        "per-split annotation cache (default); ram = "
                        "annotations + bounded decode-to-RAM image cache "
                        "(small datasets); disk = annotations + "
                        "decoded-image .npy cache on fast local disk "
                        "under the staging root")
    p.add_argument("--prefetch", default=None,
                   help="comma-separated splits to front-load in the staging "
                        "queue, e.g. val,test, so evaluation never reads "
                        "from the source mount")

    # diagnostics (W8.5)
    p.add_argument("--version", action="store_true",
                   help="print pipeline version + git commit and exit")
    p.add_argument("--env", action="store_true",
                   help="print full environment report and exit")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    if args.version:
        print_version()
        return 0
    if args.env:
        print_env(args.data_dir)
        return 0

    if args.test:
        apply_test_overrides(args)
    if args.mirror_dir is None and os.path.isdir("/content/drive"):
        args.mirror_dir = DEFAULT_MIRROR_DIR

    models = order_jobs(resolve_model_names(args.models), args.order)

    datasets = [d.lower().strip() for d in args.datasets]
    unknown_ds = [d for d in datasets
                  if d not in common.DATASETS and d != "mock"]
    if unknown_ds:
        raise SystemExit(f"unknown dataset(s): {unknown_ds} "
                         f"(choose from {common.DATASETS + ['mock']})")
    if "all" in datasets:
        datasets = ["codrone", "visdrone", "dota"]
    if "mock" in datasets:
        args.data_dir = args.data_dir or str(REPO_ROOT / "data")
        ensure_mock_data(common.resolve_data_root(args.data_dir))

    steps = ([s.strip() for s in args.steps.split(",") if s.strip()]
             if args.steps else list(MODE_PRESETS[args.mode]))
    bad_steps = [s for s in steps if s not in STEPS]
    if bad_steps:
        raise SystemExit(f"unknown step(s): {bad_steps} (choose from {STEPS})")

    print(f"{'=' * 72}\n vehicle-obb unified pipeline v{PIPELINE_VERSION}\n"
          f"{'=' * 72}")
    print(f"[pipeline] models   ({len(models)}): {', '.join(models)}")
    print(f"[pipeline] datasets ({len(datasets)}): {', '.join(datasets)}")
    print(f"[pipeline] steps: {', '.join(steps)} | order: {args.order} | "
          f"test mode: {args.test}")
    print(f"[pipeline] runs dir: {args.runs_dir} | "
          f"auto-resume: {args.auto_resume} | mirror: {args.mirror_dir}")

    state = PipelineState(Path(args.runs_dir) / STATE_FILENAME)
    if args.auto_resume:
        if state.load():
            print(f"[pipeline] resuming job queue from {state.path}")
    state.data["models"] = models
    state.data["datasets"] = datasets
    state.data["steps"] = steps
    state.ensure_jobs(datasets, models)
    state.save()

    handles: Dict[str, StageHandle] = {}
    try:
        if "stage" in steps:
            run_stage_step(datasets, args, handles)

        if "train" in steps:
            for ds in datasets:
                for m in models:
                    job = state.job(ds, m)
                    if args.auto_resume and job["status"] == "done":
                        print(f"[pipeline] skip {ds}/{m}: already done")
                        continue
                    print(f"\n[pipeline] === train {m} on {ds} ===")
                    run_train_job(ds, m, args, handles, state)

        if "eval" in steps:
            run_eval_step(datasets, models, args, state, handles)

        for suite_step in ("cross", "summary", "export", "report"):
            if suite_step in steps:
                run_suite_step(suite_step, datasets, models, args, state,
                               handles)

    except KeyboardInterrupt:
        print("\n[pipeline] interrupted — state saved; relaunch with "
              "--auto-resume to continue")
        state.save()
        return 0

    finally:
        for h in handles.values():
            try:
                h.wait(timeout=5)
            except Exception:
                pass

    failed = [j for j in state.data["jobs"] if j["status"] == "failed"]
    done = [j for j in state.data["jobs"] if j["status"] == "done"]
    print(f"\n{'=' * 72}\n pipeline finished: {len(done)} done, "
          f"{len(failed)} failed\n{'=' * 72}")
    for line in state.summary_lines():
        print(line)
    if failed:
        print(f"\n[!] {len(failed)} job(s) failed — tracebacks in "
              f"{state.path} and each run's run_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
