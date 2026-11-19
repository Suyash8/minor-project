#!/usr/bin/env python3
"""
eval_all_v43.py — Validation-only v4.3 full-accounting evaluator across all trained runs.

Discovers every trained run under `runs/` (preferring `best.pt`, falling back to
`last.pt` if interrupted), skips pairs that already have a valid
`v4.3-full-accounting` eval JSON (unless `--force` or `--quick`), backs up prior
eval JSONs to `eval_<dataset>_<split>.json.bak-v42`, runs the unified evaluator
(`training/evaluate.py`), and exports consolidated outputs in the repo root:
  - runs_v4_full_metrics.json
  - runs_v4_full_metrics_overall.csv
  - runs_v4_full_metrics_perclass.csv
  - runs_v4_full_metrics_bundle.zip
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import shutil
import sys
import time
import traceback
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(REPO_ROOT), str(REPO_ROOT / "training")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import common  # noqa: E402
from src.training.dispatcher import get_model_family, get_tuned_defaults  # noqa: E402

METRIC_VERSION = "v4.3-full-accounting"

OVERALL_COLS = [
    "run",
    "dataset",
    "model",
    "family",
    "weights_used",
    "imgsz",
    "map50",
    "map75",
    "map50_95",
    "ap_small",
    "ap_medium",
    "ap_large",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "iou_mean",
    "iou_pairs",
    "instances",
    "predictions",
    "tp",
    "fp",
    "fn",
    "speed_ms_inference",
    "imgs_s",
    "parameters_m",
    "angle_mae",
]

PERCLASS_COLS = [
    "run",
    "dataset",
    "model",
    "class_id",
    "class",
    "instances",
    "tp",
    "fp",
    "fn",
    "precision",
    "recall",
    "f1",
    "iou_mean",
    "ap50",
    "ap75",
    "ap50_95",
    "ap_small",
    "ap_medium",
    "ap_large",
]


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def _resolve_runs_dir(explicit: Optional[str]) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    local_runs = REPO_ROOT / "runs"
    if local_runs.is_dir():
        return local_runs
    env_runs = os.environ.get("VEHICLE_OBB_RUNS")
    if env_runs:
        return Path(env_runs).expanduser().resolve()
    fallback = Path(common.default_runs_dir())
    if fallback.is_dir():
        return fallback
    return local_runs


def discover_runs(runs_dir: Path) -> List[Tuple[Path, Path, str]]:
    """Discover trained runs under `runs_dir`.

    Globs `best.pt` recursively, then checks for `last.pt` in run directories
    without `best.pt`. Handles both `<run>/weights/{best,last}.pt` and flat
    `<run>/{best,last}.pt` layouts.
    """
    if not runs_dir.is_dir():
        return []

    candidates: Dict[Path, Dict[str, Path]] = {}
    for pt_name in ("best.pt", "last.pt"):
        for p in sorted(runs_dir.rglob(pt_name)):
            if not p.is_file():
                continue
            if any(part.startswith(".") or ".bak" in part for part in p.relative_to(runs_dir).parts):
                continue
            run_dir = p.parent.parent if p.parent.name == "weights" else p.parent
            slot = candidates.setdefault(run_dir.resolve(), {})
            if pt_name not in slot or p.parent.name == "weights":
                slot[pt_name] = p

    discovered: List[Tuple[Path, Path, str]] = []
    for run_dir in sorted(candidates):
        slot = candidates[run_dir]
        if "best.pt" in slot:
            discovered.append((run_dir, slot["best.pt"], "best.pt"))
        elif "last.pt" in slot:
            discovered.append((run_dir, slot["last.pt"], "last.pt"))
    return discovered


def parse_run_metadata(run_dir: Path) -> Tuple[str, str, int]:
    """Resolve `(model, dataset, imgsz)` from `run_manifest.json` or dir name."""
    manifest = _load_json(run_dir / "run_manifest.json") or {}
    model = manifest.get("model") or manifest.get("model_name")
    dataset = manifest.get("dataset") or manifest.get("dataset_name")

    if not model or not dataset:
        name = run_dir.name
        if "_" in name:
            ds_part, model_part = name.split("_", 1)
            dataset = dataset or ds_part
            model = model or model_part

    dataset_str = str(dataset).lower().strip() if dataset else "unknown"
    model_str = str(model).strip() if model else run_dir.name

    tuned = get_tuned_defaults(model_str)
    imgsz_val = tuned.get("img_size")
    if not imgsz_val:
        rt = manifest.get("resolved_tuning") or manifest.get("train_kwargs") or {}
        imgsz_val = (
            rt.get("img_size")
            or rt.get("imgsz")
            or manifest.get("imgsz")
            or manifest.get("img_size")
        )
    try:
        imgsz = int(imgsz_val) if imgsz_val else 1024
    except (TypeError, ValueError):
        imgsz = 1024

    return model_str, dataset_str, imgsz


def _run_with_v43_capture(
    evaluate_mod: Any,
    run_dir: Path,
    weights_path: Path,
    dataset: str,
    split: str,
    model_name: str,
    imgsz: int,
    batch: int,
    device: Optional[str],
    data_root: Optional[str],
    max_batches: Optional[int],
    save: bool,
) -> Dict[str, Any]:
    """Run unified evaluation while stashing v4.3 `compute_map_metrics` output.

    If `training/evaluate.py` already attaches `per_class`/`accounting` directly,
    that result is kept; otherwise the stashed output is merged in so Change 3
    works even against an unpatched evaluator.
    """
    import src.metrics.detection_map as det_map

    stashed: Dict[str, Any] = {}
    orig_compute = det_map.compute_map_metrics

    def _wrapped_compute(all_gt, all_preds, class_names):
        out = orig_compute(all_gt, all_preds, class_names)
        stashed["per_class"] = out.get("per_class", [])
        stashed["accounting"] = out.get("accounting", {})
        return out

    det_map.compute_map_metrics = _wrapped_compute
    try:
        try:
            default_w = common.find_weights(str(run_dir))
            same_weights = Path(default_w).resolve() == weights_path.resolve()
        except FileNotFoundError:
            same_weights = False

        if same_weights:
            res = evaluate_mod.evaluate_run(
                run_dir=str(run_dir),
                dataset=dataset,
                split=split,
                imgsz=imgsz,
                batch=batch,
                device=device,
                data_root=data_root,
                model_name=model_name,
                max_batches=max_batches,
                save=save,
            )
        else:
            family = get_model_family(model_name)
            if family == "yolo":
                res = evaluate_mod.evaluate_yolo_unified(
                    weights=str(weights_path),
                    dataset=dataset,
                    split=split,
                    data_root=data_root,
                    imgsz=imgsz,
                    batch=batch,
                    device=device,
                    max_batches=max_batches,
                    out_dir=str(run_dir),
                    save=save,
                    model_name=model_name,
                )
            else:
                res = evaluate_mod.evaluate_generic(
                    model_name=model_name,
                    weights=str(weights_path),
                    dataset=dataset,
                    split=split,
                    data_root=data_root,
                    imgsz=imgsz,
                    batch=batch,
                    device=device,
                    max_batches=max_batches,
                    out_dir=str(run_dir),
                    save=save,
                )
    finally:
        det_map.compute_map_metrics = orig_compute

    if (not res.get("per_class") or "tp" not in res["per_class"][0]) and stashed.get("per_class"):
        res["per_class"] = stashed["per_class"]
    if not res.get("accounting") and stashed.get("accounting"):
        res["accounting"] = stashed["accounting"]
    return res


def _csv_val(val: Any) -> Any:
    return "" if val is None else val


def _overall_row(entry: Dict[str, Any]) -> Dict[str, Any]:
    acc = entry.get("accounting") or {}
    speed = entry.get("speed_ms") or {}
    params = entry.get("parameters")
    params_m = (
        round(float(params) / 1e6, 4)
        if isinstance(params, (int, float))
        else entry.get("parameters_m")
    )
    imgs_s = entry.get("imgs_per_sec")
    if imgs_s is None:
        imgs_s = entry.get("imgs_s")
    weights_used = entry.get("weights_used") or entry.get("evaluated_from")
    if not weights_used and entry.get("weights_selection"):
        weights_used = f"{entry['weights_selection']}.pt"

    row = {
        "run": entry.get("run", ""),
        "dataset": entry.get("dataset", ""),
        "model": entry.get("model", ""),
        "family": entry.get("family", ""),
        "weights_used": weights_used or "",
        "imgsz": entry.get("imgsz", ""),
        "map50": entry.get("map50"),
        "map75": entry.get("map75"),
        "map50_95": entry.get("map50_95"),
        "ap_small": entry.get("ap_small"),
        "ap_medium": entry.get("ap_medium"),
        "ap_large": entry.get("ap_large"),
        "macro_precision": acc.get("macro_precision"),
        "macro_recall": acc.get("macro_recall"),
        "macro_f1": acc.get("macro_f1"),
        "iou_mean": acc.get("iou_mean"),
        "iou_pairs": acc.get("iou_pairs"),
        "instances": acc.get("instances"),
        "predictions": acc.get("predictions"),
        "tp": acc.get("tp"),
        "fp": acc.get("fp"),
        "fn": acc.get("fn"),
        "speed_ms_inference": speed.get("inference"),
        "imgs_s": imgs_s,
        "parameters_m": params_m,
        "angle_mae": entry.get("angle_mae"),
    }
    return {k: _csv_val(row[k]) for k in OVERALL_COLS}


def _perclass_rows(entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    run_name = entry.get("run", "")
    dataset = entry.get("dataset", "")
    model = entry.get("model", "")
    for d in entry.get("per_class") or []:
        raw_row = {
            "run": run_name,
            "dataset": dataset,
            "model": model,
            "class_id": d.get("class_id"),
            "class": d.get("name"),
            "instances": d.get("instances"),
            "tp": d.get("tp"),
            "fp": d.get("fp"),
            "fn": d.get("fn"),
            "precision": d.get("precision"),
            "recall": d.get("recall"),
            "f1": d.get("f1"),
            "iou_mean": d.get("iou_mean"),
            "ap50": d.get("ap50"),
            "ap75": d.get("ap75"),
            "ap50_95": d.get("ap50_95"),
            "ap_small": d.get("ap_small"),
            "ap_medium": d.get("ap_medium"),
            "ap_large": d.get("ap_large"),
        }
        rows.append({k: _csv_val(raw_row[k]) for k in PERCLASS_COLS})
    return rows


def _write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def _write_outputs(
    out_dir: Path,
    split: str,
    quick: bool,
    runs: List[Dict[str, Any]],
    errors: List[Dict[str, Any]],
) -> Tuple[Path, Path, Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "runs_v4_full_metrics.json"
    overall_path = out_dir / "runs_v4_full_metrics_overall.csv"
    perclass_path = out_dir / "runs_v4_full_metrics_perclass.csv"
    bundle_path = out_dir / "runs_v4_full_metrics_bundle.zip"

    payload = {
        "metric_version": METRIC_VERSION,
        "split": split,
        "quick": bool(quick),
        "generated": datetime.now(timezone.utc).isoformat(),
        "n_runs": len(runs),
        "n_errors": len(errors),
        "runs": runs,
        "errors": errors,
    }
    _save_json(json_path, payload)

    overall_rows = [_overall_row(r) for r in runs]
    perclass_rows: List[Dict[str, Any]] = []
    for r in runs:
        perclass_rows.extend(_perclass_rows(r))

    _write_csv(overall_path, OVERALL_COLS, overall_rows)
    _write_csv(perclass_path, PERCLASS_COLS, perclass_rows)

    tmp_zip = bundle_path.with_name(f".{bundle_path.name}.{os.getpid()}.tmp")
    with zipfile.ZipFile(tmp_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in (json_path, overall_path, perclass_path):
            zf.write(p, arcname=p.name)
    os.replace(tmp_zip, bundle_path)

    return json_path, overall_path, perclass_path, bundle_path


def _format_progress(
    tag: str,
    run_name: str,
    model: str,
    dataset: str,
    imgsz: int,
    elapsed_s: float,
    res: Dict[str, Any],
) -> str:
    acc = res.get("accounting") or {}
    miou = acc.get("iou_mean")
    miou_s = f"{miou:.4f}" if isinstance(miou, (int, float)) else "n/a"
    map50 = res.get("map50")
    map50_s = f"{map50:.4f}" if isinstance(map50, (int, float)) else "n/a"
    return (
        f"[{tag}] {run_name} | model={model} | dataset={dataset} | imgsz={imgsz} | "
        f"elapsed={elapsed_s:.1f}s | mAP50={map50_s} | mIoU={miou_s} | "
        f"TP={acc.get('tp', '?')} FP={acc.get('fp', '?')} FN={acc.get('fn', '?')} "
        f"inst={acc.get('instances', '?')}"
    )


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="v4.3 full-accounting validation-only evaluation across all trained runs"
    )
    p.add_argument("--split", default="test", choices=common.SPLITS)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--quick", action="store_true", help="40-sample smoke pass (does not overwrite eval JSONs)")
    p.add_argument("--force", action="store_true", help="re-evaluate even if v4.3 JSON exists")
    p.add_argument("--datasets", nargs="+", default=None, help="optional dataset filter")
    p.add_argument("--device", default="auto")
    p.add_argument("--runs-dir", default=None, help="optional runs directory override (default: <repo>/runs)")
    p.add_argument("--data-root", default=None, help="optional cleaned-data root override")
    p.add_argument("--out-dir", default=None, help="optional output directory override (default: repo root)")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    runs_dir = _resolve_runs_dir(args.runs_dir)
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else REPO_ROOT
    data_root = common.resolve_data_root(args.data_root)
    ds_filter = {d.lower().strip() for d in args.datasets} if args.datasets else None

    discovered = discover_runs(runs_dir)
    if not discovered:
        print(f"[eval_all_v43][warn] No trained runs (best.pt/last.pt) discovered under {runs_dir}", flush=True)

    runs_results: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    evaluated_count = 0
    resumed_count = 0
    evaluate_mod = None
    torch_mod = None

    max_batches = max(1, (40 + max(args.batch, 1) - 1) // max(args.batch, 1)) if args.quick else None

    for run_dir, weights_path, weights_used in discovered:
        run_name = run_dir.name
        model, dataset, imgsz = parse_run_metadata(run_dir)
        if ds_filter is not None and dataset not in ds_filter:
            continue

        eff_split = "val" if (dataset == "mock" and args.split == "test") else args.split
        eval_json = run_dir / f"eval_{dataset}_{eff_split}.json"

        # RESUME: skip existing valid v4.3 JSON unless --force or --quick
        if not args.force and not args.quick and eval_json.is_file():
            existing = _load_json(eval_json)
            if (
                existing is not None
                and existing.get("metric_version") == METRIC_VERSION
                and not existing.get("quick_smoke")
                and (
                    "weights_fingerprint" not in existing
                    or common.eval_weights_match(existing, str(weights_path))
                )
            ):
                entry = dict(existing)
                entry.setdefault("evaluated_from", weights_used)
                entry["weights_used"] = weights_used
                entry["run"] = run_name
                runs_results.append(entry)
                resumed_count += 1
                print(
                    _format_progress("resume", run_name, model, dataset, imgsz, 0.0, entry),
                    flush=True,
                )
                continue

        # BACKUP prior non-v4.3 (or missing backup) eval JSON before overwriting
        if not args.quick and eval_json.is_file():
            bak_path = run_dir / f"eval_{dataset}_{eff_split}.json.bak-v42"
            existing = _load_json(eval_json)
            if not bak_path.exists() or (existing and existing.get("metric_version") != METRIC_VERSION):
                shutil.copy2(eval_json, bak_path)

        t0 = time.time()
        try:
            if evaluate_mod is None:
                import evaluate as evaluate_mod  # type: ignore  # noqa: E402
            if torch_mod is None:
                import torch as torch_mod  # type: ignore  # noqa: E402

            res = _run_with_v43_capture(
                evaluate_mod=evaluate_mod,
                run_dir=run_dir,
                weights_path=weights_path,
                dataset=dataset,
                split=eff_split,
                model_name=model,
                imgsz=imgsz,
                batch=args.batch,
                device=args.device,
                data_root=data_root,
                max_batches=max_batches,
                save=(not args.quick),
            )
            elapsed = time.time() - t0

            res["metric_version"] = METRIC_VERSION
            res["evaluated_from"] = weights_used
            res["weights_used"] = weights_used
            if args.quick:
                res["quick_smoke"] = True

            if not args.quick and eval_json.is_file():
                on_disk = _load_json(eval_json) or {}
                if (
                    on_disk.get("metric_version") != METRIC_VERSION
                    or on_disk.get("evaluated_from") != weights_used
                    or "accounting" not in on_disk
                ):
                    _save_json(eval_json, res)

            entry = dict(res)
            entry["run"] = run_name
            runs_results.append(entry)
            evaluated_count += 1
            tag = "quick" if args.quick else "eval"
            print(
                _format_progress(tag, run_name, model, dataset, imgsz, elapsed, entry),
                flush=True,
            )
        except Exception as exc:
            traceback.print_exc()
            errors.append(
                {
                    "run": run_name,
                    "dataset": dataset,
                    "model": model,
                    "weights_used": weights_used,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            if torch_mod is not None:
                try:
                    if torch_mod.cuda.is_available():
                        torch_mod.cuda.empty_cache()
                except Exception:
                    pass
        finally:
            gc.collect()
            if torch_mod is not None:
                try:
                    if torch_mod.cuda.is_available():
                        torch_mod.cuda.empty_cache()
                except Exception:
                    pass

    json_p, overall_p, perclass_p, zip_p = _write_outputs(
        out_dir=out_dir,
        split=args.split,
        quick=args.quick,
        runs=runs_results,
        errors=errors,
    )

    print("\n" + "=" * 72, flush=True)
    print(
        f"[eval_all_v43] Summary: evaluated={evaluated_count} | resumed={resumed_count} | "
        f"total_runs={len(runs_results)} | errors={len(errors)}",
        flush=True,
    )
    if errors:
        print("[eval_all_v43] Errors:", flush=True)
        for err in errors:
            print(f"  - {err['run']}: {err['error']}", flush=True)
    print("[eval_all_v43] Output files:", flush=True)
    print(f"  1. {json_p}", flush=True)
    print(f"  2. {overall_p}", flush=True)
    print(f"  3. {perclass_p}", flush=True)
    print(f"  4. {zip_p}", flush=True)
    print("=" * 72, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
