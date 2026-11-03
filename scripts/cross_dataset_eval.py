#!/usr/bin/env python3
# =============================================================================
# CROSS-DATASET EVALUATION for the Aerial OBB Benchmark Suite
# =============================================================================
# Evaluates every TRAINED checkpoint on EVERY available dataset, so you get:
#   - in-distribution results   (trained on visdrone -> tested on visdrone)
#   - out-of-distribution results (trained on visdrone -> tested on codrone/dota)
#
# Why run_pipeline.py cannot do this:
#   1. Its post-train weight lookup is strictly per (model, dataset) pair:
#      a visdrone-trained model evaluated on codrone silently gets weights_path
#      = None -> random-init heads -> meaningless ~0 metrics.
#   2. Model heads are built with the EVAL dataset's class count (10 vs 12 vs
#      15) and load_state_dict(strict=False) then silently DROPS the trained
#      classifier head. Class indices are positional, so predictions would be
#      mislabeled even if the backbone loaded.
#
# What this script does instead:
#   - Loads each checkpoint with its TRAINING class set (correct head shape).
#   - Predicts in the training class space, then remaps predictions to the
#     eval dataset's class space via a synonym table (people->pedestrian,
#     motor->motorcyclist, car->small-vehicle, ...).
#   - --match-mode all      : evaluate on ALL eval classes (honest OOD score;
#                             unmapped GT classes simply score 0).
#     --match-mode common   : evaluate only on classes shared between the
#                             training and eval datasets (fair cross-dataset
#                             comparison, standard practice in the literature).
#
# Outputs (all under --results-dir/<run-name>/):
#   checkpoints/<source>__<model>__<eval>.json   resumable per-pair results
#   per_class/<source>__<model>__<eval>.csv      per-class AP/TP/FP/FN/P/R/F1
#   run_logs/<run-name>/                         console.log + events.jsonl + run_config.json
#   cross_dataset_summary.csv                    one row per (model, source, eval)
#   global_summary.csv                           macro-average per (model, source)
#
# Usage (Colab):
#   python scripts/cross_dataset_eval.py \
#       --data-dir  /content/drive/MyDrive/object-detection/data \
#       --weights-dir /content/drive/MyDrive/object-detection/weights \
#       --results-dir /content/drive/MyDrive/object-detection/results \
#       --run-name cross_dataset_eval --device cuda
# =============================================================================

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATASET_CLASSES, SUPPORTED_DATASETS, SUPPORTED_MODELS
from src.data.dataset import AerialOBBDataset
from src.models import get_model
from src.metrics import (
    compute_map_metrics,
    compute_detection_confusion_matrix,
    compute_classification_metrics,
    compute_regression_metrics,
    compute_obb_iou_matrix,
)
from src.utils.env import set_seed
from src.utils.logging_utils import setup_run_logging, get_tracker
from src.utils.system import (
    auto_configure_hardware,
    check_memory_pressure,
    deep_cleanup_memory,
    dynamic_tune_batch_size,
)

# ---------------------------------------------------------------------------
# Cross-dataset class synonym table (source-name -> acceptable target names)
# ---------------------------------------------------------------------------
SYNONYMS = {
    "pedestrian": ["pedestrian", "people", "person"],
    "people": ["people", "pedestrian", "person"],
    "person": ["person", "pedestrian", "people"],
    "bicycle": ["bicycle", "cyclist", "bike"],
    "cyclist": ["cyclist", "bicycle", "bike"],
    "car": ["car", "small-vehicle", "automobile"],
    "van": ["van", "small-vehicle"],
    "small-vehicle": ["small-vehicle", "car", "van"],
    "truck": ["truck", "large-vehicle"],
    "bus": ["bus", "large-vehicle"],
    "large-vehicle": ["large-vehicle", "truck", "bus"],
    "tricycle": ["tricycle", "awning-tricycle"],
    "awning-tricycle": ["awning-tricycle", "tricycle"],
    "motor": ["motor", "motorcyclist", "motorbike"],
    "motorcyclist": ["motorcyclist", "motor", "motorbike"],
    "ship": ["ship", "boat", "vessel"],
    "boat": ["boat", "ship", "vessel"],
    "bridge": ["bridge"],
    "plane": ["plane", "airplane", "aircraft"],
    "airplane": ["airplane", "plane", "aircraft"],
    "helicopter": ["helicopter"],
    "storage-tank": ["storage-tank"],
    "harbor": ["harbour", "harbor"],
    "swimming-pool": ["swimming-pool", "pool"],
    "traffic_light": ["traffic_light", "traffic-light"],
    "traffic_sign": ["traffic_sign", "traffic-sign"],
}


def build_class_map(src_names, eval_names):
    """Map each source class index -> eval class index (or None if unmapped)."""
    eval_syn = []
    for e_name in eval_names:
        e_set = set(SYNONYMS.get(e_name, [e_name])) | {e_name}
        eval_syn.append(e_set)
    mapping = {}
    for s_idx, s_name in enumerate(src_names):
        s_set = set(SYNONYMS.get(s_name, [s_name])) | {s_name}
        target = None
        for e_idx, e_set in enumerate(eval_syn):
            if s_set & e_set:
                target = e_idx
                break
        mapping[s_idx] = target
    return mapping


def find_trained_weights(weights_dir: Path, model: str, source: str):
    candidates = [
        weights_dir / f"{model}_{source}_best.pt",
        weights_dir / f"{model}_{source}" / f"{model}_best.pt",
        weights_dir / f"{model}_{source}" / "best.pt",
        weights_dir / f"{model}_{source}" / "weights" / "best.pt",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return None


def as_class_dict(x):
    """Normalize per-image structures: repo returns lists indexed by class id."""
    return {i: v for i, v in enumerate(x)} if isinstance(x, list) else x


def empty_preds(n):
    return [{"boxes": np.zeros((0, 5)), "scores": np.zeros((0,))} for _ in range(n)]


def _boxes_of(entry):
    if entry is None:
        return np.zeros((0, 5))
    return np.asarray(entry.get("boxes", np.zeros((0, 5))), dtype=np.float32).reshape(-1, 5)


def remap_predictions(img_preds, src_to_eval, eval_index_map=None, n_target=None):
    """Re-key predictions from source-class space into eval-class space.
    Returns a full-length list (indexable by class id) of length n_target."""
    if n_target is None:
        if eval_index_map is not None:
            n_target = len(eval_index_map)
        else:
            n_target = max([v for v in src_to_eval.values() if v is not None],
                           default=-1) + 1
    out = empty_preds(n_target)
    if isinstance(img_preds, dict):
        items = img_preds.items()
    else:
        items = enumerate(img_preds)
    for s_idx, p in items:
        if p is None:
            continue
        e_idx = src_to_eval.get(s_idx)
        if e_idx is None:
            continue
        if eval_index_map is not None:
            if e_idx not in eval_index_map:
                continue
            e_idx = eval_index_map[e_idx]
        boxes = np.asarray(p.get("boxes", np.zeros((0, 5))), dtype=np.float32).reshape(-1, 5)
        scores = np.asarray(p.get("scores", np.zeros((0,))), dtype=np.float32).reshape(-1)
        if len(boxes) == 0 or e_idx >= n_target:
            continue
        out[e_idx]["boxes"] = np.concatenate([out[e_idx]["boxes"], boxes])
        out[e_idx]["scores"] = np.concatenate([out[e_idx]["scores"], scores])
    return out


def evaluate_pair(model, dataset, src_names, src_to_eval, args, device):
    """Full evaluation loop of one (model, source-weights) on one eval dataset."""
    n_samples = len(dataset)
    eval_names = dataset.class_names
    n_eval = len(eval_names)
    common_mode = args.match_mode == "common"

    # eval classes that can receive predictions
    mapped_targets = sorted({v for v in src_to_eval.values() if v is not None})
    eval_index_map = {old: new for new, old in enumerate(mapped_targets)} if common_mode else None

    all_gt, all_preds = [], []
    matched_gt_boxes, matched_pred_boxes = [], []

    bs = max(1, args.batch_size or 8)
    start = time.time()
    sample_idx, batch_idx = 0, 0

    while sample_idx < n_samples:
        if device.startswith("cuda"):
            new_bs, _ = dynamic_tune_batch_size(bs, device, 1, 32)
            if new_bs != bs:
                bs = new_bs
        mem = check_memory_pressure(critical_ram_gb=1.0, max_usage_pct=90.0, auto_clean=True)
        if mem.get("should_throttle") and bs > 1:
            bs = max(1, bs // 2)

        end = min(sample_idx + bs, n_samples)
        imgs = [dataset.get_image(i) for i in range(sample_idx, end)]
        gts = [dataset.get_ground_truth(i, img_width=im.size[0], img_height=im.size[1])
               for i, im in zip(range(sample_idx, end), imgs)]
        # normalize: get_ground_truth may return a list indexed by class id
        gts = [{c: v for c, v in enumerate(g)} if isinstance(g, list) else g for g in gts]

        raw_preds = model.predict(imgs, conf_thresh=args.conf_thresh,
                                  iou_thresh=args.iou_thresh,
                                  class_names=list(src_names),
                                  ground_truth_hints=None)
        raw_preds = [as_class_dict(rp) for rp in raw_preds]

        for gt, rp in zip(gts, raw_preds):
            if common_mode:
                n_sub = len(mapped_targets)
                gt_sub = [None] * n_sub
                gt_list = gt if isinstance(gt, list) else [gt.get(i) for i in range(n_eval)]
                for c, v in enumerate(gt_list):
                    if v is not None and c in eval_index_map:
                        gt_sub[eval_index_map[c]] = v
                preds_sub = remap_predictions(rp, src_to_eval, eval_index_map, n_target=n_sub)
                all_gt.append(gt_sub)
                all_preds.append(preds_sub)
                classes = [i for i, v in enumerate(gt_sub) if v is not None]
            else:
                all_gt.append(list(gt) if isinstance(gt, list) else gt)
                all_preds.append(remap_predictions(rp, src_to_eval, None, n_target=n_eval))
                classes = range(n_eval)

            for c in classes:
                g = _boxes_of(all_gt[-1][c] if c < len(all_gt[-1]) else None)
                p = _boxes_of(all_preds[-1][c] if c < len(all_preds[-1]) else None)
                if len(g) and len(p):
                    ious = compute_obb_iou_matrix(g, p)
                    for pi in range(len(p)):
                        bi = int(np.argmax(ious[:, pi]))
                        if ious[bi, pi] >= args.iou_thresh:
                            matched_gt_boxes.append(g[bi])
                            matched_pred_boxes.append(p[pi])

        sample_idx = end
        batch_idx += 1
        if batch_idx % max(1, (n_samples // 20)) == 0 or sample_idx == n_samples or batch_idx == 1:
            el = time.time() - start
            print(f"        {sample_idx}/{n_samples} ({100.0*sample_idx/n_samples:.1f}%) "
                  f"| {sample_idx/max(el,1e-4):.1f} FPS | elapsed {el:.0f}s", flush=True)
        if mem.get("status") in ("warning", "recovered"):
            deep_cleanup_memory()

    def _count_gt(seq):
        n = 0
        for entry in seq:
            if entry is None:
                continue
            n += len(np.asarray(entry.get("boxes", []), dtype=np.float32).reshape(-1, 5))
        return n

    total_gt = sum(_count_gt(g if isinstance(g, list) else list(g.values())) for g in all_gt)
    map_res = compute_map_metrics(all_gt, all_preds, dataset.class_names if not common_mode
                                  else [dataset.class_names[i] for i in mapped_targets])
    cm, cm_labels = compute_detection_confusion_matrix(
        all_gt, all_preds,
        dataset.class_names if not common_mode else [dataset.class_names[i] for i in mapped_targets],
        iou_threshold=args.iou_thresh)
    cls_res = compute_classification_metrics(cm, cm_labels)
    reg_res = compute_regression_metrics(
        np.array(matched_gt_boxes) if matched_gt_boxes else np.zeros((0, 5)),
        np.array(matched_pred_boxes) if matched_pred_boxes else np.zeros((0, 5)))

    if n_samples > 0:
        perf = model.benchmark_latency(dataset.get_image(0), num_warmup=2, num_runs=5)
    else:
        perf = {"fps": 0.0, "mean_latency_ms": 0.0}
    deep_cleanup_memory()
    return map_res, cls_res, reg_res, perf, total_gt, mapped_targets, n_eval


def sanitize(v):
    if v is None:
        return None
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
        return None
    return v


def main():
    ap = argparse.ArgumentParser(description="Cross-dataset OBB evaluation")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--weights-dir", required=True)
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--run-name", default="cross_dataset_eval")
    ap.add_argument("--models", nargs="+", default=None,
                    help="Models to include. Default: every model with a found checkpoint.")
    ap.add_argument("--source-datasets", nargs="+", default=None,
                    help="Only use weights trained on these datasets. Default: all found.")
    ap.add_argument("--datasets", nargs="+", default=None,
                    help="Eval datasets. Default: all SUPPORTED_DATASETS.")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--conf-thresh", type=float, default=0.25)
    ap.add_argument("--iou-thresh", type=float, default=0.50)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--match-mode", default="all", choices=["all", "common"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true", help="Re-run pairs with existing checkpoints.")
    args = ap.parse_args()

    set_seed(args.seed)
    data_dir = Path(args.data_dir)
    weights_dir = Path(args.weights_dir)
    out_dir = Path(args.results_dir) / args.run_name
    ckpt_dir = out_dir / "checkpoints"
    per_class_dir = out_dir / "per_class"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    per_class_dir.mkdir(parents=True, exist_ok=True)

    hw = auto_configure_hardware(device=args.device, requested_eval_batch_size=args.batch_size)
    device = hw["device"]
    args.batch_size = args.batch_size or hw["eval_batch_size"]
    print(f"[*] Device: {device} | eval batch size: {args.batch_size}")

    # ---- Run logging & tracking (console mirror + JSONL events + config) ----
    setup_run_logging(run_id=args.run_name, results_dir=out_dir,
                      config=vars(args))
    tracker = get_tracker()
    tracker.metric(name="device", value=device)

    eval_datasets = args.datasets or list(SUPPORTED_DATASETS)
    tracker.metric(name="eval_datasets", value=eval_datasets)
    wanted_models = args.models

    # ---- discover trained checkpoints -------------------------------------
    jobs = []
    for src in (args.source_datasets or list(SUPPORTED_DATASETS)):
        for model in (wanted_models or list(SUPPORTED_MODELS)):
            w = find_trained_weights(weights_dir, model, src)
            if w:
                jobs.append((model, src, w))
    if not jobs:
        print("[!] No trained checkpoints found in", weights_dir)
        print("    Expected files like:  <weights-dir>/<model>_<dataset>_best.pt")
        sys.exit(1)
    print(f"[*] Found {len(jobs)} trained checkpoint(s):")
    for m, s, w in jobs:
        print(f"      {m:16s} trained on {s:9s} <- {w}")
    tracker.metric(name="checkpoints_found", value=len(jobs))
    tracker.metric(name="checkpoint_jobs",
                   value=[{"model": m, "source": s} for m, s, _ in jobs])

    rows = []
    t0 = time.time()
    for model, src, weights_path in jobs:
        src_names = list(DATASET_CLASSES[src])
        for eval_ds in eval_datasets:
            tag = f"{src}__{model}__{eval_ds}"
            ckpt_file = ckpt_dir / f"{tag}.json"
            if ckpt_file.exists() and not args.force:
                try:
                    cached = json.load(open(ckpt_file))
                    if cached.get("completed"):
                        print(f"\n[SKIP] {tag} (checkpoint exists)")
                        rows.append(cached["summary"])
                        continue
                except Exception:
                    pass

            print(f"\n{'='*70}\n[ EVAL ] {model}  (trained on {src})  ->  {eval_ds}\n{'='*70}")
            tracker.job_start(model=model, dataset=eval_ds, source=src,
                              phase="cross_eval")
            try:
                dataset = AerialOBBDataset(dataset_name=eval_ds, data_dir=data_dir,
                                           split="val", max_samples=args.max_samples)
            except Exception as e:
                print(f"  [!] Dataset '{eval_ds}' unavailable ({e}). Skipping.")
                tracker.job_end(model=model, dataset=eval_ds, source=src,
                                phase="cross_eval", status="dataset_unavailable",
                                error=str(e))
                continue
            if len(dataset) == 0:
                print(f"  [!] Dataset '{eval_ds}' has 0 validation images. Skipping.")
                continue
            print(f"  val samples: {len(dataset)} | classes: {len(dataset.class_names)}")

            src_to_eval = build_class_map(src_names, dataset.class_names)
            n_mapped = sum(1 for v in src_to_eval.values() if v is not None)
            print(f"  class mapping ({args.match_mode} mode): "
                  f"{n_mapped}/{len(src_names)} source classes mapped -> "
                  f"{len(set(v for v in src_to_eval.values() if v is not None))} eval classes")

            try:
                model_obj = get_model(model, device=device,
                                      num_classes=len(src_names),
                                      weights_path=weights_path)
                if not model_obj.is_loaded:
                    model_obj.load()
            except Exception as e:
                print(f"  [!] Failed to load weights: {e}")
                tracker.error(f"weight load failed: {e}", model=model, dataset=eval_ds)
                tracker.job_end(model=model, dataset=eval_ds, source=src,
                                phase="cross_eval", status="load_failed",
                                error=str(e))
                continue

            try:
                map_res, cls_res, reg_res, perf, total_gt, mapped_targets, n_eval = \
                    evaluate_pair(model_obj, dataset, src_names, src_to_eval, args, device)
            except Exception as e:
                print(f"  [!] Evaluation failed: {e}")
                import traceback
                traceback.print_exc()
                tracker.error(f"evaluation failed: {e}", model=model, dataset=eval_ds)
                tracker.job_end(model=model, dataset=eval_ds, source=src,
                                phase="cross_eval", status="failed",
                                error=str(e))
                continue

            summary = {
                "model": model, "source_dataset": src, "eval_dataset": eval_ds,
                "match_mode": args.match_mode,
                "mapped_source_classes": n_mapped,
                "evaluated_classes": len(mapped_targets) if args.match_mode == "common" else n_eval,
                "total_eval_classes": n_eval,
                "total_gt_boxes": total_gt,
                "map50": sanitize(map_res["map50"]),
                "map75": sanitize(map_res["map75"]),
                "map50_95": sanitize(map_res["map50_95"]),
                "ap_small": sanitize(map_res["ap_small"]),
                "ap_medium": sanitize(map_res["ap_medium"]),
                "ap_large": sanitize(map_res["ap_large"]),
                "precision": sanitize(cls_res["macro_precision"]),
                "recall": sanitize(cls_res["macro_recall"]),
                "f1": sanitize(cls_res["macro_f1"]),
                "accuracy": sanitize(cls_res["accuracy"]),
                "specificity": sanitize(cls_res["macro_specificity"]),
                "angle_mae": sanitize(reg_res["angle_mae"]),
                "angle_rmse": sanitize(reg_res["angle_rmse"]),
                "pearson_r": sanitize(reg_res["angle_pearson_r"]),
                "spearman_rho": sanitize(reg_res["angle_spearman_rho"]),
                "r2_score": sanitize(reg_res["angle_r2"]),
                "center_offset_mae": sanitize(reg_res["center_offset_mae"]),
                "center_offset_rmse": sanitize(reg_res["center_offset_rmse"]),
                "aspect_ratio_r2": sanitize(reg_res["aspect_ratio_r2"]),
                "fps": sanitize(perf["fps"]),
                "mean_latency_ms": sanitize(perf["mean_latency_ms"]),
            }
            rows.append(summary)

            # per-class CSV (feeds the class_wise_results xlsx sheet directly)
            per_class_rows = []
            names_out = (dataset.class_names if args.match_mode == "all"
                         else [dataset.class_names[i] for i in mapped_targets])
            for c_name in names_out:
                pc = cls_res["per_class"].get(c_name, {})
                per_class_rows.append({
                    "class": c_name,
                    "AP50": sanitize(map_res["per_class_ap50"].get(c_name)),
                    "AP75": sanitize(map_res["per_class_ap75"].get(c_name)),
                    "TP": pc.get("tp"), "FP": pc.get("fp"), "FN": pc.get("fn"),
                    "Precision": pc.get("precision"), "Recall": pc.get("recall"),
                    "F1": pc.get("f1"),
                })
            pd.DataFrame(per_class_rows).to_csv(per_class_dir / f"{tag}.csv", index=False)

            with open(ckpt_file, "w") as f:
                json.dump({"completed": True, "summary": summary,
                           "class_map": {str(k): v for k, v in src_to_eval.items()}}, f, indent=2)

            print(f"  [RESULT] mAP50={summary['map50']} | F1={summary['f1']} | "
                  f"P={summary['precision']} | R={summary['recall']} | FPS={summary['fps']}")
            tracker.job_end(model=model, dataset=eval_ds, source=src,
                            phase="cross_eval", status="ok",
                            metrics={"map50": summary["map50"],
                                     "map75": summary["map75"],
                                     "f1": summary["f1"],
                                     "precision": summary["precision"],
                                     "recall": summary["recall"],
                                     "angle_mae": summary["angle_mae"],
                                     "fps": summary["fps"],
                                     "total_gt_boxes": total_gt},
                            weights=weights_path)

    # ---- write summary tables ---------------------------------------------
    if not rows:
        print("[!] No evaluation results were produced.")
        sys.exit(1)
    df = pd.DataFrame(rows)
    detail_csv = out_dir / "cross_dataset_summary.csv"
    df.to_csv(detail_csv, index=False)
    print(f"\n[✓] Per-pair results -> {detail_csv}")
    tracker.metric(name="pairs_evaluated", value=len(df))

    metric_cols = ["map50", "map75", "map50_95", "precision", "recall", "f1",
                   "accuracy", "angle_mae", "fps", "mean_latency_ms"]
    g = df.groupby(["model", "source_dataset"])[metric_cols].mean().reset_index()
    g["eval_datasets_covered"] = df.groupby(["model", "source_dataset"]).size().values
    global_csv = out_dir / "global_summary.csv"
    g.to_csv(global_csv, index=False)
    print(f"[✓] Global per-model averages -> {global_csv}")

    cols = ["model", "source_dataset", "eval_dataset", "map50", "f1", "precision",
            "recall", "accuracy", "fps"]
    print("\n================ CROSS-DATASET RESULTS ================")
    print(df[cols].sort_values(["source_dataset", "eval_dataset", "map50"],
                               ascending=[True, True, False]).to_string(index=False))
    print("\n================ GLOBAL (macro-average over eval datasets) ================")
    print(g.to_string(index=False))
    print(f"\nTotal time: {time.time()-t0:.0f}s")
    tracker.run_end(status="completed")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        get_tracker().run_end(status="interrupted", error="KeyboardInterrupt")
        raise
    except SystemExit as e:
        get_tracker().run_end(status="exit", error=f"code={e.code}")
        raise
    except Exception as e:
        get_tracker().error(f"fatal: {e}", exc=e)
        get_tracker().run_end(status="failed", error=str(e))
        raise
    finally:
        get_tracker().close()
