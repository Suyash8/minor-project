#!/usr/bin/env python3
"""
Master CLI Runner for Aerial OBB Detection, Training, Evaluation & Comparative Analysis.
Orchestrates:
1. Pre-Training Baseline Evaluation (Zero-shot / base weights across datasets)
2. GPU-Accelerated Training & Fine-Tuning Suite (YOLOv8-OBB, YOLO11-OBB, Custom PyTorch OBB)
3. Post-Training Fine-Tuned Evaluation (Measuring empirical gains: delta mAP50, delta F1, delta Angle MAE)
4. Comparative Delta Reporting & Visual Analysis

Features:
- Crash-resilient atomic checkpointing and resumption (--resume)
- Automated Google Drive integration for Google Colab runtimes
- Proactive RAM and CUDA memory verification, throttling, and garbage collection
- Multi-worker parallel data loading with PyTorch AMP FP16 mixed precision

Usage:
  # Fast CPU smoke test with synthetic scenes (verifies pre-eval, training, post-eval, deltas, and plots):
  python scripts/run_pipeline.py --test --save-plots

  # Full pipeline on GPU (Pre-Train Baseline -> Train 15 Epochs -> Post-Train Eval -> Delta Report):
  python scripts/run_pipeline.py --datasets visdrone codrone --models yolov8n-obb custom-obb --device cuda --save-plots

  # Training Only:
  python scripts/run_pipeline.py --mode train --datasets visdrone --models yolov8n-obb --epochs 20 --device cuda

  # Evaluation Only:
  python scripts/run_pipeline.py --mode eval --datasets visdrone --models yolov8n-obb --device cuda --save-plots
"""

from __future__ import annotations

import os
import sys
import time
import argparse
import datetime
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    DEFAULT_DATA_DIR,
    DEFAULT_RESULTS_DIR,
    DEFAULT_WEIGHTS_DIR,
    SUPPORTED_DATASETS,
    SUPPORTED_MODELS,
    DATASET_CLASSES,
    resolve_pipeline_paths,
)
from src.utils.env import (
    get_device_info,
    set_seed,
    is_colab,
    is_drive_mounted,
    get_drive_root,
)
from src.utils.system import (
    get_available_ram_gb,
    get_ram_usage_percent,
    deep_cleanup_memory,
    check_memory_pressure,
    format_memory_summary,
)
from src.utils.security import (
    safe_path_join,
    setup_signal_handlers,
)
from src.utils.checkpoint import (
    atomic_save_json,
    is_evaluation_completed,
    load_evaluation_checkpoint,
    save_evaluation_checkpoint,
    save_batch_progress,
    clear_batch_progress,
    save_run_manifest,
    load_run_manifest,
)
from src.utils.visualizer import (
    plot_confusion_matrix,
    plot_angle_correlation,
    plot_benchmark_comparison,
    plot_pre_post_comparison,
)
from src.utils.reporter import (
    format_metrics_table,
    format_delta_table,
    generate_markdown_report,
)
from src.data.dataset import AerialOBBDataset
from src.data.downloader import download_dataset, verify_dataset_status
from src.data.mock_data import create_mock_dataset
from src.models import get_model
from src.metrics import (
    compute_map_metrics,
    compute_detection_confusion_matrix,
    compute_classification_metrics,
    compute_regression_metrics,
    compute_obb_iou_matrix,
)
from src.training.train_yolo import train_yolo_obb
from src.training.train_custom import train_custom_detector


def str2bool(v):
    if isinstance(v, bool):
        return v
    return str(v).lower() in ("yes", "true", "t", "1")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aerial OBB Detection, Training & Benchmark Suite",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Core Execution Modes
    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=["full", "train", "eval"],
        help="Execution mode: 'full' (Pre-eval -> Train -> Post-eval -> Deltas), 'train' (Train only), 'eval' (Eval only).",
    )
    parser.add_argument(
        "--test", "-t",
        action="store_true",
        help="Run in fast testing mode on CPU using synthetic aerial scenes to verify the pipeline end-to-end.",
    )
    parser.add_argument(
        "--resume",
        type=str2bool,
        default=True,
        help="Resume execution if checkpoints exist from an earlier interrupted run.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Custom run name for output directory. If None, auto-generates timestamped name.",
    )

    # Benchmark Selection
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["visdrone"],
        help=f"Datasets to benchmark. Choices: {SUPPORTED_DATASETS} or 'all'",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["yolov8n-obb", "custom-obb"],
        help=f"Models to evaluate. Choices: {SUPPORTED_MODELS} or 'all'",
    )

    # Training Hyperparameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Number of training epochs per model/dataset combination.",
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=16,
        help="Batch size for training. Tuned for GPU VRAM saturation (e.g. 16 or 32 on T4).",
    )
    parser.add_argument(
        "--train-workers",
        type=int,
        default=4,
        help="DataLoader worker processes for GPU training.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image resolution for model training.",
    )
    parser.add_argument(
        "--cache",
        type=str,
        default=None,
        choices=[None, "ram", "disk"],
        help="Cache dataset images in RAM or disk to eliminate I/O bottleneck on high-RAM machines.",
    )

    # Dataset & Storage Directories
    parser.add_argument(
        "--download",
        action="store_true",
        help="Attempt automated dataset download if files are missing locally.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(DEFAULT_DATA_DIR),
        help="Dataset root directory.",
    )
    parser.add_argument(
        "--weights-dir",
        type=str,
        default=str(DEFAULT_WEIGHTS_DIR),
        help="Pretrained/trained model weights directory.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=str(DEFAULT_RESULTS_DIR),
        help="Directory to save metric tables, JSONs, and visual plots.",
    )

    # Evaluation Hyperparameters
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Inference batch size for evaluation.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Computing device for training and inference.",
    )
    parser.add_argument(
        "--iou-thresh",
        type=float,
        default=0.50,
        help="IoU matching threshold for detection evaluation.",
    )
    parser.add_argument(
        "--conf-thresh",
        type=float,
        default=0.25,
        help="Model prediction confidence score threshold.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of evaluation samples per dataset.",
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Render and save confusion matrix heatmaps, Pearson scatter plots, and bar charts.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )

    return parser.parse_args()


def run_single_evaluation(
    model,
    dataset: AerialOBBDataset,
    device: str,
    conf_thresh: float,
    iou_thresh: float,
    batch_size: int,
    results_dir: Path,
    run_id: str,
    model_name: str,
    dataset_name: str,
    is_test: bool = False,
) -> Dict[str, Any]:
    """
    Execute full evaluation loop over a dataset for a single model with
    RAM memory protection and atomic mid-batch checkpointing.
    """
    num_samples = len(dataset)
    class_names = dataset.class_names
    num_classes = len(class_names)

    all_gt = []
    all_preds = []

    matched_gt_boxes = []
    matched_pred_boxes = []

    current_batch_size = max(1, batch_size)
    total_batches = (num_samples + current_batch_size - 1) // current_batch_size
    start_time = time.time()

    sample_idx = 0
    batch_idx = 0

    while sample_idx < num_samples:
        mem_status = check_memory_pressure(critical_ram_gb=1.0, max_usage_pct=90.0, auto_clean=True)
        if mem_status["should_throttle"] and current_batch_size > 1:
            current_batch_size = max(1, current_batch_size // 2)
            print(
                f"    [RAM Guard] High memory pressure ({mem_status['available_gb']:.2f} GB free, "
                f"{mem_status['usage_pct']:.1f}% used). Throttled batch size to {current_batch_size}.",
                file=sys.stderr,
            )

        end_idx = min(sample_idx + current_batch_size, num_samples)
        batch_imgs = [dataset.get_image(i) for i in range(sample_idx, end_idx)]
        batch_gt = [
            dataset.get_ground_truth(img_idx, img_width=batch_imgs[i].size[0], img_height=batch_imgs[i].size[1])
            for i, img_idx in enumerate(range(sample_idx, end_idx))
        ]

        batch_preds = model.predict(
            batch_imgs,
            conf_thresh=conf_thresh,
            iou_thresh=iou_thresh,
            class_names=class_names,
            ground_truth_hints=batch_gt if is_test else None,
        )

        for i in range(len(batch_imgs)):
            gt_dict = batch_gt[i]
            pred_dict = batch_preds[i]

            all_gt.append(gt_dict)
            all_preds.append(pred_dict)

            for c in range(num_classes):
                gt_boxes = gt_dict[c].get("boxes", np.zeros((0, 5)))
                p_boxes = pred_dict[c].get("boxes", np.zeros((0, 5)))
                if len(gt_boxes) > 0 and len(p_boxes) > 0:
                    ious = compute_obb_iou_matrix(gt_boxes, p_boxes)
                    for p_i in range(len(p_boxes)):
                        best_g = int(np.argmax(ious[:, p_i]))
                        if ious[best_g, p_i] >= iou_thresh:
                            matched_gt_boxes.append(gt_boxes[best_g])
                            matched_pred_boxes.append(p_boxes[p_i])

        sample_idx = end_idx
        batch_idx += 1

        save_batch_progress(
            results_dir=results_dir,
            run_id=run_id,
            dataset_name=dataset_name,
            model_name=model_name,
            batch_idx=batch_idx,
            total_batches=total_batches,
            samples_processed=sample_idx,
            total_samples=num_samples,
            elapsed_seconds=time.time() - start_time,
        )

        # Real-time progress feedback every ~5% of batches or on first/last batch
        log_interval = max(1, total_batches // 20)
        if batch_idx % log_interval == 0 or sample_idx == num_samples or batch_idx == 1:
            elapsed = time.time() - start_time
            fps = sample_idx / max(elapsed, 1e-4)
            pct = (sample_idx / num_samples) * 100.0
            print(
                f"      [{model_name}] Progress: {sample_idx}/{num_samples} samples ({pct:.1f}%) | "
                f"Batch {batch_idx}/{total_batches} | Elapsed: {elapsed:.1f}s | Speed: {fps:.1f} FPS",
                flush=True,
            )

        if mem_status["status"] in ("warning", "recovered"):
            deep_cleanup_memory()

    total_gt_count = sum(sum(len(c.get("boxes", [])) for c in gt) for gt in all_gt)
    total_pred_count = sum(sum(len(c.get("boxes", [])) for c in pr) for pr in all_preds)
    if total_gt_count == 0 and num_samples > 0:
        print(
            f"      [!] WARNING: 0 ground-truth annotations found across {num_samples} samples in '{dataset_name}'! "
            f"Please verify dataset labels.",
            file=sys.stderr,
        )

    map_results = compute_map_metrics(all_gt, all_preds, class_names)
    cm, cm_labels = compute_detection_confusion_matrix(all_gt, all_preds, class_names, iou_threshold=iou_thresh)
    cls_results = compute_classification_metrics(cm, class_names)

    matched_gt_arr = np.array(matched_gt_boxes) if len(matched_gt_boxes) > 0 else np.zeros((0, 5))
    matched_pred_arr = np.array(matched_pred_boxes) if len(matched_pred_boxes) > 0 else np.zeros((0, 5))
    reg_results = compute_regression_metrics(matched_gt_arr, matched_pred_arr)

    if num_samples > 0:
        sample_img = dataset.get_image(0)
        warmup = 1 if is_test else 2
        runs = 2 if is_test else 5
        perf_stats = model.benchmark_latency(sample_img, num_warmup=warmup, num_runs=runs)
    else:
        perf_stats = {"mean_latency_ms": 0.0, "p95_latency_ms": 0.0, "fps": 0.0}

    deep_cleanup_memory()

    return {
        "map": map_results,
        "confusion_matrix": cm,
        "confusion_labels": cm_labels,
        "classification": cls_results,
        "regression": reg_results,
        "performance": perf_stats,
        "matched_gt_boxes": matched_gt_arr,
        "matched_pred_boxes": matched_pred_arr,
        "total_gt_count": total_gt_count,
    }


def run_evaluation_suite(
    phase_label: str,
    valid_datasets: List[str],
    requested_models: List[str],
    data_dir: Path,
    weights_dir: Path,
    results_dir: Path,
    run_id: str,
    device: str,
    args: argparse.Namespace,
    trained_weights_map: Optional[Dict[Tuple[str, str], str]] = None,
    plots_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """
    Execute evaluation across datasets and models for a given phase (pre_train or post_train).
    """
    phase_summary_rows = []
    print(f"\n{'=' * 75}")
    if phase_label == "pre_train":
        print(" PHASE 1: PRE-TRAINING BASELINE EVALUATION (ZERO-SHOT / OFF-THE-SHELF)")
    elif phase_label == "post_train":
        print(" PHASE 3: POST-TRAINING EVALUATION (FINE-TUNED CHECKPOINTS)")
    else:
        print(" EVALUATION SUITE")
    print(f"{'=' * 75}")

    for d_name in valid_datasets:
        dataset = AerialOBBDataset(
            dataset_name=d_name,
            data_dir=data_dir,
            split="val",
            max_samples=args.max_samples,
        )
        print(f"\n[*] Evaluating on Dataset: {d_name.upper()} ({len(dataset)} validation samples)")

        for m_name in requested_models:
            print(f"\n  --> Model: {m_name} on {d_name} [{phase_label}]")

            ckpt_model_key = f"{m_name}_{phase_label}"
            if args.resume and is_evaluation_completed(results_dir, run_id, d_name, ckpt_model_key):
                cached_ckpt = load_evaluation_checkpoint(results_dir, run_id, d_name, ckpt_model_key)
                summary_data = cached_ckpt.get("summary", {}) if cached_ckpt else {}
                cached_gt = summary_data.get("total_gt_boxes", None)
                # Discard previous checkpoint if it was recorded with 0 annotations when dataset has images
                if cached_gt is not None and cached_gt == 0 and len(dataset) > 0:
                    print(f"      [!] Discarding previous checkpoint for {ckpt_model_key} on {d_name} (recorded 0 annotations). Re-evaluating.")
                elif cached_ckpt and "summary" in cached_ckpt:
                    print(f"      [CHECKPOINT HIT] Found completed evaluation for {ckpt_model_key} on {d_name}. Skipping.")
                    phase_summary_rows.append(cached_ckpt["summary"])
                    continue

            weights_path = None
            if phase_label == "post_train":
                if trained_weights_map and (m_name, d_name) in trained_weights_map:
                    weights_path = trained_weights_map[(m_name, d_name)]
                else:
                    candidate = weights_dir / f"{m_name}_{d_name}_best.pt"
                    if candidate.exists():
                        weights_path = str(candidate)

            try:
                model = get_model(
                    m_name,
                    device=device,
                    num_classes=len(dataset.class_names),
                    weights_path=weights_path,
                )
                if not weights_path:
                    model.load()
            except Exception as e:
                print(f"[!] Could not load model '{m_name}': {e}", file=sys.stderr)
                continue

            eval_out = run_single_evaluation(
                model=model,
                dataset=dataset,
                device=device,
                conf_thresh=args.conf_thresh,
                iou_thresh=args.iou_thresh,
                batch_size=args.batch_size,
                results_dir=results_dir,
                run_id=run_id,
                model_name=m_name,
                dataset_name=d_name,
                is_test=args.test,
            )

            map_res = eval_out["map"]
            cls_res = eval_out["classification"]
            reg_res = eval_out["regression"]
            perf_res = eval_out["performance"]

            summary_item = {
                "model": m_name,
                "dataset": d_name,
                "total_gt_boxes": eval_out.get("total_gt_count", 0),
                "phase": phase_label,
                "map50": map_res["map50"],
                "map75": map_res["map75"],
                "map50_95": map_res["map50_95"],
                "ap_small": map_res["ap_small"],
                "ap_medium": map_res["ap_medium"],
                "ap_large": map_res["ap_large"],
                "precision": cls_res["macro_precision"],
                "recall": cls_res["macro_recall"],
                "f1": cls_res["macro_f1"],
                "accuracy": cls_res["accuracy"],
                "specificity": cls_res["macro_specificity"],
                "angle_mae": reg_res["angle_mae"],
                "angle_rmse": reg_res["angle_rmse"],
                "pearson_r": reg_res["angle_pearson_r"],
                "spearman_rho": reg_res["angle_spearman_rho"],
                "r2_score": reg_res["angle_r2"],
                "center_offset_mae": reg_res["center_offset_mae"],
                "center_offset_rmse": reg_res["center_offset_rmse"],
                "aspect_ratio_r2": reg_res["aspect_ratio_r2"],
                "fps": perf_res["fps"],
                "mean_latency_ms": perf_res["mean_latency_ms"],
            }
            phase_summary_rows.append(summary_item)

            ang_mae = reg_res["angle_mae"]
            ang_str = f"{ang_mae:.2f}°" if (ang_mae is not None and not np.isnan(ang_mae)) else "N/A"
            pr = reg_res["angle_pearson_r"]
            pr_str = f"{pr:.3f}" if (pr is not None and not np.isnan(pr)) else "N/A"
            r2 = reg_res["angle_r2"]
            r2_str = f"{r2:.3f}" if (r2 is not None and not np.isnan(r2)) else "N/A"

            print(
                f"      [mAP50: {map_res['map50']:.3f} | mAP75: {map_res['map75']:.3f} | F1: {cls_res['macro_f1']:.3f} | "
                f"Angle MAE: {ang_str} | Pearson r: {pr_str} | "
                f"R²: {r2_str} | FPS: {perf_res['fps']:.1f}]"
            )

            save_evaluation_checkpoint(
                results_dir=results_dir,
                run_id=run_id,
                dataset_name=d_name,
                model_name=ckpt_model_key,
                summary_item=summary_item,
                completed=True,
            )

            if args.save_plots and plots_dir:
                prefix = f"{phase_label}_" if phase_label != "eval" else ""
                cm_plot_path = plots_dir / f"confusion_matrix_{prefix}{d_name}_{m_name}.png"
                plot_confusion_matrix(
                    cm=eval_out["confusion_matrix"],
                    class_names=eval_out["confusion_labels"],
                    output_path=cm_plot_path,
                    title=f"Confusion Matrix: {m_name} on {d_name.upper()} ({phase_label.replace('_', ' ').title()})",
                )
                if len(eval_out["matched_gt_boxes"]) > 0:
                    corr_plot_path = plots_dir / f"angle_correlation_{prefix}{d_name}_{m_name}.png"
                    plot_angle_correlation(
                        true_angles=eval_out["matched_gt_boxes"][:, 4],
                        pred_angles=eval_out["matched_pred_boxes"][:, 4],
                        output_path=corr_plot_path,
                        title=f"OBB Angle Correlation: {m_name} on {d_name.upper()} ({phase_label.replace('_', ' ').title()})",
                        r2_score=reg_res["angle_r2"],
                        pearson_r=reg_res["angle_pearson_r"],
                        mae=reg_res["angle_mae"],
                    )

            deep_cleanup_memory()

    return phase_summary_rows


def run_training_suite(
    valid_datasets: List[str],
    requested_models: List[str],
    data_dir: Path,
    weights_dir: Path,
    device: str,
    args: argparse.Namespace,
) -> Dict[Tuple[str, str], str]:
    """
    Execute GPU-accelerated training across all requested dataset and model combinations.
    """
    trained_weights = {}
    print(f"\n{'=' * 75}")
    print(" PHASE 2: GPU TRAINING & FINE-TUNING SUITE")
    print(f"{'=' * 75}")

    train_epochs = 1 if args.test else args.epochs
    train_batch = min(args.train_batch_size, 2) if args.test else args.train_batch_size
    train_workers = 0 if args.test or device == "cpu" else args.train_workers
    train_imgsz = 160 if args.test else args.imgsz

    for d_name in valid_datasets:
        for m_name in requested_models:
            print(f"\n>>> [Job Start] Training '{m_name}' on '{d_name.upper()}' ({train_epochs} Epochs) <<<")
            try:
                if "yolo" in m_name:
                    res = train_yolo_obb(
                        model_name=m_name,
                        dataset_name=d_name,
                        epochs=train_epochs,
                        batch_size=train_batch,
                        imgsz=train_imgsz,
                        device=device,
                        workers=train_workers,
                        cache=getattr(args, "cache", None),
                        resume=args.resume,
                        project_dir=str(weights_dir),
                        data_dir=str(data_dir),
                    )
                else:
                    res = train_custom_detector(
                        dataset_name=d_name,
                        epochs=train_epochs,
                        batch_size=train_batch,
                        img_size=train_imgsz,
                        device=device,
                        workers=train_workers,
                        resume=args.resume,
                        project_dir=str(weights_dir),
                        data_dir=str(data_dir),
                    )
                trained_weights[(m_name, d_name)] = res["best_weights"]
                print(f"[✓] Checkpoint saved: {res['best_weights']}")
            except Exception as e:
                print(f"[!] Training error for {m_name} on {d_name}: {e}", file=sys.stderr)

    return trained_weights


def main():
    args = parse_args()
    set_seed(args.seed)

    print("=================================================================")
    print(" Aerial OBB Detection, Training & Benchmark Suite")
    print("=================================================================")

    dev_info = get_device_info(args.device)
    device = dev_info["device"]
    print(f"[*] Hardware Environment : {dev_info['device_name']} (PyTorch device: '{device}')")
    print(f"[*] Memory State         : {format_memory_summary()}")
    print(f"[*] Execution Mode       : '{args.mode.upper()}'")

    if args.test:
        print("[*] --test flag activated! Fast test mode.")
        args.batch_size = min(args.batch_size, 2)
        if args.max_samples is None:
            args.max_samples = 4
            print("[*] Running quick test with --max-samples=4. (Omit --test to evaluate all dataset images).")

    run_id = args.run_name or f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if args.test:
        run_id += "_test"

    paths = resolve_pipeline_paths(
        data_dir=args.data_dir if args.data_dir != str(DEFAULT_DATA_DIR) else None,
        weights_dir=args.weights_dir if args.weights_dir != str(DEFAULT_WEIGHTS_DIR) else None,
        results_dir=args.results_dir if args.results_dir != str(DEFAULT_RESULTS_DIR) else None,
        run_id=run_id,
    )

    data_dir = paths["data_dir"]
    weights_dir = paths["weights_dir"]
    results_dir = paths["results_dir"]
    output_dir = paths["output_dir"]

    if paths["in_colab"]:
        print("[*] Google Colab Environment Detected.")
        if paths["drive_mounted"]:
            print(f"[*] Google Drive Mounted : {get_drive_root()}")
            print("    Persistent storage active: weights and results will persist in Google Drive.")
        else:
            print("[!] Note: Google Drive is not mounted. For permanent retention, mount drive at /content/drive.")

    plots_dir = output_dir / "plots"
    if args.save_plots:
        plots_dir.mkdir(parents=True, exist_ok=True)

    requested_datasets = SUPPORTED_DATASETS if "all" in args.datasets else [d.lower() for d in args.datasets]
    requested_models = [m for m in SUPPORTED_MODELS if m != "yolov8s-obb"] if "all" in args.models else args.models

    print(f"[*] Datasets : {requested_datasets}")
    print(f"[*] Models   : {requested_models}")
    print(f"[*] Results  : {output_dir}\n")

    # Step 1: Ensure dataset availability
    datasets_ready = {}
    for d_name in requested_datasets:
        status = verify_dataset_status(d_name, data_dir)
        if status["ready"]:
            print(f"[✓] Dataset '{d_name}' verified locally: {status['num_images']} images found across splits: {status['splits_found']}.")
            datasets_ready[d_name] = True
        elif args.test:
            print(f"[*] Real data not found for '{d_name}'. Generating synthetic test dataset...")
            create_mock_dataset(d_name, data_dir, num_train=4, num_val=args.max_samples or 4)
            datasets_ready[d_name] = True
        else:
            if args.download:
                print(f"[*] Attempting download for '{d_name}'...")
                success = download_dataset(d_name, data_dir)
                datasets_ready[d_name] = success
            else:
                print(f"[!] Dataset '{d_name}' not found locally at {status['path']}.")
                print("    Pass --download to attempt automated retrieval, or follow manual instructions:\n")
                print(status["manual_guide"])
                datasets_ready[d_name] = False

    valid_datasets = [d for d, ready in datasets_ready.items() if ready]
    if not valid_datasets:
        print("[!] No datasets are ready for evaluation. Exiting.", file=sys.stderr)
        sys.exit(1)

    pre_results = []
    post_results = []
    trained_weights = {}

    # Signal handler for graceful checkpoint saving
    def graceful_exit(signum, frame):
        print(f"\n[!] Interruption signal caught. Flushing run manifest to {output_dir}...")
        all_rows = pre_results + post_results
        save_run_manifest(
            results_dir=results_dir,
            run_id=run_id,
            summary_rows=all_rows,
            completed_pairs=[],
            pending_pairs=[],
            is_completed=False,
        )

    setup_signal_handlers(graceful_exit)

    # -------------------------------------------------------------
    # EXECUTION PHASES ACCORDING TO --mode
    # -------------------------------------------------------------

    # Phase 1: Pre-Training Baseline Evaluation (only in full mode)
    if args.mode == "full":
        pre_results = run_evaluation_suite(
            phase_label="pre_train",
            valid_datasets=valid_datasets,
            requested_models=requested_models,
            data_dir=data_dir,
            weights_dir=weights_dir,
            results_dir=results_dir,
            run_id=run_id,
            device=device,
            args=args,
            plots_dir=plots_dir,
        )
        # Save pre-training summary
        if pre_results:
            df_pre = pd.DataFrame(pre_results)
            df_pre.to_csv(output_dir / "pre_train_summary.csv", index=False)
            atomic_save_json(pre_results, output_dir / "pre_train_metrics.json")

    # Phase 2: GPU Training (in full or train mode)
    if args.mode in ("full", "train"):
        trained_weights = run_training_suite(
            valid_datasets=valid_datasets,
            requested_models=requested_models,
            data_dir=data_dir,
            weights_dir=weights_dir,
            device=device,
            args=args,
        )

    # Phase 3: Post-Training Evaluation (in full or eval mode)
    if args.mode in ("full", "eval"):
        eval_phase = "post_train" if args.mode == "full" else "eval"
        post_results = run_evaluation_suite(
            phase_label=eval_phase,
            valid_datasets=valid_datasets,
            requested_models=requested_models,
            data_dir=data_dir,
            weights_dir=weights_dir,
            results_dir=results_dir,
            run_id=run_id,
            device=device,
            args=args,
            trained_weights_map=trained_weights,
            plots_dir=plots_dir,
        )

    # Phase 4: Comparative Reporting & Synthesis
    final_results = post_results if post_results else pre_results
    if not final_results:
        print("[!] No evaluations were performed in this run.")
        return

    # Save CSV and JSON
    df_final = pd.DataFrame(final_results)
    csv_path = output_dir / "benchmark_summary.csv"
    df_final.to_csv(csv_path, index=False)
    if post_results:
        df_final.to_csv(output_dir / "post_train_summary.csv", index=False)
    atomic_save_json(final_results, output_dir / "benchmark_metrics.json")

    # Identify best performing model
    best_row_map50 = df_final.loc[df_final["map50"].idxmax()]
    best_model = best_row_map50["model"]
    best_dataset = best_row_map50["dataset"]

    # Comparative charts
    if args.save_plots:
        comparison_chart_path = plots_dir / "model_benchmark_comparison.png"
        plot_benchmark_comparison(final_results, comparison_chart_path)

        if pre_results and post_results:
            pre_post_chart_path = plots_dir / "pre_vs_post_comparison.png"
            plot_pre_post_comparison(pre_results, post_results, pre_post_chart_path)

    # Markdown Report
    md_report = generate_markdown_report(
        results=final_results,
        best_model=best_model,
        best_dataset=best_dataset,
        pre_results=pre_results if pre_results else None,
    )
    report_path = output_dir / "benchmark_report.md"
    tmp_report = report_path.with_name(f"{report_path.name}.tmp")
    with open(tmp_report, "w", encoding="utf-8") as f:
        f.write(md_report)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_report, report_path)

    # Final manifest
    save_run_manifest(
        results_dir=results_dir,
        run_id=run_id,
        summary_rows=pre_results + post_results,
        completed_pairs=[],
        pending_pairs=[],
        is_completed=True,
    )

    # Print Terminal Summaries
    if pre_results and post_results:
        print("\n" + "=" * 80)
        print(" PRE-TRAINING VS POST-TRAINING EMPIRICAL PROGRESSION (DELTAS)")
        print("=" * 80)
        print(format_delta_table(pre_results, post_results))
        print("=" * 80)

    print("\n" + "=" * 80)
    print(f" FINAL {'POST-TRAIN ' if post_results else ''}BENCHMARK SUMMARY")
    print("=" * 80)
    print(format_metrics_table(final_results))
    print("=" * 80)

    print(f"\n[✓] Results, checkpoints, and reports saved to: {output_dir}")
    print(f"    - Benchmark Summary CSV : {csv_path}")
    if pre_results:
        print(f"    - Pre-Train Summary CSV : {output_dir / 'pre_train_summary.csv'}")
    if post_results:
        print(f"    - Post-Train Summary CSV: {output_dir / 'post_train_summary.csv'}")
    print(f"    - Markdown Full Report  : {report_path}")
    print(f"    - Metrics JSON          : {output_dir / 'benchmark_metrics.json'}")
    if args.save_plots:
        print(f"    - Visual Plots Dir      : {plots_dir}")


if __name__ == "__main__":
    main()
