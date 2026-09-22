#!/usr/bin/env python3
"""
Master CLI Runner for Aerial OBB Detection, Evaluation & Comparative Analysis.
Orchestrates dataset verification/download, multi-model evaluation, comprehensive metrics computation,
confusion matrix extraction, continuous angle/box regression analysis, and report generation.

Includes:
- Crash-resilient atomic checkpointing and resumption (--resume)
- Automated Google Drive integration for Google Colab runtimes
- Proactive RAM and CUDA memory verification, throttling, and garbage collection
- Graceful shutdown signal handling (SIGINT, SIGTERM)

Usage:
  # Instant CPU smoke test on laptop:
  python scripts/run_pipeline.py --test --save-plots

  # Run full benchmark across datasets:
  python scripts/run_pipeline.py --datasets codrone visdrone --models yolov8n-obb custom-obb --save-plots

  # In Google Colab with GPU:
  python scripts/run_pipeline.py --datasets codrone --models yolov8n-obb yolo11n-obb --device cuda --save-plots
"""

import os
import sys
import time
import argparse
import datetime
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
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
)
from src.utils.reporter import format_metrics_table, generate_markdown_report
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


def str2bool(v):
    if isinstance(v, bool):
        return v
    return str(v).lower() in ("yes", "true", "t", "1")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aerial OBB Detection & Benchmark Suite",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Core Execution Modes
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
        default=["codrone"],
        help=f"Datasets to benchmark. Choices: {SUPPORTED_DATASETS} or 'all'",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["yolov8n-obb", "custom-obb"],
        help=f"Models to evaluate. Choices: {SUPPORTED_MODELS} or 'all'",
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
        help="Pretrained model weights directory.",
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
        help="Inference batch size.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Computing device for inference.",
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

    print(f"    [*] Starting evaluation: {num_samples} samples | Initial batch size: {current_batch_size}")
    print(f"    [*] Initial Memory: {format_memory_summary()}")

    sample_idx = 0
    batch_idx = 0

    while sample_idx < num_samples:
        # Check system memory pressure before each batch
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

        # Model inference
        batch_preds = model.predict(
            batch_imgs,
            conf_thresh=conf_thresh,
            iou_thresh=iou_thresh,
            class_names=class_names,
            ground_truth_hints=batch_gt,
        )

        for i, img_idx in enumerate(range(sample_idx, end_idx)):
            img = batch_imgs[i]
            w_img, h_img = img.size
            gt_dict = dataset.get_ground_truth(img_idx, img_width=w_img, img_height=h_img)
            pred_dict = batch_preds[i]

            all_gt.append(gt_dict)
            all_preds.append(pred_dict)

            # Extract matched TP boxes for continuous regression metrics (angle, box offset)
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

        # Periodically save batch progress
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

        # Periodic cleanup if warning pressure exists
        if mem_status["status"] in ("warning", "recovered"):
            deep_cleanup_memory()

    # Clear mid-batch progress once inference is fully complete
    clear_batch_progress(results_dir, run_id, dataset_name, model_name)

    # 1. Detection mAP Metrics
    map_results = compute_map_metrics(all_gt, all_preds, class_names)

    # 2. Multi-class Confusion Matrix with Background
    cm, cm_labels = compute_detection_confusion_matrix(all_gt, all_preds, class_names, iou_threshold=iou_thresh)

    # 3. Classification Metrics (Accuracy, Precision, Recall, F1, Specificity)
    cls_results = compute_classification_metrics(cm, class_names)

    # 4. Continuous Regression & Correlation Metrics
    matched_gt_arr = np.array(matched_gt_boxes) if len(matched_gt_boxes) > 0 else np.zeros((0, 5))
    matched_pred_arr = np.array(matched_pred_boxes) if len(matched_pred_boxes) > 0 else np.zeros((0, 5))
    reg_results = compute_regression_metrics(matched_gt_arr, matched_pred_arr)

    # 5. Latency & FPS Benchmark
    if num_samples > 0:
        sample_img = dataset.get_image(0)
        warmup = 1 if is_test else 2
        runs = 2 if is_test else 5
        perf_stats = model.benchmark_latency(sample_img, num_warmup=warmup, num_runs=runs)
    else:
        perf_stats = {"mean_latency_ms": 0.0, "p95_latency_ms": 0.0, "fps": 0.0}

    # Final deep cleanup
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
    }


def main():
    args = parse_args()
    set_seed(args.seed)

    print("=================================================================")
    print(" Aerial OBB Detection & Benchmark Evaluation Suite")
    print("=================================================================")

    dev_info = get_device_info(args.device)
    device = dev_info["device"]
    print(f"[*] Hardware Environment : {dev_info['device_name']} (PyTorch device: '{device}')")
    print(f"[*] Memory State         : {format_memory_summary()}")

    if args.test:
        print("[*] --test flag activated! Fast CPU test mode with synthetic aerial data.")
        args.batch_size = min(args.batch_size, 2)
        if args.max_samples is None:
            args.max_samples = 4

    run_id = args.run_name or f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if args.test:
        run_id += "_test"

    # Setup directories via path resolver (incorporates Google Drive when in Colab)
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
            print("    Persistent storage is active: weights and results will persist in Google Drive.")
        else:
            print("[!] Note: Google Drive is not mounted. For permanent retention, mount drive at /content/drive.")

    plots_dir = output_dir / "plots"
    if args.save_plots:
        plots_dir.mkdir(parents=True, exist_ok=True)

    # Resolve datasets and models
    requested_datasets = SUPPORTED_DATASETS if "all" in args.datasets else [d.lower() for d in args.datasets]
    requested_models = SUPPORTED_MODELS if "all" in args.models else args.models

    print(f"[*] Datasets to evaluate  : {requested_datasets}")
    print(f"[*] Models to evaluate    : {requested_models}")
    print(f"[*] Results Directory     : {output_dir}\n")

    # Step 1: Ensure dataset availability
    datasets_ready = {}
    for d_name in requested_datasets:
        if args.test:
            print(f"[*] Generating synthetic test dataset for '{d_name}'...")
            create_mock_dataset(d_name, data_dir, num_train=3, num_val=args.max_samples)
            datasets_ready[d_name] = True
        else:
            status = verify_dataset_status(d_name, data_dir)
            if not status["ready"]:
                if args.download:
                    print(f"[*] Attempting download for '{d_name}'...")
                    success = download_dataset(d_name, data_dir)
                    datasets_ready[d_name] = success
                else:
                    print(f"[!] Dataset '{d_name}' not found locally at {status['path']}.")
                    print("    Pass --download to attempt automated retrieval, or follow manual instructions:\n")
                    print(status["manual_guide"])
                    datasets_ready[d_name] = False
            else:
                datasets_ready[d_name] = True

    valid_datasets = [d for d, ready in datasets_ready.items() if ready]
    if not valid_datasets:
        print("[!] No datasets are ready for evaluation. Exiting.", file=sys.stderr)
        sys.exit(1)

    # Queue evaluation pairs
    total_pairs = [{"dataset": d, "model": m} for d in valid_datasets for m in requested_models]
    completed_pairs = []
    pending_pairs = list(total_pairs)
    all_summary_rows = []

    # Check for existing run manifest if resuming
    if args.resume:
        manifest = load_run_manifest(results_dir, run_id)
        if manifest:
            print(f"[*] Existing run manifest detected for '{run_id}'. Resuming previous progress...")

    # Register graceful signal handler to commit run state on SIGINT/SIGTERM
    def graceful_exit(signum, frame):
        print(f"\n[!] Signal handler invoked. Flushing partial run manifest to {output_dir}...")
        save_run_manifest(
            results_dir=results_dir,
            run_id=run_id,
            summary_rows=all_summary_rows,
            completed_pairs=completed_pairs,
            pending_pairs=pending_pairs,
            is_completed=False,
        )

    setup_signal_handlers(graceful_exit)

    # Step 2: Evaluation Loop
    for d_name in valid_datasets:
        print(f"\n{'#' * 65}")
        print(f" Evaluating on Dataset: {d_name.upper()}")
        print(f"{'#' * 65}")

        dataset = AerialOBBDataset(
            dataset_name=d_name,
            data_dir=data_dir,
            split="val",
            max_samples=args.max_samples,
        )
        print(f"[*] Loaded {len(dataset)} validation images for {d_name}.")

        for m_name in requested_models:
            current_pair = {"dataset": d_name, "model": m_name}
            print(f"\n--> Model: {m_name} on {d_name}")

            # Check if this evaluation was already completed
            if args.resume and is_evaluation_completed(results_dir, run_id, d_name, m_name):
                print(f"    [CHECKPOINT HIT] Found completed evaluation for {m_name} on {d_name}. Skipping.")
                cached_ckpt = load_evaluation_checkpoint(results_dir, run_id, d_name, m_name)
                if cached_ckpt and "summary" in cached_ckpt:
                    all_summary_rows.append(cached_ckpt["summary"])
                    completed_pairs.append(current_pair)
                    if current_pair in pending_pairs:
                        pending_pairs.remove(current_pair)
                    continue

            try:
                model = get_model(m_name, device=device, num_classes=len(dataset.class_names))
                model.load()
            except Exception as e:
                print(f"[!] Could not load model '{m_name}': {e}", file=sys.stderr)
                continue

            # Run evaluation
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
            all_summary_rows.append(summary_item)
            completed_pairs.append(current_pair)
            if current_pair in pending_pairs:
                pending_pairs.remove(current_pair)

            print(
                f"    [mAP50: {map_res['map50']:.3f} | mAP75: {map_res['map75']:.3f} | F1: {cls_res['macro_f1']:.3f} | "
                f"Angle MAE: {reg_res['angle_mae']:.2f}° | Pearson r: {reg_res['angle_pearson_r']:.3f} | "
                f"R²: {reg_res['angle_r2']:.3f} | FPS: {perf_res['fps']:.1f}]"
            )

            # Persist checkpoint immediately after model-dataset completion
            save_evaluation_checkpoint(
                results_dir=results_dir,
                run_id=run_id,
                dataset_name=d_name,
                model_name=m_name,
                summary_item=summary_item,
                completed=True,
            )

            # Update master manifest
            save_run_manifest(
                results_dir=results_dir,
                run_id=run_id,
                summary_rows=all_summary_rows,
                completed_pairs=completed_pairs,
                pending_pairs=pending_pairs,
                is_completed=len(pending_pairs) == 0,
            )

            # Save visual plots
            if args.save_plots:
                cm_plot_path = plots_dir / f"confusion_matrix_{d_name}_{m_name}.png"
                plot_confusion_matrix(
                    cm=eval_out["confusion_matrix"],
                    class_names=eval_out["confusion_labels"],
                    output_path=cm_plot_path,
                    title=f"Confusion Matrix: {m_name} on {d_name.upper()}",
                )

                if len(eval_out["matched_gt_boxes"]) > 0:
                    corr_plot_path = plots_dir / f"angle_correlation_{d_name}_{m_name}.png"
                    plot_angle_correlation(
                        true_angles=eval_out["matched_gt_boxes"][:, 4],
                        pred_angles=eval_out["matched_pred_boxes"][:, 4],
                        output_path=corr_plot_path,
                        title=f"OBB Angle Correlation: {m_name} on {d_name.upper()}",
                        r2_score=reg_res["angle_r2"],
                        pearson_r=reg_res["angle_pearson_r"],
                        mae=reg_res["angle_mae"],
                    )

            # Clean memory before loading next model
            deep_cleanup_memory()

    if not all_summary_rows:
        print("[!] No evaluations completed.", file=sys.stderr)
        sys.exit(1)

    # Save summary table CSV and JSON atomically
    df_summary = pd.DataFrame(all_summary_rows)
    csv_path = output_dir / "benchmark_summary.csv"
    tmp_csv = csv_path.with_name(f"{csv_path.name}.tmp")
    df_summary.to_csv(tmp_csv, index=False)
    os.replace(tmp_csv, csv_path)

    atomic_save_json(all_summary_rows, output_dir / "benchmark_metrics.json")

    # Identify best model and dataset
    best_row_map50 = df_summary.loc[df_summary["map50"].idxmax()]
    best_model = best_row_map50["model"]
    best_dataset = best_row_map50["dataset"]

    # Save comparative bar chart
    if args.save_plots:
        comparison_chart_path = plots_dir / "model_benchmark_comparison.png"
        plot_benchmark_comparison(all_summary_rows, comparison_chart_path)

    # Generate Markdown Report atomically
    md_report = generate_markdown_report(all_summary_rows, best_model=best_model, best_dataset=best_dataset)
    report_path = output_dir / "benchmark_report.md"
    tmp_report = report_path.with_name(f"{report_path.name}.tmp")
    with open(tmp_report, "w", encoding="utf-8") as f:
        f.write(md_report)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_report, report_path)

    # Update manifest to completed state
    save_run_manifest(
        results_dir=results_dir,
        run_id=run_id,
        summary_rows=all_summary_rows,
        completed_pairs=completed_pairs,
        pending_pairs=[],
        is_completed=True,
    )

    # Print Final Summary Table to Terminal
    print("\n" + "=" * 80)
    print(" FINAL BENCHMARK SUMMARY")
    print("=" * 80)
    print(format_metrics_table(all_summary_rows))
    print("=" * 80)
    print(f"\n[✓] Results, checkpoints, and plots successfully saved to: {output_dir}")
    print(f"    - CSV Summary      : {csv_path}")
    print(f"    - JSON Metrics     : {output_dir / 'benchmark_metrics.json'}")
    print(f"    - Markdown Report  : {report_path}")
    print(f"    - Master Manifest  : {output_dir / 'run_manifest.json'}")
    if args.save_plots:
        print(f"    - Visual Plots Dir : {plots_dir}")


if __name__ == "__main__":
    main()
