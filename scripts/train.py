#!/usr/bin/env python3
"""
Dedicated Training CLI for Aerial OBB Detection.
Trains YOLOv8-OBB, YOLO11-OBB, and Custom PyTorch OBB detectors across CoDrone, VisDrone, and DOTA
with full GPU acceleration, mixed-precision FP16, parallel multi-worker data loading,
and per-epoch checkpointing to Google Drive.

Usage:
  # Train a single model on VisDrone:
  python scripts/train.py --model yolov8n-obb --dataset visdrone --epochs 20 --device cuda

  # Train all 3 models across all 3 datasets:
  python scripts/train.py --models yolov8n-obb yolo11n-obb custom-obb --datasets codrone visdrone dota --epochs 15 --device cuda
"""

from __future__ import annotations

import os
import sys
import argparse
import time
from pathlib import Path
from typing import List

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import SUPPORTED_MODELS, SUPPORTED_DATASETS, resolve_pipeline_paths
from src.training.train_yolo import train_yolo_obb
from src.training.train_custom import train_custom_detector
from src.utils.env import get_device_info, set_seed, is_colab, is_drive_mounted, get_drive_root


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Aerial OBB Models with Full GPU Acceleration & Epoch Checkpointing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["yolov8n-obb"],
        help=f"Models to train. Choices: {SUPPORTED_MODELS} or 'all'",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["visdrone"],
        help=f"Datasets to train on. Choices: {SUPPORTED_DATASETS} or 'all'",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs per model/dataset combination.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for training. Tune to saturate GPU VRAM (e.g. 16 or 32 on T4).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image input size for training.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Target hardware device for training.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of background data-loading worker processes.",
    )
    parser.add_argument(
        "--cache",
        type=str,
        default=None,
        choices=[None, "ram", "disk"],
        help="Cache dataset images in RAM or disk to eliminate I/O bottleneck on high-RAM machines.",
    )
    parser.add_argument(
        "--resume",
        type=lambda x: str(x).lower() in ("true", "1", "yes"),
        default=True,
        help="Resume training if previous epoch checkpoints exist.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Custom dataset root directory.",
    )
    parser.add_argument(
        "--weights-dir",
        type=str,
        default=None,
        help="Custom directory to store trained weights and epoch checkpoints.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    print("=" * 70)
    print(" Aerial OBB Detection: Multi-Model & Multi-Dataset Training Suite")
    print("=" * 70)

    dev_info = get_device_info(args.device)
    print(f"[*] Compute Environment : {dev_info['device_name']} (Device: '{dev_info['device']}')")

    paths = resolve_pipeline_paths(data_dir=args.data_dir, weights_dir=args.weights_dir)
    if paths["in_colab"] and paths["drive_mounted"]:
        print(f"[*] Google Drive Active : {get_drive_root()}")
        print("    All epoch checkpoints will automatically persist to Google Drive.")

    requested_datasets = SUPPORTED_DATASETS if "all" in args.datasets else [d.lower() for d in args.datasets]
    requested_models = [m for m in SUPPORTED_MODELS if m != "yolov8s-obb"] if "all" in args.models else args.models

    print(f"[*] Datasets to Train On : {requested_datasets}")
    print(f"[*] Models to Train      : {requested_models}")
    print(f"[*] Total Combinations   : {len(requested_datasets) * len(requested_models)}")
    print(f"[*] Epochs per Model     : {args.epochs}")
    print(f"[*] Batch Size           : {args.batch_size}")
    print(f"[*] Data Load Workers    : {args.workers}\n")

    overall_start = time.time()
    results = []

    for d_name in requested_datasets:
        for m_name in requested_models:
            print(f"\n>>> [Job Start] Training '{m_name}' on '{d_name.upper()}' <<<")
            try:
                if "yolo" in m_name:
                    res = train_yolo_obb(
                        model_name=m_name,
                        dataset_name=d_name,
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        imgsz=args.imgsz,
                        device=args.device,
                        workers=args.workers,
                        cache=args.cache,
                        resume=args.resume,
                        project_dir=str(paths["weights_dir"]),
                        data_dir=str(paths["data_dir"]),
                    )
                else:
                    res = train_custom_detector(
                        dataset_name=d_name,
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        img_size=args.imgsz,
                        device=args.device,
                        workers=args.workers,
                        resume=args.resume,
                        project_dir=str(paths["weights_dir"]),
                        data_dir=str(paths["data_dir"]),
                    )
                results.append(res)
            except Exception as e:
                print(f"[!] Error training {m_name} on {d_name}: {e}", file=sys.stderr)

    total_time_min = (time.time() - overall_start) / 60.0
    print("\n" + "=" * 70)
    print(f" ALL TRAINING JOBS COMPLETE ({total_time_min:.2f} total minutes)")
    print("=" * 70)
    for r in results:
        print(f"  - {r['model_name']} on {r['dataset_name']} -> Best Weights: {r['best_weights']}")
    print(f"\nTrained weights are ready for benchmark evaluation via scripts/run_pipeline.py!")


if __name__ == "__main__":
    main()
