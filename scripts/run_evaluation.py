
from __future__ import annotations
#!/usr/bin/env python3
"""
Dedicated evaluation script for evaluating a single model checkpoint against a dataset.
"""

import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEFAULT_DATA_DIR, DEFAULT_RESULTS_DIR
from src.utils.env import get_device_info, set_seed
from src.data.dataset import AerialOBBDataset
from src.models import get_model
from scripts.run_pipeline import run_single_evaluation

def main():
    parser = argparse.ArgumentParser(description="Evaluate a specific model on an aerial dataset.")
    parser.add_argument("--model", type=str, default="yolov8n-obb", help="Model name or architecture.")
    parser.add_argument("--weights", type=str, default=None, help="Path to checkpoint file (.pt).")
    parser.add_argument("--dataset", type=str, default="codrone", help="Target dataset name.")
    parser.add_argument("--split", type=str, default="val", help="Dataset split (val/test).")
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR), help="Data directory.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size.")
    parser.add_argument("--device", type=str, default="auto", help="Inference device.")
    parser.add_argument("--iou-thresh", type=float, default=0.50, help="Matching IoU threshold.")
    parser.add_argument("--conf-thresh", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of images evaluated.")

    args = parser.parse_args()
    set_seed(42)

    dev_info = get_device_info(args.device)
    device = dev_info["device"]

    dataset = AerialOBBDataset(
        dataset_name=args.dataset,
        data_dir=Path(args.data_dir),
        split=args.split,
        max_samples=args.max_samples,
    )

    if len(dataset) == 0:
        print(f"[!] Dataset {args.dataset} has 0 images in split {args.split} at {args.data_dir}.", file=sys.stderr)
        sys.exit(1)

    print(f"[*] Evaluating {args.model} on {args.dataset} ({len(dataset)} images)...")
    model = get_model(args.model, device=device, num_classes=len(dataset.class_names))
    model.load(args.weights)

    out = run_single_evaluation(
        model=model,
        dataset=dataset,
        device=device,
        conf_thresh=args.conf_thresh,
        iou_thresh=args.iou_thresh,
        batch_size=args.batch_size,
    )

    map_res = out["map"]
    cls_res = out["classification"]
    reg_res = out["regression"]
    perf_res = out["performance"]

    print("\n" + "=" * 55)
    print(" EVALUATION RESULTS")
    print("=" * 55)
    print(f"  mAP@0.50        : {map_res['map50']:.4f}")
    print(f"  mAP@0.75        : {map_res['map75']:.4f}")
    print(f"  mAP@0.50:0.95   : {map_res['map50_95']:.4f}")
    print(f"  AP (Small)      : {map_res['ap_small']:.4f}")
    print(f"  Precision       : {cls_res['macro_precision']:.4f}")
    print(f"  Recall          : {cls_res['macro_recall']:.4f}")
    print(f"  F1-Score        : {cls_res['macro_f1']:.4f}")
    print(f"  Accuracy        : {cls_res['accuracy']:.4f}")
    print(f"  Angle MAE       : {reg_res['angle_mae']:.2f}°")
    print(f"  Pearson r       : {reg_res['angle_pearson_r']:.4f}")
    print(f"  R² Score        : {reg_res['angle_r2']:.4f}")
    print(f"  Throughput      : {perf_res['fps']:.1f} FPS ({perf_res['mean_latency_ms']:.1f} ms)")
    print("=" * 55)

if __name__ == "__main__":
    main()
