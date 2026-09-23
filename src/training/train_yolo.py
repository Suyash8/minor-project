"""
Full GPU-accelerated training engine for Ultralytics YOLO-OBB models (YOLOv8-OBB & YOLO11-OBB).
Configured for maximum T4 GPU throughput with Tensor Core FP16 AMP, multi-worker parallel data loading,
per-epoch checkpoints, and Google Drive auto-persistence.
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Dict, Any, Optional
import torch

from src.config import DEFAULT_WEIGHTS_DIR, resolve_pipeline_paths
from src.data.yolo_converter import prepare_dataset_for_yolo_training
from src.utils.env import is_colab, is_drive_mounted, get_drive_root


def train_yolo_obb(
    model_name: str = "yolov8n-obb",
    dataset_name: str = "visdrone",
    epochs: int = 20,
    batch_size: int = 16,
    imgsz: int = 640,
    device: str = "auto",
    workers: int = 4,
    lr0: float = 0.01,
    cache: Optional[str] = None,
    pretrained_weights: Optional[str] = None,
    resume: bool = True,
    project_dir: Optional[str] = None,
    data_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Train a YOLO-OBB model on a given dataset with full GPU acceleration and epoch checkpointing.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("Ultralytics is required for YOLO training. Run: pip install ultralytics")

    # 1. Resolve Computing Device
    if device == "auto":
        dev = "0" if torch.cuda.is_available() else "cpu"
    elif device.startswith("cuda"):
        dev = "0"
    else:
        dev = "cpu"

    print(f"\n{'=' * 70}")
    print(f" Starting GPU Training: {model_name} on {dataset_name.upper()}")
    print(f"{'=' * 70}")
    print(f"[*] Target Hardware : {'NVIDIA CUDA GPU (device 0)' if dev != 'cpu' else 'CPU'}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"[*] GPU Name        : {gpu_name} ({gpu_mem:.2f} GB VRAM)")
    print(f"[*] Training Config : {epochs} Epochs | Batch Size: {batch_size} | Image Size: {imgsz}px | Workers: {workers}")

    # 2. Prepare Dataset YAML
    print(f"[*] Verifying YOLO-OBB dataset format for '{dataset_name}'...")
    yaml_path = prepare_dataset_for_yolo_training(dataset_name, data_dir=Path(data_dir) if data_dir else None)
    print(f"[✓] Dataset YAML ready at: {yaml_path}")

    # 3. Setup Project & Weights Directory
    paths = resolve_pipeline_paths()
    weights_dir = Path(project_dir or paths["weights_dir"])
    weights_dir.mkdir(parents=True, exist_ok=True)

    run_name = f"{model_name}_{dataset_name}"
    save_dir = weights_dir / run_name
    last_pt = save_dir / "weights" / "last.pt"
    best_pt = save_dir / "weights" / "best.pt"

    # Check for resumption
    can_resume = resume and last_pt.exists()
    if can_resume:
        print(f"[*] Found previous checkpoint at: {last_pt}. Resuming training seamlessly...")
        model = YOLO(str(last_pt))
    else:
        init_weights = pretrained_weights
        if not init_weights:
            candidate_weights = weights_dir / f"{model_name}.pt"
            if candidate_weights.exists():
                init_weights = str(candidate_weights)
            else:
                init_weights = f"{model_name}.pt"
        print(f"[*] Initializing model with base weights: {init_weights}")
        model = YOLO(init_weights)

    # 4. Execute Native Ultralytics Training
    start_time = time.time()
    train_args = {
        "data": str(yaml_path),
        "epochs": epochs,
        "batch": batch_size,
        "imgsz": imgsz,
        "device": dev,
        "workers": workers if dev != "cpu" else 0,
        "lr0": lr0,
        "save_period": 1,        # Save checkpoint after EVERY single epoch
        "save": True,
        "project": str(weights_dir),
        "name": run_name,
        "exist_ok": True,
        "verbose": True,
        "resume": can_resume,
    }
    if cache and str(cache).lower() in ("ram", "disk", "true", "1"):
        train_args["cache"] = str(cache).lower() if str(cache).lower() in ("ram", "disk") else True

    results = model.train(**train_args)
    elapsed_minutes = (time.time() - start_time) / 60.0

    print(f"\n[✓] Training completed in {elapsed_minutes:.2f} minutes!")

    # 5. Persist Best and Last Checkpoints
    final_best = save_dir / "weights" / "best.pt"
    final_last = save_dir / "weights" / "last.pt"

    target_best = weights_dir / f"{model_name}_{dataset_name}_best.pt"
    if final_best.exists() and final_best.resolve() != target_best.resolve():
        shutil.copy2(final_best, target_best)
        print(f"[✓] Saved best model checkpoint to: {target_best}")

    # Synchronize to Google Drive if active
    if is_drive_mounted():
        drive_weights = get_drive_root() / "weights"
        drive_weights.mkdir(parents=True, exist_ok=True)
        if target_best.exists():
            dest = drive_weights / target_best.name
            if target_best.resolve() != dest.resolve():
                shutil.copy2(target_best, dest)
                print(f"[✓] Mirrored checkpoint to Google Drive: {dest}")

    return {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "best_weights": str(target_best if target_best.exists() else final_best),
        "last_weights": str(final_last),
        "elapsed_minutes": elapsed_minutes,
        "epochs_completed": epochs,
    }
