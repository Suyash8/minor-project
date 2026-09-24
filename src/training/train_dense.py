"""
GPU-accelerated PyTorch training engine for Dense Grid-Based OBB Detectors.
Supports: LSKNet, Swin-OBB, STD (Spatial Transform Decoupling), RVSA, Custom-OBB, and Native YOLO11-OBB.
Features multi-task loss (focal classification, box regression, continuous circular angle loss),
AdamW optimizer with CosineAnnealingLR, PyTorch AMP mixed precision, multi-worker DataLoader,
and per-epoch checkpointing with automatic Google Drive persistence.
"""

from __future__ import annotations

import os
import time
import math
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.config import DATASET_CLASSES, DEFAULT_WEIGHTS_DIR, resolve_pipeline_paths
from src.data.dataset import AerialOBBDataset
from src.models import get_model
from src.training.dataset import UnifiedOBBDataset, obb_collate_fn
from src.utils.checkpoint import atomic_save_json
from src.utils.env import is_drive_mounted, get_drive_root


class OBBMaskLoss(nn.Module):
    """
    Composite Multi-Task Loss for Dense Oriented Object Detection:
    - Focal Loss for multi-class classification
    - Smooth L1 Loss for center offset & dimension regression
    - Circular continuous trigonometric Loss for orientation angles
    - Binary Cross Entropy for objectness presence
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(
        self,
        cls_logits: torch.Tensor,
        reg_preds: torch.Tensor,
        cls_targets: torch.Tensor,
        reg_targets: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        # 1. Classification Focal Loss
        p = torch.sigmoid(cls_logits)
        pt = p * cls_targets + (1.0 - p) * (1.0 - cls_targets)
        w = self.alpha * cls_targets + (1.0 - self.alpha) * (1.0 - cls_targets)
        focal_weight = w * ((1.0 - pt) ** self.gamma)
        cls_loss = F.binary_cross_entropy_with_logits(cls_logits, cls_targets, reduction="none")
        cls_loss = (focal_weight * cls_loss).mean() * 10.0

        # 2. Regression Loss on Object Anchors
        obj_mask = reg_targets[:, 6:7]  # (B, 1, H, W)
        num_objs = obj_mask.sum().clamp(min=1.0)

        # Center (dx, dy)
        center_loss = (F.smooth_l1_loss(reg_preds[:, 0:2], reg_targets[:, 0:2], reduction="none") * obj_mask).sum() / num_objs

        # Dimensions (log_w, log_h)
        dim_loss = (F.smooth_l1_loss(reg_preds[:, 2:4], reg_targets[:, 2:4], reduction="none") * obj_mask).sum() / num_objs

        # Orientation continuous circular angle loss (sin, cos vectors)
        pred_sin, pred_cos = reg_preds[:, 4:5], reg_preds[:, 5:6]
        tgt_sin, tgt_cos = reg_targets[:, 4:5], reg_targets[:, 5:6]
        angle_loss = (
            F.mse_loss(pred_sin, tgt_sin, reduction="none") * obj_mask +
            F.mse_loss(pred_cos, tgt_cos, reduction="none") * obj_mask
        ).sum() / num_objs

        # Objectness presence loss
        obj_loss = F.binary_cross_entropy_with_logits(reg_preds[:, 6:7], obj_mask)

        total_loss = cls_loss + 2.0 * center_loss + 1.5 * dim_loss + 2.5 * angle_loss + obj_loss

        return {
            "total_loss": total_loss,
            "cls_loss": cls_loss.detach(),
            "center_loss": center_loss.detach(),
            "dim_loss": dim_loss.detach(),
            "angle_loss": angle_loss.detach(),
            "obj_loss": obj_loss.detach(),
        }


def encode_dense_targets(
    targets: List[Dict[str, torch.Tensor]],
    batch_size: int,
    num_classes: int,
    feat_h: int,
    feat_w: int,
    img_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Project ground-truth bounding box annotations to the spatial feature grid (feat_h, feat_w).
    Returns:
        cls_targets: (B, num_classes, feat_h, feat_w)
        reg_targets: (B, 7, feat_h, feat_w) -> [dx, dy, log_w, log_h, sin_a, cos_a, obj_mask]
    """
    stride_x = img_size / float(feat_w)
    stride_y = img_size / float(feat_h)

    cls_targets = torch.zeros((batch_size, num_classes, feat_h, feat_w), device=device)
    reg_targets = torch.zeros((batch_size, 7, feat_h, feat_w), device=device)

    for b, tgt in enumerate(targets):
        boxes = tgt["boxes"].to(device)
        labels = tgt["labels"].to(device)
        if len(boxes) == 0:
            continue

        for box, lbl in zip(boxes, labels):
            cx = box[0]
            cy = box[1]
            w = box[2]
            h = box[3]
            ang_deg = box[4] % 180.0

            gx = int(cx / stride_x)
            gy = int(cy / stride_y)

            if 0 <= gx < feat_w and 0 <= gy < feat_h:
                c_idx = int(lbl.item())
                if 0 <= c_idx < num_classes:
                    cls_targets[b, c_idx, gy, gx] = 1.0

                    dx = (cx / stride_x) - (gx + 0.5)
                    dy = (cy / stride_y) - (gy + 0.5)
                    log_w = torch.log(torch.clamp(w / (stride_x * 3.0), min=0.1))
                    log_h = torch.log(torch.clamp(h / (stride_y * 6.0), min=0.1))
                    ang_rad = torch.deg2rad(ang_deg)
                    sin_a = torch.sin(2.0 * ang_rad)
                    cos_a = torch.cos(2.0 * ang_rad)

                    reg_targets[b, 0, gy, gx] = dx
                    reg_targets[b, 1, gy, gx] = dy
                    reg_targets[b, 2, gy, gx] = log_w
                    reg_targets[b, 3, gy, gx] = log_h
                    reg_targets[b, 4, gy, gx] = sin_a
                    reg_targets[b, 5, gy, gx] = cos_a
                    reg_targets[b, 6, gy, gx] = 1.0

    return cls_targets, reg_targets


def forward_dense_network(
    model_name: str,
    net: nn.Module,
    batch_tensor: torch.Tensor,
    num_classes: int,
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    """
    Standardize the forward pass output of all dense architectures into:
      cls_logits: (B, num_classes, H, W)
      reg_preds:  (B, 7, H, W)
      feat_h, feat_w: spatial dimensions
    """
    m_lower = model_name.lower().strip()
    B = batch_tensor.shape[0]

    if "std" in m_lower:
        preds, (feat_h, feat_w) = net(batch_tensor)
        cls_logits = preds["cls_logits"].permute(0, 2, 1).view(B, num_classes, feat_h, feat_w)
        reg_parts = torch.cat([preds["pos"], preds["scale"], preds["angle"], preds["obj_logits"]], dim=-1)
        reg_preds = reg_parts.permute(0, 2, 1).view(B, 7, feat_h, feat_w)
        return cls_logits, reg_preds, feat_h, feat_w

    elif "rvsa" in m_lower or "swin" in m_lower:
        cls_logits, reg_out, (feat_h, feat_w) = net(batch_tensor)
        cls_logits = cls_logits.permute(0, 2, 1).view(B, num_classes, feat_h, feat_w)
        reg_preds = reg_out.permute(0, 2, 1).view(B, 7, feat_h, feat_w)
        return cls_logits, reg_preds, feat_h, feat_w

    else:
        # lsknet, custom-obb, yolo11-obb
        cls_logits, reg_preds, (feat_h, feat_w) = net(batch_tensor)
        return cls_logits, reg_preds, feat_h, feat_w


def train_dense_model(
    model_name: str = "lsknet",
    dataset_name: str = "visdrone",
    epochs: int = 15,
    batch_size: int = 16,
    img_size: int = 640,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "auto",
    workers: int = 4,
    resume: bool = True,
    project_dir: Optional[str] = None,
    data_dir: Optional[str] = None,
    max_batches: Optional[int] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Unified training engine for dense grid-based OBB architectures.
    """
    # 1. Device Selection
    if device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device.startswith("cuda"):
        dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device("cpu")

    print(f"\n{'=' * 70}")
    print(f" Starting Dense OBB Training: '{model_name}' on {dataset_name.upper()}")
    print(f"{'=' * 70}")
    print(f"[*] Compute Device   : {dev}")
    if dev.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"[*] GPU Model        : {gpu_name} ({gpu_mem:.2f} GB VRAM)")
    print(f"[*] Training Config  : {epochs} Epochs | Batch: {batch_size} | LR: {lr} | Workers: {workers}")

    # 2. Datasets and Loaders
    train_raw = AerialOBBDataset(dataset_name=dataset_name, data_dir=data_dir, split="train")
    val_raw = AerialOBBDataset(dataset_name=dataset_name, data_dir=data_dir, split="val")

    labeled_train = sum(1 for s in train_raw.samples if s["label_path"] is not None and s["label_path"].exists())
    print(f"[*] Verified training set for '{dataset_name}': {len(train_raw)} images ({labeled_train} with valid label files).")

    if labeled_train == 0:
        raise RuntimeError(
            f"Cannot train '{model_name}' on '{dataset_name}': 0 ground truth annotations were found. "
            f"Please verify that annotations exist in the dataset directory."
        )

    train_ds = UnifiedOBBDataset(train_raw, img_size=img_size)
    val_ds = UnifiedOBBDataset(val_raw, img_size=img_size)

    num_classes = len(train_raw.class_names)
    num_workers = workers if dev.type == "cuda" else 0

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=obb_collate_fn,
        pin_memory=(dev.type == "cuda"),
        drop_last=True if len(train_ds) > batch_size else False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=obb_collate_fn,
        pin_memory=(dev.type == "cuda"),
    )

    # 3. Model, Loss, Optimizer, Scaler
    detector = get_model(model_name, device=str(dev), num_classes=num_classes)
    detector.load()
    net = detector.net.to(dev)

    criterion = OBBMaskLoss().to(dev)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    use_bf16 = False
    if dev.type == "cuda" and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
        use_bf16 = True
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

    scaler = None
    if dev.type == "cuda":
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            scaler = torch.amp.GradScaler("cuda", enabled=not use_bf16)
        elif hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "GradScaler"):
            scaler = torch.cuda.amp.GradScaler(enabled=not use_bf16)

    # 4. Checkpoint Storage Directory
    paths = resolve_pipeline_paths(weights_dir=project_dir, data_dir=data_dir)
    save_dir = paths["weights_dir"] / f"{model_name}_{dataset_name}"
    save_dir.mkdir(parents=True, exist_ok=True)
    best_weights_path = save_dir / f"{model_name}_best.pt"
    last_weights_path = save_dir / f"{model_name}_last.pt"

    start_epoch = 1
    best_val_loss = float("inf")
    history = []

    # Resume support
    if resume and last_weights_path.exists():
        try:
            ckpt = torch.load(last_weights_path, map_location=dev)
            net.load_state_dict(ckpt["model_state"], strict=False)
            if "optimizer_state" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state"])
            start_epoch = ckpt.get("epoch", 0) + 1
            best_val_loss = ckpt.get("best_val_loss", float("inf"))
            history = ckpt.get("history", [])
            print(f"[✓] Successfully resumed training from epoch {start_epoch} (Best val loss: {best_val_loss:.4f})")
        except Exception as e:
            print(f"[!] Could not resume from checkpoint: {e}. Starting fresh.")

    # 5. Training Loop
    total_train_start = time.time()

    for epoch in range(start_epoch, epochs + 1):
        net.train()
        epoch_start = time.time()
        running_loss = 0.0
        running_cls = 0.0
        running_box = 0.0
        running_ang = 0.0
        batch_count = 0

        for b_idx, (imgs, targets) in enumerate(train_loader):
            if max_batches is not None and b_idx >= max_batches:
                break
            imgs = imgs.to(dev, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=dev.type, dtype=amp_dtype, enabled=(dev.type == "cuda")):
                cls_logits, reg_preds, feat_h, feat_w = forward_dense_network(
                    model_name=model_name, net=net, batch_tensor=imgs, num_classes=num_classes
                )
                cls_tgt, reg_tgt = encode_dense_targets(
                    targets=targets,
                    batch_size=imgs.shape[0],
                    num_classes=num_classes,
                    feat_h=feat_h,
                    feat_w=feat_w,
                    img_size=img_size,
                    device=dev,
                )
                losses = criterion(cls_logits, reg_preds, cls_tgt, reg_tgt)
                loss = losses["total_loss"]

            if scaler is not None and not use_bf16:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=10.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=10.0)
                optimizer.step()

            running_loss += loss.item()
            running_cls += losses["cls_loss"].item()
            running_box += (losses["center_loss"] + losses["dim_loss"]).item()
            running_ang += losses["angle_loss"].item()
            batch_count += 1

        scheduler.step()

        train_loss = running_loss / max(batch_count, 1)
        train_cls = running_cls / max(batch_count, 1)
        train_box = running_box / max(batch_count, 1)
        train_ang = running_ang / max(batch_count, 1)

        # Validation Phase
        net.eval()
        val_loss = 0.0
        val_batch_count = 0

        with torch.no_grad():
            for v_idx, (imgs, targets) in enumerate(val_loader):
                if max_batches is not None and v_idx >= max_batches:
                    break
                imgs = imgs.to(dev, non_blocking=True)
                cls_logits, reg_preds, feat_h, feat_w = forward_dense_network(
                    model_name=model_name, net=net, batch_tensor=imgs, num_classes=num_classes
                )
                cls_tgt, reg_tgt = encode_dense_targets(
                    targets=targets,
                    batch_size=imgs.shape[0],
                    num_classes=num_classes,
                    feat_h=feat_h,
                    feat_w=feat_w,
                    img_size=img_size,
                    device=dev,
                )
                v_losses = criterion(cls_logits, reg_preds, cls_tgt, reg_tgt)
                val_loss += v_losses["total_loss"].item()
                val_batch_count += 1

        val_loss = val_loss / max(val_batch_count, 1)
        epoch_time = time.time() - epoch_start
        cur_lr = optimizer.param_groups[0]["lr"]

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_cls": round(train_cls, 4),
            "train_box": round(train_box, 4),
            "train_ang": round(train_ang, 4),
            "val_loss": round(val_loss, 4),
            "lr": round(cur_lr, 6),
            "time_sec": round(epoch_time, 2),
            "is_best": is_best,
        }
        history.append(epoch_record)

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train: {train_loss:.4f} (cls: {train_cls:.3f}, box: {train_box:.3f}, ang: {train_ang:.3f}) | "
            f"Val: {val_loss:.4f} {'(*BEST*)' if is_best else ''} | "
            f"LR: {cur_lr:.2e} | Time: {epoch_time:.1f}s"
        )

        # Save Checkpoint
        checkpoint_data = {
            "epoch": epoch,
            "model_name": model_name,
            "dataset_name": dataset_name,
            "model_state": net.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "history": history,
            "num_classes": num_classes,
        }

        torch.save(checkpoint_data, last_weights_path)
        if is_best:
            torch.save(checkpoint_data, best_weights_path)
            # Automatic Google Drive Persistence
            if is_drive_mounted():
                drive_save = get_drive_root() / "weights" / f"{model_name}_{dataset_name}"
                drive_save.mkdir(parents=True, exist_ok=True)
                shutil.copy2(best_weights_path, drive_save / f"{model_name}_best.pt")

    # Save training history JSON
    history_path = save_dir / "train_history.json"
    atomic_save_json(history, history_path)

    total_time = time.time() - total_train_start
    print(f"\n[✓] Dense Training Completed in {total_time:.1f}s ({total_time / 60:.1f} min)")
    print(f"    - Best Model Weights : {best_weights_path}")
    print(f"    - Final Best Val Loss: {best_val_loss:.4f}")

    return {
        "best_weights": str(best_weights_path),
        "last_weights": str(last_weights_path),
        "epochs_trained": epochs,
        "best_val_loss": best_val_loss,
        "history": history,
    }
