"""
GPU-accelerated PyTorch training engine for DETR-Based Oriented Object Detectors.
Supports: ARS-DETR, AO2-DETR, RiO-DETR, RHINO, and OrientedFormer.
Features Hungarian Bipartite Matching (scipy.optimize.linear_sum_assignment),
Multi-Task Loss (Cross-Entropy/Focal classification, L1 box regression, periodic continuous angle loss),
AdamW optimizer with CosineAnnealingLR, PyTorch AMP mixed precision,
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
from scipy.optimize import linear_sum_assignment

from src.config import DATASET_CLASSES, DEFAULT_WEIGHTS_DIR, resolve_pipeline_paths
from src.data.dataset import AerialOBBDataset
from src.models import get_model
from src.training.dataset import UnifiedOBBDataset, obb_collate_fn
from src.utils.checkpoint import atomic_save_json
from src.utils.env import is_drive_mounted, get_drive_root


def extract_detr_predictions(
    outputs: Dict[str, torch.Tensor],
    model_name: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Standardize the forward pass output of any DETR architecture:
    Returns:
        cls_logits: (B, Q, num_classes + 1)
        boxes_norm: (B, Q, 4) in [0, 1] (cx, cy, w, h)
        angle_rad:  (B, Q, 1) angle in radians
    """
    cls_logits = outputs["cls_logits"]

    if ("box_norm" in outputs or "boxes_norm" in outputs) and "angle_vec" in outputs:
        # ARS-DETR, RiO-DETR, RHINO
        boxes_norm = outputs["boxes_norm"] if "boxes_norm" in outputs else outputs["box_norm"]
        ang_vec = outputs["angle_vec"]
        angle_rad = torch.atan2(ang_vec[..., 0:1], ang_vec[..., 1:2])
        return cls_logits, boxes_norm, angle_rad

    elif "box_deltas" in outputs:
        # AO2-DETR
        deltas = outputs["box_deltas"]
        cx = torch.clamp(0.5 + deltas[..., 0:1] * 0.4, 0.0, 1.0)
        cy = torch.clamp(0.5 + deltas[..., 1:2] * 0.4, 0.0, 1.0)
        w = torch.clamp(torch.exp(torch.clamp(deltas[..., 2:3], -2.0, 2.0)) * 0.3, 0.01, 1.0)
        h = torch.clamp(torch.exp(torch.clamp(deltas[..., 3:4], -2.0, 2.0)) * 0.3, 0.01, 1.0)
        boxes_norm = torch.cat([cx, cy, w, h], dim=-1)
        angle_rad = deltas[..., 4:5]
        return cls_logits, boxes_norm, angle_rad

    elif "boxes" in outputs:
        # OrientedFormer
        raw_b = outputs["boxes"]
        cx = torch.clamp(raw_b[..., 0:1], 0.0, 1.0)
        cy = torch.clamp(raw_b[..., 1:2], 0.0, 1.0)
        w = torch.clamp(torch.exp(torch.clamp(raw_b[..., 2:3], -2.0, 2.0)) * 0.2, 0.01, 1.0)
        h = torch.clamp(torch.exp(torch.clamp(raw_b[..., 3:4], -2.0, 2.0)) * 0.2, 0.01, 1.0)
        boxes_norm = torch.cat([cx, cy, w, h], dim=-1)
        angle_rad = raw_b[..., 4:5]
        return cls_logits, boxes_norm, angle_rad

    else:
        raise ValueError(f"Unrecognized output format for DETR model: {list(outputs.keys())}")


class HungarianOBBMatcher(nn.Module):
    """
    Computes bipartite assignment between predicted object queries and ground-truth targets.
    Cost = Cost_class + Cost_box + Cost_angle
    """

    def __init__(self, cost_class: float = 1.0, cost_bbox: float = 5.0, cost_angle: float = 2.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_angle = cost_angle

    @torch.no_grad()
    def forward(
        self,
        pred_logits: torch.Tensor,
        pred_boxes: torch.Tensor,
        pred_angles: torch.Tensor,
        targets: List[Dict[str, torch.Tensor]],
        img_size: int,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Returns list of (query_indices, target_indices) for each image in the batch.
        """
        B, Q, num_classes_p1 = pred_logits.shape
        indices = []

        prob = F.softmax(pred_logits, dim=-1)

        for b in range(B):
            tgt_boxes = targets[b]["boxes"]
            tgt_labels = targets[b]["labels"]

            if len(tgt_labels) == 0:
                indices.append((
                    torch.empty(0, dtype=torch.int64, device=pred_logits.device),
                    torch.empty(0, dtype=torch.int64, device=pred_logits.device),
                ))
                continue

            # Normalized target boxes: [cx, cy, w, h] in [0, 1]
            tgt_b_norm = torch.zeros((len(tgt_boxes), 4), device=pred_boxes.device)
            tgt_b_norm[:, 0] = tgt_boxes[:, 0] / float(img_size)
            tgt_b_norm[:, 1] = tgt_boxes[:, 1] / float(img_size)
            tgt_b_norm[:, 2] = tgt_boxes[:, 2] / float(img_size)
            tgt_b_norm[:, 3] = tgt_boxes[:, 3] / float(img_size)

            tgt_ang_rad = torch.deg2rad(tgt_boxes[:, 4:5]).to(pred_angles.device)

            # 1. Classification Cost
            tgt_cls = tgt_labels.to(pred_logits.device)
            cost_class = -prob[b, :, tgt_cls]  # (Q, M)

            # 2. Box L1 Cost
            cost_bbox = torch.cdist(pred_boxes[b], tgt_b_norm, p=1)  # (Q, M)

            # 3. Angle Continuous Cost
            diff_ang = 2.0 * (pred_angles[b] - tgt_ang_rad.T)  # (Q, M)
            cost_angle = 1.0 - torch.cos(diff_ang)

            cost_matrix = (
                self.cost_class * cost_class +
                self.cost_bbox * cost_bbox +
                self.cost_angle * cost_angle
            ).cpu().numpy()

            q_ind, t_ind = linear_sum_assignment(cost_matrix)
            indices.append((
                torch.as_tensor(q_ind, dtype=torch.int64, device=pred_logits.device),
                torch.as_tensor(t_ind, dtype=torch.int64, device=pred_logits.device),
            ))

        return indices


class DETROBBLoss(nn.Module):
    """
    DETR Loss for Oriented Object Detection:
      - Cross Entropy on all queries (matched -> target class, unmatched -> background class)
      - Smooth L1 Loss on matched normalized coordinates (cx, cy, w, h)
      - Continuous Circular Trigonometric Loss on matched angles (sin, cos)
    """

    def __init__(self, num_classes: int, loss_class: float = 1.0, loss_bbox: float = 5.0, loss_angle: float = 2.5):
        super().__init__()
        self.num_classes = num_classes
        self.loss_class = loss_class
        self.loss_bbox = loss_bbox
        self.loss_angle = loss_angle
        self.matcher = HungarianOBBMatcher()

    def forward(
        self,
        pred_logits: torch.Tensor,
        pred_boxes: torch.Tensor,
        pred_angles: torch.Tensor,
        targets: List[Dict[str, torch.Tensor]],
        img_size: int,
    ) -> Dict[str, torch.Tensor]:
        B, Q, num_classes_p1 = pred_logits.shape
        bg_idx = self.num_classes  # Background class index

        indices = self.matcher(pred_logits, pred_boxes, pred_angles, targets, img_size)

        # 1. Target classification tensor initialized to background
        target_classes = torch.full((B, Q), bg_idx, dtype=torch.int64, device=pred_logits.device)
        for b, (q_idx, t_idx) in enumerate(indices):
            if len(q_idx) > 0:
                target_classes[b, q_idx] = targets[b]["labels"][t_idx].to(pred_logits.device)

        cls_loss = F.cross_entropy(pred_logits.view(-1, num_classes_p1), target_classes.view(-1))

        # 2. Matched Bounding Box & Angle Losses
        box_losses = []
        angle_losses = []
        num_matched = 0

        for b, (q_idx, t_idx) in enumerate(indices):
            if len(q_idx) == 0:
                continue

            num_matched += len(q_idx)
            m_pred_box = pred_boxes[b, q_idx]
            m_pred_ang = pred_angles[b, q_idx]

            raw_boxes = targets[b]["boxes"][t_idx]
            tgt_b_norm = torch.zeros_like(m_pred_box)
            tgt_b_norm[:, 0] = raw_boxes[:, 0] / float(img_size)
            tgt_b_norm[:, 1] = raw_boxes[:, 1] / float(img_size)
            tgt_b_norm[:, 2] = raw_boxes[:, 2] / float(img_size)
            tgt_b_norm[:, 3] = raw_boxes[:, 3] / float(img_size)

            tgt_ang_rad = torch.deg2rad(raw_boxes[:, 4:5]).to(m_pred_ang.device)

            # Box L1 Loss
            box_l1 = F.smooth_l1_loss(m_pred_box, tgt_b_norm, reduction="sum")
            box_losses.append(box_l1)

            # Angle periodic continuous loss
            sin_p, cos_p = torch.sin(2.0 * m_pred_ang), torch.cos(2.0 * m_pred_ang)
            sin_t, cos_t = torch.sin(2.0 * tgt_ang_rad), torch.cos(2.0 * tgt_ang_rad)
            ang_l = (F.mse_loss(sin_p, sin_t, reduction="sum") + F.mse_loss(cos_p, cos_t, reduction="sum"))
            angle_losses.append(ang_l)

        denom = max(num_matched, 1)
        box_loss = (torch.stack(box_losses).sum() / denom) if box_losses else torch.tensor(0.0, device=pred_logits.device)
        angle_loss = (torch.stack(angle_losses).sum() / denom) if angle_losses else torch.tensor(0.0, device=pred_logits.device)

        total_loss = (
            self.loss_class * cls_loss +
            self.loss_bbox * box_loss +
            self.loss_angle * angle_loss
        )

        return {
            "total_loss": total_loss,
            "cls_loss": cls_loss.detach(),
            "box_loss": box_loss.detach(),
            "angle_loss": angle_loss.detach(),
        }


def train_detr_model(
    model_name: str = "ars-detr",
    dataset_name: str = "visdrone",
    epochs: int = 15,
    batch_size: int = 8,
    img_size: int = 640,
    lr: float = 1e-4,
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
    Unified training engine for DETR-based Oriented Object Detectors.
    """
    # 1. Device Selection
    if device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device.startswith("cuda"):
        dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device("cpu")

    print(f"\n{'=' * 70}")
    print(f" Starting DETR OBB Training: '{model_name}' on {dataset_name.upper()}")
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

    # 3. Model, Loss, Optimizer
    detector = get_model(model_name, device=str(dev), num_classes=num_classes)
    detector.load()
    net = detector.net.to(dev)

    criterion = DETROBBLoss(num_classes=num_classes).to(dev)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

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
                outputs = net(imgs)
                pred_logits, pred_boxes, pred_angles = extract_detr_predictions(outputs, model_name=model_name)
                losses = criterion(pred_logits, pred_boxes, pred_angles, targets, img_size=img_size)
                loss = losses["total_loss"]

            if scaler is not None and not use_bf16:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
                optimizer.step()

            running_loss += loss.item()
            running_cls += losses["cls_loss"].item()
            running_box += losses["box_loss"].item()
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
                outputs = net(imgs)
                pred_logits, pred_boxes, pred_angles = extract_detr_predictions(outputs, model_name=model_name)
                v_losses = criterion(pred_logits, pred_boxes, pred_angles, targets, img_size=img_size)
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
    print(f"\n[✓] DETR Training Completed in {total_time:.1f}s ({total_time / 60:.1f} min)")
    print(f"    - Best Model Weights : {best_weights_path}")
    print(f"    - Final Best Val Loss: {best_val_loss:.4f}")

    return {
        "best_weights": str(best_weights_path),
        "last_weights": str(last_weights_path),
        "epochs_trained": epochs,
        "best_val_loss": best_val_loss,
        "history": history,
    }
