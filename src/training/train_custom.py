"""
Full GPU-accelerated PyTorch training engine for CustomOBBDetector.
Features multi-task loss (focal classification, box regression, continuous circular angle loss),
AdamW optimizer with CosineAnnealingLR, PyTorch AMP mixed precision, multi-worker DataLoader,
and per-epoch checkpointing to local and Google Drive storage.
"""

from __future__ import annotations

import os
import time
import math
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from src.config import DATASET_CLASSES, DEFAULT_WEIGHTS_DIR, resolve_pipeline_paths
from src.data.dataset import AerialOBBDataset
from src.models.custom_obb import PyTorchOBBNet
from src.utils.checkpoint import atomic_save_json
from src.utils.env import is_drive_mounted, get_drive_root


class PyTorchOBBDataset(Dataset):
    """
    PyTorch Dataset wrapper around AerialOBBDataset for batch training.
    Encodes oriented bounding boxes into target feature grids (stride 8).
    """

    def __init__(self, raw_dataset: AerialOBBDataset, img_size: int = 640):
        self.raw = raw_dataset
        self.img_size = img_size
        self.stride = 8
        self.feat_size = img_size // self.stride
        self.num_classes = len(self.raw.class_names)

    def __len__(self) -> int:
        return len(self.raw)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img = self.raw.get_image(idx)
        orig_w, orig_h = img.size

        # Resize image to model input size
        resized = img.resize((self.img_size, self.img_size), Image.Resampling.BILINEAR)
        img_arr = np.array(resized, dtype=np.float32) / 255.0
        # (H, W, C) -> (C, H, W)
        img_tensor = torch.from_numpy(img_arr).permute(2, 0, 1)

        # Ground truth targets
        gt_info = self.raw.get_ground_truth(idx, img_width=orig_w, img_height=orig_h)

        # Target classification map: (num_classes, feat_h, feat_w)
        cls_target = torch.zeros((self.num_classes, self.feat_size, self.feat_size), dtype=torch.float32)
        # Target regression map: (7, feat_h, feat_w) -> [dx, dy, log_w, log_h, sin_a, cos_a, obj_mask]
        reg_target = torch.zeros((7, self.feat_size, self.feat_size), dtype=torch.float32)
        obj_mask = torch.zeros((self.feat_size, self.feat_size), dtype=torch.float32)

        scale_x = self.img_size / float(orig_w)
        scale_y = self.img_size / float(orig_h)

        for c_idx in range(self.num_classes):
            boxes = gt_info[c_idx].get("boxes", np.zeros((0, 5)))
            for box in boxes:
                cx = float(box[0]) * scale_x
                cy = float(box[1]) * scale_y
                w = max(float(box[2]) * scale_x, 4.0)
                h = max(float(box[3]) * scale_y, 4.0)
                angle_deg = float(box[4]) % 180.0
                angle_rad = math.radians(angle_deg)

                grid_x = int(cx // self.stride)
                grid_y = int(cy // self.stride)

                if 0 <= grid_x < self.feat_size and 0 <= grid_y < self.feat_size:
                    cls_target[c_idx, grid_y, grid_x] = 1.0
                    obj_mask[grid_y, grid_x] = 1.0

                    dx = (cx / self.stride) - (grid_x + 0.5)
                    dy = (cy / self.stride) - (grid_y + 0.5)
                    log_w = math.log(max(w / (self.stride * 3.0), 0.1))
                    log_h = math.log(max(h / (self.stride * 6.0), 0.1))
                    sin_a = math.sin(2.0 * angle_rad)
                    cos_a = math.cos(2.0 * angle_rad)

                    reg_target[:, grid_y, grid_x] = torch.tensor(
                        [dx, dy, log_w, log_h, sin_a, cos_a, 1.0],
                        dtype=torch.float32,
                    )

        return img_tensor, cls_target, reg_target


class OBBMaskLoss(nn.Module):
    """
    Composite Multi-Task Loss for Oriented Object Detection:
    - Focal Loss for multi-class classification
    - Smooth L1 Loss for center offset & dimension regression
    - Circular continuous trigonometric Loss for orientation angles
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
        obj_mask = reg_targets[:, 6:7] # (B, 1, H, W)
        num_objs = obj_mask.sum().clamp(min=1.0)

        # Center (dx, dy)
        center_loss = (F.smooth_l1_loss(reg_preds[:, 0:2], reg_targets[:, 0:2], reduction="none") * obj_mask).sum() / num_objs

        # Dimensions (log_w, log_h)
        dim_loss = (F.smooth_l1_loss(reg_preds[:, 2:4], reg_targets[:, 2:4], reduction="none") * obj_mask).sum() / num_objs

        # Orientation continuous circular angle loss (sin, cos vectors)
        pred_sin, pred_cos = reg_preds[:, 4:5], reg_preds[:, 5:6]
        tgt_sin, tgt_cos = reg_targets[:, 4:5], reg_targets[:, 5:6]
        angle_loss = (F.mse_loss(pred_sin, tgt_sin, reduction="none") * obj_mask +
                      F.mse_loss(pred_cos, tgt_cos, reduction="none") * obj_mask).sum() / num_objs

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


def train_custom_detector(
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
) -> Dict[str, Any]:
    """
    Train custom PyTorch OBB network with full GPU acceleration, AMP mixed precision,
    cosine learning rate schedule, and per-epoch checkpointing.
    """
    # 1. Device Selection
    if device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device.startswith("cuda"):
        dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device("cpu")

    print(f"\n{'=' * 70}")
    print(f" Starting Custom PyTorch OBB Training on {dataset_name.upper()}")
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

    train_ds = PyTorchOBBDataset(train_raw, img_size=img_size)
    val_ds = PyTorchOBBDataset(val_raw, img_size=img_size)

    num_classes = len(train_raw.class_names)
    num_workers = workers if dev.type == "cuda" else 0

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(dev.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(dev.type == "cuda"),
    )

    # 3. Model, Loss, Optimizer, Scaler
    model = PyTorchOBBNet(num_classes=num_classes).to(dev)
    criterion = OBBMaskLoss().to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    device_type = "cuda" if dev.type == "cuda" else "cpu"
    use_bf16 = False
    if dev.type == "cuda" and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
        use_bf16 = True
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler(device_type, enabled=(dev.type == "cuda" and not use_bf16))
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=(dev.type == "cuda" and not use_bf16))

    # 4. Checkpoint Setup
    paths = resolve_pipeline_paths()
    weights_dir = Path(project_dir or paths["weights_dir"])
    weights_dir.mkdir(parents=True, exist_ok=True)

    best_pt = weights_dir / f"custom-obb_{dataset_name}_best.pt"
    last_pt = weights_dir / f"custom-obb_{dataset_name}_last.pt"

    start_epoch = 1
    best_loss = float("inf")

    if resume and last_pt.exists():
        print(f"[*] Loading previous checkpoint from {last_pt}...")
        checkpoint = torch.load(last_pt, map_location=dev)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_loss = checkpoint.get("best_loss", float("inf"))
        print(f"[✓] Resumed at epoch {start_epoch}")

    # 5. Training Loop
    total_start = time.time()

    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.time()
        model.train()
        train_loss_accum = 0.0

        for b_idx, (imgs, cls_tgt, reg_tgt) in enumerate(train_loader):
            imgs = imgs.to(dev, non_blocking=True)
            cls_tgt = cls_tgt.to(dev, non_blocking=True)
            reg_tgt = reg_tgt.to(dev, non_blocking=True)

            optimizer.zero_grad()
            amp_ctx = torch.amp.autocast(device_type, dtype=amp_dtype, enabled=(dev.type == "cuda")) if hasattr(torch, "amp") and hasattr(torch.amp, "autocast") else torch.cuda.amp.autocast(enabled=(dev.type == "cuda"))
            with amp_ctx:
                cls_logits, reg_out = model(imgs)
                loss_dict = criterion(cls_logits, reg_out, cls_tgt, reg_tgt)
                loss = loss_dict["total_loss"]

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss_accum += loss.item()

            if (b_idx + 1) % max(1, len(train_loader) // 4) == 0:
                print(
                    f"    Epoch [{epoch:02d}/{epochs:02d}] | Batch [{b_idx + 1:03d}/{len(train_loader):03d}] | "
                    f"Loss: {loss.item():.4f} (cls: {loss_dict['cls_loss']:.3f}, "
                    f"box: {loss_dict['center_loss'] + loss_dict['dim_loss']:.3f}, "
                    f"ang: {loss_dict['angle_loss']:.3f})"
                )

        scheduler.step()
        avg_train_loss = train_loss_accum / max(1, len(train_loader))

        # Validation Pass
        model.eval()
        val_loss_accum = 0.0
        with torch.no_grad():
            for imgs, cls_tgt, reg_tgt in val_loader:
                imgs = imgs.to(dev, non_blocking=True)
                cls_tgt = cls_tgt.to(dev, non_blocking=True)
                reg_tgt = reg_tgt.to(dev, non_blocking=True)

                amp_ctx_val = torch.amp.autocast(device_type, dtype=amp_dtype, enabled=(dev.type == "cuda")) if hasattr(torch, "amp") and hasattr(torch.amp, "autocast") else torch.cuda.amp.autocast(enabled=(dev.type == "cuda"))
                with amp_ctx_val:
                    cls_logits, reg_out = model(imgs)
                    loss_dict = criterion(cls_logits, reg_out, cls_tgt, reg_tgt)
                    val_loss_accum += loss_dict["total_loss"].item()

        avg_val_loss = val_loss_accum / max(1, len(val_loader))
        epoch_sec = time.time() - epoch_start

        print(
            f"--> Epoch {epoch:02d} Complete in {epoch_sec:.1f}s | "
            f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}"
        )

        # Save Checkpoint after EVERY Epoch
        checkpoint_data = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_loss": avg_val_loss,
            "best_loss": best_loss,
            "class_names": train_raw.class_names,
        }
        torch.save(checkpoint_data, last_pt)

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(checkpoint_data, best_pt)
            print(f"    [★] New best validation loss: {best_loss:.4f}! Saved {best_pt.name}")

            # Mirror to Drive if active and not already on Drive
            if is_drive_mounted():
                drive_weights = get_drive_root() / "weights"
                drive_weights.mkdir(parents=True, exist_ok=True)
                dest = drive_weights / best_pt.name
                if best_pt.resolve() != dest.resolve():
                    shutil.copy2(best_pt, dest)

    total_min = (time.time() - total_start) / 60.0
    print(f"\n[✓] Custom OBB Training finished in {total_min:.2f} minutes. Best Val Loss: {best_loss:.4f}")

    return {
        "model_name": "custom-obb",
        "dataset_name": dataset_name,
        "best_weights": str(best_pt),
        "last_weights": str(last_pt),
        "best_loss": best_loss,
        "total_minutes": total_min,
    }
