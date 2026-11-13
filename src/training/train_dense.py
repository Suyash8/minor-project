"""
Dense Grid-Based Oriented Object Detection (OBB) PyTorch Training Engine (v4)
==============================================================================
Algorithmic & Upstream Framework Provenance:
    1. MMRotate Benchmark Suite: https://github.com/open-mmlab/mmrotate
       - Rotated FCOS / Rotated ATSS anchor-free training regimes
    2. LSKNet (ICCV 2023 / IJCV 2024): https://github.com/zcablii/LSKNet
    3. STD (AAAI 2024): https://github.com/yuhongtian17/Spatial-Transform-Decoupling
    4. RVSA (CVPR 2023 / TPAMI 2024): https://github.com/ViTAE-Transformer/Remote-Sensing-RVSA
    5. Swin-OBB (ICCV 2021 / MMRotate): https://github.com/microsoft/Swin-Transformer

Supported Architectures:
    LSKNet, Swin-OBB, STD, RVSA, Custom-OBB, Native YOLO11-OBB.

v4 performance & resilience features (W4/W7, shared via engine_common):
    - bf16 AMP on A100/L4-class GPUs, fp16 + GradScaler on T4 (train AND val)
    - TF32 + cudnn.benchmark; channels_last; optional torch.compile
    - --batch means EFFECTIVE batch; gradient accumulation over micro-batches
    - VRAM-aware micro-batch scaling + OOM catch -> empty_cache -> split/retry
    - EMA of weights (validation runs on EMA weights)
    - Atomic checkpoints (model+optimizer+scaler+scheduler+EMA+RNG+config)
      every epoch, throttled background mirror to Drive
    - Early stopping on val mAP50 (not val loss)
    - OBB-aware augmentation + letterbox via the shared data layer
"""

from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.config import resolve_pipeline_paths
from src.data.augment import AugmentConfig
from src.data.dataset import AerialOBBDataset
from src.metrics.detection_map import compute_map_metrics
from src.models import get_model
from src.training.dataset import UnifiedOBBDataset, obb_collate_fn, _worker_init
from src.training.decode import (W_PRIOR, H_PRIOR, MIN_PRIOR_FRAC,
                                 decode_dense_grid, to_metric_format)
from src.training.engine_common import (
    AmpContext,
    DriveMirror,
    EpochLogger,
    ModelEMA,
    STOP,
    gpu_mem_gb,
    load_rng_state,
    maybe_channels_last,
    maybe_compile,
    resolved_config_snapshot,
    restore_from_mirror,
    rng_state_dict,
    save_training_checkpoint,
    seed_all,
    setup_perf_flags,
)
from src.utils.system import auto_configure_hardware


class GracefulStop(Exception):
    """Raised when SIGINT/SIGTERM asks the engine to checkpoint and exit."""


class OBBMaskLoss(nn.Module):
    """
    Composite Multi-Task Loss for Dense Oriented Object Detection:
    - Focal Loss for multi-class classification
    - Smooth L1 Loss for center offset & dimension regression
    - Circular continuous trigonometric Loss for orientation angles
    - Pos-weighted Binary Cross Entropy for objectness presence
      (dense grids are ~97% negative cells; an unweighted BCE collapses
      recall to ~0 — commit a447499)
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

        center_loss = (F.smooth_l1_loss(reg_preds[:, 0:2], reg_targets[:, 0:2], reduction="none") * obj_mask).sum() / num_objs
        dim_loss = (F.smooth_l1_loss(reg_preds[:, 2:4], reg_targets[:, 2:4], reduction="none") * obj_mask).sum() / num_objs

        pred_sin, pred_cos = reg_preds[:, 4:5], reg_preds[:, 5:6]
        tgt_sin, tgt_cos = reg_targets[:, 4:5], reg_targets[:, 5:6]
        angle_loss = (
            F.mse_loss(pred_sin, tgt_sin, reduction="none") * obj_mask +
            F.mse_loss(pred_cos, tgt_cos, reduction="none") * obj_mask
        ).sum() / num_objs

        # Objectness presence loss (pos-weighted; commit a447499)
        neg_mask = 1.0 - obj_mask
        num_pos = obj_mask.sum()
        num_neg = neg_mask.sum()
        pos_w = (num_neg / num_pos.clamp(min=1.0)).clamp(max=100.0)
        obj_bce = F.binary_cross_entropy_with_logits(reg_preds[:, 6:7], obj_mask, reduction="none")
        obj_loss = (obj_bce * (obj_mask * pos_w + neg_mask)).sum() / (num_pos * pos_w + num_neg).clamp(min=1.0)

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

    # Vectorized (review C): the old per-box Python loop did a .to(device)
    # plus GPU .item() sync PER BOX - with 100+ boxes per VisDrone image
    # that dominated the data path. Everything is computed on the targets'
    # own device and moved once per image.
    for b, tgt in enumerate(targets):
        boxes = tgt["boxes"]
        labels = tgt["labels"]
        if len(boxes) == 0:
            continue
        if not torch.is_tensor(boxes):
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
        if not torch.is_tensor(labels):
            labels = torch.as_tensor(labels)
        boxes = boxes.float()
        labels = labels.long()

        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        ang_deg = boxes[:, 4] % 180.0
        gx = (cx / stride_x).long()
        gy = (cy / stride_y).long()
        valid = ((gx >= 0) & (gx < feat_w) & (gy >= 0) & (gy < feat_h)
                 & (labels >= 0) & (labels < num_classes))
        if not bool(valid.any()):
            continue
        gx, gy, lbl = gx[valid], gy[valid], labels[valid]
        cx, cy, w, h, ang_deg = (cx[valid], cy[valid], w[valid], h[valid],
                                 ang_deg[valid])

        dx = (cx / stride_x) - (gx.float() + 0.5)
        dy = (cy / stride_y) - (gy.float() + 0.5)
        log_w = torch.log(torch.clamp(w / (stride_x * W_PRIOR),
                                           min=MIN_PRIOR_FRAC))
        log_h = torch.log(torch.clamp(h / (stride_y * H_PRIOR),
                                           min=MIN_PRIOR_FRAC))
        ang_rad = torch.deg2rad(ang_deg)
        sin_a = torch.sin(2.0 * ang_rad)
        cos_a = torch.cos(2.0 * ang_rad)
        ones = torch.ones_like(dx)
        vals = torch.stack([dx, dy, log_w, log_h, sin_a, cos_a, ones], dim=0)

        gy_d, gx_d, lbl_d, vals_d = (gy.to(device), gx.to(device),
                                     lbl.to(device), vals.to(device))
        cls_targets[b, lbl_d, gy_d, gx_d] = 1.0
        reg_targets[b, :, gy_d, gx_d] = vals_d

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


# W4.6 spec lookup: GPU-class micro-batch reference points at 1024 px,
# keyed by (name-substring, min-VRAM GB) checked in order; memory scales
# with batch * img_size^2 between classes.
_GPU_CLASS_TABLE = [
    ("a100", 40.0, 32),
    ("h100", 80.0, 48),
    ("l4",   24.0, 16),
    ("t4",   16.0, 8),
]


def scale_batch_for_gpu(base_batch: int, img_size: int, vram_gb: float,
                        gpu_name: str = "") -> int:
    """VRAM-aware micro-batch scaling (W4.6) via the T4/L4/A100 class lookup
    (torch.cuda.get_device_properties supplies name + memory); unknown GPUs
    fall back to the img_size^2-scaled linear estimate against 16 GB."""
    if vram_gb <= 0:
        return max(1, min(base_batch, 2))
    name = gpu_name.lower()
    for key, min_vram, ref_batch in _GPU_CLASS_TABLE:
        if key in name and vram_gb >= min_vram * 0.9:
            scaled = ref_batch * (vram_gb / min_vram) * \
                (1024.0 / max(img_size, 1)) ** 2
            return int(max(1, min(64, round(scaled))))
    scaled = base_batch * (vram_gb / 16.0) * (1024.0 / max(img_size, 1)) ** 2
    return int(max(1, min(64, round(scaled))))


def _normalize_on_gpu(imgs: torch.Tensor, device: torch.device) -> torch.Tensor:
    imgs = imgs.to(device, non_blocking=True)
    if imgs.dtype == torch.uint8:
        imgs = imgs.float().div_(255.0)
    return imgs


@torch.no_grad()
def validate_dense(
    net: nn.Module,
    model_name: str,
    val_loader: DataLoader,
    criterion: OBBMaskLoss,
    dev: torch.device,
    amp: AmpContext,
    img_size: int,
    num_classes: int,
    class_names: List[str],
    max_batches: Optional[int] = None,
    compute_map: bool = True,
    conf_thresh: float = 0.05,
) -> Tuple[float, float, float]:
    """Validation pass: loss (always) + mAP50 / mAP50-95 via the shared
    decode + metric protocol (when compute_map). Runs under AMP."""
    net.eval()
    val_loss, n_batches = 0.0, 0
    all_gt: List[List[Dict[str, Any]]] = []
    all_pred: List[List[Dict[str, Any]]] = []

    total_val_batches = len(val_loader) if max_batches is None else min(len(val_loader), max_batches)
    for v_idx, (imgs, targets) in enumerate(val_loader):
        if max_batches is not None and v_idx >= max_batches:
            break
        if compute_map:
            print(f"\r  [val] batch {v_idx + 1}/{total_val_batches}...", end="", flush=True)
        imgs = _normalize_on_gpu(imgs, dev)
        with amp.autocast():
            cls_logits, reg_preds, feat_h, feat_w = forward_dense_network(
                model_name=model_name, net=net, batch_tensor=imgs,
                num_classes=num_classes)
            cls_tgt, reg_tgt = encode_dense_targets(
                targets=targets, batch_size=imgs.shape[0], num_classes=num_classes,
                feat_h=feat_h, feat_w=feat_w, img_size=img_size, device=dev)
            v_losses = criterion(cls_logits, reg_preds, cls_tgt, reg_tgt)
        val_loss += float(v_losses["total_loss"])
        n_batches += 1

        if compute_map:
            decoded = decode_dense_grid(cls_logits.float(), reg_preds.float(),
                                        img_size=img_size, conf_thresh=conf_thresh)
            all_pred.extend(to_metric_format(decoded, num_classes))
            for tgt in targets:
                gt_img = [{"boxes": np.zeros((0, 5), dtype=np.float32)}
                          for _ in range(num_classes)]
                for box, lbl in zip(tgt["boxes"].numpy(), tgt["labels"].numpy()):
                    if 0 <= int(lbl) < num_classes:
                        gt_img[int(lbl)]["boxes"] = np.append(
                            gt_img[int(lbl)]["boxes"],
                            np.array([box], dtype=np.float32), axis=0)
                all_gt.append(gt_img)

    val_loss = val_loss / max(n_batches, 1)
    if compute_map and all_gt:
        metrics = compute_map_metrics(all_gt, all_pred, class_names)
        return val_loss, float(metrics["map50"]), float(metrics["map50_95"])
    return val_loss, float("nan"), float("nan")


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
    patience: Optional[int] = None,
    # ---- v4 additions -----------------------------------------------------
    micro_batch: Optional[int] = None,
    ema: bool = True,
    compile_model: bool = False,
    deterministic: bool = False,
    seed: int = 42,
    resize_mode: str = "letterbox",
    augment: bool = True,
    mirror_dir: Optional[str] = None,
    mirror_every: int = 5,
    staging=None,
    run_dir: Optional[str] = None,
    val_map: bool = True,
    val_map_max_batches: Optional[int] = None,
    val_map_every: int = 3,
    test_mode: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Unified training engine for dense grid-based OBB architectures.

    `batch_size` is the EFFECTIVE batch (gradient accumulation over
    micro-batches when VRAM requires); `micro_batch` overrides the per-step
    batch explicitly.
    """
    seed_all(seed)
    setup_perf_flags(deterministic=deterministic)
    STOP.install()

    # 1. Device & workers (batch is governed by MODEL_TUNING + VRAM scaling,
    # not by the legacy auto-config, per the v4 tuning contract)
    hw_config = auto_configure_hardware(
        device=device,
        requested_train_batch_size=None,
        requested_workers=workers,
    )
    if device == "auto":
        dev = torch.device("cuda" if hw_config["device"] == "cuda" else "cpu")
    elif device.startswith("cuda"):
        dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device("cpu")
    workers = hw_config["workers"] if dev.type == "cuda" else 0

    vram_gb = hw_config.get("total_vram_gb", 0.0)
    if micro_batch is None:
        micro_batch = min(batch_size, scale_batch_for_gpu(batch_size, img_size, vram_gb, hw_config.get("gpu_name", ""))) \
            if dev.type == "cuda" else min(batch_size, 4)
    micro_batch = max(1, micro_batch)
    accum_steps = max(1, math.ceil(batch_size / micro_batch))

    print(f"\n{'=' * 70}")
    print(f" Starting Dense OBB Training: '{model_name}' on {dataset_name.upper()}")
    print(f"{'=' * 70}")
    print(f"[*] Compute Device   : {dev} ({hw_config.get('gpu_name', 'CPU')})")
    print(f"[*] Training Config  : {epochs} epochs | effective batch {batch_size} "
          f"= {micro_batch} x {accum_steps} accum | imgsz {img_size} | lr {lr} | workers {workers}")
    if dev.type == "cuda":
        print(f"[*] AMP dtype        : {'bf16' if (hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_bf16_supported()) else 'fp16'}")

    # 2. Datasets and Loaders (shared fast data layer, W3)
    max_train_samples = kwargs.get("max_train_samples", None)
    max_val_samples = kwargs.get("max_val_samples", None)
    # W3.7 (review E): real cache strategy. `none` disables the annotation
    # cache; `annotations` (default) keeps it; `ram` adds the bounded
    # decode-to-RAM image cache; `disk` = annotations + NVMe staging (the
    # stage step handles the disk tier).
    cache_mode = kwargs.get("cache_mode", "annotations")
    use_ann_cache = cache_mode != "none"
    ram_cache = cache_mode == "ram"
    disk_cache_dir = None
    if cache_mode == "disk":
        # Real disk tier (review): decoded-image .npy cache on fast local
        # disk under the staging root.
        from training.staging import stage_root as _sr
        disk_cache_dir = os.path.join(_sr(None), "imgcache")

    train_raw = AerialOBBDataset(dataset_name=dataset_name, data_dir=data_dir,
                                 split="train", max_samples=max_train_samples,
                                 use_cache=use_ann_cache, ram_cache=ram_cache,
                                 disk_cache_dir=disk_cache_dir)
    val_raw = AerialOBBDataset(dataset_name=dataset_name, data_dir=data_dir,
                               split="val", max_samples=max_val_samples,
                               use_cache=use_ann_cache, ram_cache=ram_cache,
                               disk_cache_dir=disk_cache_dir)
    if staging is not None:
        # W2 fix (review A1): source-rooted inventory + staged-or-source
        # reads with a cross-process miss counter. The dataset is built from
        # the SOURCE root (run.py passes staging.src_root as data_dir), so
        # the listing is complete from second 0; each read lands on the
        # staged copy once that copy is size-complete.
        from training.staging import SourceResolver
        resolver = SourceResolver(staging.src_root, staging.dst_root,
                                  staging.source_reads_value)
        train_raw.set_resolver(resolver)
        val_raw.set_resolver(resolver)

    labeled_train = sum(1 for s in train_raw.samples
                        if s["label_path"] is not None and s["label_path"].exists())
    print(f"[*] Verified training set for '{dataset_name}': {len(train_raw)} images "
          f"({labeled_train} with valid label files).")
    if labeled_train == 0:
        raise RuntimeError(
            f"Cannot train '{model_name}' on '{dataset_name}': 0 ground truth "
            f"annotations were found. Please verify that annotations exist.")

    train_ds = UnifiedOBBDataset(train_raw, img_size=img_size, augment=augment,
                                 aug_cfg=AugmentConfig.dense_default(),
                                 resize_mode=resize_mode,
                                 normalize_on_gpu=True, seed=seed)
    val_ds = UnifiedOBBDataset(val_raw, img_size=img_size, augment=False,
                               resize_mode=resize_mode,
                               normalize_on_gpu=True, seed=seed)

    num_classes = len(train_raw.class_names)
    loader_kwargs: Dict[str, Any] = {}
    if workers > 0:
        loader_kwargs.update(num_workers=workers, pin_memory=True,
                             persistent_workers=True, prefetch_factor=4,
                             worker_init_fn=_worker_init)
    else:
        loader_kwargs.update(num_workers=0, pin_memory=(dev.type == "cuda"))

    train_loader = DataLoader(
        train_ds, batch_size=micro_batch, shuffle=True,
        collate_fn=obb_collate_fn,
        drop_last=len(train_ds) > micro_batch,
        **loader_kwargs)
    # Val loader is constructed ONCE and reused across epochs (W3.5)
    val_loader = DataLoader(
        val_ds, batch_size=micro_batch, shuffle=False,
        collate_fn=obb_collate_fn, **loader_kwargs)

    # 3. Model, Loss, Optimizer, AMP
    detector = get_model(model_name, device=str(dev), num_classes=num_classes)
    detector.load()
    net = detector.net.to(dev)
    net = maybe_channels_last(net, dev, enabled=(dev.type == "cuda"))
    net = maybe_compile(
        net, compile_model,
        probe=(lambda: net(torch.zeros(
            1, 3, img_size, img_size, device=dev)) if compile_model else None))

    criterion = OBBMaskLoss().to(dev)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5)
    amp = AmpContext(dev)
    ema_tracker = ModelEMA(net) if ema else None

    # 4. Checkpoint layout: runs/<name>/{weights/best.pt,weights/last.pt,
    # results.csv} when run_dir is given (hard constraint §4.4), else the
    # legacy weights-dir layout for direct callers.
    if run_dir is not None:
        run_path = Path(run_dir)
        weights_out = run_path / "weights"
        weights_out.mkdir(parents=True, exist_ok=True)
        best_weights_path = weights_out / "best.pt"
        last_weights_path = weights_out / "last.pt"
        logger = EpochLogger(run_path)
    else:
        paths = resolve_pipeline_paths(weights_dir=project_dir, data_dir=data_dir)
        save_dir = paths["weights_dir"] / f"{model_name}_{dataset_name}"
        save_dir.mkdir(parents=True, exist_ok=True)
        run_path = save_dir
        best_weights_path = save_dir / f"{model_name}_best.pt"
        last_weights_path = save_dir / f"{model_name}_last.pt"
        logger = EpochLogger(save_dir)

    # Per-run mirror subdir (review A3): models no longer overwrite each
    # other's last.pt/best.pt/results.csv in one shared mirror dir.
    run_subdir = f"{dataset_name}_{model_name}"
    if run_dir is not None:
        # W7.1: pull surviving checkpoints back from the mirror when the
        # local run dir was wiped (e.g. Colab session death).
        restore_from_mirror(mirror_dir, run_subdir, run_path)
    mirror = DriveMirror(mirror_dir, every_n_epochs=mirror_every,
                         run_subdir=run_subdir)

    config_snapshot = resolved_config_snapshot(
        model_name=model_name, dataset_name=dataset_name, epochs=epochs,
        effective_batch=batch_size, micro_batch=micro_batch,
        accum_steps=accum_steps, img_size=img_size, lr=lr,
        weight_decay=weight_decay, patience=patience, seed=seed,
        resize_mode=resize_mode, augment=augment, ema=ema,
        deterministic=deterministic, compile=compile_model,
    )

    start_epoch = 1
    best_map50 = -1.0
    best_val_loss = float("inf")
    epochs_since_best = 0  # restored from the checkpoint on resume (review D)
    history: List[Dict[str, Any]] = []

    if resume and last_weights_path.exists():
        try:
            ckpt = torch.load(last_weights_path, map_location=dev, weights_only=False)
            missing, unexpected = net.load_state_dict(ckpt["model_state"],
                                                      strict=False)
            if missing or unexpected:
                print(f"[resume][warn] state-dict mismatch: "
                      f"{len(missing)} missing, {len(unexpected)} unexpected keys")
            epochs_since_best = ckpt.get("epochs_since_best", 0)
            if "optimizer_state" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state"])
            if "scheduler_state" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler_state"])
            if ckpt.get("scaler_state") is not None:
                amp.load_state_dict(ckpt["scaler_state"])
            if ema_tracker is not None and ckpt.get("ema") is not None:
                ema_tracker.load_state_dict(ckpt["ema"])
            if "rng" in ckpt:
                load_rng_state(ckpt["rng"])
            ckpt_epoch = ckpt.get("epoch", 0)
            start_epoch = ckpt_epoch + 1
            best_map50 = ckpt.get("best_map50", -1.0)
            best_val_loss = ckpt.get("best_val_loss", float("inf"))
            history = ckpt.get("history", [])
            print(f"[✓] Resumed from epoch {ckpt_epoch} "
                  f"(best val mAP50: {best_map50:.4f}, best val loss: {best_val_loss:.4f})")
            if start_epoch > epochs:
                print(f"[*] Checkpoint already reached epoch {ckpt_epoch} >= target "
                      f"{epochs}. Nothing to do for '{model_name}'.")
        except Exception as e:
            print(f"[!] Could not resume from checkpoint: {e}. Starting fresh.")

    # 5. Training Loop
    total_train_start = time.time()
    oom_events = 0
    cur_micro = micro_batch   # persistent micro-batch; halved on OOM (W4.5)
    loader_micro = micro_batch
    loader_was_perm = False  # R1: fixed-perm sampler must not freeze order

    def _train_step(step_imgs: torch.Tensor, step_targets,
                    grad_weight: float = 1.0) -> float:
        step_imgs = _normalize_on_gpu(step_imgs, dev)
        with amp.autocast():
            cls_logits, reg_preds, feat_h, feat_w = forward_dense_network(
                model_name=model_name, net=net, batch_tensor=step_imgs,
                num_classes=num_classes)
            cls_tgt, reg_tgt = encode_dense_targets(
                targets=step_targets, batch_size=step_imgs.shape[0],
                num_classes=num_classes, feat_h=feat_h, feat_w=feat_w,
                img_size=img_size, device=dev)
            losses = criterion(cls_logits, reg_preds, cls_tgt, reg_tgt)
            # grad_weight (review A4): an OOM-split half of size n_half must
            # contribute n_half/n of the full-batch mean gradient, not the
            # full 1/accum_steps share.
            loss = losses["total_loss"] * grad_weight / accum_steps
        if amp.scaler is not None:
            amp.scaler.scale(loss).backward()
        else:
            loss.backward()
        return float(losses["total_loss"].detach()), losses

    for epoch in range(start_epoch, epochs + 1):
        net.train()
        train_ds.set_epoch(epoch)
        epoch_start = time.time()
        run_loss = run_cls = run_box = run_ang = 0.0
        batch_count = 0
        data_wait_s = 0.0  # W3 acceptance stat: time spent waiting on data
        optimizer.zero_grad(set_to_none=True)
        micro_step = 0

        # W2.2 (review A1): while staging is incomplete, feed the queue this
        # epoch's exact consumption order and follow it with the sampler, so
        # copy priority == read order. Once staging completes, the persistent
        # shuffled loader runs.
        if staging is not None and not staging.status().complete:
            g = torch.Generator().manual_seed((seed or 0) + epoch)
            perm = torch.randperm(len(train_ds), generator=g).tolist()
            rels = train_raw.sample_relpaths()
            if len(rels) == len(train_ds):
                staging.reprioritize([rels[i] for i in perm],
                                     dataset=dataset_name)
            kw = dict(loader_kwargs)
            kw.pop("persistent_workers", None)
            train_loader = DataLoader(
                train_ds, batch_size=cur_micro, sampler=perm,
                collate_fn=obb_collate_fn,
                drop_last=len(train_ds) > cur_micro, **kw)
            loader_micro = cur_micro
            loader_was_perm = True
        elif loader_micro != cur_micro or loader_was_perm:
            # OOM halved the micro-batch mid-run, or staging just completed
            # (review R1): a reused perm-LIST sampler freezes the batch
            # order for every remaining epoch - rebuild with shuffle=True
            # (RandomSampler draws a fresh permutation every epoch).
            train_loader = DataLoader(
                train_ds, batch_size=cur_micro, shuffle=True,
                collate_fn=obb_collate_fn,
                drop_last=len(train_ds) > cur_micro, **loader_kwargs)
            loader_micro = cur_micro
            loader_was_perm = False

        t_prev = time.time()
        for b_idx, (imgs, targets) in enumerate(train_loader):
            data_wait_s += time.time() - t_prev
            if max_batches is not None and b_idx >= max_batches:
                break
            if STOP.requested():
                break
            # Inline chunking (review/main numerical equivalence): after a
            # persistent OOM halving the DataLoader is only rebuilt at the
            # next epoch, so until then loader batches arrive LARGER than
            # cur_micro. Split them here so every optimizer micro-step is
            # exactly cur_micro images and the gradient flow matches a run
            # that started at the halved micro-batch.
            if imgs.shape[0] > cur_micro:
                n_chunks = math.ceil(imgs.shape[0] / cur_micro)
                chunk_imgs = list(imgs.chunk(n_chunks, dim=0))
                sizes = [c.shape[0] for c in chunk_imgs]
                chunk_tgts, _off = [], 0
                for _s in sizes:
                    chunk_tgts.append(targets[_off:_off + _s])
                    _off += _s
                chunks = list(zip(chunk_imgs, chunk_tgts))
            else:
                chunks = [(imgs, targets)]
            for c_idx, (c_imgs, c_tgts) in enumerate(chunks):
                if c_imgs.shape[0] == 0:
                    continue
                oom = False
                try:
                    loss_val, losses = _train_step(c_imgs, c_tgts)
                except torch.cuda.OutOfMemoryError:
                    oom = True
                if oom:
                    # Retry OUTSIDE the except handler (review A4): the failed
                    # backward graph must be released before retrying, or the
                    # retry OOMs again. The partial accumulation group is
                    # discarded because a failed backward leaves unsafe grads.
                    oom_events += 1
                    torch.cuda.empty_cache()
                    optimizer.zero_grad(set_to_none=True)
                    micro_step = 0
                    new_micro = max(1, cur_micro // 2)
                    if new_micro < cur_micro:
                        cur_micro = new_micro
                        accum_steps = max(1, math.ceil(batch_size / cur_micro))
                        print(f"[oom-recovery] OOM at epoch {epoch} batch {b_idx}; "
                              f"micro-batch halved persistently to {cur_micro} "
                              f"(accum x{accum_steps}); retrying split halves",
                              flush=True)
                    else:
                        print(f"[oom-recovery] OOM at epoch {epoch} batch {b_idx}; "
                              f"retrying split halves", flush=True)
                    n = c_imgs.shape[0]
                    halves_i = c_imgs.chunk(2, dim=0)
                    halves_t = [c_tgts[: len(halves_i[0])], c_tgts[len(halves_i[0]):]]
                    sub: List[Tuple[float, int]] = []
                    losses = None
                    halves_ok = 0
                    for im, tg in zip(halves_i, halves_t):
                        if im.shape[0] == 0:
                            continue
                        try:
                            # grad_weight (review/main numerical equivalence):
                            # _train_step divides by accum_steps, so scale the
                            # n_half/n share BACK UP by accum_steps - each half
                            # then contributes exactly its share of the full-
                            # batch mean gradient, identical to having trained
                            # at the halved micro-batch from the start.
                            lv, losses = _train_step(
                                im, tg,
                                grad_weight=(im.shape[0] / n) * accum_steps)
                            sub.append((lv, im.shape[0]))
                            halves_ok += 1
                        except torch.cuda.OutOfMemoryError:
                            torch.cuda.empty_cache()
                            oom_events += 1
                            print("[oom-recovery] OOM persists on half-batch; "
                                  "half skipped (logged, not silently averaged)",
                                  flush=True)
                    if not sub:
                        t_prev = time.time()
                        continue  # nothing trainable; group already discarded
                    loss_val = sum(l * w for l, w in sub) / sum(w for _, w in sub)
                    if halves_ok == len([h for h in halves_i if h.shape[0]]):
                        # Both halves trained: the retry IS a complete effective
                        # batch - close the accumulation group so the optimizer
                        # steps on it, instead of leaking these grads into the
                        # next batch's group.
                        micro_step = accum_steps - 1

                if loss_val is None or (isinstance(loss_val, float)
                                        and math.isnan(loss_val)):
                    # Never let a skipped/NaN batch poison the epoch average.
                    t_prev = time.time()
                    continue

                micro_step += 1
                if (micro_step % accum_steps == 0
                        or (b_idx == len(train_loader) - 1
                            and c_idx == len(chunks) - 1)):
                    if amp.scaler is not None:
                        amp.scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=10.0)
                        amp.scaler.step(optimizer)
                        amp.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=10.0)
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    if ema_tracker is not None:
                        ema_tracker.update(net)

                # Per-IMAGE weighting (review/main equivalence): an OOM-
                # retried chunk logs one loss for 2+ images while plain
                # chunks log one per micro; weighting by images makes the
                # epoch mean identical across chunking histories.
                n_img = c_imgs.shape[0]
                run_loss += loss_val * n_img
                if losses is not None:
                    run_cls += float(losses["cls_loss"]) * n_img
                    run_box += float(losses["center_loss"]
                                     + losses["dim_loss"]) * n_img
                    run_ang += float(losses["angle_loss"]) * n_img
                batch_count += n_img
                t_prev = time.time()

                total_train_batches = len(train_loader) if max_batches is None else min(len(train_loader), max_batches)
                log_interval = max(1, total_train_batches // 10)
                if b_idx % log_interval == 0 or b_idx == total_train_batches - 1 or b_idx == 0:
                    elapsed_b = time.time() - epoch_start
                    fps = (b_idx + 1) * c_imgs.shape[0] / max(elapsed_b, 1e-4)
                    staged_note = ""
                    if staging is not None:
                        st = staging.status()
                        staged_note = f" | staged {st.done}/{st.total} (drive reads {st.drive_reads})"
                    print(
                        f"  [{model_name}] Epoch [{epoch:02d}/{epochs:02d}] "
                        f"Batch {b_idx + 1}/{total_train_batches} "
                        f"({(b_idx + 1) / total_train_batches * 100:.1f}%) | "
                        f"Loss: {loss_val:.4f} | {fps:.1f} imgs/s{staged_note}",
                        flush=True,
                    )

        if STOP.requested():
            print("[signal] stop requested — saving checkpoint before exit")
            _save_epoch_checkpoint(
                # Save the last COMPLETED epoch (review D): the interrupted
                # epoch is partial and must be re-run on resume.
                last_weights_path, epoch=epoch - 1, model_name=model_name,
                dataset_name=dataset_name, net=net, optimizer=optimizer,
                scheduler=scheduler, amp=amp, ema_tracker=ema_tracker,
                best_map50=best_map50, best_val_loss=best_val_loss,
                history=history, num_classes=num_classes,
                config_snapshot=config_snapshot, staging=staging,
                epochs_since_best=epochs_since_best)
            mirror.submit(last_weights_path)
            mirror.flush()
            raise GracefulStop()

        scheduler.step()

        train_loss = run_loss / max(batch_count, 1)
        train_cls = run_cls / max(batch_count, 1)
        train_box = run_box / max(batch_count, 1)
        train_ang = run_ang / max(batch_count, 1)
        epoch_imgs = batch_count  # batch_count is per-image now
        epoch_time = time.time() - epoch_start

        # --- validation (EMA weights when enabled; W4.7) -------------------
        eval_net = net
        backup_state = None
        if ema_tracker is not None:
            backup_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            net.load_state_dict(ema_tracker.ema_state, strict=False)
        try:
            val_loss, val_map50, val_map5095 = validate_dense(
                eval_net, model_name, val_loader, criterion, dev, amp,
                img_size=img_size, num_classes=num_classes,
                class_names=list(train_raw.class_names),
                max_batches=(max_batches if max_batches is not None
                             else val_map_max_batches),
                # Review C: full-decode + shapely NMS + mAP on the whole val
                # set EVERY epoch costs many minutes; compute on a cadence
                # and always on the final epoch (early stopping falls back
                # to val loss, loudly, on off epochs).
                compute_map=(val_map and (epoch % val_map_every == 0
                                          or epoch == epochs)
                             and not (test_mode and epoch < epochs)),
            )
        finally:
            if backup_state is not None:
                net.load_state_dict(backup_state, strict=False)

        # Early stopping / best selection on val mAP50 ONLY (W4.10, review
        # R2): on off-cadence epochs (val_map_every > 1) mAP was not
        # measured, so best.pt, best_map50 and patience must not move at
        # all - selecting by val loss there can crown a lower-mAP epoch.
        map_evaluated = (val_map
                         and (epoch % val_map_every == 0 or epoch == epochs)
                         and not (test_mode and epoch < epochs))
        if not val_map:
            # mAP selection explicitly disabled: loss is the chosen
            # criterion (loud once), NOT a cadence gap.
            if epoch == start_epoch:
                print("[early-stop][warn] val_map=False: best-selection "
                      "uses val loss by explicit choice (not W4.10 mAP50)",
                      flush=True)
            is_best = val_loss < best_val_loss
        elif not map_evaluated:
            is_best = False
        elif not math.isnan(val_map50):
            is_best = val_map50 > best_map50
        else:
            # mAP WAS evaluated and came back NaN (collapse): loud, counts
            # as no improvement (patience advances) - never loss-selected.
            print(f"[early-stop][warn] epoch {epoch}: val mAP50 is NaN on "
                  f"an mAP-evaluated epoch; best/patience treat it as no "
                  f"improvement (loss is NOT substituted, W4.10)",
                  flush=True)
            is_best = False
        if is_best:
            best_map50 = val_map50 if not math.isnan(val_map50) else best_map50
            best_val_loss = val_loss
            epochs_since_best = 0
        elif map_evaluated or not val_map:
            epochs_since_best += 1

        cur_lr = optimizer.param_groups[0]["lr"]
        epoch_record = {
            "epoch": epoch, "lr": round(cur_lr, 6),
            "train_loss": round(train_loss, 4), "train_cls": round(train_cls, 4),
            "train_box": round(train_box, 4), "train_ang": round(train_ang, 4),
            "val_loss": round(val_loss, 4),
            "metrics/mAP50(B)": round(val_map50, 5) if not math.isnan(val_map50) else "",
            "metrics/mAP50-95(B)": round(val_map5095, 5) if not math.isnan(val_map5095) else "",
            "imgs_per_sec": round(epoch_imgs / max(epoch_time, 1e-4), 1),
            "gpu_mem_gb": gpu_mem_gb(), "time": round(epoch_time, 2),
            "data_wait_s": round(data_wait_s, 2),  # W3 acceptance stat
        }
        logger.log(epoch_record)
        history.append(epoch_record)

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train: {train_loss:.4f} (cls {train_cls:.3f}, box {train_box:.3f}, "
            f"ang {train_ang:.3f}) | Val: {val_loss:.4f} | "
            f"mAP50: {val_map50 if not math.isnan(val_map50) else float('nan'):.4f} "
            f"{'(*BEST*)' if is_best else ''} | LR: {cur_lr:.2e} | "
            f"{epoch_record['imgs_per_sec']} imgs/s | {epoch_time:.1f}s"
        )

        _save_epoch_checkpoint(
            last_weights_path, epoch=epoch, model_name=model_name,
            dataset_name=dataset_name, net=net, optimizer=optimizer,
            scheduler=scheduler, amp=amp, ema_tracker=ema_tracker,
            best_map50=best_map50, best_val_loss=best_val_loss,
            history=history, num_classes=num_classes,
            config_snapshot=config_snapshot, staging=staging,
            epochs_since_best=epochs_since_best)
        if is_best:
            save_training_checkpoint(
                best_weights_path,
                epoch=epoch, model_name=model_name, dataset_name=dataset_name,
                model_state=(ema_tracker.ema_state if ema_tracker is not None
                             else net.state_dict()),
                best_map50=best_map50, best_val_loss=best_val_loss,
                num_classes=num_classes, config=config_snapshot)

        mirror.submit_epoch(epoch, last_weights_path, logger.csv_path)
        if is_best:
            mirror.submit(best_weights_path)

        if patience and epochs_since_best >= patience:
            print(f"[!] Early stopping at epoch {epoch}: no val mAP50 "
                  f"improvement for {patience} epochs")
            break

    mirror.flush()
    total_time = time.time() - total_train_start
    print(f"\n[✓] Dense Training Completed in {total_time:.1f}s "
          f"({total_time / 60:.1f} min) | OOM recoveries: {oom_events}")
    print(f"    - Best weights : {best_weights_path}")
    print(f"    - Best val mAP50: {best_map50:.4f} | best val loss: {best_val_loss:.4f}")

    return {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "best_weights": str(best_weights_path),
        "last_weights": str(last_weights_path),
        "epochs_trained": epochs,
        "best_map50": best_map50,
        "best_val_loss": best_val_loss,
        "oom_events": oom_events,
        "history": history,
        "run_dir": str(run_path),
        "resolved_config": config_snapshot,
    }


def _save_epoch_checkpoint(path: Path, *, epoch: int, model_name: str,
                           dataset_name: str, net: nn.Module,
                           optimizer, scheduler, amp: AmpContext,
                           ema_tracker: Optional[ModelEMA],
                           best_map50: float, best_val_loss: float,
                           history: List[Dict[str, Any]], num_classes: int,
                           config_snapshot: Dict[str, Any], staging,
                           epochs_since_best: int = 0) -> None:
    save_training_checkpoint(
        path,
        epoch=epoch, model_name=model_name, dataset_name=dataset_name,
        model_state=net.state_dict(),
        optimizer_state=optimizer.state_dict(),
        scheduler_state=scheduler.state_dict(),
        scaler_state=amp.state_dict(),
        ema=(ema_tracker.state_dict() if ema_tracker is not None else None),
        rng=rng_state_dict(),
        best_map50=best_map50, best_val_loss=best_val_loss,
        epochs_since_best=epochs_since_best,
        history=history, num_classes=num_classes,
        config=config_snapshot,
        staging_status=(staging.status().__dict__ if staging is not None else None),
    )
