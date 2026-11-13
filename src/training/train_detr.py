"""
Transformer & DETR-Based Oriented Object Detection (OBB) PyTorch Training Engine (v4)
======================================================================================
Algorithmic & Upstream Framework Provenance:
    1. DETR / Deformable-DETR:
       - "End-to-End Object Detection with Transformers" (ECCV 2020) / https://github.com/facebookresearch/detr
       - "Deformable DETR" (ICLR 2021) / https://github.com/fundamentalvision/Deformable-DETR
    2. AO2-DETR (IEEE TCSVT 2023): https://github.com/Ixiaohuihuihui/AO2-DETR
    3. ARS-DETR (IEEE TGRS 2024 / ICCV 2023): https://github.com/httle/ARS-DETR
    4. RiO-DETR (ECCV 2024 Oral): https://github.com/RicePasteM/RiO-DETR
    5. OrientedFormer (IEEE TGRS 2024): https://github.com/wokaikaixinxin/OrientedFormer
    6. RHINO / AI4RS (CVPR 2024): https://github.com/wokaikaixinxin/ai4rs

Supported Architectures:
    ARS-DETR, AO2-DETR, RiO-DETR, RHINO, OrientedFormer.

v4 performance & resilience features (W4/W7, shared via engine_common):
    - bf16/fp16 AMP (train AND val), TF32 + cudnn.benchmark, optional torch.compile
    - gradient accumulation: --batch is the EFFECTIVE batch (DETR needs
      effective batch >= 8-16 even when VRAM forces micro-batch 2-4)
    - OOM catch -> empty_cache -> split/retry, never crashes the run
    - EMA of weights; early stopping on val mAP50 (loss and mAP anti-correlate
      late in DETR training)
    - atomic full-state checkpoints + throttled Drive mirror
    - linear LR warmup (query-collapse guard) then cosine decay
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
from scipy.optimize import linear_sum_assignment

from src.config import resolve_pipeline_paths
from src.data.augment import AugmentConfig
from src.data.dataset import AerialOBBDataset
from src.metrics.detection_map import compute_map_metrics
from src.models import get_model
from src.training.dataset import UnifiedOBBDataset, obb_collate_fn, _worker_init
from src.training.decode import decode_detr, to_metric_format
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
from src.training.train_dense import (
    GracefulStop,
    _normalize_on_gpu,
    scale_batch_for_gpu,
)
from src.utils.system import auto_configure_hardware


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

            tgt_b_norm = torch.zeros((len(tgt_boxes), 4), device=pred_boxes.device)
            tgt_b_norm[:, 0] = tgt_boxes[:, 0] / float(img_size)
            tgt_b_norm[:, 1] = tgt_boxes[:, 1] / float(img_size)
            tgt_b_norm[:, 2] = tgt_boxes[:, 2] / float(img_size)
            tgt_b_norm[:, 3] = tgt_boxes[:, 3] / float(img_size)

            tgt_ang_rad = torch.deg2rad(tgt_boxes[:, 4:5]).to(pred_angles.device)

            cost_class = -prob[b, :, tgt_labels.to(pred_logits.device)]  # (Q, M)
            cost_bbox = torch.cdist(pred_boxes[b], tgt_b_norm, p=1)  # (Q, M)
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
      - Cross Entropy on all queries (matched -> target class, unmatched -> background)
      - Smooth L1 Loss on matched normalized coordinates (cx, cy, w, h)
      - Continuous Circular Trigonometric Loss on matched angles (sin, cos)
    """

    def __init__(self, num_classes: int, loss_class: float = 1.0,
                 loss_bbox: float = 5.0, loss_angle: float = 2.5):
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
        bg_idx = self.num_classes

        indices = self.matcher(pred_logits, pred_boxes, pred_angles, targets, img_size)

        target_classes = torch.full((B, Q), bg_idx, dtype=torch.int64,
                                    device=pred_logits.device)
        for b, (q_idx, t_idx) in enumerate(indices):
            if len(q_idx) > 0:
                tgt_lbls = targets[b]["labels"].to(pred_logits.device)
                target_classes[b, q_idx] = tgt_lbls[t_idx]

        cls_loss = F.cross_entropy(pred_logits.view(-1, num_classes_p1),
                                   target_classes.view(-1))

        box_losses = []
        angle_losses = []
        num_matched = 0

        for b, (q_idx, t_idx) in enumerate(indices):
            if len(q_idx) == 0:
                continue

            num_matched += len(q_idx)
            m_pred_box = pred_boxes[b, q_idx]
            m_pred_ang = pred_angles[b, q_idx]

            raw_boxes = targets[b]["boxes"].to(pred_logits.device)[t_idx]
            tgt_b_norm = torch.zeros_like(m_pred_box)
            tgt_b_norm[:, 0] = raw_boxes[:, 0] / float(img_size)
            tgt_b_norm[:, 1] = raw_boxes[:, 1] / float(img_size)
            tgt_b_norm[:, 2] = raw_boxes[:, 2] / float(img_size)
            tgt_b_norm[:, 3] = raw_boxes[:, 3] / float(img_size)

            tgt_ang_rad = torch.deg2rad(raw_boxes[:, 4:5])

            box_losses.append(F.smooth_l1_loss(m_pred_box, tgt_b_norm, reduction="sum"))

            sin_p, cos_p = torch.sin(2.0 * m_pred_ang), torch.cos(2.0 * m_pred_ang)
            sin_t, cos_t = torch.sin(2.0 * tgt_ang_rad), torch.cos(2.0 * tgt_ang_rad)
            angle_losses.append(
                F.mse_loss(sin_p, sin_t, reduction="sum") +
                F.mse_loss(cos_p, cos_t, reduction="sum"))

        denom = max(num_matched, 1)
        box_loss = (torch.stack(box_losses).sum() / denom) if box_losses else torch.tensor(0.0, device=pred_logits.device)
        angle_loss = (torch.stack(angle_losses).sum() / denom) if angle_losses else torch.tensor(0.0, device=pred_logits.device)

        total_loss = (
            self.loss_class * cls_loss +
            self.loss_bbox * box_loss +
            self.loss_angle * angle_loss
        )

        # Real query-collapse signals (review C): Hungarian matching always
        # matches min(Q, M) queries, so matched==0 can never fire. Track
        # signals that CAN fire: the fraction of queries predicting
        # background, and the mean max foreground probability of the matched
        # queries (collapse drives it toward 0).
        with torch.no_grad():
            probs = pred_logits.softmax(-1)
            bg_fraction = float((probs[..., bg_idx] > 0.5).float().mean())
            fg = [probs[b, q_idx, :bg_idx].max(-1).values.mean()
                  for b, (q_idx, t_idx) in enumerate(indices) if len(q_idx) > 0]
            matched_fg_prob = float(torch.stack(fg).mean()) if fg else 0.0

        return {
            "total_loss": total_loss,
            "cls_loss": cls_loss.detach(),
            "box_loss": box_loss.detach(),
            "angle_loss": angle_loss.detach(),
            "matched_queries": num_matched,
            "bg_fraction": bg_fraction,
            "matched_fg_prob": matched_fg_prob,
        }


@torch.no_grad()
def validate_detr(
    net: nn.Module,
    model_name: str,
    val_loader: DataLoader,
    criterion: DETROBBLoss,
    dev: torch.device,
    amp: AmpContext,
    img_size: int,
    num_classes: int,
    class_names: List[str],
    max_batches: Optional[int] = None,
    compute_map: bool = True,
    conf_thresh: float = 0.05,
) -> Tuple[float, float, float]:
    """Validation pass: loss + mAP50 / mAP50-95 via the shared DETR decode."""
    net.eval()
    val_loss, n_batches = 0.0, 0
    all_gt: List[List[Dict[str, Any]]] = []
    all_pred: List[List[Dict[str, Any]]] = []

    for v_idx, (imgs, targets) in enumerate(val_loader):
        if max_batches is not None and v_idx >= max_batches:
            break
        imgs = _normalize_on_gpu(imgs, dev)
        with amp.autocast():
            outputs = net(imgs)
            pred_logits, pred_boxes, pred_angles = extract_detr_predictions(
                outputs, model_name=model_name)
            v_losses = criterion(pred_logits, pred_boxes, pred_angles, targets,
                                 img_size=img_size)
        val_loss += float(v_losses["total_loss"])
        n_batches += 1

        if compute_map:
            decoded = decode_detr(pred_logits.float(), pred_boxes.float(),
                                  pred_angles.float(), img_size=img_size,
                                  num_classes=num_classes, conf_thresh=conf_thresh)
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
    warmup_epochs: int = 0,
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
    """Unified training engine for DETR-based Oriented Object Detectors.

    `batch_size` is the EFFECTIVE batch (W4.5: DETR keeps effective batch
    >= 8 via gradient accumulation over micro-batches when VRAM requires).
    """
    seed_all(seed)
    setup_perf_flags(deterministic=deterministic)
    STOP.install()

    # 1. Device & workers
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
            if dev.type == "cuda" else min(batch_size, 2)
    # Query-collapse guard: never drop below micro-batch 2 on GPU, 1 on CPU.
    micro_batch = max(1 if dev.type == "cpu" else 2, micro_batch)
    accum_steps = max(1, math.ceil(batch_size / micro_batch))

    print(f"\n{'=' * 70}")
    print(f" Starting DETR OBB Training: '{model_name}' on {dataset_name.upper()}")
    print(f"{'=' * 70}")
    print(f"[*] Compute Device   : {dev} ({hw_config.get('gpu_name', 'CPU')})")
    print(f"[*] Training Config  : {epochs} epochs | effective batch {batch_size} "
          f"= {micro_batch} x {accum_steps} accum | imgsz {img_size} | lr {lr} "
          f"| warmup {warmup_epochs} | workers {workers}")

    # 2. Datasets and Loaders (shared fast data layer; DETR gets the lighter
    # augmentation schedule, W3.4)
    max_train_samples = kwargs.get("max_train_samples", None)
    max_val_samples = kwargs.get("max_val_samples", None)
    # W3.7 (review E): real cache strategy (see train_dense).
    cache_mode = kwargs.get("cache_mode", "annotations")
    use_ann_cache = cache_mode != "none"
    ram_cache = cache_mode == "ram"
    disk_cache_dir = None
    if cache_mode == "disk":
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
        # reads with a cross-process miss counter.
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
                                 aug_cfg=AugmentConfig.detr_light(),
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
    val_loader = DataLoader(
        val_ds, batch_size=micro_batch, shuffle=False,
        collate_fn=obb_collate_fn, **loader_kwargs)

    # 3. Model, Loss, Optimizer (warmup + cosine)
    detector = get_model(model_name, device=str(dev), num_classes=num_classes)
    detector.load()
    net = detector.net.to(dev)
    net = maybe_channels_last(net, dev, enabled=(dev.type == "cuda"))
    net = maybe_compile(
        net, compile_model,
        probe=(lambda: net(torch.zeros(
            1, 3, img_size, img_size, device=dev)) if compile_model else None))

    criterion = DETROBBLoss(num_classes=num_classes).to(dev)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    warmup_epochs = max(0, min(int(warmup_epochs), epochs - 1))
    base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs - warmup_epochs), eta_min=1e-6)
    if warmup_epochs > 0:
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=0.01, end_factor=1.0,
                    total_iters=warmup_epochs),
                base_scheduler,
            ],
            milestones=[warmup_epochs],
        )
    else:
        scheduler = base_scheduler

    amp = AmpContext(dev)
    ema_tracker = ModelEMA(net) if ema else None

    # 4. Checkpoint layout (§4.4 when run_dir given, else legacy layout)
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

    # Per-run mirror subdir + pull-from-mirror restore (review A3, W7.1).
    run_subdir = f"{dataset_name}_{model_name}"
    if run_dir is not None:
        restore_from_mirror(mirror_dir, run_subdir, run_path)
    mirror = DriveMirror(mirror_dir, every_n_epochs=mirror_every,
                         run_subdir=run_subdir)

    config_snapshot = resolved_config_snapshot(
        model_name=model_name, dataset_name=dataset_name, epochs=epochs,
        effective_batch=batch_size, micro_batch=micro_batch,
        accum_steps=accum_steps, img_size=img_size, lr=lr,
        weight_decay=weight_decay, warmup_epochs=warmup_epochs,
        patience=patience, seed=seed, resize_mode=resize_mode,
        augment=augment, ema=ema, deterministic=deterministic,
        compile=compile_model,
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
                  f"(best val mAP50: {best_map50:.4f})")
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
                    grad_weight: float = 1.0) -> Tuple[float, Dict[str, torch.Tensor]]:
        step_imgs = _normalize_on_gpu(step_imgs, dev)
        with amp.autocast():
            outputs = net(step_imgs)
            pred_logits, pred_boxes, pred_angles = extract_detr_predictions(
                outputs, model_name=model_name)
            losses = criterion(pred_logits, pred_boxes, pred_angles, step_targets,
                               img_size=img_size)
            # grad_weight (review A4): an OOM-split half of size n_half must
            # contribute n_half/n of the full-batch mean gradient.
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
        img_count = 0  # per-image counter for loss means (chunking-safe)
        matched_this_epoch = 0
        bg_frac_sum = 0.0
        fg_prob_sum = 0.0
        data_wait_s = 0.0  # W3 acceptance stat
        optimizer.zero_grad(set_to_none=True)
        micro_step = 0

        # W2.2 (review A1): queue the epoch's exact consumption order.
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
            # OOM halve, or staging just completed (review R1): a reused
            # perm-list sampler freezes batch order for every remaining
            # epoch - rebuild with shuffle=True.
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
            # Inline chunking (see train_dense): after a persistent OOM
            # halving, oversized loader batches are split here so every
            # micro-step is exactly cur_micro images.
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
                    # Retry OUTSIDE the except handler; discard the accumulation
                    # group; halve the micro-batch persistently (review A4).
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
                            # grad_weight: scale the n_half/n share back up by
                            # accum_steps (see train_dense) so the retried batch
                            # is numerically identical to having trained at the
                            # halved micro-batch from the start.
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
                        continue
                    loss_val = sum(l * w for l, w in sub) / sum(w for _, w in sub)
                    if halves_ok == len([h for h in halves_i if h.shape[0]]):
                        # Complete effective batch: close the accumulation group.
                        micro_step = accum_steps - 1

                if loss_val is None or (isinstance(loss_val, float)
                                        and math.isnan(loss_val)):
                    t_prev = time.time()
                    continue

                micro_step += 1
                if (micro_step % accum_steps == 0
                        or (b_idx == len(train_loader) - 1
                            and c_idx == len(chunks) - 1)):
                    if amp.scaler is not None:
                        amp.scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
                        amp.scaler.step(optimizer)
                        amp.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    if ema_tracker is not None:
                        ema_tracker.update(net)

                n_img = c_imgs.shape[0]
                run_loss += loss_val * n_img
                if losses is not None:
                    run_cls += float(losses["cls_loss"]) * n_img
                    run_box += float(losses["box_loss"]) * n_img
                    run_ang += float(losses["angle_loss"]) * n_img
                    matched_this_epoch += int(losses.get("matched_queries", 0))
                    bg_frac_sum += float(losses.get("bg_fraction", 0.0))
                    fg_prob_sum += float(losses.get("matched_fg_prob", 0.0))
                batch_count += 1        # chunks, for ratio-stat means
                img_count += n_img      # images, for loss means
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
            # Save the last COMPLETED epoch (review D): the interrupted
            # epoch is partial and must be re-run on resume.
            save_training_checkpoint(
                last_weights_path,
                epoch=epoch - 1, model_name=model_name,
                dataset_name=dataset_name, model_state=net.state_dict(),
                optimizer_state=optimizer.state_dict(),
                scheduler_state=scheduler.state_dict(),
                scaler_state=amp.state_dict(),
                ema=(ema_tracker.state_dict() if ema_tracker is not None else None),
                rng=rng_state_dict(),
                best_map50=best_map50, best_val_loss=best_val_loss,
                epochs_since_best=epochs_since_best,
                history=history, num_classes=num_classes,
                config=config_snapshot,
                staging_status=(staging.status().__dict__ if staging is not None else None))
            mirror.submit(last_weights_path)
            mirror.flush()
            raise GracefulStop()

        scheduler.step()

        train_loss = run_loss / max(img_count, 1)
        train_cls = run_cls / max(img_count, 1)
        train_box = run_box / max(img_count, 1)
        train_ang = run_ang / max(img_count, 1)
        epoch_imgs = img_count
        epoch_time = time.time() - epoch_start

        # Query-collapse early warning (historical DETR failure mode, W5.1).
        # Review C: matched==0 can never fire (Hungarian always matches
        # min(Q, M)); watch signals that CAN fire instead.
        if batch_count > 0:
            mean_fg_prob = fg_prob_sum / batch_count
            mean_bg_frac = bg_frac_sum / batch_count
            if epoch > warmup_epochs and (mean_fg_prob < 0.3
                                          or mean_bg_frac > 0.98):
                print(f"[!] WARNING: possible query collapse in epoch {epoch}: "
                      f"matched-query mean foreground prob {mean_fg_prob:.3f}, "
                      f"background fraction {mean_bg_frac:.3f}; check lr/warmup "
                      f"for '{model_name}'", flush=True)

        # --- validation on EMA weights (W4.7) ------------------------------
        backup_state = None
        if ema_tracker is not None:
            backup_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            net.load_state_dict(ema_tracker.ema_state, strict=False)
        try:
            val_loss, val_map50, val_map5095 = validate_detr(
                net, model_name, val_loader, criterion, dev, amp,
                img_size=img_size, num_classes=num_classes,
                class_names=list(train_raw.class_names),
                max_batches=(max_batches if max_batches is not None
                             else val_map_max_batches),
                # Review C: mAP on a cadence + final epoch, not every epoch.
                compute_map=(val_map and (epoch % val_map_every == 0
                                          or epoch == epochs)
                             and not (test_mode and epoch < epochs)),
            )
        finally:
            if backup_state is not None:
                net.load_state_dict(backup_state, strict=False)

        # Early stop on val mAP50 ONLY (W4.10, review R2): off-cadence
        # epochs (val_map_every > 1) did not measure mAP, so best.pt,
        # best_map50 and patience must not move on loss there.
        map_evaluated = (val_map
                         and (epoch % val_map_every == 0 or epoch == epochs)
                         and not (test_mode and epoch < epochs))
        if not val_map:
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
            print(f"[early-stop][warn] epoch {epoch}: val mAP50 is NaN on "
                  f"an mAP-evaluated epoch; treated as no improvement "
                  f"(loss is NOT substituted, W4.10)", flush=True)
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

        save_training_checkpoint(
            last_weights_path,
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
            staging_status=(staging.status().__dict__ if staging is not None else None))
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
    print(f"\n[✓] DETR Training Completed in {total_time:.1f}s "
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
