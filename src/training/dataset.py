"""
Unified PyTorch Dataset and Batch Collation for Aerial OBB Detection Training (v4).

One shared reader for the dense, DETR and custom engines (W3.6):
- OpenCV decode via AerialOBBDataset.get_image_array (uint8 until the GPU
  normalizes — set normalize_on_gpu=True to skip CPU-side float conversion),
- aspect-ratio-correct letterbox resize by default (resize_mode="stretch"
  reproduces the exact v3 behaviour for comparison runs),
- optional OBB-aware augmentation (src/data/augment.py) with correct corner
  transforms; dense engines use the full schedule, DETR a lighter one,
- labels come from AerialOBBDataset's fingerprinted annotation cache, so
  label text is never re-parsed per epoch.

Public contract (unchanged): UnifiedOBBDataset wraps an AerialOBBDataset and
obb_collate_fn batches samples into (images, targets) where targets carry
boxes (M, 5) [cx, cy, w, h, angle_deg] in output pixel coords and labels (M,).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.augment import (
    AugmentConfig,
    apply_augmentations,
    boxes_to_corners,
    corners_to_boxes,
    letterbox,
)
from src.data.dataset import AerialOBBDataset


def _worker_init(_worker_id: int) -> None:
    """Keep OpenCV single-threaded inside DataLoader workers (it otherwise
    spawns its own thread pool per worker and thrashes Colab's 2 vCPUs)."""
    try:
        import cv2
        cv2.setNumThreads(0)
    except ImportError:
        pass


class UnifiedOBBDataset(Dataset):
    """
    Standard PyTorch Dataset for Oriented Object Detection training.
    Works for Dense anchor-free detectors and DETR query architectures.
    """

    def __init__(
        self,
        raw_dataset: AerialOBBDataset,
        img_size: int = 640,
        augment: bool = False,
        aug_cfg: Optional[AugmentConfig] = None,
        resize_mode: str = "letterbox",
        normalize_on_gpu: bool = False,
        seed: int = 0,
    ):
        if resize_mode not in ("letterbox", "stretch"):
            raise ValueError(f"resize_mode must be 'letterbox' or 'stretch', got {resize_mode!r}")
        self.raw = raw_dataset
        self.img_size = img_size
        self.num_classes = len(self.raw.class_names)
        self.augment = augment
        self.aug_cfg = aug_cfg or AugmentConfig.dense_default()
        self.resize_mode = resize_mode
        self.normalize_on_gpu = normalize_on_gpu
        self._seed = seed
        self._epoch = 0

    def __len__(self) -> int:
        return len(self.raw)

    def set_epoch(self, epoch: int) -> None:
        """Seed augmentation deterministically per epoch when --deterministic
        is used (called by the engines at each epoch start)."""
        self._epoch = epoch

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img = self.raw.get_image_array(idx)  # (H, W, 3) uint8 RGB
        orig_h, orig_w = img.shape[:2]

        gt_info = self.raw.get_ground_truth(idx, img_width=orig_w, img_height=orig_h)
        boxes_list, labels_list = [], []
        for c_idx in range(self.num_classes):
            boxes = gt_info[c_idx].get("boxes", np.zeros((0, 5)))
            for box in boxes:
                boxes_list.append([float(v) for v in box[:5]])
                labels_list.append(c_idx)
        boxes = np.array(boxes_list, dtype=np.float32).reshape(-1, 5)
        corners = boxes_to_corners(boxes)

        # --- geometric transforms on image + corner polygons --------------
        if self.augment:
            rng = np.random.default_rng(
                (self._seed * 1_000_003 + self._epoch * 10_007 + idx) & 0xFFFFFFFF)
            img, corners = apply_augmentations(img, corners, self.aug_cfg, rng)

        if self.resize_mode == "letterbox":
            img, corners = letterbox(img, corners, self.img_size)
        else:  # stretch (exact v3 behaviour)
            h, w = img.shape[:2]
            try:
                import cv2
                img = cv2.resize(img, (self.img_size, self.img_size),
                                 interpolation=cv2.INTER_LINEAR)
            except ImportError:  # pragma: no cover
                from PIL import Image
                img = np.array(Image.fromarray(img).resize(
                    (self.img_size, self.img_size)))
            if len(corners):
                corners = corners * np.array(
                    [self.img_size / float(w), self.img_size / float(h)],
                    dtype=np.float32)

        labels_in = np.array(labels_list, dtype=np.int64)
        boxes_out, keep = corners_to_boxes(corners, self.img_size, self.img_size,
                                           return_keep=True)
        labels_out = labels_in[keep] if len(keep) else np.zeros((0,), dtype=np.int64)

        if len(boxes_out):
            boxes_tensor = torch.from_numpy(boxes_out.astype(np.float32))
            labels_tensor = torch.from_numpy(labels_out.astype(np.int64))
        else:
            boxes_tensor = torch.zeros((0, 5), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)

        img_chw = np.ascontiguousarray(img.transpose(2, 0, 1))  # (3, H, W)
        if self.normalize_on_gpu:
            img_tensor = torch.from_numpy(img_chw)  # uint8; engines normalize on-GPU
        else:
            img_tensor = torch.from_numpy(img_chw).float().div_(255.0)

        return {
            "image": img_tensor,
            "boxes": boxes_tensor,        # (M, 5) cx, cy, w, h, angle_deg in [0, img_size]
            "labels": labels_tensor,      # (M,) class indices
            "orig_size": (orig_w, orig_h),
            "index": idx,                 # raw dataset index (eval GT lookup, B1)
        }


def obb_collate_fn(batch: List[Dict[str, Any]]) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
    """
    Collate function batching images into a 4D tensor and packaging annotations
    into a per-image target dictionary list. uint8 images stay uint8 (engines
    normalize on-GPU to cut CPU time and PCIe traffic).
    """
    images = torch.stack([item["image"] for item in batch], dim=0)
    targets = [
        {
            "boxes": item["boxes"],
            "labels": item["labels"],
            "orig_size": item["orig_size"],
            "index": item["index"],
        }
        for item in batch
    ]
    return images, targets
