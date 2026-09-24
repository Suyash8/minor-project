"""
Unified PyTorch Dataset and Batch Collation for Aerial OBB Detection Training.
Provides standardized image resizing, tensor normalization, and coordinate scaling
compatible with both Dense grid-based and DETR query-based models.
"""

from __future__ import annotations

import math
from typing import Dict, Any, List, Tuple
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.dataset import AerialOBBDataset


class UnifiedOBBDataset(Dataset):
    """
    Standard PyTorch Dataset for Oriented Object Detection training.
    Works seamlessly for Dense anchor-free detectors and DETR query architectures.
    """

    def __init__(self, raw_dataset: AerialOBBDataset, img_size: int = 640):
        self.raw = raw_dataset
        self.img_size = img_size
        self.num_classes = len(self.raw.class_names)

    def __len__(self) -> int:
        return len(self.raw)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img = self.raw.get_image(idx)
        orig_w, orig_h = img.size

        # Resize image to target model input size
        if img.mode != "RGB":
            img = img.convert("RGB")
        resized = img.resize((self.img_size, self.img_size), Image.Resampling.BILINEAR)
        img_arr = np.array(resized, dtype=np.float32) / 255.0
        # (H, W, C) -> (C, H, W)
        img_tensor = torch.from_numpy(img_arr).permute(2, 0, 1)

        gt_info = self.raw.get_ground_truth(idx, img_width=orig_w, img_height=orig_h)

        boxes_list = []
        labels_list = []
        scale_x = self.img_size / float(orig_w) if orig_w > 0 else 1.0
        scale_y = self.img_size / float(orig_h) if orig_h > 0 else 1.0

        for c_idx in range(self.num_classes):
            boxes = gt_info[c_idx].get("boxes", np.zeros((0, 5)))
            for box in boxes:
                cx = float(box[0]) * scale_x
                cy = float(box[1]) * scale_y
                w = max(float(box[2]) * scale_x, 4.0)
                h = max(float(box[3]) * scale_y, 4.0)
                angle_deg = float(box[4]) % 180.0
                boxes_list.append([cx, cy, w, h, angle_deg])
                labels_list.append(c_idx)

        if len(boxes_list) > 0:
            boxes_tensor = torch.tensor(boxes_list, dtype=torch.float32)
            labels_tensor = torch.tensor(labels_list, dtype=torch.int64)
        else:
            boxes_tensor = torch.zeros((0, 5), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)

        return {
            "image": img_tensor,
            "boxes": boxes_tensor,        # (M, 5) [cx, cy, w, h, angle_deg] in pixel coords [0, img_size]
            "labels": labels_tensor,      # (M,) class indices
            "orig_size": (orig_w, orig_h),
        }


def obb_collate_fn(batch: List[Dict[str, Any]]) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
    """
    Collate function batching images into a 4D tensor and packaging annotations
    into a per-image target dictionary list.
    """
    images = torch.stack([item["image"] for item in batch], dim=0)
    targets = [
        {
            "boxes": item["boxes"],
            "labels": item["labels"],
            "orig_size": item["orig_size"],
        }
        for item in batch
    ]
    return images, targets
