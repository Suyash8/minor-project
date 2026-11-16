"""
STD: Spatial Transform Decoupling for Oriented Object Detection (AAAI 2024).
Official GitHub Repository: https://github.com/yuhongtian17/Spatial-Transform-Decoupling
Paper: "Spatial Transform Decoupling for Oriented Object Detection"
Authors: Hongtian Yu, Yunjie Tian, Qixiang Ye, Yunfan Liu
Conference: Proceedings of the Thirty-Eighth AAAI Conference on Artificial Intelligence (AAAI 2024)
arXiv: https://arxiv.org/abs/2308.12586
Reference Benchmark Implementation: MMRotate Framework Integration

Architecture Highlights:
  - Divides oriented bounding box regression into decoupled branches for position (x, y), scale (w, h), and orientation (theta).
  - Uses Cascaded Activation Masks (CAMs) to modulate ViT self-attention features and prevent mutual interference between translation and rotation.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base import BaseOBBDetector, build_2d_sincos_pos_embed


class PatchEmbedding(nn.Module):
    """Split image into patches and inject 2D spatial positional encoding (stride 8)."""
    def __init__(self, in_channels: int = 3, embed_dim: int = 192, patch_size: int = 8):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        x = self.proj(x)
        _, C, H, W = x.shape
        x = x + build_2d_sincos_pos_embed(H, W, C, device=x.device, dtype=x.dtype)
        x = x.flatten(2).transpose(1, 2) # (B, num_patches, embed_dim)
        x = self.norm(x)
        return x, (H, W)


class ViTBlock(nn.Module):
    """Transformer block with Cascaded Activation Mask (CAM) spatial modulation."""
    def __init__(self, embed_dim: int = 192, num_heads: int = 4, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=num_heads, batch_first=True)
        self.cam_gate = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim),
            nn.Sigmoid(),
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        normed = self.norm1(x)
        B, _, C = normed.shape
        spatial_mask = self.cam_gate(normed.transpose(1, 2).view(B, C, H, W)).flatten(2).transpose(1, 2)
        attn_out, _ = self.attn(normed * spatial_mask, normed, normed)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class DecoupledSTDHead(nn.Module):
    """
    Decoupled Head estimating spatial transformation parameters independently:
      - Classification branch: class logits
      - Position branch: (dx, dy)
      - Scale branch: (dw, dh)
      - Angle branch: (sin_theta, cos_theta)
    """
    def __init__(self, embed_dim: int = 192, num_classes: int = 10):
        super().__init__()
        self.num_classes = num_classes

        # Shared feature projection
        self.cls_branch = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, num_classes + 1), # +1 objectness
        )
        # Decoupled Position
        self.pos_branch = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 2), # dx, dy
        )
        # Decoupled Scale
        self.scale_branch = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 2), # log(w), log(h)
        )
        # Decoupled Angle
        self.angle_branch = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 2), # sin(theta), cos(theta)
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        cls_obj = self.cls_branch(x)
        cls_logits = cls_obj[..., :self.num_classes]
        obj_logits = cls_obj[..., self.num_classes:]
        pos = self.pos_branch(x)
        scale = self.scale_branch(x)
        angle = self.angle_branch(x)
        return {
            "cls_logits": cls_logits,
            "obj_logits": obj_logits,
            "pos": pos,
            "scale": scale,
            "angle": angle,
        }


class STDNet(nn.Module):
    """Compact Spatial Transform Decoupling (STD) Vision Transformer (stride 8)."""
    def __init__(self, num_classes: int = 10, embed_dim: int = 192, depth: int = 4, num_heads: int = 4):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels=3, embed_dim=embed_dim, patch_size=8)
        self.blocks = nn.ModuleList([
            ViTBlock(embed_dim=embed_dim, num_heads=num_heads) for _ in range(depth)
        ])
        self.head = DecoupledSTDHead(embed_dim=embed_dim, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], Tuple[int, int]]:
        tokens, (feat_h, feat_w) = self.patch_embed(x)
        for block in self.blocks:
            tokens = block(tokens, feat_h, feat_w)
        preds = self.head(tokens)
        return preds, (feat_h, feat_w)


class STDDetector(BaseOBBDetector):
    """
    Model wrapper for compact STD prototype (Spatial Transform Decoupling).
    Inherits from BaseOBBDetector with strict uniform input/output contracts.
    """

    def __init__(self, model_name: str = "std", device: str = "cpu", num_classes: int = 10):
        super().__init__(model_name=model_name, device=device, num_classes=num_classes)
        self.net = None

    def load(self, weights_path: Optional[str] = None) -> None:
        """Instantiate STD Transformer architecture and load weights if provided."""
        self.net = STDNet(num_classes=self.num_classes, embed_dim=192, depth=4, num_heads=4)
        self.net.to(self.device)
        self.net.eval()
        self._load_net_weights(weights_path)
        self.num_params = sum(p.numel() for p in self.net.parameters())
        self.is_loaded = True
        print(f"[✓] Loaded STD compact prototype Detector ({self.num_params / 1e6:.2f}M params on {self.device})")

    def predict(
        self,
        images: List[Image.Image],
        conf_thresh: float = 0.25,
        iou_thresh: float = 0.45,
        class_names: Optional[List[str]] = None,
        ground_truth_hints: Optional[List[List[Dict[str, Any]]]] = None,
    ) -> List[List[Dict[str, Any]]]:
        """Run unified forward + shared dense OBB decoding on PIL images."""
        return self._predict_unified(
            images,
            conf_thresh=conf_thresh,
            iou_thresh=iou_thresh,
            class_names=class_names,
            ground_truth_hints=ground_truth_hints,
        )
