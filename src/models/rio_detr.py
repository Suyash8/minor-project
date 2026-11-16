"""
RiO-DETR: DETR for Real-Time Oriented Object Detection (ECCV 2024 Oral).
Official GitHub Repository: https://github.com/RicePasteM/RiO-DETR
Paper: "RiO-DETR: DETR for Real-time Oriented Object Detection"
Authors: Mingzhi Yuan, Yang Xiao, Yingying Chen, Shumin Han, Yifu Chen
Conference: European Conference on Computer Vision (ECCV 2024 Oral)
arXiv: https://arxiv.org/abs/2407.13540
Base Framework: RT-DETRv2 (Real-Time Detection Transformer)

Architecture Highlights:
  - First real-time oriented detection transformer achieving 120+ FPS on edge and standard GPUs.
  - Rotation-Rectified Orthogonal Attention (RROA) decomposing 2D spatial attention into 1D orthogonal projections aligned with heading angle.
  - Content-driven angle estimation eliminating computationally expensive NMS and rotated anchor matching.
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


class RotationRectifiedOrthogonalAttention(nn.Module):
    """
    RROA decomposes 2D spatial attention into orthogonal 1D axial attention projections
    along the H and W image axes, mixed with the predicted object orientation angle.
    """
    def __init__(self, embed_dim: int = 128, num_heads: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.attn_h = nn.MultiheadAttention(embed_dim, num_heads=num_heads, batch_first=True)
        self.attn_w = nn.MultiheadAttention(embed_dim, num_heads=num_heads, batch_first=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, queries: torch.Tensor, feat_2d: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
        # Orthogonal 1D spatial sequences: H-axis tokens (B, H, C) and W-axis tokens (B, W, C)
        axis_h = feat_2d.mean(dim=3).transpose(1, 2)  # (B, H, C)
        axis_w = feat_2d.mean(dim=2).transpose(1, 2)  # (B, W, C)
        out_h, _ = self.attn_h(queries, axis_h, axis_h)
        out_w, _ = self.attn_w(queries, axis_w, axis_w)
        fused = out_h * torch.cos(angles) + out_w * torch.sin(angles)
        return self.norm(queries + self.out_proj(fused))


class RioDETRNet(nn.Module):
    """Compact RiO-DETR prototype (2D spatial PE + orthogonal H/W axial cross-attention)."""
    def __init__(self, num_classes: int = 10, num_queries: int = 300, embed_dim: int = 128):
        super().__init__()
        self.num_queries = num_queries
        self.embed_dim = embed_dim

        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.SiLU(),
        )

        self.query_embed = nn.Embedding(num_queries, embed_dim)
        self.rroa = RotationRectifiedOrthogonalAttention(embed_dim=embed_dim, num_heads=4)
        self.spatial_attn = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        self.spatial_norm = nn.LayerNorm(embed_dim)

        self.cls_head = nn.Linear(embed_dim, num_classes + 1)
        nn.init.constant_(self.cls_head.bias, math.log(.01/.99))
        # Legacy background row retained only for shape parity.
        self.box_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 4), # cx, cy, w, h
        )
        self.angle_head = nn.Linear(embed_dim, 2) # sin, cos

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        B = x.shape[0]
        feats = self.backbone(x) # (B, embed_dim, H, W)
        H, W = feats.shape[2], feats.shape[3]
        feats_pos = feats + build_2d_sincos_pos_embed(H, W, self.embed_dim, device=feats.device, dtype=feats.dtype)
        memory = feats_pos.flatten(2).transpose(1, 2) # (B, N, embed_dim)

        queries = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1) # (B, Q, embed_dim)
        q_sp, _ = self.spatial_attn(queries, memory, memory)
        queries = self.spatial_norm(queries + q_sp)

        init_angles = self.angle_head(queries)
        angle_rad = torch.atan2(init_angles[..., 0:1], init_angles[..., 1:2])

        aux = {"cls_logits": self.cls_head(queries), "box_norm": self.box_head(queries).sigmoid(), "angle_vec": init_angles}
        refined = self.rroa(queries, feats_pos, angle_rad)

        return {
            "aux_outputs": [aux],
            "cls_logits": self.cls_head(refined),
            "box_norm": torch.sigmoid(self.box_head(refined)),
            "angle_vec": self.angle_head(refined),
        }


class RioDETRDetector(BaseOBBDetector):
    """
    Model wrapper for compact RiO-DETR prototype (Real-Time Oriented DETR style).
    Inherits from BaseOBBDetector with strict uniform input/output contracts.
    """

    def __init__(self, model_name: str = "rio-detr", device: str = "cpu", num_classes: int = 10):
        super().__init__(model_name=model_name, device=device, num_classes=num_classes)
        self.net = None

    def load(self, weights_path: Optional[str] = None) -> None:
        """Instantiate RiO-DETR architecture and load weights if provided."""
        self.net = RioDETRNet(num_classes=self.num_classes, num_queries=300, embed_dim=128)
        self.net.to(self.device)
        self.net.eval()
        self._load_net_weights(weights_path)
        self.num_params = sum(p.numel() for p in self.net.parameters())
        self.is_loaded = True
        print(f"[✓] Loaded RiO-DETR compact axial Detector ({self.num_params / 1e6:.2f}M params on {self.device})")

    def predict(
        self,
        images: List[Image.Image],
        conf_thresh: float = 0.25,
        iou_thresh: float = 0.45,
        class_names: Optional[List[str]] = None,
        ground_truth_hints: Optional[List[List[Dict[str, Any]]]] = None,
    ) -> List[List[Dict[str, Any]]]:
        """Run unified forward + shared DETR OBB decoding on PIL images."""
        return self._predict_unified(
            images,
            conf_thresh=conf_thresh,
            iou_thresh=iou_thresh,
            class_names=class_names,
            ground_truth_hints=ground_truth_hints,
        )
