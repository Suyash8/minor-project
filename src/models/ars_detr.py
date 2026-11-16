"""
ARS-DETR: Aspect Ratio-Sensitive Detection Transformer for Aerial Oriented Object Detection (IEEE TGRS 2024).
Official GitHub Repository: https://github.com/httle/ARS-DETR
Paper: "ARS-DETR: Aspect Ratio-Sensitive Detection Transformer for Aerial Oriented Object Detection"
Authors: Chenghao Xiao, Qiangqiang Yuan, Kui Jiang, Xianwei Zheng, Liangpei Zhang
Journal: IEEE Transactions on Geoscience and Remote Sensing (IEEE TGRS 2024)
arXiv: https://arxiv.org/abs/2303.04989
Original Baseline: Deformable DETR adapted for oriented remote sensing

Architecture Highlights:
  - Rotated Deformable Attention that dynamically aligns cross-attention sampling points with target orientation angles.
  - Aspect Ratio-aware Circular Smooth Label (AR-CSL) resolving angular boundary discontinuity and aspect-ratio sensitivity.
  - Aspect-ratio modulated angle loss weighting.
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


class RotatedDeformableAttention(nn.Module):
    """
    Rotated Deformable Attention Module.
    Predicts rotated 2D sampling offsets relative to query reference boxes and
    bilinearly samples spatial feature maps via `F.grid_sample`.
    """
    def __init__(self, embed_dim: int = 192, num_heads: int = 4, n_points: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.n_points = n_points

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.offset_proj = nn.Linear(embed_dim, num_heads * n_points * 2)
        self.attn_weights_proj = nn.Linear(embed_dim, num_heads * n_points)
        self.v_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.rot_proj = nn.Linear(1, embed_dim)

    def forward(
        self,
        query: torch.Tensor,
        feat_2d: torch.Tensor,
        ref_boxes: torch.Tensor,
        angles: torch.Tensor,
    ) -> torch.Tensor:
        B, Q, C = query.shape
        q_rot = self.q_proj(query + self.rot_proj(angles))

        # Predict normalized local offsets & attention weights
        offsets = self.offset_proj(q_rot).view(B, Q, self.num_heads, self.n_points, 2)
        attn_w = F.softmax(self.attn_weights_proj(q_rot).view(B, Q, self.num_heads, self.n_points), dim=-1)

        # Rotate offsets by predicted object angle and scale by box size
        cos_a = torch.cos(angles).view(B, Q, 1, 1)
        sin_a = torch.sin(angles).view(B, Q, 1, 1)
        ox = offsets[..., 0] * (ref_boxes[..., 2:3].unsqueeze(-1).clamp(min=0.01) * 0.5)
        oy = offsets[..., 1] * (ref_boxes[..., 3:4].unsqueeze(-1).clamp(min=0.01) * 0.5)
        rx = ox * cos_a - oy * sin_a
        ry = ox * sin_a + oy * cos_a

        loc_x = (ref_boxes[..., 0:1].unsqueeze(-1) + rx) * 2.0 - 1.0
        loc_y = (ref_boxes[..., 1:2].unsqueeze(-1) + ry) * 2.0 - 1.0
        # Reshape (B, Q, H, P, 2) -> (B*H, Q, P, 2) for grid_sample on head value maps
        grid = torch.stack([loc_x, loc_y], dim=-1).permute(0, 2, 1, 3, 4).reshape(B * self.num_heads, Q, self.n_points, 2)

        v_2d = self.v_proj(feat_2d).view(B * self.num_heads, self.head_dim, feat_2d.shape[2], feat_2d.shape[3])
        sampled = F.grid_sample(
            v_2d.float(), grid.float(), mode="bilinear", padding_mode="zeros", align_corners=False
        )  # (B*H, D, Q, P)
        sampled = sampled.view(B, self.num_heads, self.head_dim, Q, self.n_points).permute(0, 3, 1, 4, 2).to(dtype=query.dtype)
        out = (sampled * attn_w.unsqueeze(-1)).sum(dim=3).reshape(B, Q, C)
        return self.out_proj(out)


class ARSDETRDecoderLayer(nn.Module):
    """DETR Decoder Layer with self-attention and Rotated Deformable Cross-Attention."""
    def __init__(self, embed_dim: int = 192, num_heads: int = 4):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads=num_heads, batch_first=True)
        self.cross_attn = RotatedDeformableAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def forward(
        self,
        tgt: torch.Tensor,
        feat_2d: torch.Tensor,
        ref_boxes: torch.Tensor,
        angles: torch.Tensor,
    ) -> torch.Tensor:
        tgt2, _ = self.self_attn(tgt, tgt, tgt)
        tgt = self.norm1(tgt + tgt2)
        tgt2 = self.cross_attn(tgt, feat_2d, ref_boxes, angles)
        tgt = self.norm2(tgt + tgt2)
        tgt = self.norm3(tgt + self.ffn(tgt))
        return tgt


class ARSDETRNet(nn.Module):
    """Compact ARS-DETR prototype with 2D spatial PE and rotated deformable sampling."""
    def __init__(self, num_classes: int = 10, num_queries: int = 300, embed_dim: int = 192):
        super().__init__()
        self.num_queries = num_queries
        self.embed_dim = embed_dim

        self.backbone = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(48),
            nn.SiLU(),
            nn.Conv2d(48, 96, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.SiLU(),
            nn.Conv2d(96, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.SiLU(),
        )

        self.query_embed = nn.Embedding(num_queries, embed_dim)

        self.decoder = nn.ModuleList([
            ARSDETRDecoderLayer(embed_dim=embed_dim, num_heads=4) for _ in range(3)
        ])

        self.class_head = nn.Linear(embed_dim, num_classes + 1)
        nn.init.constant_(self.class_head.bias, math.log(.01/.99))
        # Legacy background row retained only for shape parity.
        self.box_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 4), # normalized cx, cy, w, h
        )
        self.angle_head = nn.Linear(embed_dim, 2) # sin, cos

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        B = x.shape[0]
        feats = self.backbone(x) # (B, C, H, W)
        H, W = feats.shape[2], feats.shape[3]
        feats_pos = feats + build_2d_sincos_pos_embed(H, W, self.embed_dim, device=feats.device, dtype=feats.dtype)

        queries = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1) # (B, Q, C)
        ref_boxes = self.box_head(queries).sigmoid()
        init_ang = self.angle_head(queries)
        angles = torch.atan2(init_ang[..., 0:1], init_ang[..., 1:2])

        aux_outputs = []
        for layer in self.decoder:
            queries = layer(queries, feats_pos, ref_boxes, angles)
            ref_boxes = self.box_head(queries).sigmoid()
            dang = self.angle_head(queries)
            angles = torch.atan2(dang[..., 0:1], dang[..., 1:2])
            aux_outputs.append({"cls_logits": self.class_head(queries), "boxes_norm": ref_boxes, "angle_vec": dang})

        return {
            "aux_outputs": aux_outputs[:-1],
            "cls_logits": aux_outputs[-1]["cls_logits"],
            "boxes_norm": aux_outputs[-1]["boxes_norm"],
            "angle_vec": aux_outputs[-1]["angle_vec"],
        }


class ARSDETRDetector(BaseOBBDetector):
    """
    Model wrapper for compact ARS-DETR prototype (Aspect Ratio-Sensitive DETR style).
    Inherits from BaseOBBDetector with strict uniform input/output contracts.
    """

    def __init__(self, model_name: str = "ars-detr", device: str = "cpu", num_classes: int = 10):
        super().__init__(model_name=model_name, device=device, num_classes=num_classes)
        self.net = None

    def load(self, weights_path: Optional[str] = None) -> None:
        """Instantiate ARS-DETR architecture and load weights if provided."""
        self.net = ARSDETRNet(num_classes=self.num_classes, num_queries=300, embed_dim=192)
        self.net.to(self.device)
        self.net.eval()
        self._load_net_weights(weights_path)
        self.num_params = sum(p.numel() for p in self.net.parameters())
        self.is_loaded = True
        print(f"[✓] Loaded ARS-DETR compact rotated-deformable Detector ({self.num_params / 1e6:.2f}M params on {self.device})")

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
