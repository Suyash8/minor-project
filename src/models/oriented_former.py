"""
OrientedFormer: End-to-End Oriented Object Detection with Gaussian Queries (IEEE TGRS 2024).
Official GitHub Repository: https://github.com/wokaikaixinxin/OrientedFormer
Upstream Comprehensive Benchmark: https://github.com/wokaikaixinxin/ai4rs
Paper: "OrientedFormer: End-to-End Oriented Object Detection with Gaussian Queries"
Authors: Kaixinxin Wo, Menglong Yan, Xiangrong Zhang, Licheng Jiao
Journal: IEEE Transactions on Geoscience and Remote Sensing (IEEE TGRS 2024)
DOI: 10.1109/TGRS.2024.3377778
Framework: MMDetection / MMRotate integration

Architecture Highlights:
  - Gaussian Positional Encoding (GPE) modeling oriented bounding boxes as 2D Gaussian query priors.
  - Wasserstein Self-Attention (WSA) using Gaussian Wasserstein distance metric to capture spatial and angular geometric relations.
  - Oriented Cross-Attention (OCA) aligning sampling features directly with oriented target directions.
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

from src.models.base import (
    BaseOBBDetector,
    build_2d_sincos_pos_embed,
    sample_rotated_roi_features,
)


def _decode_raw_gaussian_boxes(raw_b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Decode unconstrained raw box params -> normalized (cx, cy, w, h) in (0, 1) and angle_rad."""
    cx_cy = torch.sigmoid(raw_b[..., 0:2])
    wh = 0.001 + 0.999 * torch.sigmoid(raw_b[..., 2:4] - 2.5)
    return torch.cat([cx_cy, wh], dim=-1), raw_b[..., 4:5]


class GaussianPositionalEncoding(nn.Module):
    """Encodes 2D Gaussian parameters (mu_x, mu_y, w, h, theta) into query embeddings."""
    def __init__(self, embed_dim: int = 192):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(5, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim),
        )

    def forward(self, gaussian_params: torch.Tensor) -> torch.Tensor:
        return self.proj(gaussian_params)


class WassersteinSelfAttention(nn.Module):
    """Self-attention biased by pairwise 2D Gaussian Wasserstein-2 distances between query priors."""
    def __init__(self, embed_dim: int = 192, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=num_heads, batch_first=True)
        self.w_scale = nn.Parameter(torch.ones(num_heads))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, queries: torch.Tensor, raw_gaussians: torch.Tensor) -> torch.Tensor:
        B, Q, _ = queries.shape
        boxes_norm, ang = _decode_raw_gaussian_boxes(raw_gaussians)
        centers = boxes_norm[..., 0:2]  # (B, Q, 2)
        cos_a = torch.cos(ang[..., 0])
        sin_a = torch.sin(ang[..., 0])
        sx = (boxes_norm[..., 2] * 0.5).square()
        sy = (boxes_norm[..., 3] * 0.5).square()
        # 2x2 rotated covariance Sigma^(1/2) elements (sym: [a, b; b, c])
        s_xx = sx * cos_a.square() + sy * sin_a.square()
        s_yy = sx * sin_a.square() + sy * cos_a.square()
        s_xy = (sx - sy) * sin_a * cos_a

        dc = centers.unsqueeze(2) - centers.unsqueeze(1)  # (B, Q, Q, 2)
        center_dist2 = dc.square().sum(dim=-1)
        cov_diff2 = (
            (s_xx.unsqueeze(2) - s_xx.unsqueeze(1)).square()
            + (s_yy.unsqueeze(2) - s_yy.unsqueeze(1)).square()
            + 2.0 * (s_xy.unsqueeze(2) - s_xy.unsqueeze(1)).square()
        )
        w2_dist = torch.sqrt((center_dist2 + cov_diff2).clamp(min=1e-6))  # (B, Q, Q)
        bias = -self.w_scale.view(1, self.num_heads, 1, 1) * w2_dist.unsqueeze(1)  # (B, H, Q, Q)
        attn_mask = bias.reshape(B * self.num_heads, Q, Q)
        out, _ = self.attn(queries, queries, queries, attn_mask=attn_mask, need_weights=False)
        return self.norm(queries + out)


class OrientedCrossAttention(nn.Module):
    """Cross-attention combining oriented rotated feature sampling with 2D spatial memory attention."""
    def __init__(self, embed_dim: int = 192, num_heads: int = 4):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads=num_heads, batch_first=True)
        self.roi_proj = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        feat_2d: torch.Tensor,
        raw_gaussians: torch.Tensor,
    ) -> torch.Tensor:
        out, _ = self.cross_attn(query, memory, memory)
        boxes_norm, ang = _decode_raw_gaussian_boxes(raw_gaussians)
        roi_feat = sample_rotated_roi_features(feat_2d, boxes_norm, ang, grid_k=3)
        return self.norm(query + out + self.roi_proj(roi_feat))


class OrientedFormerNet(nn.Module):
    """Compact OrientedFormer prototype with GPE, Wasserstein self-attention, and oriented cross-attention."""
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

        # Spread Gaussian query priors across image in raw logit coords
        init_g = torch.zeros(num_queries, 5)
        init_g[:, 0:2].uniform_(-2.0, 2.0)
        init_g[:, 2:4].uniform_(-1.5, 0.5)
        init_g[:, 4].uniform_(-math.pi / 2.0, math.pi / 2.0)
        self.init_gaussian_queries = nn.Parameter(init_g)
        self.gpe = GaussianPositionalEncoding(embed_dim=embed_dim)

        self.wsa = WassersteinSelfAttention(embed_dim=embed_dim, num_heads=4)
        self.oca = OrientedCrossAttention(embed_dim=embed_dim, num_heads=4)

        self.cls_head = nn.Linear(embed_dim, num_classes + 1)
        nn.init.constant_(self.cls_head.bias, math.log(.01/.99))
        # Legacy background row retained only for shape parity.
        self.gaussian_reg_head = nn.Linear(embed_dim, 5)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        B = x.shape[0]
        feats = self.backbone(x)
        H, W = feats.shape[2], feats.shape[3]
        feats_pos = feats + build_2d_sincos_pos_embed(H, W, self.embed_dim, device=feats.device, dtype=feats.dtype)
        memory = feats_pos.flatten(2).transpose(1, 2) # (B, N, embed_dim)

        g_priors = self.init_gaussian_queries.unsqueeze(0).repeat(B, 1, 1) # (B, Q, 5)
        q_pos = self.gpe(g_priors)

        queries = self.wsa(q_pos, g_priors)
        aux_boxes = g_priors + self.gaussian_reg_head(queries)
        aux = {"cls_logits": self.cls_head(queries), "boxes": aux_boxes}
        queries = self.oca(queries, memory, feats_pos, aux_boxes)

        cls_logits = self.cls_head(queries)
        final_boxes = aux_boxes + self.gaussian_reg_head(queries)

        return {
            "aux_outputs": [aux],
            "cls_logits": cls_logits,
            "boxes": final_boxes,
        }


class OrientedFormerDetector(BaseOBBDetector):
    """
    Model wrapper for compact OrientedFormer prototype (IEEE TGRS 2024 style).
    Inherits from BaseOBBDetector with strict uniform input/output contracts.
    """

    def __init__(self, model_name: str = "oriented-former", device: str = "cpu", num_classes: int = 10):
        super().__init__(model_name=model_name, device=device, num_classes=num_classes)
        self.net = None

    def load(self, weights_path: Optional[str] = None) -> None:
        """Instantiate OrientedFormer architecture and load weights if provided."""
        self.net = OrientedFormerNet(num_classes=self.num_classes, num_queries=300, embed_dim=192)
        self.net.to(self.device)
        self.net.eval()
        self._load_net_weights(weights_path)
        self.num_params = sum(p.numel() for p in self.net.parameters())
        self.is_loaded = True
        print(f"[✓] Loaded OrientedFormer compact Gaussian-Guided Detector ({self.num_params / 1e6:.2f}M params on {self.device})")

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
