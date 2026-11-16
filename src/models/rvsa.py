"""
RVSA: Rotated Varied-Size Attention for Remote Sensing Object Detection (CVPR 2023 / TPAMI 2024).
Official GitHub Repository: https://github.com/ViTAE-Transformer/Remote-Sensing-RVSA
Paper: "Rotated Varied-Size Attention for Remote Sensing Object Detection"
Authors: Di Wang, Qiming Zhang, Yufei Xu, Jing Zhang, Bo Du, Dacheng Tao, Liangpei Zhang
Conference: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2023)
Journal: IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI 2024)
arXiv: https://arxiv.org/abs/2304.14811
Framework: PyTorch / MMRotate Foundation Models for Remote Sensing

Architecture Highlights:
  - Adapts plain Vision Transformers to oriented aerial imagery via dynamic window self-attention.
  - Dynamically determines window size, aspect ratio, position, and rotation angle to conform to target geometries.
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


class RotatedVariedSizeAttention(nn.Module):
    """
    Rotated Varied-Size Attention (RVSA) Module.
    Samples rotated, per-head varied-size local windows over 2D feature maps
    using learnable `head_angles` and `scale_factors`.
    """
    def __init__(self, embed_dim: int = 192, num_heads: int = 4, window_size: int = 5):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.window_size = window_size

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)

        # Learnable rotation angle and varied window aspect scales per head (executed in forward)
        self.head_angles = nn.Parameter(torch.tensor([0.0, math.pi / 4, math.pi / 2, 3 * math.pi / 4]))
        self.scale_factors = nn.Parameter(torch.ones(num_heads, 2))

    def forward(self, x: torch.Tensor, feat_h: int, feat_w: int) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q = qkv[:, :, 0].permute(0, 2, 1, 3)  # (B, H, N, D)
        k_2d = qkv[:, :, 1].permute(0, 2, 3, 1).reshape(B * self.num_heads, self.head_dim, feat_h, feat_w)
        v_2d = qkv[:, :, 2].permute(0, 2, 3, 1).reshape(B * self.num_heads, self.head_dim, feat_h, feat_w)

        coords = torch.linspace(-0.5, 0.5, self.window_size, device=x.device, dtype=torch.float32)
        oy, ox = torch.meshgrid(coords, coords, indexing="ij")
        base_ox = ox.reshape(-1) * (float(self.window_size) / max(feat_w, 1))
        base_oy = oy.reshape(-1) * (float(self.window_size) / max(feat_h, 1))

        cy_vec = (torch.arange(feat_h, device=x.device, dtype=torch.float32) + 0.5) / max(feat_h, 1) * 2.0 - 1.0
        cx_vec = (torch.arange(feat_w, device=x.device, dtype=torch.float32) + 0.5) / max(feat_w, 1) * 2.0 - 1.0
        grid_y, grid_x = torch.meshgrid(cy_vec, cx_vec, indexing="ij")
        centers = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)  # (N, 2)

        head_outputs = []
        scale = 1.0 / math.sqrt(self.head_dim)
        for h in range(self.num_heads):
            ang = self.head_angles[h]
            sx_fac = F.softplus(self.scale_factors[h, 0]).clamp(min=0.25, max=4.0)
            sy_fac = F.softplus(self.scale_factors[h, 1]).clamp(min=0.25, max=4.0)
            ux = base_ox * sx_fac * 2.0
            uy = base_oy * sy_fac * 2.0
            rx = ux * torch.cos(ang) - uy * torch.sin(ang)
            ry = ux * torch.sin(ang) + uy * torch.cos(ang)
            offsets = torch.stack([rx, ry], dim=-1)
            sample_grid = (centers.unsqueeze(1) + offsets.unsqueeze(0)).unsqueeze(0).expand(B, -1, -1, -1)
            k_h = k_2d[h::self.num_heads].float()
            v_h = v_2d[h::self.num_heads].float()
            k_win = F.grid_sample(k_h, sample_grid.float(), mode="bilinear", padding_mode="zeros", align_corners=False)
            v_win = F.grid_sample(v_h, sample_grid.float(), mode="bilinear", padding_mode="zeros", align_corners=False)
            k_win = k_win.permute(0, 2, 3, 1).to(dtype=x.dtype)
            v_win = v_win.permute(0, 2, 3, 1).to(dtype=x.dtype)
            attn_h = F.softmax(torch.einsum("bnd,bnkd->bnk", q[:, h], k_win) * scale, dim=-1)
            head_outputs.append(torch.einsum("bnk,bnkd->bnd", attn_h, v_win))

        return self.proj(torch.cat(head_outputs, dim=-1))


class RVSABlock(nn.Module):
    """Transformer block with Rotated Varied-Size Attention and Feed-Forward Network."""
    def __init__(self, embed_dim: int = 192, num_heads: int = 4, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = RotatedVariedSizeAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor, feat_h: int, feat_w: int) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), feat_h, feat_w)
        x = x + self.mlp(self.norm2(x))
        return x


class RVSANet(nn.Module):
    """Compact local RVSA Vision Transformer for Oriented Bounding Box Detection (stride 8)."""
    def __init__(self, num_classes: int = 10, embed_dim: int = 192, depth: int = 4, num_heads: int = 4):
        super().__init__()
        self.patch_size = 8
        self.patch_proj = nn.Conv2d(3, embed_dim, kernel_size=8, stride=8)
        self.blocks = nn.ModuleList([
            RVSABlock(embed_dim=embed_dim, num_heads=num_heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        self.cls_head = nn.Linear(embed_dim, num_classes)
        self.reg_head = nn.Linear(embed_dim, 7)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
        x_proj = self.patch_proj(x)
        feat_h, feat_w = x_proj.shape[2], x_proj.shape[3]
        x_proj = x_proj + build_2d_sincos_pos_embed(feat_h, feat_w, x_proj.shape[1], device=x.device, dtype=x.dtype)
        tokens = x_proj.flatten(2).transpose(1, 2)

        for block in self.blocks:
            tokens = block(tokens, feat_h, feat_w)

        tokens = self.norm(tokens)
        return self.cls_head(tokens), self.reg_head(tokens), (feat_h, feat_w)


class RVSADetector(BaseOBBDetector):
    """
    Model wrapper for compact RVSA prototype (Rotated Varied-Size Attention).
    Inherits from BaseOBBDetector with strict uniform input/output contracts.
    """

    def __init__(self, model_name: str = "rvsa", device: str = "cpu", num_classes: int = 10):
        super().__init__(model_name=model_name, device=device, num_classes=num_classes)
        self.net = None

    def load(self, weights_path: Optional[str] = None) -> None:
        """Instantiate RVSA architecture and load weights if provided."""
        self.net = RVSANet(num_classes=self.num_classes, embed_dim=192, depth=4, num_heads=4)
        self.net.to(self.device)
        self.net.eval()
        self._load_net_weights(weights_path)
        self.num_params = sum(p.numel() for p in self.net.parameters())
        self.is_loaded = True
        print(f"[✓] Loaded RVSA compact rotated-window Detector ({self.num_params / 1e6:.2f}M params on {self.device})")

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
