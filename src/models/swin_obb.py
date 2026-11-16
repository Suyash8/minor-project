"""
Swin-Transformer for Oriented Object Detection (Swin-OBB)
=========================================================
Foundational Paper Reference:
    "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"
    Authors: Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, Baining Guo
    Venue: IEEE/CVF International Conference on Computer Vision (ICCV), 2021 (Best Paper Award).
    Official Repository: https://github.com/microsoft/Swin-Transformer
    Detection Codebase: https://github.com/SwinTransformer/Swin-Transformer-Object-Detection

Aerial & Remote Sensing OBB Standard:
    Implemented in MMRotate benchmark suite:
    "MMRotate: A Rotated Object Detection Benchmark using PyTorch"
    Repository: https://github.com/open-mmlab/mmrotate
    Configurations: configs/rotated_fcos/rotated_fcos_swin-t_fpn_1x_dota_le90.py

Architecture:
    1. Hierarchical Shifted Window Self-Attention (W-MSA / SW-MSA): Restricts self-attention computation to local
       windows while shifting partitions between successive layers for cross-window interactions (linear computational complexity).
    2. Multi-Scale FPN Integration: Produces multiscale feature representations for multi-scale aerial target detection.
    3. Continuous Angle Regression Head: Regresses [cx, cy, w, h, sin(2θ), cos(2θ)] with boundary-unconstrained periodic angle representation.
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


def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, int, int]:
    """Partition (B, H, W, C) into local windows ((B*nW), ws*ws, C)."""
    B, H, W, C = x.shape
    pad_b = (window_size - H % window_size) % window_size
    pad_r = (window_size - W % window_size) % window_size
    if pad_b or pad_r:
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
    Hp, Wp = H + pad_b, W + pad_r
    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size * window_size, C)
    return windows, Hp, Wp


def window_reverse(windows: torch.Tensor, window_size: int, Hp: int, Wp: int, H: int, W: int) -> torch.Tensor:
    """Invert `window_partition` back to (B, H, W, C)."""
    B = int(windows.shape[0] // ((Hp // window_size) * (Wp // window_size)))
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)
    return x[:, :H, :W, :].contiguous()


class SwinBlock(nn.Module):
    """Swin Transformer block with genuine W-MSA / SW-MSA shifted-window self-attention."""
    def __init__(self, embed_dim: int = 96, num_heads: int = 3, window_size: int = 7, shift_size: int = 0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def _shifted_mask(self, Hp: int, Wp: int, device: torch.device, dtype: torch.dtype) -> Optional[torch.Tensor]:
        ws, ss = self.window_size, self.shift_size
        if ss <= 0 or min(Hp, Wp) <= ss:
            return None
        img_mask = torch.zeros((1, Hp, Wp, 1), device=device, dtype=torch.float32)
        h_slices = (slice(0, -ws), slice(-ws, -ss), slice(-ss, None))
        w_slices = (slice(0, -ws), slice(-ws, -ss), slice(-ss, None))
        cnt = 0
        for h_s in h_slices:
            for w_s in w_slices:
                img_mask[:, h_s, w_s, :] = cnt
                cnt += 1
        mask_wins, _, _ = window_partition(img_mask, ws)
        mask_wins = mask_wins.squeeze(-1)
        attn_diff = mask_wins.unsqueeze(1) - mask_wins.unsqueeze(2)
        return torch.where(
            attn_diff != 0,
            torch.tensor(-100.0, device=device, dtype=dtype),
            torch.tensor(0.0, device=device, dtype=dtype),
        )

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        shortcut = x
        x_4d = self.norm1(x).view(B, H, W, C)
        eff_shift = self.shift_size if min(H, W) > self.window_size else 0
        shifted = torch.roll(x_4d, shifts=(-eff_shift, -eff_shift), dims=(1, 2)) if eff_shift > 0 else x_4d
        wins, Hp, Wp = window_partition(shifted, self.window_size)
        mask = self._shifted_mask(Hp, Wp, x.device, x.dtype) if eff_shift > 0 else None
        if mask is not None:
            nW = mask.shape[0]
            attn_mask = mask.unsqueeze(0).expand(B, -1, -1, -1).reshape(B * nW, wins.shape[1], wins.shape[1])
            attn_mask = attn_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1).reshape(B * nW * self.num_heads, wins.shape[1], wins.shape[1])
        else:
            attn_mask = None
        attn_wins, _ = self.attn(wins, wins, wins, attn_mask=attn_mask, need_weights=False)
        shifted_back = window_reverse(attn_wins, self.window_size, Hp, Wp, H, W)
        x_back = torch.roll(shifted_back, shifts=(eff_shift, eff_shift), dims=(1, 2)) if eff_shift > 0 else shifted_back
        x = shortcut + x_back.view(B, N, C)
        x = x + self.mlp(self.norm2(x))
        return x


class PatchMerging(nn.Module):
    """Merges 2x2 neighboring patches to halve spatial resolution and double channels."""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim * 4, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> Tuple[torch.Tensor, int, int]:
        B, N, C = x.shape
        x = x.view(B, H, W, C)
        # Pad if needed
        pad_h = (2 - H % 2) % 2
        pad_w = (2 - W % 2) % 2
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
            H, W = H + pad_h, W + pad_w

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, (H // 2) * (W // 2), 4 * C)
        x = self.norm(self.linear(x))
        return x, H // 2, W // 2


class SwinOBBNet(nn.Module):
    """Compact hierarchical Swin-style OBB detector (stride-8 FPN output, W-MSA + SW-MSA)."""
    def __init__(self, num_classes: int = 10, embed_dim: int = 96, window_size: int = 7):
        super().__init__()
        self.patch_size = 4
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=4, stride=4)
        self.norm0 = nn.LayerNorm(embed_dim)

        ws_half = window_size // 2
        self.stage1 = nn.ModuleList([
            SwinBlock(embed_dim=embed_dim, num_heads=3, window_size=window_size, shift_size=(0 if i % 2 == 0 else ws_half))
            for i in range(2)
        ])
        self.merge1 = PatchMerging(in_dim=embed_dim, out_dim=embed_dim * 2)  # stride 8
        self.stage2 = nn.ModuleList([
            SwinBlock(embed_dim=embed_dim * 2, num_heads=6, window_size=window_size, shift_size=(0 if i % 2 == 0 else ws_half))
            for i in range(2)
        ])
        self.merge2 = PatchMerging(in_dim=embed_dim * 2, out_dim=embed_dim * 4)  # stride 16
        self.stage3 = nn.ModuleList([
            SwinBlock(embed_dim=embed_dim * 4, num_heads=12, window_size=window_size, shift_size=(0 if i % 2 == 0 else ws_half))
            for i in range(2)
        ])

        self.fpn_lat3 = nn.Conv2d(embed_dim * 4, embed_dim * 2, kernel_size=1)
        self.fpn_out = nn.Sequential(
            nn.Conv2d(embed_dim * 2, embed_dim * 2, kernel_size=3, padding=1),
            nn.SiLU(),
        )

        out_dim = embed_dim * 2
        self.cls_head = nn.Linear(out_dim, num_classes)
        self.reg_head = nn.Linear(out_dim, 7)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
        B = x.shape[0]
        x = self.patch_embed(x)
        H1, W1 = x.shape[2], x.shape[3]
        x = x + build_2d_sincos_pos_embed(H1, W1, x.shape[1], device=x.device, dtype=x.dtype)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm0(x)

        for block in self.stage1:
            x = block(x, H1, W1)

        x2, H2, W2 = self.merge1(x, H1, W1)
        for block in self.stage2:
            x2 = block(x2, H2, W2)

        x3, H3, W3 = self.merge2(x2, H2, W2)
        for block in self.stage3:
            x3 = block(x3, H3, W3)

        f2 = x2.transpose(1, 2).view(B, -1, H2, W2)
        f3 = self.fpn_lat3(x3.transpose(1, 2).view(B, -1, H3, W3))
        f3_up = F.interpolate(f3, size=(H2, W2), mode="bilinear", align_corners=False)
        fused = self.fpn_out(f2 + f3_up).flatten(2).transpose(1, 2)

        return self.cls_head(fused), self.reg_head(fused), (H2, W2)


class SwinOBBDetector(BaseOBBDetector):
    """
    Model wrapper for compact Swin-OBB prototype (Shifted Window Transformer).
    Inherits from BaseOBBDetector with strict uniform input/output contracts.
    """

    def __init__(self, model_name: str = "swin-obb", device: str = "cpu", num_classes: int = 10):
        super().__init__(model_name=model_name, device=device, num_classes=num_classes)
        self.net = None

    def load(self, weights_path: Optional[str] = None) -> None:
        """Instantiate Swin-OBB architecture and load weights if provided."""
        self.net = SwinOBBNet(num_classes=self.num_classes, embed_dim=96)
        self.net.to(self.device)
        self.net.eval()
        self._load_net_weights(weights_path)
        self.num_params = sum(p.numel() for p in self.net.parameters())
        self.is_loaded = True
        print(f"[✓] Loaded Swin-OBB compact shifted-window Detector ({self.num_params / 1e6:.2f}M params on {self.device})")

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
