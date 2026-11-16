"""
AO2-DETR: Arbitrary-Oriented Object Detection Transformer
=========================================================
Paper Reference:
    "AO2-DETR: Arbitrary-Oriented Object Detection Transformer"
    Authors: Lanjun Dai, Hong Liu, Hao Tang, Zehao Wu, Shengchao Chen
    Venue: IEEE Transactions on Circuits and Systems for Video Technology (TCSVT), 2023.
    DOI: 10.1109/TCSVT.2023.3243144
    Official GitHub Repository: https://github.com/Ixiaohuihuihui/AO2-DETR

Key Innovations & Architecture:
    1. Oriented Proposal Generation (OPG): Directly extracts oriented object candidates from multi-scale feature maps,
       avoiding standard axis-aligned prior box bias.
    2. Adaptive Oriented Proposal Refinement (AOPR): Iteratively refines oriented proposals via cross-attention with
       rotation-guided positional encoding, mitigating misalignment between oriented objects and receptive fields.
    3. Fully End-to-End: Eliminates complex post-processing such as Oriented Non-Maximum Suppression (Rotated NMS).
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


class OrientedProposalGenerator(nn.Module):
    """Generates oriented bounding box query priors + scores from 2D spatial feature maps."""
    def __init__(self, embed_dim: int = 192):
        super().__init__()
        self.proj = nn.Conv2d(embed_dim, 5, kernel_size=1)  # raw (cx, cy, w, h, angle) logits
        self.score = nn.Conv2d(embed_dim, 1, kernel_size=1)

    def forward(self, feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # feat: (B, C, H, W) -> proposals (B, 5, H, W), score (B, 1, H, W)
        return self.proj(feat), self.score(feat)


class AdaptiveOrientedProposalRefinement(nn.Module):
    """Refines oriented queries using self-attention, 2D-encoded cross-attention, and rotated ROI sampling."""
    def __init__(self, embed_dim: int = 192, num_heads: int = 4):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads=num_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads=num_heads, batch_first=True)
        self.roi_proj = nn.Linear(embed_dim, embed_dim)
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
        queries: torch.Tensor,
        memory: torch.Tensor,
        feat_map: torch.Tensor,
        proposals: torch.Tensor,
    ) -> torch.Tensor:
        q2, _ = self.self_attn(queries, queries, queries)
        queries = self.norm1(queries + q2)
        q2, _ = self.cross_attn(queries, memory, memory)
        # Rotation-guided spatial sampling around current proposal boxes
        cx_cy = torch.sigmoid(proposals[..., 0:2])
        wh = 0.001 + 0.999 * torch.sigmoid(proposals[..., 2:4] - 2.5)
        roi_feat = sample_rotated_roi_features(
            feat_map, torch.cat([cx_cy, wh], dim=-1), proposals[..., 4:5], grid_k=3
        )
        queries = self.norm2(queries + q2 + self.roi_proj(roi_feat))
        queries = self.norm3(queries + self.ffn(queries))
        return queries


class AO2DETRNet(nn.Module):
    """Compact AO2-DETR prototype with active OPG, 2D spatial PE, and rotated AOPR refinement."""
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

        self.opg = OrientedProposalGenerator(embed_dim=embed_dim)
        self.opg_query_proj = nn.Linear(embed_dim + 5, embed_dim)
        self.query_embed = nn.Embedding(num_queries, embed_dim)

        self.aopr_layers = nn.ModuleList([
            AdaptiveOrientedProposalRefinement(embed_dim=embed_dim, num_heads=4) for _ in range(3)
        ])

        self.cls_head = nn.Linear(embed_dim, num_classes + 1)
        nn.init.constant_(self.cls_head.bias, math.log(.01/.99))
        # Legacy background row retained only for shape parity.
        self.reg_head = nn.Linear(embed_dim, 5)  # iterative refinement on (cx, cy, w, h, angle) logits
        nn.init.normal_(self.reg_head.weight, std=0.01)
        nn.init.zeros_(self.reg_head.bias)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        B = x.shape[0]
        feats = self.backbone(x)  # (B, embed_dim, H, W)
        H, W = feats.shape[2], feats.shape[3]
        pos_2d = build_2d_sincos_pos_embed(H, W, self.embed_dim, device=feats.device, dtype=feats.dtype)
        feats_pos = feats + pos_2d
        memory = feats_pos.flatten(2).transpose(1, 2)  # (B, N, embed_dim)

        # Active Oriented Proposal Generation (top-Q spatial proposals + center offsets)
        opg_raw, opg_scores = self.opg(feats_pos)  # (B, 5, H, W), (B, 1, H, W)
        gy, gx = torch.meshgrid(
            (torch.arange(H, device=x.device, dtype=feats.dtype) + 0.5) / max(H, 1),
            (torch.arange(W, device=x.device, dtype=feats.dtype) + 0.5) / max(W, 1),
            indexing="ij",
        )
        grid_norm = torch.stack([gx, gy], dim=0).unsqueeze(0).expand(B, -1, -1, -1).clamp(1e-3, 1.0 - 1e-3)
        grid_logits = torch.log(grid_norm / (1.0 - grid_norm))
        opg_props = opg_raw.clone()
        opg_props[:, :2] = opg_raw[:, :2] + grid_logits
        flat_props = opg_props.flatten(2).transpose(1, 2)  # (B, N, 5)
        flat_scores = opg_scores.flatten(1)  # (B, N)
        N = flat_scores.shape[1]
        if N >= self.num_queries:
            topk_idx = flat_scores.topk(self.num_queries, dim=1).indices.sort(dim=1).values
        else:
            reps = (self.num_queries + N - 1) // N
            topk_idx = (torch.arange(N * reps, device=x.device) % N)[:self.num_queries].unsqueeze(0).expand(B, -1)
        sel_feats = torch.gather(memory, 1, topk_idx.unsqueeze(-1).expand(-1, -1, self.embed_dim))
        sel_props = torch.gather(flat_props, 1, topk_idx.unsqueeze(-1).expand(-1, -1, 5))
        sel_scores = torch.gather(flat_scores, 1, topk_idx)
        sel_feats = sel_feats * (0.5 + 0.5 * torch.sigmoid(sel_scores)).unsqueeze(-1)

        queries = self.opg_query_proj(torch.cat([sel_feats, sel_props], dim=-1)) + self.query_embed.weight.unsqueeze(0)
        proposals = sel_props

        aux_outputs = []
        for layer in self.aopr_layers:
            queries = layer(queries, memory, feats_pos, proposals)
            proposals = proposals + self.reg_head(queries)
            aux_outputs.append({"cls_logits": self.cls_head(queries), "box_deltas": proposals})

        return {
            "aux_outputs": aux_outputs[:-1],
            "cls_logits": aux_outputs[-1]["cls_logits"],
            "box_deltas": aux_outputs[-1]["box_deltas"],
        }


class AO2DETRDetector(BaseOBBDetector):
    """
    Model wrapper for compact AO2-DETR prototype (IEEE TCSVT 2023 style).
    Inherits from BaseOBBDetector with strict uniform input/output contracts.
    """

    def __init__(self, model_name: str = "ao2-detr", device: str = "cpu", num_classes: int = 10):
        super().__init__(model_name=model_name, device=device, num_classes=num_classes)
        self.net = None

    def load(self, weights_path: Optional[str] = None) -> None:
        """Instantiate AO2-DETR architecture and load weights if provided."""
        self.net = AO2DETRNet(num_classes=self.num_classes, num_queries=300, embed_dim=192)
        self.net.to(self.device)
        self.net.eval()
        self._load_net_weights(weights_path)
        self.num_params = sum(p.numel() for p in self.net.parameters())
        self.is_loaded = True
        print(f"[✓] Loaded AO2-DETR compact OPG/AOPR Detector ({self.num_params / 1e6:.2f}M params on {self.device})")

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
