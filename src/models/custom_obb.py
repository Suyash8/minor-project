"""
Custom Anchor-Free Oriented Bounding Box Detector Baseline (FCOS-OBB / CenterNet-OBB Style)
===========================================================================================
Architectural Paradigm & Literature Reference:
    Based on the anchor-free single-stage detection principles established in:
    1. "FCOS: Fully Convolutional One-Stage Object Detection" (ICCV 2019)
       Authors: Zhi Tian, Chunhua Shen, Hao Chen, Tong He
       Official Repository: https://github.com/tianzhi0549/FCOS
    2. "Objects as Points" (CenterNet, 2019)
       Authors: Xingyi Zhou, Dequan Wang, Philipp Krähenbühl
       Official Repository: https://github.com/xingyizhou/CenterNet
    3. MMRotate Anchor-Free Baselines (Rotated FCOS):
       Reference: https://github.com/open-mmlab/mmrotate/tree/main/configs/rotated_fcos

Design & Formulation:
    - Pure PyTorch native architecture running on CPU/GPU without CUDA extensions.
    - Decoupled classification and bounding box regression branches.
    - Continuous Angle Representation: regresses [sin(2θ), cos(2θ)] or continuous θ with smooth L1/Gaussian loss,
      eliminating angle boundary discontinuity (Poincare / periodicity issues) common in [0, pi] and [-pi/2, pi/2] schemes.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base import BaseOBBDetector

class SmallOBBBackbone(nn.Module):
    """Lightweight convolutional backbone with 3 downsampling stages."""
    def __init__(self, in_channels: int = 3, base_channels: int = 32):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.SiLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.SiLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

class OrientedDetectionHead(nn.Module):
    """
    Decoupled head predicting class logits, box dimensions, and orientation angles.
    """
    def __init__(self, in_channels: int = 128, num_classes: int = 10):
        super().__init__()
        self.num_classes = num_classes
        # Classification branch
        self.cls_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(in_channels, num_classes, kernel_size=1),
        )
        # Bounding box & angle regression branch: (dx, dy, dw, dh, angle_sin, angle_cos, objectness)
        self.reg_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(in_channels, 7, kernel_size=1),
        )

    def forward(self, feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
        cls_logits = self.cls_head(feat)
        reg_out = self.reg_head(feat)
        return cls_logits, reg_out, (int(feat.shape[2]), int(feat.shape[3]))

class PyTorchOBBNet(nn.Module):
    """Full end-to-end oriented detector network."""
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.backbone = SmallOBBBackbone(in_channels=3, base_channels=32)
        self.head = OrientedDetectionHead(in_channels=128, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
        feat = self.backbone(x)
        return self.head(feat)

class CustomOBBDetector(BaseOBBDetector):
    """
    Model wrapper for the native PyTorch oriented detector baseline.
    """

    def __init__(self, model_name: str = "custom-obb", device: str = "cpu", num_classes: int = 10):
        super().__init__(model_name=model_name, device=device)
        self.num_classes = num_classes
        self.net = None

    def load(self, weights_path: Optional[str] = None) -> None:
        """Instantiate network architecture and load weights if provided."""
        self.net = PyTorchOBBNet(num_classes=self.num_classes)
        self.net.to(self.device)
        self.net.eval()

        self._load_net_weights(weights_path)

        self.num_params = sum(p.numel() for p in self.net.parameters())
        self.is_loaded = True
        print(f"[✓] Initialized Custom PyTorch OBB Detector ({self.num_params / 1e6:.2f}M params on {self.device})")

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
