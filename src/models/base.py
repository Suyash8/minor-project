"""
Base Interface for Oriented Bounding Box Detectors.
Defines standardized prediction signatures, latency benchmarking, and metadata extraction.
All models inherit from BaseOBBDetector to guarantee a strict, uniform API:
  - Input: images: List[Image.Image] (only images is strictly required).
  - Output: List[List[Dict[str, Any]]] where output[img_idx][class_idx] = {'boxes': (N, 5), 'scores': (N,)}.
"""

from __future__ import annotations

import time
import math
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from PIL import Image

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


_STATE_PREFIXES = ("_orig_mod.", "module.")


def unwrap_torch_module(module: Any) -> Any:
    """Return the underlying module beneath torch.compile / DataParallel wrappers."""
    while hasattr(module, "_orig_mod") or hasattr(module, "module"):
        if hasattr(module, "_orig_mod"):
            module = module._orig_mod
        elif hasattr(module, "module"):
            module = module.module
        else:
            break
    return module


def normalize_state_dict_keys(state: Dict[str, Any]) -> Dict[str, Any]:
    """Strip wrapper prefixes (e.g. torch.compile `_orig_mod.`, DDP `module.`)."""
    cleaned: Dict[str, Any] = {}
    for k, v in state.items():
        nk = k
        changed = True
        while changed:
            changed = False
            for prefix in _STATE_PREFIXES:
                if nk.startswith(prefix):
                    nk = nk[len(prefix):]
                    changed = True
        cleaned[nk] = v
    return cleaned


class BaseOBBDetector(ABC):
    """
    Abstract base class for all oriented bounding box detectors.
    Ensures uniform software contract across all CNN, Transformer, and hybrid architectures.
    """

    def __init__(self, model_name: str, device: str = "cpu", num_classes: int = 10):
        self.model_name = model_name
        self.device = device
        self.num_classes = num_classes
        self.num_params = 0
        self.is_loaded = False

    @abstractmethod
    def load(self, weights_path: Optional[str] = None) -> None:
        """Load model architecture and weights."""
        pass

    def _load_net_weights(self, weights_path: Optional[str]) -> None:
        """Load a PyTorch checkpoint into `self.net`, stripping wrapper prefixes cleanly."""
        if not (weights_path and Path(weights_path).exists()):
            return
        checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)
        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            state = checkpoint["model_state"]
        elif isinstance(checkpoint, dict):
            state = checkpoint
        else:
            self.net = unwrap_torch_module(checkpoint).to(self.device)
            self.net.eval()
            return
        cleaned = normalize_state_dict_keys(state)
        self.net.load_state_dict(cleaned, strict=True)

    @abstractmethod
    def predict(
        self,
        images: List[Image.Image],
        conf_thresh: float = 0.25,
        iou_thresh: float = 0.45,
        class_names: Optional[List[str]] = None,
        ground_truth_hints: Optional[List[List[Dict[str, Any]]]] = None,
    ) -> List[List[Dict[str, Any]]]:
        """
        Run inference on a batch of PIL images.

        Args:
            images: List of PIL Image objects (the only strictly required argument).
            conf_thresh: Confidence score threshold (default 0.25).
            iou_thresh: Non-maximum suppression IoU threshold (default 0.45).
            class_names: Optional list of target class strings.
            ground_truth_hints: Optional hints used during fast pipeline smoke tests.

        Returns:
            list over images, where each image has a list over classes:
            output[img_idx][class_idx] = {
                'boxes': ndarray of shape (N, 5) with [cx, cy, w, h, angle_deg],
                'scores': ndarray of shape (N,)
            }
        """
        pass

    def _preprocess_images(
        self,
        images: List[Image.Image],
        target_size: int = 640,
    ) -> Tuple[Any, List[Tuple[int, int]]]:
        """
        Standardize and batch a list of PIL Images into a normalized PyTorch tensor.
        Returns:
            tensor: (B, 3, target_size, target_size) on self.device
            orig_dims: List of (width, height) tuples for image coordinate denormalization
        """
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is required for detector image preprocessing.")

        tensors = []
        orig_dims = []

        for img in images:
            w, h = img.size
            orig_dims.append((w, h))

            if img.mode != "RGB":
                img = img.convert("RGB")

            resized = img.resize((target_size, target_size), Image.Resampling.BILINEAR)
            arr = np.array(resized, dtype=np.float32) / 255.0
            # (H, W, C) -> (C, H, W)
            tensor = torch.from_numpy(arr).permute(2, 0, 1)
            tensors.append(tensor)

        batched = torch.stack(tensors, dim=0).to(self.device)
        return batched, orig_dims

    def _empty_results(self, batch_size: int, num_classes: int) -> List[List[Dict[str, Any]]]:
        """Create empty prediction container matching canonical format."""
        return [
            [
                {
                    "boxes": np.zeros((0, 5), dtype=np.float32),
                    "scores": np.zeros(0, dtype=np.float32),
                }
                for _ in range(num_classes)
            ]
            for _ in range(batch_size)
        ]

    def _simulate_if_hinted(
        self,
        img_idx: int,
        orig_w: int,
        orig_h: int,
        num_classes: int,
        conf_thresh: float,
        ground_truth_hints: Optional[List[List[Dict[str, Any]]]],
        recall_rate: float = 0.80,
        noise_std: float = 2.5,
    ) -> Optional[List[Dict[str, Any]]]:
        """Legacy argument is rejected rather than fabricating predictions."""
        if ground_truth_hints is not None:
            raise ValueError("Ground-truth hints cannot be used for production inference")
        return None

    def benchmark_latency(
        self,
        sample_img: Image.Image,
        num_warmup: int = 3,
        num_runs: int = 15,
    ) -> Dict[str, float]:
        """
        Benchmark average inference latency and frames per second (FPS).
        """
        # Warmup
        for _ in range(num_warmup):
            _ = self.predict([sample_img])

        latencies = []
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = self.predict([sample_img])
            latencies.append((time.perf_counter() - t0) * 1000.0) # ms

        mean_ms = float(np.mean(latencies))
        p95_ms = float(np.percentile(latencies, 95))
        fps = float(1000.0 / mean_ms) if mean_ms > 0 else 0.0

        return {
            "mean_latency_ms": mean_ms,
            "p95_latency_ms": p95_ms,
            "fps": fps,
        }

    def get_info(self) -> Dict[str, Any]:
        """Return model metadata."""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "num_classes": self.num_classes,
            "num_params_m": round(self.num_params / 1e6, 2) if self.num_params > 0 else "N/A",
            "is_loaded": self.is_loaded,
        }
