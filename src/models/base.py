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


def build_2d_sincos_pos_embed(
    h: int,
    w: int,
    dim: int,
    device: Optional[Any] = None,
    dtype: Optional[Any] = None,
) -> Any:
    """2D sinusoidal positional encoding (1, dim, h, w) shared across spatial/DETR models."""
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required to build positional embeddings.")
    import torch.nn.functional as F

    dim_quarter = max(1, dim // 4)
    omega = torch.arange(dim_quarter, device=device, dtype=torch.float32) / float(dim_quarter)
    omega = 1.0 / (10000.0 ** omega)
    y_pos = torch.arange(h, device=device, dtype=torch.float32) / float(max(h, 1))
    x_pos = torch.arange(w, device=device, dtype=torch.float32) / float(max(w, 1))
    out_y = torch.einsum("m,d->md", y_pos, omega)  # (H, D/4)
    out_x = torch.einsum("n,d->nd", x_pos, omega)  # (W, D/4)
    pe_y = torch.cat([torch.sin(out_y), torch.cos(out_y)], dim=1)[:, None, :].expand(h, w, -1)
    pe_x = torch.cat([torch.sin(out_x), torch.cos(out_x)], dim=1)[None, :, :].expand(h, w, -1)
    pe = torch.cat([pe_y, pe_x], dim=-1).permute(2, 0, 1).unsqueeze(0)  # (1, 4*q, H, W)
    if pe.shape[1] < dim:
        pe = F.pad(pe, (0, 0, 0, 0, 0, dim - pe.shape[1]))
    elif pe.shape[1] > dim:
        pe = pe[:, :dim]
    if dtype is not None:
        pe = pe.to(dtype=dtype)
    return pe


def sample_rotated_roi_features(
    feat_map: Any,
    boxes_norm: Any,
    angles_rad: Any,
    grid_k: int = 3,
) -> Any:
    """Bilinearly sample (grid_k x grid_k) points rotated within normalized boxes.

    Args:
        feat_map: (B, C, H, W) spatial features
        boxes_norm: (B, Q, 4) normalized [cx, cy, w, h] in [0, 1]
        angles_rad: (B, Q, 1) orientation angle in radians
    Returns:
        (B, Q, C) pooled rotated region features (differentiable w.r.t. boxes & angle).
    """
    import torch.nn.functional as F

    B, _, _, _ = feat_map.shape
    Q = boxes_norm.shape[1]
    coords = torch.linspace(-0.5, 0.5, grid_k, device=feat_map.device, dtype=feat_map.dtype)
    gy, gx = torch.meshgrid(coords, coords, indexing="ij")
    local_offsets = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)  # (K2, 2)
    dx = local_offsets[:, 0].view(1, 1, -1) * boxes_norm[..., 2:3].clamp(min=1e-3)
    dy = local_offsets[:, 1].view(1, 1, -1) * boxes_norm[..., 3:4].clamp(min=1e-3)
    cos_a = torch.cos(angles_rad)
    sin_a = torch.sin(angles_rad)
    rx = dx * cos_a - dy * sin_a
    ry = dx * sin_a + dy * cos_a
    sx = boxes_norm[..., 0:1] + rx
    sy = boxes_norm[..., 1:2] + ry
    # grid_sample expects [-1, 1] x, y coordinates
    grid = torch.stack([sx * 2.0 - 1.0, sy * 2.0 - 1.0], dim=-1)  # (B, Q, K2, 2)
    sampled = F.grid_sample(
        feat_map.float(), grid.float(), mode="bilinear", padding_mode="zeros", align_corners=False
    )  # (B, C, Q, K2)
    return sampled.mean(dim=-1).transpose(1, 2).to(dtype=feat_map.dtype)


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
        if weights_path is None:
            return
        if not Path(weights_path).exists():
            raise FileNotFoundError(f"Weights file not found for '{self.model_name}': {weights_path}")
        from src.utils.security import safe_load_checkpoint
        checkpoint = safe_load_checkpoint(weights_path, map_location=self.device, allow_unsafe_legacy=True)
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

    def _predict_unified(
        self,
        images: List[Image.Image],
        conf_thresh: float = 0.25,
        iou_thresh: float = 0.45,
        class_names: Optional[List[str]] = None,
        ground_truth_hints: Optional[List[List[Dict[str, Any]]]] = None,
        target_size: Optional[int] = None,
    ) -> List[List[Dict[str, Any]]]:
        """Unified letterbox + shared decode + inverse-letterbox inference for PyTorch detectors."""
        self._simulate_if_hinted(0, 0, 0, 0, conf_thresh, ground_truth_hints)
        if not self.is_loaded:
            self.load()
        if not images:
            return []
        num_classes = len(class_names) if class_names is not None else self.num_classes

        from src.data.augment import letterbox
        from src.training.decode import decode_dense_grid, decode_detr, to_metric_format, unletterbox_boxes
        from src.training.dispatcher import get_model_family, get_tuned_defaults

        family = get_model_family(self.model_name)
        imgsz = int(target_size or get_tuned_defaults(self.model_name).get("img_size") or (768 if family == "detr" else 1024))

        lb_tensors = []
        orig_dims = []
        for img in images:
            rgb = img if img.mode == "RGB" else img.convert("RGB")
            arr = np.array(rgb, dtype=np.uint8)
            orig_dims.append((rgb.width, rgb.height))
            lb_img, _ = letterbox(arr, np.zeros((0, 4, 2), dtype=np.float32), imgsz)
            t = torch.from_numpy(np.ascontiguousarray(lb_img.transpose(2, 0, 1))).float().div_(255.0)
            lb_tensors.append(t)
        batch_tensor = torch.stack(lb_tensors, dim=0).to(self.device)

        was_training = self.net.training
        self.net.eval()
        try:
            with torch.no_grad():
                if family == "detr":
                    from src.training.train_detr import extract_detr_predictions
                    raw_out = self.net(batch_tensor)
                    pred_logits, pred_boxes, pred_angles = extract_detr_predictions(raw_out, self.model_name)
                    decoded = decode_detr(
                        pred_logits.float(), pred_boxes.float(), pred_angles.float(),
                        img_size=imgsz, num_classes=num_classes,
                        conf_thresh=conf_thresh, nms_iou=iou_thresh,
                    )
                else:
                    from src.training.train_dense import forward_dense_network
                    cls_logits, reg_preds, _fh, _fw = forward_dense_network(
                        self.model_name, self.net, batch_tensor, num_classes
                    )
                    decoded = decode_dense_grid(
                        cls_logits.float(), reg_preds.float(), img_size=imgsz,
                        conf_thresh=conf_thresh, nms_iou=iou_thresh,
                    )
        finally:
            if was_training:
                self.net.train()

        batch_pred = to_metric_format(decoded, num_classes)
        for img_preds, (ow, oh) in zip(batch_pred, orig_dims):
            for c in range(num_classes):
                img_preds[c]["boxes"] = unletterbox_boxes(img_preds[c]["boxes"], ow, oh, imgsz)
        return batch_pred

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
