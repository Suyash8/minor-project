"""
Base Interface for Oriented Bounding Box Detectors.
Defines standardized prediction signatures, latency benchmarking, and metadata extraction.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np
from PIL import Image

class BaseOBBDetector(ABC):
    """
    Abstract base class for all oriented bounding box detectors.
    """

    def __init__(self, model_name: str, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.num_params = 0
        self.is_loaded = False

    @abstractmethod
    def load(self, weights_path: Optional[str] = None) -> None:
        """Load model architecture and weights."""
        pass

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
        Returns:
            list over images, where each image has a list over classes:
            output[img_idx][class_idx] = {
                'boxes': ndarray of shape (N, 5) with [cx, cy, w, h, angle_deg],
                'scores': ndarray of shape (N,)
            }
        """
        pass

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
            "num_params_m": round(self.num_params / 1e6, 2) if self.num_params > 0 else "N/A",
        }
