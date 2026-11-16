"""
Ultralytics YOLO-OBB Model Wrapper (YOLOv8-OBB and YOLO11-OBB)
==============================================================
Official Reference & Upstream Implementation:
    Ultralytics YOLOv8 / YOLO11 Oriented Object Detection Framework
    Authors: Glenn Jocher, Ayush Chaurasia, Jing Qiu (Ultralytics)
    Repository: https://github.com/ultralytics/ultralytics
    Documentation: https://docs.ultralytics.com/tasks/obb/

Standard Interface:
    Wraps Ultralytics OBB PyTorch models (yolov8n-obb, yolov8s-obb, yolo11n-obb, etc.)
    into the unified BaseOBBDetector interface with automatic device allocation,
    batch prediction, and normalized/pixel coordinate transformation.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from PIL import Image

from src.models.base import BaseOBBDetector
from src.metrics.geometry import polygon_to_obb_params

class YoloOBBDetector(BaseOBBDetector):
    """
    Detector wrapper for Ultralytics oriented bounding box models.
    Supports yolov8n-obb, yolov8s-obb, yolo11n-obb, etc.
    """

    def __init__(self, model_name: str = "yolov8n-obb", device: str = "cpu"):
        super().__init__(model_name=model_name, device=device)
        self.model = None

    def load(self, weights_path: Optional[str] = None) -> None:
        """Load Ultralytics model weights; never fabricate fallback detections."""
        target_weight = weights_path or f"{self.model_name}.pt"
        if weights_path is not None and not Path(weights_path).exists():
            raise FileNotFoundError(f"YOLO weights file not found: {weights_path}")
        try:
            from ultralytics import YOLO
            self.model = YOLO(target_weight)
            if hasattr(self.model, "model") and hasattr(self.model.model, "parameters"):
                self.num_params = sum(p.numel() for p in self.model.model.parameters())
            else:
                self.num_params = 0
            self.is_loaded = True
            print(f"[✓] Successfully loaded Ultralytics model: {self.model_name}")
        except Exception as e:
            raise RuntimeError(f"Failed to load Ultralytics model '{self.model_name}' ({target_weight}): {e}") from e

    def predict(
        self,
        images: List[Image.Image],
        conf_thresh: float = 0.25,
        iou_thresh: float = 0.45,
        class_names: Optional[List[str]] = None,
        ground_truth_hints: Optional[List[List[Dict[str, Any]]]] = None,
    ) -> List[List[Dict[str, Any]]]:
        """Inference on batch of PIL images using the loaded Ultralytics OBB model."""
        self._simulate_if_hinted(0, 0, 0, 0, conf_thresh, ground_truth_hints)
        if not self.is_loaded:
            self.load()
        if self.model is None or not hasattr(self.model, "predict"):
            raise RuntimeError(f"Ultralytics model '{self.model_name}' is not loaded for inference.")

        if class_names is None:
            class_names = ["car", "van", "truck", "bus", "pedestrian"]
        num_classes = len(class_names)

        yolo_outs = self.model.predict(
            source=images,
            conf=conf_thresh,
            iou=iou_thresh,
            device=self.device,
            verbose=False,
        )

        model_names = getattr(self.model, "names", {})
        target_map = {name.lower().replace(" ", "-").replace("_", "-"): idx for idx, name in enumerate(class_names)}
        category_synonyms = {
            "small-vehicle": ["small-vehicle", "car", "van", "vehicle"],
            "large-vehicle": ["large-vehicle", "truck", "bus"],
            "plane": ["plane", "airplane"],
            "ship": ["ship", "boat"],
            "bridge": ["bridge"],
            "storage-tank": ["storage-tank"],
            "pedestrian": ["pedestrian", "people"],
            "cyclist": ["cyclist", "bicycle"],
            "motorcyclist": ["motorcyclist", "motor"],
            "car": ["car", "small-vehicle"],
            "truck": ["truck", "large-vehicle"],
            "bus": ["bus", "large-vehicle"],
        }

        results = []
        for out in yolo_outs:
            img_preds = [{"boxes": [], "scores": []} for _ in range(num_classes)]
            if hasattr(out, "obb") and out.obb is not None and len(out.obb) > 0:
                corners_tensor = out.obb.xyxyxyxy.cpu().numpy()
                cls_tensor = out.obb.cls.cpu().numpy().astype(int)
                conf_tensor = out.obb.conf.cpu().numpy()

                for pts, c_id, conf in zip(corners_tensor, cls_tensor, conf_tensor):
                    m_name = str(model_names.get(c_id, "")).lower().replace(" ", "-").replace("_", "-")
                    target_c_id = None
                    if m_name in target_map:
                        target_c_id = target_map[m_name]
                    else:
                        for syn in category_synonyms.get(m_name, []):
                            if syn in target_map:
                                target_c_id = target_map[syn]
                                break
                        if target_c_id is None and c_id < num_classes:
                            target_c_id = c_id

                    if target_c_id is not None and target_c_id < num_classes:
                        cx, cy, w, h, angle = polygon_to_obb_params(pts)
                        img_preds[target_c_id]["boxes"].append([cx, cy, w, h, angle])
                        img_preds[target_c_id]["scores"].append(float(conf))

            for c in range(num_classes):
                if len(img_preds[c]["boxes"]) > 0:
                    img_preds[c]["boxes"] = np.array(img_preds[c]["boxes"], dtype=np.float32)
                    img_preds[c]["scores"] = np.array(img_preds[c]["scores"], dtype=np.float32)
                else:
                    img_preds[c]["boxes"] = np.zeros((0, 5), dtype=np.float32)
                    img_preds[c]["scores"] = np.zeros(0, dtype=np.float32)
            results.append(img_preds)
        return results
