"""
Unified Aerial OBB Dataset Reader.
Parses annotations from YOLO-OBB, DOTA polygon, and VisDrone formats into a standardized internal representation.
"""

import math
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from PIL import Image

from src.config import DATASET_CLASSES

def polygon_to_obb_params(pts: np.ndarray) -> Tuple[float, float, float, float, float]:
    """
    Fit a minimum-area rotated rectangle to 4 corner points.
    Returns (cx, cy, w, h, angle_deg).
    Angle is in degrees clockwise relative to positive x-axis.
    """
    cx = float(np.mean(pts[:, 0]))
    cy = float(np.mean(pts[:, 1]))

    # Edge vectors
    v0 = pts[1] - pts[0]
    v1 = pts[2] - pts[1]

    len0 = float(np.linalg.norm(v0))
    len1 = float(np.linalg.norm(v1))

    # Long edge determines primary orientation
    if len0 >= len1:
        w = len0
        h = max(len1, 1.0)
        angle_rad = math.atan2(v0[1], v0[0])
    else:
        w = len1
        h = max(len0, 1.0)
        angle_rad = math.atan2(v1[1], v1[0])

    angle_deg = math.degrees(angle_rad) % 180.0
    return cx, cy, w, h, angle_deg

class AerialOBBDataset:
    """
    Dataset loader for aerial OBB imagery.
    Loads images and parses ground truth labels across formats.
    """

    def __init__(
        self,
        dataset_name: str,
        data_dir: Path,
        split: str = "val",
        max_samples: Optional[int] = None,
    ):
        self.dataset_name = dataset_name.lower()
        self.data_dir = Path(data_dir) / self.dataset_name
        self.split = split
        self.class_names = DATASET_CLASSES.get(self.dataset_name, ["car", "van", "truck", "bus", "pedestrian"])
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}

        self.samples = []
        self._discover_samples(max_samples)

    def _discover_samples(self, max_samples: Optional[int] = None) -> None:
        """Find image and label file pairs."""
        img_dir = self.data_dir / "images" / self.split
        lbl_dir = self.data_dir / "labels" / self.split

        if not img_dir.exists():
            # Fallback search
            img_dir = self.data_dir / self.split / "images"
            lbl_dir = self.data_dir / self.split / "labels"
        if not img_dir.exists():
            img_dir = self.data_dir / self.split
            lbl_dir = self.data_dir / self.split

        if not img_dir.exists():
            return

        all_imgs = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
        if max_samples:
            all_imgs = all_imgs[:max_samples]

        for img_path in all_imgs:
            lbl_path = lbl_dir / f"{img_path.stem}.txt" if lbl_dir.exists() else None
            self.samples.append({
                "image_path": img_path,
                "label_path": lbl_path if (lbl_path and lbl_path.exists()) else None,
            })

    def __len__(self) -> int:
        return len(self.samples)

    def get_ground_truth(self, idx: int, img_width: int, img_height: int) -> List[Dict[str, Any]]:
        """
        Parse annotations for sample at idx into per-class boxes:
        gt[class_idx] = {'boxes': ndarray of [cx, cy, w, h, angle_deg]}
        """
        num_classes = len(self.class_names)
        gt_by_class = [{"boxes": []} for _ in range(num_classes)]

        if idx >= len(self.samples):
            return [{"boxes": np.zeros((0, 5), dtype=np.float32)} for _ in range(num_classes)]

        lbl_path = self.samples[idx]["label_path"]
        if not lbl_path or not lbl_path.exists():
            return [{"boxes": np.zeros((0, 5), dtype=np.float32)} for _ in range(num_classes)]

        with open(lbl_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Format 1: VisDrone CSV format (left, top, width, height, score, category, trunc, occl)
            if "," in line:
                subparts = [p.strip() for p in line.split(",")]
                if len(subparts) >= 6:
                    try:
                        x, y, w, h = float(subparts[0]), float(subparts[1]), float(subparts[2]), float(subparts[3])
                        score = int(subparts[4])
                        cls_idx = int(subparts[5]) - 1 # VisDrone categories are 1-indexed
                        # Skip ignored classes (cls_idx == -1 or score == 0)
                        if score == 1 and 0 <= cls_idx < num_classes:
                            cx = x + w / 2.0
                            cy = y + h / 2.0
                            gt_by_class[cls_idx]["boxes"].append([cx, cy, w, h, 0.0])
                    except (ValueError, IndexError):
                        continue
                continue

            parts = line.split()

            # Format 2: YOLO-OBB (class_idx x1 y1 x2 y2 x3 y3 x4 y4 normalized)
            # Exactly 9 parts, with no alphabetical class name in parts[8]
            if len(parts) == 9 and not any(ch.isalpha() or ch == "-" for ch in parts[8]):
                try:
                    cls_idx = int(parts[0])
                    if cls_idx < num_classes:
                        pts = np.array([float(p) for p in parts[1:]]).reshape(4, 2)
                        pts[:, 0] *= img_width
                        pts[:, 1] *= img_height
                        cx, cy, w, h, angle = polygon_to_obb_params(pts)
                        gt_by_class[cls_idx]["boxes"].append([cx, cy, w, h, angle])
                except (ValueError, IndexError):
                    pass

            # Format 3: DOTA / CODrone (x1 y1 x2 y2 x3 y3 x4 y4 class_name [difficult])
            elif len(parts) >= 9:
                try:
                    pts = np.array([float(p) for p in parts[:8]]).reshape(4, 2)
                    raw_cls = parts[8].lower()
                    raw_cls_underscore = raw_cls.replace("-", "_")
                    raw_cls_hyphen = raw_cls.replace("_", "-")

                    # Normalize category synonyms
                    synonyms = {
                        "people": "pedestrian",
                        "motor": "motorcyclist",
                        "bicycle": "cyclist",
                        "traffic_signs": "traffic_sign",
                        "traffic_lights": "traffic_light",
                        "traffic-signs": "traffic-sign",
                        "traffic-lights": "traffic-light",
                    }
                    norm_cls = synonyms.get(raw_cls, raw_cls)
                    norm_cls_underscore = synonyms.get(raw_cls_underscore, raw_cls_underscore)

                    cls_idx = None
                    for candidate in [raw_cls, raw_cls_underscore, raw_cls_hyphen, norm_cls, norm_cls_underscore]:
                        if candidate in self.class_to_idx:
                            cls_idx = self.class_to_idx[candidate]
                            break

                    if cls_idx is not None and cls_idx < num_classes:
                        cx, cy, w, h, angle = polygon_to_obb_params(pts)
                        gt_by_class[cls_idx]["boxes"].append([cx, cy, w, h, angle])
                except (ValueError, IndexError):
                    continue


        # Convert to numpy arrays
        for c in range(num_classes):
            if len(gt_by_class[c]["boxes"]) > 0:
                gt_by_class[c]["boxes"] = np.array(gt_by_class[c]["boxes"], dtype=np.float32)
            else:
                gt_by_class[c]["boxes"] = np.zeros((0, 5), dtype=np.float32)

        return gt_by_class

    def get_image(self, idx: int) -> Image.Image:
        """Load and return PIL Image."""
        img_path = self.samples[idx]["image_path"]
        return Image.open(img_path).convert("RGB")
