"""
Unified Aerial OBB Dataset Reader.
Parses annotations from YOLO-OBB, DOTA polygon, and VisDrone formats into a standardized internal representation.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from PIL import Image

from src.config import DATASET_CLASSES, DATASET_STATIC_PATHS

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
        """Find image and label file pairs directly using static dataset paths."""
        static_info = DATASET_STATIC_PATHS.get(self.dataset_name, {}).get(self.split)
        img_dir = None
        lbl_dir = None

        if static_info:
            direct_img = self.data_dir / static_info["images"]
            direct_lbl = (self.data_dir / static_info["labels"]) if static_info.get("labels") else None
            if direct_img.exists() and direct_img.is_dir():
                img_dir = direct_img
                lbl_dir = direct_lbl

        # Fallback dynamic search if static directory is not found
        if not img_dir:
            candidate_img_dirs = [
                self.data_dir / "images" / self.split,
                self.data_dir / self.split / "images",
                self.data_dir / f"VisDrone2019-DET-{self.split}" / "images",
                self.data_dir / f"VisDrone2019-DET-{self.split}",
                self.data_dir / "yolo_obb" / "images" / self.split,
                self.data_dir / self.split,
            ]
            best_imgs = []
            for d in candidate_img_dirs:
                if d.exists() and d.is_dir():
                    imgs = [p for p in d.iterdir() if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")]
                    if len(imgs) > len(best_imgs):
                        best_imgs = imgs
                        img_dir = d
            if not img_dir:
                return
            all_imgs = sorted(best_imgs)
        else:
            all_imgs = sorted([p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")])

        if max_samples:
            all_imgs = all_imgs[:max_samples]

        # Comprehensive candidate label directories
        candidate_lbl_dirs = []
        if lbl_dir and lbl_dir.exists() and lbl_dir.is_dir():
            candidate_lbl_dirs.append(lbl_dir)

        candidate_lbl_dirs.extend([
            self.data_dir / "labels" / self.split,
            self.data_dir / self.split / "labels",
            self.data_dir / "labelTxt" / self.split,
            self.data_dir / self.split / "labelTxt",
            self.data_dir / "labels-v1.5" / self.split,
            self.data_dir / self.split / "labels-v1.5",
            self.data_dir / "labelTxt-v1.5" / self.split,
            self.data_dir / self.split / "labelTxt-v1.5",
            self.data_dir / self.split / "annfile",
            self.data_dir / "annfile" / self.split,
            self.data_dir / self.split / "xml_labels",
            self.data_dir / f"VisDrone2019-DET-{self.split}" / "annotations",
            self.data_dir / f"VisDrone2019-DET-{self.split}" / "labels",
            self.data_dir / "annotations" / self.split,
            self.data_dir / "yolo_obb" / "labels" / self.split,
            self.data_dir / self.split,
        ])

        # Filter and prioritize directories containing actual .txt annotation files
        seen_dirs = set()
        valid_lbl_dirs = []
        for d in candidate_lbl_dirs:
            resolved_d = d.resolve() if d.exists() else None
            if resolved_d and resolved_d.is_dir() and str(resolved_d) not in seen_dirs:
                seen_dirs.add(str(resolved_d))
                txt_count = len(list(resolved_d.glob("*.txt")))
                if txt_count > 0:
                    valid_lbl_dirs.append((resolved_d, txt_count))

        # Sort candidate directories with the most .txt files first
        valid_lbl_dirs.sort(key=lambda item: item[1], reverse=True)
        active_lbl_dirs = [item[0] for item in valid_lbl_dirs]

        for img_path in all_imgs:
            lbl_path = None
            stem = img_path.stem
            for ld in active_lbl_dirs:
                candidate = ld / f"{stem}.txt"
                if candidate.exists() and candidate.is_file():
                    lbl_path = candidate
                    break
            self.samples.append({
                "image_path": img_path,
                "label_path": lbl_path,
            })

        labeled_count = sum(1 for s in self.samples if s["label_path"] is not None)
        if labeled_count == 0 and len(self.samples) > 0 and self.split != "test":
            print(
                f"[!] WARNING: Found {len(self.samples)} images for '{self.dataset_name}' ({self.split}) "
                f"but 0 label files across candidate directories!",
                file=sys.stderr,
            )

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
