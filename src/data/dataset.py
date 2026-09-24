"""
Unified Aerial OBB Dataset Reader.
Parses annotations from YOLO-OBB, DOTA polygon, and VisDrone formats into a standardized internal representation.
"""

from __future__ import annotations

import math
import sys
import zipfile
import xml.etree.ElementTree as ET
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

        # Also search in repository local data/ directory if self.data_dir points to an external mount (e.g. Google Drive)
        repo_data_dir = Path(__file__).resolve().parent.parent.parent / "data" / self.dataset_name
        if repo_data_dir.exists() and repo_data_dir.resolve() != self.data_dir.resolve():
            candidate_lbl_dirs.extend([
                repo_data_dir / f"VisDrone2019-DET-{self.split}" / "annotations",
                repo_data_dir / f"VisDrone2019-DET-{self.split}" / "labels",
                repo_data_dir / "annotations" / self.split,
                repo_data_dir / self.split / "annfile",
                repo_data_dir / "labels" / self.split,
                repo_data_dir / self.split / "labels",
            ])

        # Filter and prioritize directories containing actual non-empty annotation files
        seen_dirs = set()
        valid_lbl_dirs = []
        for d in candidate_lbl_dirs:
            resolved_d = d.resolve() if d.exists() else None
            if resolved_d and resolved_d.is_dir() and str(resolved_d) not in seen_dirs:
                seen_dirs.add(str(resolved_d))
                txt_count = sum(1 for p in resolved_d.glob("*.txt") if p.is_file() and p.stat().st_size > 0)
                xml_count = sum(1 for p in resolved_d.glob("*.xml") if p.is_file() and p.stat().st_size > 0)
                total_valid = txt_count + xml_count
                if total_valid > 0:
                    valid_lbl_dirs.append((resolved_d, total_valid))

        # Sort candidate directories with the most valid non-empty files first
        valid_lbl_dirs.sort(key=lambda item: item[1], reverse=True)
        active_lbl_dirs = [item[0] for item in valid_lbl_dirs]

        # Dynamic fallback: if candidate paths had no non-empty labels, scan self.data_dir recursively
        if len(active_lbl_dirs) == 0 and len(all_imgs) > 0:
            sample_stems = set(img.stem for img in all_imgs[:10])
            for p in self.data_dir.rglob("*.txt"):
                if p.stem in sample_stems and p.stat().st_size > 0:
                    cand_d = p.parent
                    if cand_d not in active_lbl_dirs:
                        active_lbl_dirs.append(cand_d)

            # Auto-extract bundled annotations from assets/ if available
            if len(active_lbl_dirs) == 0:
                repo_root = Path(__file__).resolve().parent.parent.parent
                bundled_zip = repo_root / "assets" / f"{self.dataset_name}_annotations.zip"
                if bundled_zip.exists():
                    print(f"[*] Auto-extracting bundled {self.dataset_name} annotations from {bundled_zip.name} into {self.data_dir}...")
                    try:
                        with zipfile.ZipFile(bundled_zip, "r") as z:
                            z.extractall(self.data_dir)
                        for d in candidate_lbl_dirs:
                            if d.exists() and d.is_dir():
                                if any(p.is_file() and p.stat().st_size > 0 for p in d.glob("*.txt")) or any(p.is_file() and p.stat().st_size > 0 for p in d.glob("*.xml")):
                                    if d.resolve() not in [p.resolve() for p in active_lbl_dirs]:
                                        active_lbl_dirs.append(d)
                    except Exception as e:
                        print(f"[!] Failed to extract bundled annotations: {e}")

            # Auto-extract missing annotations from any present zip archives if needed
            if len(active_lbl_dirs) == 0:
                for zip_path in sorted(list(self.data_dir.glob("*.zip")) + list(self.data_dir.parent.glob("*.zip"))):
                    if self.split in zip_path.name.lower() or self.dataset_name in zip_path.name.lower():
                        try:
                            with zipfile.ZipFile(zip_path, "r") as z:
                                ann_members = [
                                    m for m in z.namelist()
                                    if ("annotation" in m.lower() or "annfile" in m.lower() or "label" in m.lower())
                                    and (m.endswith(".txt") or m.endswith(".xml"))
                                ]
                                if ann_members:
                                    print(f"[*] Auto-extracting missing annotations from {zip_path.name}...")
                                    z.extractall(self.data_dir, members=ann_members)
                                    for m in ann_members:
                                        cand_dir = (self.data_dir / m).parent
                                        if cand_dir.exists() and cand_dir not in active_lbl_dirs:
                                            active_lbl_dirs.append(cand_dir)
                        except Exception:
                            pass

        for img_path in all_imgs:
            lbl_path = None
            stem = img_path.stem
            # 1. Prefer non-empty label file
            for ld in active_lbl_dirs:
                for ext in (".txt", ".xml"):
                    candidate = ld / f"{stem}{ext}"
                    if candidate.exists() and candidate.is_file() and candidate.stat().st_size > 0:
                        lbl_path = candidate
                        break
                if lbl_path is not None:
                    break

            # 2. Fallback to any matching candidate file
            if lbl_path is None:
                for ld in active_lbl_dirs:
                    for ext in (".txt", ".xml"):
                        candidate = ld / f"{stem}{ext}"
                        if candidate.exists() and candidate.is_file():
                            lbl_path = candidate
                            break
                    if lbl_path is not None:
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

        # Handle XML annotations (Pascal VOC / CoDrone robndbox XML)
        if lbl_path.suffix.lower() == ".xml":
            try:
                tree = ET.parse(lbl_path)
                root = tree.getroot()
                for obj in root.findall("object"):
                    name_elem = obj.find("name")
                    if name_elem is None or not name_elem.text:
                        continue
                    raw_cls = name_elem.text.strip().lower()
                    raw_cls_underscore = raw_cls.replace("-", "_")
                    raw_cls_hyphen = raw_cls.replace("_", "-")
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

                    if cls_idx is None or cls_idx >= num_classes:
                        continue

                    bnd = obj.find("bndbox")
                    if bnd is not None:
                        if bnd.find("x0") is not None:
                            try:
                                pts = np.array([
                                    [float(bnd.find("x0").text), float(bnd.find("y0").text)],
                                    [float(bnd.find("x1").text), float(bnd.find("y1").text)],
                                    [float(bnd.find("x2").text), float(bnd.find("y2").text)],
                                    [float(bnd.find("x3").text), float(bnd.find("y3").text)],
                                ], dtype=np.float32)
                                cx, cy, w, h, angle = polygon_to_obb_params(pts)
                                gt_by_class[cls_idx]["boxes"].append([cx, cy, w, h, angle])
                            except (ValueError, TypeError, AttributeError):
                                pass
                        elif bnd.find("xmin") is not None:
                            try:
                                xmin = float(bnd.find("xmin").text)
                                ymin = float(bnd.find("ymin").text)
                                xmax = float(bnd.find("xmax").text)
                                ymax = float(bnd.find("ymax").text)
                                cx = (xmin + xmax) / 2.0
                                cy = (ymin + ymax) / 2.0
                                w = max(xmax - xmin, 1.0)
                                h = max(ymax - ymin, 1.0)
                                gt_by_class[cls_idx]["boxes"].append([cx, cy, w, h, 0.0])
                            except (ValueError, TypeError, AttributeError):
                                pass
            except Exception:
                pass

            for c in range(num_classes):
                if len(gt_by_class[c]["boxes"]) > 0:
                    gt_by_class[c]["boxes"] = np.array(gt_by_class[c]["boxes"], dtype=np.float32)
                else:
                    gt_by_class[c]["boxes"] = np.zeros((0, 5), dtype=np.float32)
            return gt_by_class

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
