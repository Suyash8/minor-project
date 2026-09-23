"""
Dataset format converter to standard normalized YOLO-OBB format.
Converts CoDrone, VisDrone, and DOTA into:
    class_idx x1 y1 x2 y2 x3 y3 x4 y4
where all (x_i, y_i) are normalized to [0.0, 1.0] by image width and height.
Generates dynamic dataset YAML configuration files for native Ultralytics training.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from PIL import Image
import yaml

from src.config import DATASET_CLASSES, DEFAULT_DATA_DIR, DATASET_STATIC_PATHS


def normalize_box_corners(pts: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    """Normalize 4 polygon corners to [0.0, 1.0] and clip to image boundaries."""
    norm_pts = pts.copy().astype(np.float32)
    norm_pts[:, 0] = np.clip(norm_pts[:, 0] / float(img_w), 0.0, 1.0)
    norm_pts[:, 1] = np.clip(norm_pts[:, 1] / float(img_h), 0.0, 1.0)
    return norm_pts


def resolve_ultralytics_label_dir(img_dir: Path) -> Path:
    """
    Ultralytics resolves image paths and replaces '/images/' with '/labels/'.
    Find the exact directory Ultralytics will expect labels to reside in.
    """
    real_img = img_dir.resolve()
    parts = list(real_img.parts)
    if "images" in parts:
        idx = len(parts) - 1 - parts[::-1].index("images")
        parts[idx] = "labels"
        target = Path(*parts)
    else:
        target = real_img.parent / "labels" / real_img.name

    # If the target exists as a symlink to another directory (e.g. annfile), unlink it
    if target.is_symlink():
        target.unlink()

    target.mkdir(parents=True, exist_ok=True)
    return target


def convert_visdrone_to_yolo_obb(visdrone_dir: Path) -> Path:
    """
    Convert VisDrone annotations (CSV: left, top, w, h, score, category, trunc, occl)
    into normalized YOLO-OBB format: class_idx x1 y1 x2 y2 x3 y3 x4 y4.
    """
    visdrone_dir = Path(visdrone_dir).resolve()
    classes = DATASET_CLASSES["visdrone"]
    class_to_idx = {c: i for i, c in enumerate(classes)}

    out_yaml = visdrone_dir / "visdrone_yolo_obb.yaml"

    for split in ["train", "val"]:
        static_p = DATASET_STATIC_PATHS["visdrone"].get(split, {})
        direct_img = (visdrone_dir / static_p["images"]) if static_p.get("images") else None
        direct_ann = (visdrone_dir / static_p["labels"]) if static_p.get("labels") else None

        if direct_img and direct_img.exists() and direct_img.is_dir():
            img_dir = direct_img
            ann_dir = direct_ann
        else:
            candidate_img_dirs = [
                visdrone_dir / "images" / split,
                visdrone_dir / split / "images",
                visdrone_dir / f"VisDrone2019-DET-{split}" / "images",
                visdrone_dir / f"VisDrone2019-DET-{split}",
                visdrone_dir / split,
            ]
            img_dir = next((d for d in candidate_img_dirs if d.exists() and d.is_dir()), None)
            if not img_dir:
                continue

            candidate_ann_dirs = [
                visdrone_dir / "annotations" / split,
                visdrone_dir / split / "annotations",
                visdrone_dir / f"VisDrone2019-DET-{split}" / "annotations",
                visdrone_dir / f"VisDrone2019-DET-{split}" / "labels",
                visdrone_dir / "labels" / split,
                visdrone_dir / split,
            ]
            ann_dir = next((d for d in candidate_ann_dirs if d.exists() and d.is_dir()), None)

        out_lbl_dir = resolve_ultralytics_label_dir(img_dir)

        # Also mirror to visdrone/labels/split if separate
        fallback_lbl_dir = visdrone_dir / "labels" / split
        if fallback_lbl_dir != out_lbl_dir and not fallback_lbl_dir.is_symlink():
            fallback_lbl_dir.mkdir(parents=True, exist_ok=True)

        img_files = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
        for img_path in img_files:
            out_lbl_file = out_lbl_dir / f"{img_path.stem}.txt"
            if out_lbl_file.exists() and out_lbl_file.stat().st_size > 0:
                continue

            ann_file = ann_dir / f"{img_path.stem}.txt" if ann_dir.exists() else None
            if not ann_file or not ann_file.exists():
                out_lbl_file.write_text("", encoding="utf-8")
                continue

            try:
                with Image.open(img_path) as img:
                    w_img, h_img = img.size
            except Exception:
                continue

            obb_lines = []
            with open(ann_file, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    # Check if already normalized YOLO-OBB: class_id x1 y1 x2 y2 x3 y3 x4 y4
                    parts_space = line.split()
                    if len(parts_space) == 9:
                        try:
                            c_id = int(parts_space[0])
                            coords = [float(p) for p in parts_space[1:]]
                            if all(0.0 <= c <= 1.05 for c in coords) and 0 <= c_id < len(classes):
                                obb_lines.append(f"{line}\n")
                                continue
                        except ValueError:
                            pass

                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 6:
                        try:
                            x, y, w, h = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
                            score = int(parts[4])
                            cat_id = int(parts[5]) - 1  # 1-indexed in VisDrone

                            # score==1 means considered object; skip ignored classes
                            if score == 1 and 0 <= cat_id < len(classes):
                                pts = np.array([
                                    [x, y],
                                    [x + w, y],
                                    [x + w, y + h],
                                    [x, y + h],
                                ], dtype=np.float32)
                                norm_pts = normalize_box_corners(pts, w_img, h_img)
                                pts_str = " ".join(f"{coord:.6f}" for coord in norm_pts.flatten())
                                obb_lines.append(f"{cat_id} {pts_str}\n")
                        except (ValueError, IndexError):
                            continue

            out_lbl_file.write_text("".join(obb_lines), encoding="utf-8")

    # Generate dataset YAML for Ultralytics
    yaml_dict = {
        "path": str(visdrone_dir),
        "train": DATASET_STATIC_PATHS["visdrone"]["train"]["images"],
        "val": DATASET_STATIC_PATHS["visdrone"]["val"]["images"],
        "names": {i: c for i, c in enumerate(classes)},
    }
    with open(out_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(yaml_dict, f, sort_keys=False)

    return out_yaml


def convert_dota_to_yolo_obb(dota_dir: Path) -> Path:
    """
    Convert DOTA annotations (x1 y1 x2 y2 x3 y3 x4 y4 class_name difficult)
    into normalized YOLO-OBB format: class_idx x1 y1 x2 y2 x3 y3 x4 y4.
    """
    dota_dir = Path(dota_dir).resolve()
    classes = DATASET_CLASSES["dota"]
    class_to_idx = {c: i for i, c in enumerate(classes)}

    out_yaml = dota_dir / "dota_yolo_obb.yaml"

    for split in ["train", "val"]:
        img_dir = dota_dir / "images" / split
        lbl_dir = dota_dir / "labels" / split
        if not img_dir.exists() or not lbl_dir.exists():
            continue

        out_lbl_dir = resolve_ultralytics_label_dir(img_dir)

        img_files = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
        for img_path in img_files:
            out_lbl_file = out_lbl_dir / f"{img_path.stem}.txt"
            if out_lbl_file.exists() and out_lbl_file.stat().st_size > 0:
                continue

            lbl_file = lbl_dir / f"{img_path.stem}.txt"
            if not lbl_file.exists():
                out_lbl_file.write_text("", encoding="utf-8")
                continue

            try:
                with Image.open(img_path) as img:
                    w_img, h_img = img.size
            except Exception:
                continue

            obb_lines = []
            with open(lbl_file, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    # Check if already normalized YOLO-OBB: class_id x1 y1 x2 y2 x3 y3 x4 y4
                    parts_space = line.split()
                    if len(parts_space) == 9:
                        try:
                            c_id = int(parts_space[0])
                            coords = [float(p) for p in parts_space[1:]]
                            if all(0.0 <= c <= 1.05 for c in coords) and 0 <= c_id < len(classes):
                                obb_lines.append(f"{line}\n")
                                continue
                        except ValueError:
                            pass

                    parts = line.split()
                    if len(parts) >= 9 and not line.startswith(("imagesource", "gsd")):
                        try:
                            pts = np.array([float(p) for p in parts[:8]], dtype=np.float32).reshape(4, 2)
                            cls_str = parts[8].lower().replace(" ", "-").replace("_", "-")
                            if cls_str in class_to_idx:
                                cat_id = class_to_idx[cls_str]
                                norm_pts = normalize_box_corners(pts, w_img, h_img)
                                pts_str = " ".join(f"{coord:.6f}" for coord in norm_pts.flatten())
                                obb_lines.append(f"{cat_id} {pts_str}\n")
                        except (ValueError, IndexError):
                            continue

            out_lbl_file.write_text("".join(obb_lines), encoding="utf-8")

    yaml_dict = {
        "path": str(dota_dir),
        "train": DATASET_STATIC_PATHS["dota"]["train"]["images"],
        "val": DATASET_STATIC_PATHS["dota"]["val"]["images"],
        "names": {i: c for i, c in enumerate(classes)},
    }
    with open(out_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(yaml_dict, f, sort_keys=False)

    return out_yaml


def convert_codrone_to_yolo_obb(codrone_dir: Path) -> Path:
    """
    Convert CoDrone annotations into normalized YOLO-OBB format.
    Handles text annotations in labels/ and raw format.
    """
    codrone_dir = Path(codrone_dir).resolve()
    classes = DATASET_CLASSES["codrone"]
    class_to_idx = {c: i for i, c in enumerate(classes)}

    out_yaml = codrone_dir / "codrone_yolo_obb.yaml"

    for split in ["train", "val"]:
        static_p = DATASET_STATIC_PATHS["codrone"].get(split, {})
        direct_img = (codrone_dir / static_p["images"]) if static_p.get("images") else None
        direct_lbl = (codrone_dir / static_p["labels"]) if static_p.get("labels") else None

        if direct_img and direct_img.exists() and direct_img.is_dir():
            img_dir = direct_img
            lbl_dir = direct_lbl
        else:
            candidate_img_dirs = [
                codrone_dir / "images" / split,
                codrone_dir / split / "images",
                codrone_dir / split,
            ]
            img_dir = next((d for d in candidate_img_dirs if d.exists() and d.is_dir()), None)
            if not img_dir:
                continue

            candidate_lbl_dirs = [
                codrone_dir / "labels" / split,
                codrone_dir / split / "labels",
                codrone_dir / split / "annfile",
                codrone_dir / split / "xml_labels",
                codrone_dir / split,
            ]
            lbl_dir = next((d for d in candidate_lbl_dirs if d.exists() and d.is_dir()), None)

        out_lbl_dir = resolve_ultralytics_label_dir(img_dir)

        img_files = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
        for img_path in img_files:
            out_lbl_file = out_lbl_dir / f"{img_path.stem}.txt"
            if out_lbl_file.exists() and out_lbl_file.stat().st_size > 0:
                continue

            lbl_file = lbl_dir / f"{img_path.stem}.txt" if lbl_dir.exists() else None
            if not lbl_file or not lbl_file.exists():
                out_lbl_file.write_text("", encoding="utf-8")
                continue

            try:
                with Image.open(img_path) as img:
                    w_img, h_img = img.size
            except Exception:
                continue

            obb_lines = []
            with open(lbl_file, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    # Check if already normalized YOLO-OBB: class_id x1 y1 x2 y2 x3 y3 x4 y4
                    parts_space = line.split()
                    if len(parts_space) == 9:
                        try:
                            c_id = int(parts_space[0])
                            coords = [float(p) for p in parts_space[1:]]
                            if all(0.0 <= c <= 1.05 for c in coords) and 0 <= c_id < len(classes):
                                obb_lines.append(f"{line}\n")
                                continue
                        except ValueError:
                            pass

                    parts = line.split()
                    if len(parts) >= 9:
                        try:
                            pts = np.array([float(p) for p in parts[:8]], dtype=np.float32).reshape(4, 2)
                            cls_str = parts[8].lower().replace(" ", "-").replace("_", "-")
                            synonyms = {
                                "motor": "motorcyclist",
                                "people": "pedestrian",
                                "bicycle": "cyclist",
                                "traffic-signs": "traffic_sign",
                                "traffic-lights": "traffic_light",
                            }
                            cls_str = synonyms.get(cls_str, cls_str)
                            if cls_str in class_to_idx:
                                cat_id = class_to_idx[cls_str]
                                norm_pts = normalize_box_corners(pts, w_img, h_img)
                                pts_str = " ".join(f"{coord:.6f}" for coord in norm_pts.flatten())
                                obb_lines.append(f"{cat_id} {pts_str}\n")
                        except (ValueError, IndexError):
                            continue

            out_lbl_file.write_text("".join(obb_lines), encoding="utf-8")

    yaml_dict = {
        "path": str(codrone_dir),
        "train": DATASET_STATIC_PATHS["codrone"]["train"]["images"],
        "val": DATASET_STATIC_PATHS["codrone"]["val"]["images"],
        "names": {i: c for i, c in enumerate(classes)},
    }
    with open(out_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(yaml_dict, f, sort_keys=False)

    return out_yaml


def prepare_dataset_for_yolo_training(dataset_name: str, data_dir: Optional[Path] = None) -> Path:
    """
    Ensure dataset is converted and return the path to the YOLO-OBB dataset YAML.
    """
    base_dir = Path(data_dir or DEFAULT_DATA_DIR) / dataset_name
    d_lower = dataset_name.lower()

    if d_lower == "visdrone":
        return convert_visdrone_to_yolo_obb(base_dir)
    elif d_lower == "codrone":
        return convert_codrone_to_yolo_obb(base_dir)
    elif d_lower == "dota":
        return convert_dota_to_yolo_obb(base_dir)
    else:
        raise ValueError(f"Unknown dataset for YOLO-OBB preparation: {dataset_name}")
