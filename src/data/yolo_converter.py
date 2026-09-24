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
import zipfile
import xml.etree.ElementTree as ET
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

    # Auto-extract bundled annotations if missing
    train_ann = visdrone_dir / "VisDrone2019-DET-train" / "annotations"
    val_ann = visdrone_dir / "VisDrone2019-DET-val" / "annotations"
    has_train = train_ann.exists() and any(p.stat().st_size > 0 for p in train_ann.glob("*.txt"))
    has_val = val_ann.exists() and any(p.stat().st_size > 0 for p in val_ann.glob("*.txt"))
    if not has_train or not has_val:
        repo_root = Path(__file__).resolve().parent.parent.parent
        bundled_zip = repo_root / "assets" / "visdrone_annotations.zip"
        if bundled_zip.exists():
            print(f"[*] Extracting bundled VisDrone annotations from {bundled_zip.name} into {visdrone_dir}...")
            with zipfile.ZipFile(bundled_zip, "r") as z:
                z.extractall(visdrone_dir)
            print(f"[✓] VisDrone annotations extracted successfully.")

    for split in ["train", "val"]:
        static_p = DATASET_STATIC_PATHS["visdrone"].get(split, {})
        direct_img = (visdrone_dir / static_p["images"]) if static_p.get("images") else None
        direct_ann = (visdrone_dir / static_p["labels"]) if static_p.get("labels") else None

        img_dir = None
        ann_dir = None

        if direct_img and direct_img.exists() and direct_img.is_dir():
            img_dir = direct_img
            if direct_ann and direct_ann.exists() and any(p.stat().st_size > 0 for p in direct_ann.glob("*.txt")):
                ann_dir = direct_ann

        if not img_dir:
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

        if not ann_dir:
            candidate_ann_dirs = [
                visdrone_dir / f"VisDrone2019-DET-{split}" / "annotations",
                visdrone_dir / "annotations" / split,
                visdrone_dir / split / "annotations",
                visdrone_dir / "labels" / split,
                visdrone_dir / split / "labels",
                visdrone_dir / split,
            ]
            for cand in candidate_ann_dirs:
                if cand.exists() and cand.is_dir() and any(p.stat().st_size > 0 for p in cand.glob("*.txt")):
                    ann_dir = cand
                    break

        out_lbl_dir = resolve_ultralytics_label_dir(img_dir)

        # Also mirror to visdrone/labels/split if separate
        fallback_lbl_dir = visdrone_dir / "labels" / split
        if fallback_lbl_dir != out_lbl_dir and not fallback_lbl_dir.is_symlink():
            fallback_lbl_dir.mkdir(parents=True, exist_ok=True)

        img_files = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
        total_imgs = len(img_files)

        for idx, img_path in enumerate(img_files, 1):
            if idx % 1000 == 0 or idx == total_imgs or idx == 1:
                print(f"      [VisDrone {split}] Converting: {idx}/{total_imgs} images ({(idx/total_imgs)*100:.1f}%)...", flush=True)

            out_lbl_file = out_lbl_dir / f"{img_path.stem}.txt"
            if out_lbl_file.exists() and out_lbl_file.stat().st_size > 0:
                continue

            ann_file = ann_dir / f"{img_path.stem}.txt" if ann_dir and ann_dir.exists() else None
            if not ann_file or not ann_file.exists():
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

            if obb_lines:
                lbl_text = "".join(obb_lines)
                out_lbl_file.write_text(lbl_text, encoding="utf-8")
                if fallback_lbl_dir != out_lbl_dir and fallback_lbl_dir.exists():
                    (fallback_lbl_dir / f"{img_path.stem}.txt").write_text(lbl_text, encoding="utf-8")

        # Invalidate any stale Ultralytics labels.cache so it indexes the new annotations
        for cache_f in list(out_lbl_dir.parent.glob("*.cache")) + list(out_lbl_dir.glob("*.cache")):
            try:
                cache_f.unlink()
            except Exception:
                pass

    # Generate dataset YAML for Ultralytics
    train_rel = DATASET_STATIC_PATHS["visdrone"]["train"]["images"]
    if not (visdrone_dir / train_rel).exists() and (visdrone_dir / "images" / "train").exists():
        train_rel = "images/train"

    val_rel = DATASET_STATIC_PATHS["visdrone"]["val"]["images"]
    if not (visdrone_dir / val_rel).exists() and (visdrone_dir / "images" / "val").exists():
        val_rel = "images/val"

    yaml_dict = {
        "path": str(visdrone_dir),
        "train": train_rel,
        "val": val_rel,
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

            lbl_file = lbl_dir / f"{img_path.stem}.txt" if lbl_dir and lbl_dir.exists() else None
            if not lbl_file or not lbl_file.exists():
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

            if obb_lines:
                out_lbl_file.write_text("".join(obb_lines), encoding="utf-8")

    train_rel = DATASET_STATIC_PATHS["dota"]["train"]["images"]
    if not (dota_dir / train_rel).exists() and (dota_dir / "images" / "train").exists():
        train_rel = "images/train"

    val_rel = DATASET_STATIC_PATHS["dota"]["val"]["images"]
    if not (dota_dir / val_rel).exists() and (dota_dir / "images" / "val").exists():
        val_rel = "images/val"

    yaml_dict = {
        "path": str(dota_dir),
        "train": train_rel,
        "val": val_rel,
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

            lbl_file = lbl_dir / f"{img_path.stem}.txt" if lbl_dir and lbl_dir.exists() else None
            xml_file = lbl_dir / f"{img_path.stem}.xml" if lbl_dir and lbl_dir.exists() else None
            if not xml_file or not xml_file.exists():
                alt_xml = codrone_dir / split / "xml_labels" / f"{img_path.stem}.xml"
                if alt_xml.exists():
                    xml_file = alt_xml

            has_txt = lbl_file is not None and lbl_file.exists() and lbl_file.stat().st_size > 0
            has_xml = xml_file is not None and xml_file.exists() and xml_file.stat().st_size > 0

            if not has_txt and not has_xml:
                continue

            try:
                with Image.open(img_path) as img:
                    w_img, h_img = img.size
            except Exception:
                continue

            obb_lines = []
            synonyms = {
                "motor": "motorcyclist",
                "people": "pedestrian",
                "bicycle": "cyclist",
                "traffic-signs": "traffic_sign",
                "traffic-lights": "traffic_light",
            }

            if has_txt:
                with open(lbl_file, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue

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
                                cls_str = synonyms.get(cls_str, cls_str)
                                if cls_str in class_to_idx:
                                    cat_id = class_to_idx[cls_str]
                                    norm_pts = normalize_box_corners(pts, w_img, h_img)
                                    pts_str = " ".join(f"{coord:.6f}" for coord in norm_pts.flatten())
                                    obb_lines.append(f"{cat_id} {pts_str}\n")
                            except (ValueError, IndexError):
                                continue

            elif has_xml:
                try:
                    tree = ET.parse(xml_file)
                    root = tree.getroot()
                    for obj in root.findall("object"):
                        name_el = obj.find("name")
                        if name_el is None or not name_el.text:
                            continue
                        cls_str = name_el.text.strip().lower().replace(" ", "-").replace("_", "-")
                        cls_str = synonyms.get(cls_str, cls_str)
                        if cls_str not in class_to_idx:
                            continue
                        cat_id = class_to_idx[cls_str]

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
                                    norm_pts = normalize_box_corners(pts, w_img, h_img)
                                    pts_str = " ".join(f"{coord:.6f}" for coord in norm_pts.flatten())
                                    obb_lines.append(f"{cat_id} {pts_str}\n")
                                except (ValueError, TypeError, AttributeError):
                                    pass
                            elif bnd.find("xmin") is not None:
                                try:
                                    xmin = float(bnd.find("xmin").text)
                                    ymin = float(bnd.find("ymin").text)
                                    xmax = float(bnd.find("xmax").text)
                                    ymax = float(bnd.find("ymax").text)
                                    pts = np.array([
                                        [xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]
                                    ], dtype=np.float32)
                                    norm_pts = normalize_box_corners(pts, w_img, h_img)
                                    pts_str = " ".join(f"{coord:.6f}" for coord in norm_pts.flatten())
                                    obb_lines.append(f"{cat_id} {pts_str}\n")
                                except (ValueError, TypeError, AttributeError):
                                    pass
                except Exception:
                    pass

            if obb_lines:
                out_lbl_file.write_text("".join(obb_lines), encoding="utf-8")

    train_rel = DATASET_STATIC_PATHS["codrone"]["train"]["images"]
    if not (codrone_dir / train_rel).exists() and (codrone_dir / "images" / "train").exists():
        train_rel = "images/train"

    val_rel = DATASET_STATIC_PATHS["codrone"]["val"]["images"]
    if not (codrone_dir / val_rel).exists() and (codrone_dir / "images" / "val").exists():
        val_rel = "images/val"

    yaml_dict = {
        "path": str(codrone_dir),
        "train": train_rel,
        "val": val_rel,
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
