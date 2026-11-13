"""
Unified Aerial OBB Dataset Reader.
Parses annotations from YOLO-OBB, DOTA polygon, and VisDrone formats into a standardized internal representation.
"""

from __future__ import annotations

import math
import os
import pickle
import shutil
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from PIL import Image

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from src.config import DATASET_CLASSES, DATASET_STATIC_PATHS, CLEANED_DATASET_CLASSES, cleaned_data_root


def _annotation_fingerprint(label_paths: List[Path]) -> Tuple[int, int, int]:
    """Cheap cache-invalidation fingerprint: file count + total size + max mtime.
    Any label edit, add or delete changes at least one component."""
    n = 0
    total = 0
    max_mtime = 0
    for p in label_paths:
        try:
            st = p.stat()
        except OSError:
            continue
        n += 1
        total += st.st_size
        max_mtime = max(max_mtime, st.st_mtime_ns)
    return n, total, max_mtime

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
        use_cache: bool = True,
        ram_cache: bool = False,
        disk_cache_dir: Optional[str] = None,
        workers: int = 1,
    ):
        self.dataset_name = dataset_name.lower()
        self.data_dir = Path(data_dir) / self.dataset_name
        self.split = split
        # If the requested data_dir does not actually contain this dataset,
        # transparently fall back to the cleaned-data root (v4.1 build) so all
        # engines train on the same data without manual path plumbing.
        if not self.data_dir.is_dir():
            cleaned_root = cleaned_data_root()
            candidate = cleaned_root / self.dataset_name
            if candidate.is_dir():
                self.data_dir = candidate
        self.class_names = DATASET_CLASSES.get(self.dataset_name, ["car", "van", "truck", "bus", "pedestrian"])
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}

        # v4: optional read resolver for overlapped staging (training/staging.py).
        # When set, every file read is routed through it so a not-yet-staged
        # local file is transparently read from the source (Drive) instead.
        self.resolver = None
        # v4: annotation cache — parse each label file at most once per
        # fingerprint instead of re-parsing text on every epoch.
        self.use_cache = use_cache
        self._ann_cache: Optional[List[Optional[Dict[str, Any]]]] = None
        # W3.7 `--cache ram`: decoded images cached in RAM, bounded by a
        # fraction of currently-free memory (psutil when available, sysconf
        # otherwise) divided by DataLoader workers so multiprocessing never OOMs.
        self.ram_cache = ram_cache
        self._img_cache: Dict[int, np.ndarray] = {}
        self._img_cache_bytes = 0
        self._img_cache_budget = self._ram_budget_bytes(workers) if ram_cache else 0
        # W3.7 `--cache disk` (review: must be a REAL tier, not an alias of
        # `annotations`): decoded image arrays persist as .npy on fast local
        # disk (the staging root), so later epochs/models skip both the
        # source read AND the JPEG decode. Atomic tmp+rename writes; writes
        # stop when free disk falls below 2 GiB.
        self.disk_cache_dir = (Path(disk_cache_dir) / self.dataset_name / split
                               if disk_cache_dir else None)
        if self.disk_cache_dir is not None:
            self.disk_cache_dir.mkdir(parents=True, exist_ok=True)

        self.samples = []
        self._discover_samples(max_samples)
        if self.use_cache:
            self._ann_cache = self._load_or_build_cache()

    def set_resolver(self, resolver) -> None:
        """Attach a training/staging resolver (overlapped Drive->NVMe reads)."""
        self.resolver = resolver

    def sample_relpaths(self) -> List[str]:
        """Image paths of all samples, relative to the dataset root.

        Used to feed the staging queue the exact per-epoch consumption order
        (W2.2): queue relpaths are relative to <src_root>/<dataset>, which is
        exactly self.data_dir.
        """
        out: List[str] = []
        for s in self.samples:
            try:
                out.append(os.path.relpath(str(s["image_path"]),
                                           str(self.data_dir)))
            except (ValueError, KeyError):
                continue
        return out

    def _resolve_read_path(self, path: Path) -> Path:
        if self.resolver is None:
            return path
        return Path(self.resolver.resolve(str(path)))

    # ------------------------------------------------------------------
    # annotation cache (v4, W3): parse once per fingerprint, mirroring
    # Ultralytics' labels.cache behaviour for the dense/DETR engines.
    # ------------------------------------------------------------------
    def _cache_path(self) -> Optional[Path]:
        lbl_dirs = {s["label_path"].parent for s in self.samples if s.get("label_path")}
        if len(lbl_dirs) == 1:
            return lbl_dirs.pop() / "labels.cache"
        return self.data_dir / f"labels_{self.split}.cache"

    def _load_or_build_cache(self) -> Optional[List[Optional[Dict[str, Any]]]]:
        label_paths = [s["label_path"] for s in self.samples if s.get("label_path")]
        if not label_paths:
            return None
        fp = _annotation_fingerprint(label_paths)
        cache_path = self._cache_path()
        if cache_path and cache_path.exists():
            try:
                with open(cache_path, "rb") as f:
                    blob = pickle.load(f)
                if blob.get("fingerprint") == fp and len(blob.get("entries", [])) == len(self.samples):
                    return blob["entries"]
            except Exception:
                pass  # corrupt or stale cache: rebuild below

        entries: List[Optional[Dict[str, Any]]] = []
        for sample in self.samples:
            lbl = sample.get("label_path")
            if lbl is None:
                entries.append(None)
                continue
            entries.append(self._parse_label_file(self._resolve_read_path(lbl)))

        if cache_path:
            try:
                tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
                with open(tmp, "wb") as f:
                    pickle.dump({"fingerprint": fp, "entries": entries}, f,
                                protocol=pickle.HIGHEST_PROTOCOL)
                os.replace(tmp, cache_path)
            except OSError:
                pass  # read-only directory (e.g. Drive): caching just skips
        return entries

    def _parse_label_file(self, lbl_path: Path) -> Dict[str, Any]:
        """Parse one label file into cacheable form.

        Returns {'fmt': 'corners_norm' | 'cxcywha', 'items': [[cls, ...], ...]}:
        - corners_norm: YOLO-OBB normalized corner polygons (need image dims
          at read time to recover pixel geometry).
        - cxcywha: absolute-pixel (cx, cy, w, h, angle_deg) from XML / VisDrone
          CSV / DOTA polygon formats.
        """
        items_corners: List[List[float]] = []
        items_abs: List[List[float]] = []

        if lbl_path.suffix.lower() == ".xml":
            self._parse_xml(lbl_path, items_abs)
            return {"fmt": "cxcywha", "items": items_abs}

        try:
            with open(lbl_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
        except OSError:
            return {"fmt": "cxcywha", "items": []}

        synonyms = {
            "people": "pedestrian",
            "motor": "motorcyclist",
            "bicycle": "cyclist",
            "traffic_signs": "traffic_sign",
            "traffic_lights": "traffic_light",
            "traffic-signs": "traffic-sign",
            "traffic-lights": "traffic-light",
        }

        def _cls_idx(raw: str) -> Optional[int]:
            raw = raw.lower()
            for cand in (raw, raw.replace("-", "_"), raw.replace("_", "-"),
                         synonyms.get(raw, raw),
                         synonyms.get(raw.replace("-", "_"), raw.replace("-", "_"))):
                if cand in self.class_to_idx:
                    return self.class_to_idx[cand]
            return None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # VisDrone CSV (left, top, w, h, score, category, trunc, occl)
            if "," in line:
                sub = [p.strip() for p in line.split(",")]
                if len(sub) >= 6:
                    try:
                        x, y, w, h = (float(sub[i]) for i in range(4))
                        score = int(sub[4])
                        cls_idx = int(sub[5]) - 1
                        if score == 1 and 0 <= cls_idx < len(self.class_names):
                            items_abs.append([cls_idx, x + w / 2.0, y + h / 2.0,
                                              max(w, 1.0), max(h, 1.0), 0.0])
                    except (ValueError, IndexError):
                        continue
                continue

            parts = line.split()

            # YOLO-OBB (cls x1 y1 x2 y2 x3 y3 x4 y4 normalized)
            if len(parts) == 9 and not any(ch.isalpha() or ch == "-" for ch in parts[8]):
                try:
                    cls_idx = int(parts[0])
                    if 0 <= cls_idx < len(self.class_names):
                        coords = [float(p) for p in parts[1:]]
                        items_corners.append([cls_idx] + coords)
                except (ValueError, IndexError):
                    pass

            # DOTA / CODrone polygon (x1 y1 ... x4 y4 class_name [difficult])
            elif len(parts) >= 9:
                try:
                    pts = np.array([float(p) for p in parts[:8]]).reshape(4, 2)
                    cls_idx = _cls_idx(parts[8])
                    if cls_idx is not None and cls_idx < len(self.class_names):
                        cx, cy, w, h, angle = polygon_to_obb_params(pts)
                        items_abs.append([cls_idx, cx, cy, w, h, angle])
                except (ValueError, IndexError):
                    continue

        if items_corners:
            return {"fmt": "corners_norm", "items": items_corners}
        return {"fmt": "cxcywha", "items": items_abs}

    def _parse_xml(self, lbl_path: Path, items_abs: List[List[float]]) -> None:
        try:
            root = ET.parse(lbl_path).getroot()
        except Exception:
            return
        synonyms = {
            "people": "pedestrian",
            "motor": "motorcyclist",
            "bicycle": "cyclist",
            "traffic_signs": "traffic_sign",
            "traffic_lights": "traffic_light",
            "traffic-signs": "traffic-sign",
            "traffic-lights": "traffic-light",
        }
        for obj in root.findall("object"):
            name_elem = obj.find("name")
            if name_elem is None or not name_elem.text:
                continue
            raw = name_elem.text.strip().lower()
            cls_idx = None
            for cand in (raw, raw.replace("-", "_"), raw.replace("_", "-"),
                         synonyms.get(raw, raw),
                         synonyms.get(raw.replace("-", "_"), raw.replace("-", "_"))):
                if cand in self.class_to_idx:
                    cls_idx = self.class_to_idx[cand]
                    break
            if cls_idx is None or cls_idx >= len(self.class_names):
                continue
            bnd = obj.find("bndbox")
            if bnd is None:
                continue
            try:
                if bnd.find("x0") is not None:
                    pts = np.array([[
                        float(bnd.find(f"x{i}").text), float(bnd.find(f"y{i}").text)]
                        for i in range(4)], dtype=np.float32)
                    cx, cy, w, h, angle = polygon_to_obb_params(pts)
                    items_abs.append([cls_idx, cx, cy, w, h, angle])
                elif bnd.find("xmin") is not None:
                    xmin = float(bnd.find("xmin").text)
                    ymin = float(bnd.find("ymin").text)
                    xmax = float(bnd.find("xmax").text)
                    ymax = float(bnd.find("ymax").text)
                    items_abs.append([cls_idx, (xmin + xmax) / 2.0, (ymin + ymax) / 2.0,
                                      max(xmax - xmin, 1.0), max(ymax - ymin, 1.0), 0.0])
            except (ValueError, TypeError, AttributeError):
                pass

    def _discover_samples(self, max_samples: Optional[int] = None) -> None:
        """Find image and label file pairs directly using static dataset paths."""
        static_info = DATASET_STATIC_PATHS.get(self.dataset_name, {}).get(self.split)
        img_dir = None
        lbl_dir = None
        using_cleaned_layout = False

        if static_info:
            # Static paths may be a priority list (cleaned layout first, raw
            # fallbacks after) or a legacy single dict entry.
            layouts = static_info if isinstance(static_info, list) else [static_info]
            for entry in layouts:
                direct_img = self.data_dir / entry["images"]
                direct_lbl = (self.data_dir / entry["labels"]) if entry.get("labels") else None
                if direct_img.exists() and direct_img.is_dir():
                    img_dir = direct_img
                    lbl_dir = direct_lbl
                    using_cleaned_layout = entry.get("layout") == "cleaned"
                    break

        if using_cleaned_layout:
            # The cleaned v4.1 build emits a single unified 8-class space in
            # YOLO-OBB polygon format; re-target class names so the dense/DETR
            # engines train on exactly the same labels as the Ultralytics suite.
            cleaned_names = CLEANED_DATASET_CLASSES.get(self.dataset_name)
            if cleaned_names:
                self.class_names = list(cleaned_names)
                self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}

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
        Annotations for sample idx as per-class boxes:
        gt[class_idx] = {'boxes': ndarray of [cx, cy, w, h, angle_deg]}

        v4: served from the fingerprinted annotation cache when enabled, so
        label text is parsed once per dataset version, not once per epoch.
        """
        num_classes = len(self.class_names)
        empty = [{"boxes": np.zeros((0, 5), dtype=np.float32)} for _ in range(num_classes)]
        if idx >= len(self.samples):
            return empty

        if self._ann_cache is not None:
            parsed = self._ann_cache[idx]
        else:
            lbl_path = self.samples[idx]["label_path"]
            if not lbl_path:
                return empty
            resolved = self._resolve_read_path(lbl_path)
            if not resolved.exists():
                return empty
            parsed = self._parse_label_file(resolved)

        if parsed is None:
            return empty

        gt_by_class = [{"boxes": []} for _ in range(num_classes)]
        if parsed["fmt"] == "corners_norm":
            for item in parsed["items"]:
                cls_idx = int(item[0])
                if cls_idx >= num_classes:
                    continue
                pts = np.array(item[1:], dtype=np.float32).reshape(4, 2)
                pts[:, 0] *= img_width
                pts[:, 1] *= img_height
                cx, cy, w, h, angle = polygon_to_obb_params(pts)
                gt_by_class[cls_idx]["boxes"].append([cx, cy, w, h, angle])
        else:
            for item in parsed["items"]:
                cls_idx = int(item[0])
                if cls_idx < num_classes:
                    gt_by_class[cls_idx]["boxes"].append(item[1:])

        for c in range(num_classes):
            if len(gt_by_class[c]["boxes"]) > 0:
                gt_by_class[c]["boxes"] = np.array(gt_by_class[c]["boxes"], dtype=np.float32)
            else:
                gt_by_class[c]["boxes"] = np.zeros((0, 5), dtype=np.float32)
        return gt_by_class

    @staticmethod
    def _available_system_ram_bytes() -> int:
        try:
            import psutil
            return int(psutil.virtual_memory().available)
        except Exception:
            try:
                return int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
            except Exception:
                return 0

    @classmethod
    def _ram_budget_bytes(cls, workers: int = 1) -> int:
        """RAM-cache budget: bounded safe fraction (<=20% total) divided across workers."""
        avail = cls._available_system_ram_bytes()
        if avail <= 0:
            return 0
        w = max(1, workers)
        return int((avail * 0.20) / w)

    def get_image_array(self, idx: int) -> np.ndarray:
        """Load sample idx as an (H, W, 3) uint8 RGB array.

        OpenCV decode (cv2.setNumThreads(0) is applied inside DataLoader
        workers) with a PIL fallback. Reads go through the staging resolver
        when one is attached, so a not-yet-staged local file is read from the
        source mount instead of failing. With ram_cache on, decoded arrays
        are memoized up to the free-memory budget.
        """
        if self.ram_cache and idx in self._img_cache:
            return self._img_cache[idx]
        disk_path = None
        if self.disk_cache_dir is not None:
            stem = Path(self.samples[idx]["image_path"]).stem
            disk_path = self.disk_cache_dir / f"{stem}.npy"
            if disk_path.is_file():
                try:
                    return np.load(disk_path)
                except Exception as e:  # corrupt cache entry: rebuild loud
                    print(f"[dataset][warn] disk cache entry unreadable "
                          f"({disk_path}): {e}; decoding from source",
                          file=sys.stderr)
        img_path = self._resolve_read_path(self.samples[idx]["image_path"])
        if HAS_CV2:
            arr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if arr is None:
                raise FileNotFoundError(f"could not decode image: {img_path}")
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        else:
            arr = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
        if self.ram_cache and self._img_cache_budget > 0:
            nbytes = int(arr.nbytes)
            if self._img_cache_bytes + nbytes <= self._img_cache_budget:
                # Dynamic system memory safety check: never allow free memory to drop below 4 GiB
                if self._available_system_ram_bytes() > 4 * (1024 ** 3):
                    self._img_cache[idx] = arr
                    self._img_cache_bytes += nbytes
        if disk_path is not None:
            try:
                self.disk_cache_dir.mkdir(parents=True, exist_ok=True)
                free = shutil.disk_usage(str(self.disk_cache_dir)).free
                if free > 2 * 1024 ** 3:
                    tmp = disk_path.with_name(f"{disk_path.stem}.{os.getpid()}.part.npy")
                    np.save(tmp, arr)
                    os.replace(tmp, disk_path)
            except OSError as e:
                print(f"[dataset][warn] disk cache write failed "
                      f"({disk_path}): {e}", file=sys.stderr)
        return arr

    def get_image(self, idx: int) -> Image.Image:
        """Load and return a PIL Image (kept for the detector wrappers)."""
        return Image.fromarray(self.get_image_array(idx))
