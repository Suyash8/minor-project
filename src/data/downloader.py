"""
Dataset Download & Ingestion Manager for VisDrone, CODrone, and DOTA.
Handles automated retrieval, archive extraction, validation, and manual download guidance.
"""

from __future__ import annotations

import os
import sys
import shutil
import zipfile
import tarfile
from pathlib import Path
from typing import Dict, Any, Optional
import requests

from src.config import DATASET_CLASSES, DATASET_STATIC_PATHS

DATASET_METADATA: Dict[str, Dict[str, Any]] = {
    "visdrone": {
        "name": "VisDrone2019-DET",
        "description": "Multi-class drone imagery captured across urban China.",
        "auto_downloadable": True,
        "direct_urls": {
            "val": "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-val.zip",
            "train": "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-train.zip",
        },
        "manual_guide": "Download VisDrone2019-DET from https://github.com/VisDrone/VisDrone-Dataset",
    },
    "codrone": {
        "name": "CODrone (2025)",
        "description": "4K UHD oriented bounding box UAV benchmark with 30° oblique tilt and altitude shifts.",
        "auto_downloadable": False,
        "repo_url": "https://github.com/AHideoKuzeA/CODrone",
        "manual_guide": (
            "CODrone is hosted via Google Drive and Baidu Wangpan (~40GB for full 4K frames).\n"
            "1. Visit the official repository: https://github.com/AHideoKuzeA/CODrone\n"
            "2. Download the validation/test zip archives from their provided Google Drive link.\n"
            "3. Extract into: <data_dir>/codrone/ (with 'images' and 'labels' subdirectories)."
        ),
    },
    "dota": {
        "name": "DOTA (v1.0 / v2.0)",
        "description": "Large-scale dataset for oriented object detection in aerial images.",
        "auto_downloadable": False,
        "repo_url": "https://captain-whu.github.io/DOTA/dataset.html",
        "manual_guide": (
            "DOTA datasets are hosted on Baidu NetDisk and Google Drive.\n"
            "1. Register and visit: https://captain-whu.github.io/DOTA/dataset.html\n"
            "2. Download val images ('valimages.zip') and annotations ('val-label-oriented.zip').\n"
            "3. Extract into: <data_dir>/dota/."
        ),
    },
}

def download_file_with_progress(url: str, dest_path: Path) -> bool:
    """
    Download a remote file over HTTP/HTTPS with streaming and size reporting.
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[*] Downloading: {url}")
    print(f"[*] Saving to: {dest_path}")

    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))

        downloaded = 0
        chunk_size = 1024 * 1024 # 1 MB chunks

        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        pct = downloaded / total_size * 100.0
                        sys.stdout.write(f"\r    Progress: {downloaded / (1024 * 1024):.1f} MB / {total_size / (1024 * 1024):.1f} MB ({pct:.1f}%)")
                    else:
                        sys.stdout.write(f"\r    Progress: {downloaded / (1024 * 1024):.1f} MB")
                    sys.stdout.flush()

        print(f"\n[✓] Successfully downloaded: {dest_path.name}")
        return True
    except Exception as e:
        print(f"\n[!] Download failed: {e}", file=sys.stderr)
        if dest_path.exists():
            dest_path.unlink()
        return False

def extract_archive(archive_path: Path, extract_to: Path) -> bool:
    """
    Extract a .zip, .tar.gz, or .tgz archive into the destination folder.
    """
    extract_to.mkdir(parents=True, exist_ok=True)
    print(f"[*] Extracting archive: {archive_path.name} -> {extract_to}")

    try:
        if archive_path.suffix.lower() == ".zip":
            with zipfile.ZipFile(archive_path, "r") as zip_ref:
                zip_ref.extractall(extract_to)
        elif archive_path.name.endswith(".tar.gz") or archive_path.suffix.lower() == ".tgz":
            with tarfile.open(archive_path, "r:gz") as tar_ref:
                tar_ref.extractall(extract_to)
        else:
            print(f"[!] Unsupported archive format: {archive_path.suffix}", file=sys.stderr)
            return False

        print(f"[✓] Successfully extracted to: {extract_to}")
        return True
    except Exception as e:
        print(f"[!] Extraction failed: {e}", file=sys.stderr)
        return False

def verify_dataset_status(dataset_name: str, data_dir: Path) -> Dict[str, Any]:
    """
    Check if a dataset is present, valid, and count available images.
    """
    target_dir = Path(data_dir) / dataset_name.lower()
    meta = DATASET_METADATA.get(dataset_name.lower(), {})

    status = {
        "dataset": dataset_name,
        "exists": target_dir.exists(),
        "path": str(target_dir),
        "num_images": 0,
        "splits_found": [],
        "ready": False,
        "manual_guide": meta.get("manual_guide", "No guide available."),
    }

    if not target_dir.exists():
        return status

    # Fast check using explicit static dataset paths
    static_splits = DATASET_STATIC_PATHS.get(dataset_name.lower(), {})
    for split in ["val", "test", "train"]:
        count = 0
        if split in static_splits:
            rel_img = static_splits[split]["images"]
            static_p = target_dir / rel_img
            if static_p.exists() and static_p.is_dir():
                count = len([p for p in static_p.iterdir() if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")])

        # Dynamic fallback if not found at static path
        if count == 0:
            candidates = [
                target_dir / "images" / split,
                target_dir / split / "images",
                target_dir / f"VisDrone2019-DET-{split}" / "images",
                target_dir / f"VisDrone2019-DET-{split}",
                target_dir / "yolo_obb" / "images" / split,
                target_dir / split,
            ]
            for cand in candidates:
                if cand.exists() and cand.is_dir():
                    imgs = [p for p in cand.iterdir() if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")]
                    if len(imgs) > count:
                        count = len(imgs)

        if count > 0:
            status["splits_found"].append(split)
            status["num_images"] += count

    status["ready"] = status["num_images"] > 0
    return status

def download_dataset(dataset_name: str, data_dir: Path, force: bool = False) -> bool:
    """
    Automated download workflow for a requested dataset.
    """
    d_name = dataset_name.lower()
    if d_name not in DATASET_METADATA:
        print(f"[!] Unknown dataset: {dataset_name}. Supported: {list(DATASET_METADATA.keys())}", file=sys.stderr)
        return False

    meta = DATASET_METADATA[d_name]
    status = verify_dataset_status(d_name, data_dir)

    if status["ready"] and not force:
        print(f"[✓] Dataset '{d_name}' is already present and ready with {status['num_images']} images.")
        return True

    target_dir = Path(data_dir) / d_name
    target_dir.mkdir(parents=True, exist_ok=True)

    # Check if local archives exist in target_dir that need extraction
    local_zips = list(target_dir.glob("*.zip")) + list(target_dir.glob("*.tar.gz"))
    if local_zips and not status["ready"]:
        for z in local_zips:
            print(f"[*] Found local archive '{z.name}' in {target_dir}. Extracting...")
            extract_archive(z, target_dir)
        status = verify_dataset_status(d_name, data_dir)
        if status["ready"]:
            print(f"[✓] Successfully unpacked local archives for {d_name} ({status['num_images']} images ready).")
            return True

    if not meta.get("auto_downloadable", False):
        print(f"[*] Dataset '{d_name}' requires manual download due to hosting restrictions (Baidu / Google Drive).")
        print("\n" + "=" * 65)
        print(f" MANUAL DOWNLOAD INSTRUCTIONS FOR: {meta['name'].upper()}")
        print("=" * 65)
        print(meta["manual_guide"])
        print("=" * 65 + "\n")
        return False

    # Automated download flow (e.g. for VisDrone)
    direct_urls = meta.get("direct_urls", {})
    all_success = True
    for split, url in direct_urls.items():
        archive_name = Path(url).name
        archive_path = target_dir / archive_name
        success = download_file_with_progress(url, archive_path)
        if success:
            extract_success = extract_archive(archive_path, target_dir)
            if extract_success:
                try:
                    archive_path.unlink() # Cleanup archive
                except OSError:
                    pass
            else:
                all_success = False
        else:
            all_success = False

    return all_success
