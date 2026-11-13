#!/usr/bin/env python3
"""
Fast Local Staging Script (Drive FUSE -> Fast Local NVMe/SSD)
=============================================================
Copies selected cleaned datasets from Google Drive to local fast storage
(/content/data) in parallel, dramatically speeding up training epochs.

Usage:
  # 1. Use the configured list at the top of this file:
  python scripts/stage_data.py

  # 2. Or override via CLI flags:
  python scripts/stage_data.py --datasets visdrone
  python scripts/stage_data.py --datasets visdrone codrone
"""

from __future__ import annotations

import os
import sys
import time
import shutil
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==============================================================================
# CONFIGURATION ARRAY: Comment out / edit the datasets you want to stage.
# If no CLI arguments are passed, this list is used by default.
# ==============================================================================
DATASETS_TO_STAGE = [
    "visdrone",
    # "codrone",
    # "dota",
]

DEFAULT_DRIVE_SRC = Path("/content/drive/MyDrive/object-detection/cleaned_data")
DEFAULT_LOCAL_DST = Path("/content/data")


def stage_single_dataset(
    ds_name: str,
    src_root: Path,
    dst_root: Path,
    workers: int = 16,
    force: bool = False,
) -> bool:
    """
    Stage one dataset from src_root to dst_root using multi-threaded copying.
    Idempotent: skips if destination already contains files (unless force=True).
    """
    src_dir = src_root / ds_name
    dst_dir = dst_root / ds_name

    if not src_dir.is_dir():
        print(f"[!] Source dataset directory not found: {src_dir}")
        print(f"    Check that Google Drive is mounted at /content/drive and cleaned_data exists.")
        return False

    # Check if already staged (idempotency check)
    if dst_dir.is_dir() and not force:
        existing_files = list(dst_dir.glob("*/*/*.*"))
        if len(existing_files) > 50:
            print(f"[✓] '{ds_name}' is already staged at {dst_dir} ({len(existing_files)} files). Skipping.")
            return True

    print(f"[*] Staging '{ds_name.upper()}' from Drive to fast local NVMe: {dst_dir} ...")
    files_to_copy = []
    for root, _, files in os.walk(src_dir):
        for f in files:
            files_to_copy.append(Path(root) / f)

    if not files_to_copy:
        print(f"[!] No files found inside {src_dir}")
        return False

    dst_dir.mkdir(parents=True, exist_ok=True)

    def _copy_worker(src_file: Path):
        rel_path = src_file.relative_to(src_dir)
        target = dst_dir / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_file, target)

    completed = 0
    total = len(files_to_copy)
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_copy_worker, f) for f in files_to_copy]
        for f in as_completed(futures):
            completed += 1
            if completed % 250 == 0 or completed == total:
                elapsed = time.time() - start_time
                pct = completed / total * 100
                speed = completed / elapsed if elapsed > 0 else 0
                remaining = (total - completed) / speed if speed > 0 else 0
                sys.stdout.write(
                    f"\r    [{ds_name.upper()}] {completed}/{total} files ({pct:.1f}%) | "
                    f"{speed:.1f} files/s | ETA: {remaining:.0f}s"
                )
                sys.stdout.flush()

    print(f"\n[✓] Successfully staged '{ds_name}': {len(files_to_copy)} files copied in {time.time() - start_time:.1f}s.")
    return True


def stage_datasets(
    datasets: list[str] | None = None,
    src_root: Path | str = DEFAULT_DRIVE_SRC,
    dst_root: Path | str = DEFAULT_LOCAL_DST,
    workers: int = 16,
    force: bool = False,
) -> str:
    """
    Stage a list of datasets and return the resolved data root directory.
    Configures VEHICLE_OBB_DATA environment variable accordingly.
    """
    targets = datasets if datasets is not None else DATASETS_TO_STAGE
    src_path = Path(src_root).resolve()
    dst_path = Path(dst_root).resolve()

    print("=================================================================")
    print(" Fast NVMe Dataset Staging for Aerial OBB Detection")
    print("=================================================================")
    print(f"[*] Target datasets: {targets}")
    print(f"[*] Source (Drive):  {src_path}")
    print(f"[*] Dest (Local NVMe): {dst_path}\n")

    staged_any = False
    for ds in targets:
        ds_clean = ds.strip().lower()
        if not ds_clean:
            continue
        success = stage_single_dataset(
            ds_name=ds_clean,
            src_root=src_path,
            dst_root=dst_path,
            workers=workers,
            force=force,
        )
        if success:
            staged_any = True

    if staged_any and dst_path.is_dir() and any(dst_path.iterdir()):
        resolved_root = str(dst_path)
        os.environ["VEHICLE_OBB_DATA"] = resolved_root
        print("\n" + "=" * 65)
        print(f"[🚀 SPEED BOOST ACTIVE] Fast local NVMe data root:")
        print(f"   export VEHICLE_OBB_DATA={resolved_root}")
        print("=" * 65)
    else:
        resolved_root = str(src_path)
        os.environ["VEHICLE_OBB_DATA"] = resolved_root
        print("\n" + "=" * 65)
        print(f"[📁 DRIVE FALLBACK ACTIVE] Falling back to Google Drive:")
        print(f"   export VEHICLE_OBB_DATA={resolved_root}")
        print("=" * 65)

    return resolved_root


def main():
    parser = argparse.ArgumentParser(
        description="Stage aerial datasets from Google Drive to fast local NVMe disk.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Datasets to stage (e.g. visdrone codrone dota). If omitted, uses DATASETS_TO_STAGE array from script.",
    )
    parser.add_argument(
        "--src",
        type=str,
        default=str(DEFAULT_DRIVE_SRC),
        help="Source directory containing cleaned datasets (usually Google Drive).",
    )
    parser.add_argument(
        "--dst",
        type=str,
        default=str(DEFAULT_LOCAL_DST),
        help="Destination directory on fast local disk (e.g. /content/data).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Number of concurrent worker threads for file copying.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-copying even if destination dataset already exists.",
    )

    args = parser.parse_args()

    stage_datasets(
        datasets=args.datasets,
        src_root=args.src,
        dst_root=args.dst,
        workers=args.workers,
        force=args.force,
    )


if __name__ == "__main__":
    main()
