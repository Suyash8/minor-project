
from __future__ import annotations
#!/usr/bin/env python3
"""
Standalone Dataset Downloader & Verifier for VisDrone, CODrone, and DOTA.
"""

import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEFAULT_DATA_DIR, SUPPORTED_DATASETS
from src.data.downloader import download_dataset, verify_dataset_status

def main():
    parser = argparse.ArgumentParser(description="Download and verify aerial datasets.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["all"],
        help=f"Datasets to download/verify. Choices: {SUPPORTED_DATASETS} or 'all'",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(DEFAULT_DATA_DIR),
        help="Dataset destination root directory.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check status and print manual instructions without downloading.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force redownload even if dataset already exists.",
    )

    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    target_datasets = SUPPORTED_DATASETS if "all" in args.datasets else args.datasets

    print("==================================================")
    print(" Aerial Dataset Manager: Download & Verification")
    print("==================================================")
    print(f"[*] Target Directory: {data_dir}\n")

    for d_name in target_datasets:
        print(f"--- [{d_name.upper()}] ---")
        status = verify_dataset_status(d_name, data_dir)
        print(f"  Status        : {'Ready' if status['ready'] else 'Missing / Incomplete'}")
        print(f"  Images Found  : {status['num_images']}")
        print(f"  Splits Found  : {status['splits_found']}")

        if args.check_only:
            if not status["ready"]:
                print("  Download Guide:")
                print("  " + status["manual_guide"].replace("\n", "\n  "))
            continue

        if not status["ready"] or args.force:
            print(f"[*] Starting download procedure for {d_name}...")
            success = download_dataset(d_name, data_dir, force=args.force)
            if success:
                print(f"[✓] {d_name} download and setup complete!")
            else:
                print(f"[!] {d_name} automated setup could not complete automatically.")
        else:
            print(f"[✓] Dataset {d_name} is already present and validated.")
        print()

if __name__ == "__main__":
    main()
