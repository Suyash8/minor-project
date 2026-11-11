#!/usr/bin/env python3
"""
colab_quickstart.py — ONE-CELL Colab bootstrap for the training pipeline.

Paste into a Colab GPU runtime (T4/L4/A100) and edit the TRAIN_ARGS line:

    !git clone https://github.com/Suyash8/minor-project.git /content/minor-project   # or upload the zip
    %cd /content/minor-project
    !python training/colab_quickstart.py --datasets visdrone --models yolo26s-obb

It mounts Drive, verifies the cleaned datasets exist (points you to the
cleaner if not), installs requirements, and hands over to training/train.py.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "training"))
import common  # noqa: E402


def in_colab() -> bool:
    return "google.colab" in sys.modules or os.path.isdir("/content/drive")


def mount_drive():
    try:
        from google.colab import drive          # type: ignore
        drive.mount("/content/drive")
        return True
    except Exception:
        return False


def check_cleaned_data(data_root: str) -> None:
    problems = []
    for ds in common.REAL_DATASETS:
        for sp in common.SPLITS:
            d = os.path.join(data_root, ds, sp, "images")
            if not os.path.isdir(d):
                problems.append(d)
    if problems:
        print("!! cleaned datasets incomplete at", data_root)
        for p in problems:
            print("   missing:", p)
        print("""
Fix (in order):
  1. upload assets/dota_labels_all.zip from this repo to
     /content/drive/MyDrive/object-detection/ and extract it into data/dota
     (see data_cleaning/README.md, 'Run it' section)
  2. run the FULL cleaner:
       import os; os.environ['CLEAN_TEST_RUN'] = '0'
       %run data_cleaning/clean_build_v4.py
     (wait for 'copied N files to Drive' — the finish flag)
""")
        sys.exit(1)
    print("cleaned datasets OK at", data_root)


def main():
    if in_colab():
        print("== Colab detected ==")
        if not os.path.isdir("/content/drive/MyDrive"):
            mount_drive()
        try:
            gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                                  "--format=csv,noheader"],
                                 capture_output=True, text=True, timeout=15)
            print("GPU:", gpu.stdout.strip() or "NONE (use a GPU runtime!)")
        except Exception:
            print("no nvidia-smi — running on CPU (slow; pick a GPU runtime)")

        missing = []
        try:
            import ultralytics                    # noqa: F401
            import torch
            if not torch.cuda.is_available():
                print("note: CUDA not available to torch")
        except ImportError:
            missing.append("ultralytics")
        if missing:
            print("installing:", missing)
            subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                            "ultralytics>=8.4.189"], check=True)

    data_root = common.resolve_data_root()
    check_cleaned_data(data_root)

    import train
    train.main(sys.argv[1:])


if __name__ == "__main__":
    main()
