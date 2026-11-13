#!/usr/bin/env python3
"""
Colab Quickstart (v4) — single bootstrap: mount Drive, install pinned deps,
verify cleaned data, launch the unified pipeline.

Usage (one Colab cell):
    !python training/colab_quickstart.py                 # bootstrap + smoke test
    !python training/colab_quickstart.py --full          # bootstrap + full benchmark
    !python training/colab_quickstart.py --models lsknet --datasets codrone
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def sh(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def main() -> None:
    p = argparse.ArgumentParser(description="v4 Colab quickstart")
    p.add_argument("--full", action="store_true",
                   help="launch the full benchmark instead of the smoke test")
    p.add_argument("--models", nargs="+", default=None)
    p.add_argument("--datasets", nargs="+", default=None)
    a = p.parse_args()

    if os.path.isdir("/content") and "google.colab" in sys.modules:
        from google.colab import drive
        drive.mount("/content/drive")

    repo = Path.cwd()
    if not (repo / "training" / "run.py").exists():
        cand = Path("/content/drive/MyDrive/object-detection/minor-project")
        if cand.exists():
            os.chdir(cand)
            repo = cand
    print(f"[*] repo: {repo}")

    sh([sys.executable, "install.py"])          # pinned deps (W8.1)
    sh([sys.executable, "training/run.py", "--env"])  # bug-report header (W8.5)

    data = Path("/content/drive/MyDrive/object-detection/cleaned_data")
    missing = [ds for ds in ("codrone", "visdrone", "dota")
               if not (data / ds / "train" / "images").is_dir()]
    if missing and not a.models:
        print(f"[!] cleaned data missing for {missing}; run "
              f"data_cleaning/clean_build_v4.py first (mock smoke still works)")

    cmd = [sys.executable, "training/run.py"]
    cmd += ["--models", *(a.models or ["all"])]
    if a.full or a.models:
        cmd += ["--datasets", *(a.datasets or ["all"]), "--mode", "full"]
    else:
        cmd += ["--datasets", "mock", "--steps", "stage,train,eval,summary",
                "--test"]
    sh(cmd)


if __name__ == "__main__":
    main()
