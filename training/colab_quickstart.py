#!/usr/bin/env python3
"""
Colab Quickstart (v4.2) - single bootstrap: mount Drive, restore a preempted
session from the Drive mirror, install pinned deps, verify cleaned data and
launch (or RESUME) the unified pipeline.

Usage (one Colab cell; re-run the SAME cell after a preemption):
    !python training/colab_quickstart.py                 # bootstrap + smoke test
    !python training/colab_quickstart.py --full          # full benchmark (resumes)
    !python training/colab_quickstart.py --models lsknet --datasets codrone
    !python training/colab_quickstart.py --recover-only  # restore, do not launch

Recovery (C7): on every start the mirror (default
/content/drive/MyDrive/object-detection/runs_v4) is inspected; pipeline_state.json
and every surviving last.pt/best.pt/results.csv are restored into
<runs-dir>/train/<dataset>_<model>/ (never overwriting local files). When no
--models/--datasets are given and a restored queue exists, the SAME queue is
relaunched, so finished jobs are skipped and the interrupted job resumes from
its last mirrored epoch. A preemption therefore loses at most the epochs since
the last mirror (--mirror-every, default 5).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import colab_recovery  # noqa: E402


def sh(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def ensure_drive() -> bool:
    """True when Drive is available. `!python script.py` runs in a subprocess,
    where google.colab is not already imported, so the mount is attempted
    explicitly; if the kernel cannot mount from a subprocess the user must
    run drive.mount in a cell first (clear message, no crash)."""
    if os.path.isdir("/content/drive/MyDrive"):
        return True
    if not os.path.isdir("/content"):
        return False                       # not Colab
    try:
        from google.colab import drive
        drive.mount("/content/drive")
    except Exception as e:                 # noqa: BLE001
        print(f"[!] could not mount Drive from this process ({e}). Run "
              f"`from google.colab import drive; drive.mount('/content/drive')` "
              f"in a cell, then re-run this one.", flush=True)
    return os.path.isdir("/content/drive/MyDrive")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="v4.2 Colab quickstart + recovery")
    p.add_argument("--full", action="store_true",
                   help="launch the full benchmark instead of the smoke test")
    p.add_argument("--models", nargs="+", default=None)
    p.add_argument("--datasets", nargs="+", default=None)
    p.add_argument("--steps", default=None,
                   help="override the step list (default: restored queue's "
                        "steps, else the full preset)")
    p.add_argument("--epochs", type=int, default=None,
                   help="override epoch target when resuming (pass 0 to ignore "
                        "any saved --epochs override and use model-tuned defaults)")
    p.add_argument("--mirror-dir", default=colab_recovery.DEFAULT_MIRROR_DIR)
    p.add_argument("--mirror-every", type=int, default=5)
    p.add_argument("--runs-dir", default=None,
                   help="default /content/minor-project/runs on Colab, else runs")
    p.add_argument("--device", default="auto")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--cache", default="annotations",
                   choices=["none", "annotations", "ram", "disk"])
    p.add_argument("--no-recover", action="store_true",
                   help="skip restoring from the mirror")
    p.add_argument("--recover-only", action="store_true",
                   help="restore from the mirror and stop")
    a = p.parse_args(argv)

    on_colab = os.path.isdir("/content")
    drive_ok = ensure_drive()

    repo = Path.cwd()
    if not (repo / "training" / "run.py").exists():
        for cand in (Path("/content/minor-project"),
                     Path("/content/drive/MyDrive/object-detection/minor-project"),
                     Path(__file__).resolve().parent.parent):
            if (cand / "training" / "run.py").exists():
                os.chdir(cand)
                repo = cand
                break
    print(f"[*] repo: {repo}")
    runs_dir = a.runs_dir or ("/content/minor-project/runs" if on_colab
                              else str(repo / "runs"))
    mirror_dir = a.mirror_dir if (drive_ok or not on_colab) else None

    rep = {}
    if not a.no_recover:
        rep = colab_recovery.recover(mirror_dir, runs_dir)
    if a.recover_only:
        return 0

    sh([sys.executable, "install.py"])                 # pinned deps (W8.1)
    sh([sys.executable, "training/run.py", "--env"])   # bug-report header (W8.5)

    data = Path("/content/drive/MyDrive/object-detection/cleaned_data")
    missing = [ds for ds in ("codrone", "visdrone", "dota")
               if not (data / ds / "train" / "images").is_dir()]
    if missing and not a.models and on_colab:
        print(f"[!] cleaned data missing for {missing}; run "
              f"data_cleaning/clean_build_v4.py first (mock smoke still works)")

    cmd = [sys.executable, "training/run.py", "--runs-dir", runs_dir,
           "--device", a.device, "--workers", str(a.workers),
           "--cache", a.cache, "--mirror-every", str(a.mirror_every)]
    if mirror_dir:
        cmd += ["--mirror-dir", mirror_dir]

    resumed = bool(rep.get("resume_models") and rep.get("resume_datasets")
                   and not a.models and not a.datasets)
    if resumed:
        # C7: replay the interrupted queue exactly; run.py's job state skips
        # finished jobs and the trainers resume the interrupted one from its
        # restored last.pt.
        steps = a.steps or ",".join(rep.get("resume_steps") or [])
        cmd += ["--models", *rep["resume_models"],
                "--datasets", *rep["resume_datasets"]]
        if steps:
            cmd += ["--steps", steps]
        saved = rep.get("resume_invocation") or {}
        for key in ("epochs", "batch", "micro_batch", "img_size", "lr", "weight_decay", "patience", "data_dir"):
            if key == "epochs":
                if a.epochs is not None:
                    if a.epochs > 0:
                        cmd += ["--epochs", str(a.epochs)]
                    else:
                        print("[recover] clearing saved --epochs override; using model-tuned defaults", flush=True)
                    continue
                if saved.get("epochs") is not None:
                    print(f"[recover] replaying saved --epochs {saved['epochs']} from pipeline_state.json "
                          f"(pass --epochs 0 to resume up to model-tuned defaults)", flush=True)
            if saved.get(key) is not None:
                cmd += ["--"+key.replace("_","-"), str(saved[key])]
        if saved.get("test"):
            cmd += ["--test"]
        if saved.get("warm_restart"):
            cmd += ["--warm-restart"]
        print(f"[recover] resuming queue: {len(rep['resume_models'])} models x "
              f"{len(rep['resume_datasets'])} datasets, steps={steps or 'preset'}")
    elif a.full or a.models:
        cmd += ["--models", *(a.models or ["all"]),
                "--datasets", *(a.datasets or ["all"]), "--mode", "full"]
        if a.steps:
            cmd += ["--steps", a.steps]
        if a.epochs is not None and a.epochs > 0:
            cmd += ["--epochs", str(a.epochs)]
    else:
        cmd += ["--models", "all", "--datasets", "mock", "--steps",
                a.steps or "stage,train,eval,summary", "--test"]
    sh(cmd)
    return 0


if __name__ == "__main__":
    sys.exit(main())
