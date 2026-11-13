#!/usr/bin/env python3
# =============================================================================
# ARCHIVE / SHRINK RESULTS FOLDERS
# =============================================================================
# Compresses old or disposable run directories into gzipped tarballs so the
# repository stays small enough for GitHub while no data is lost.
#
# What it does (per matched run directory):
#   1. tar + gzip the whole directory into  archives/<name>.tar.gz
#   2. verify the tarball is readable
#   3. delete the original directory
# With --keep-essentials (recommended): metric JSONs/CSVs are copied back out
# of the archive into  archives/essentials/<name>/  before deletion, so the
# numbers stay directly accessible.
#
# By default it matches obvious throwaway runs: *test*, *smoke*, *debug*.
# Use --all to archive every run folder in the results root.
#
# Usage:
#   python scripts/archive_results.py --dry-run          # see the plan first
#   python scripts/archive_results.py                    # archive test/smoke runs
#   python scripts/archive_results.py --all --keep-essentials
#   python scripts/archive_results.py --restore archives/some_run.tar.gz
# =============================================================================

from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DISPOSABLE_PAT = ("test", "smoke", "debug")
ESSENTIAL_PAT = ("*.json", "*.csv", "*.md")


def _matches(name: str, archive_all: bool) -> bool:
    if archive_all:
        return True
    low = name.lower()
    return any(p in low for p in DISPOSABLE_PAT)


def _dir_size_mb(d: Path) -> float:
    return sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / 1e6


def archive_dir(d: Path, archives_dir: Path, keep_essentials: bool) -> Path:
    archives_dir.mkdir(parents=True, exist_ok=True)
    tar_path = archives_dir / f"{d.name}.tar.gz"
    if tar_path.exists():
        print(f"    [skip] {tar_path.name} already exists")
        return tar_path
    t0 = time.time()
    with tarfile.open(tar_path, "w:gz", compresslevel=9) as tar:
        tar.add(d, arcname=d.name)
    # verify archive integrity before deleting anything
    with tarfile.open(tar_path, "r:gz") as tar:
        n = len(tar.getnames())
    if keep_essentials:
        ess_dir = archives_dir / "essentials" / d.name
        ess_dir.mkdir(parents=True, exist_ok=True)
        for pat in ESSENTIAL_PAT:
            for f in d.rglob(pat):
                rel = f.relative_to(d)
                dest = ess_dir / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, dest)
    size_mb = tar_path.stat().st_size / 1e6
    print(f"    [archived] {d.name}: {n} files -> {tar_path.name} "
          f"({size_mb:.2f} MB, {time.time()-t0:.1f}s)")
    shutil.rmtree(d)
    return tar_path


def main():
    ap = argparse.ArgumentParser(description="Archive run folders to shrink the repo")
    ap.add_argument("--results-root", nargs="*", default=[
        str(PROJECT_ROOT / "results"),
        str(PROJECT_ROOT / "saves" / "results"),
    ])
    ap.add_argument("--archives-dir", default=str(PROJECT_ROOT / "archives"))
    ap.add_argument("--all", action="store_true",
                    help="Archive ALL run folders, not just test/smoke/debug ones.")
    ap.add_argument("--keep-essentials", action="store_true",
                    help="Copy metric JSON/CSV/MD files to archives/essentials/ before archiving.")
    ap.add_argument("--dry-run", action="store_true", help="Show the plan, do nothing.")
    ap.add_argument("--restore", default=None,
                    help="Extract a given archives/<name>.tar.gz back into the results root.")
    args = ap.parse_args()

    archives_dir = Path(args.archives_dir)

    if args.restore:
        tar_path = Path(args.restore)
        dest = Path(args.results_root[0]) if args.results_root else Path("results")
        dest.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(dest)
        print(f"[✓] Restored {tar_path.name} -> {dest}")
        return

    total_saved = 0.0
    plan = []
    for root in args.results_root:
        root = Path(root)
        if not root.is_dir():
            continue
        for d in sorted(root.iterdir()):
            if not d.is_dir():
                continue
            if _matches(d.name, args.all):
                plan.append((d, _dir_size_mb(d)))

    if not plan:
        print("[*] Nothing to archive (no matching run folders).")
        return

    print("[*] Archive plan:")
    for d, size in plan:
        print(f"    {d}  ({size:.2f} MB)")
    if args.dry_run:
        print("[dry-run] No changes made.")
        return

    for d, _ in plan:
        archive_dir(d, archives_dir, args.keep_essentials)
        total_saved += 1  # Count each saved source file; byte accounting follows

    print(f"\n[✓] Archived {len(plan)} folder(s) -> {archives_dir}")
    kept = sum(f.stat().st_size for f in archives_dir.rglob("*") if f.is_file()) / 1e6
    print(f"    Archive store size: {kept:.2f} MB (gitignored)")


if __name__ == "__main__":
    main()
