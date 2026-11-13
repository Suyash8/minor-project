#!/usr/bin/env python3
"""
staging.py — the ONE data staging module for the whole project (v4, W2).

Replaces both historical stagers (scripts/stage_data.py and
training/common.py::stage_local), which disagreed on idempotency and always
blocked training behind a full Drive -> NVMe copy.

Design
------
1. **Overlapped, not blocking.** `start_background_stage()` returns a
   StageHandle immediately. A thread pool streams files from the (slow)
   source root — typically the Google Drive FUSE mount — to the (fast) local
   staging root — typically Colab NVMe at /content/data — while training is
   already running.
2. **Priority order = consumption order.** The copy queue is ordered
   train -> val -> test, and `reprioritize(ordered_rel_paths)` lets the
   trainer push the current epoch's shuffled file order to the front, so
   Drive-read fallbacks only happen for the first few batches.
3. **Race-safe reads.** Copies land as `<name>.part-<pid>` and are
   atomically `os.replace()`d into place. Readers never see a half-written
   file: `resolve_reader(local, source)` returns the local path only when it
   exists and is complete, otherwise the source path (slow but correct).
4. **Exact idempotency.** A file counts as staged only when it exists with
   the exact source size. A per-dataset `.stage_manifest.json` (relpath ->
   size) is written on completion. No ">50 files" or "count matches"
   heuristics.
5. **Persistent cache.** Once staged, files stay on the local disk; later
   epochs, later models and later runs read at NVMe speed with zero source
   reads.
6. **Disk guard.** `check_disk_guard()` aborts with a clear message when the
   staging target lacks free space (dataset size x 1.1).

CLI (blocking, replaces scripts/stage_data.py):
    python -m training.staging --datasets visdrone dota
    python -m training.staging --datasets all --force
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent

REAL_DATASETS = ["codrone", "visdrone", "dota"]
SPLITS = ["train", "val", "test"]

DEFAULT_STAGE_ROOT = "/content/data"
DRIVE_SOURCE = "/content/drive/MyDrive/object-detection/cleaned_data"
LEGACY_STAGE_ROOT = "/content/vehicle_obb_data"

_MANIFEST_NAME = ".stage_manifest.json"


# ---------------------------------------------------------------------------
# roots
# ---------------------------------------------------------------------------
def source_root(explicit: Optional[str] = None) -> str:
    """Where the cleaned datasets come FROM (slow, persistent storage)."""
    if explicit:
        return os.path.abspath(os.path.expanduser(explicit))
    env = os.environ.get("VEHICLE_OBB_SOURCE")
    if env:
        return os.path.abspath(os.path.expanduser(env))
    if os.path.isdir(DRIVE_SOURCE):
        return DRIVE_SOURCE
    # Non-Colab fallback: the ordinary data root doubles as the source; when
    # source == stage root, staging is a verified no-op.
    return os.environ.get("VEHICLE_OBB_DATA", str(REPO_ROOT / "data" / "cleaned_data"))


def stage_root(explicit: Optional[str] = None) -> str:
    """Where datasets are staged TO (fast local disk). Priority per spec §2.2."""
    if explicit:
        return os.path.abspath(os.path.expanduser(explicit))
    env = os.environ.get("VEHICLE_OBB_STAGE")
    if env:
        return os.path.abspath(os.path.expanduser(env))
    # Only default to /content/data on Colab-like systems; elsewhere stage
    # next to the source (repo-local), which keeps Windows/Anaconda working.
    if os.path.exists("/content"):
        return DEFAULT_STAGE_ROOT
    return os.path.join(source_root(), "")


def same_root(a: str, b: str) -> bool:
    try:
        return os.path.realpath(a) == os.path.realpath(b)
    except OSError:
        return os.path.abspath(a) == os.path.abspath(b)


# ---------------------------------------------------------------------------
# file inventory / exact idempotency
# ---------------------------------------------------------------------------
def list_dataset_files(src_ds_dir: str) -> List[Tuple[str, int]]:
    """All files under a dataset dir as (relpath, size_bytes), stable order."""
    out: List[Tuple[str, int]] = []
    for root, _dirs, files in os.walk(src_ds_dir):
        for f in files:
            p = os.path.join(root, f)
            try:
                out.append((os.path.relpath(p, src_ds_dir), os.path.getsize(p)))
            except OSError:
                continue
    out.sort()
    return out


def _file_complete(dst: str, size: int) -> bool:
    try:
        return os.path.getsize(dst) == size
    except OSError:
        return False


def pending_files(src_ds_dir: str, dst_ds_dir: str,
                  inventory: Optional[List[Tuple[str, int]]] = None) -> List[Tuple[str, int]]:
    """Files not yet staged exactly (missing or size mismatch)."""
    inv = inventory if inventory is not None else list_dataset_files(src_ds_dir)
    return [(rel, size) for rel, size in inv
            if not _file_complete(os.path.join(dst_ds_dir, rel), size)]


class StagedStatus:
    """Snapshot of staging progress for one or more datasets."""

    def __init__(self, done: int, total: int, drive_reads: int = 0):
        self.done = done
        self.total = total
        self.drive_reads = drive_reads

    @property
    def complete(self) -> bool:
        return self.total > 0 and self.done >= self.total

    @property
    def fraction(self) -> float:
        return (self.done / self.total) if self.total else 1.0

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (f"StagedStatus(done={self.done}/{self.total}, "
                f"complete={self.complete}, drive_reads={self.drive_reads})")


def stage_status(dataset: str, src_root: Optional[str] = None,
                 dst_root: Optional[str] = None) -> StagedStatus:
    """Exact staged-file count for one dataset (existence + size match)."""
    src = os.path.join(source_root(src_root), dataset)
    dst = os.path.join(stage_root(dst_root), dataset)
    if not os.path.isdir(src):
        return StagedStatus(0, 0)
    inv = list_dataset_files(src)
    done = sum(1 for rel, size in inv if _file_complete(os.path.join(dst, rel), size))
    return StagedStatus(done, len(inv))


# ---------------------------------------------------------------------------
# disk guard
# ---------------------------------------------------------------------------
def check_disk_guard(datasets: Sequence[str], src: str, dst: str,
                     margin: float = 1.1) -> None:
    """Abort with a clear message when the staging target lacks free space."""
    # Review R5b: guard on PENDING bytes (files not yet staged exactly),
    # not the full dataset - a resumed run with 90% already staged must not
    # demand 1.1x the whole dataset again.
    needed = 0
    for ds in datasets:
        ds_dir = os.path.join(src, ds)
        if not os.path.isdir(ds_dir):
            continue
        for _rel, size in pending_files(ds_dir, os.path.join(dst, ds)):
            needed += size
    if needed == 0 or same_root(src, dst):
        return
    usage = shutil.disk_usage(dst if os.path.isdir(dst) else os.path.dirname(dst) or "/")
    required = int(needed * margin)
    if usage.free < required:
        gb = 1024 ** 3
        raise RuntimeError(
            f"[stage] Not enough free space at {dst}: need ~{required / gb:.1f} GB "
            f"({needed / gb:.1f} GB data x {margin}), only {usage.free / gb:.1f} GB free. "
            f"Prune other datasets under {dst} or stage fewer datasets at once."
        )


# ---------------------------------------------------------------------------
# race-safe reads
# ---------------------------------------------------------------------------
class SourceResolver:
    """Picklable read resolver for overlapped staging (W2).

    Datasets are always built from the SOURCE root, so the sample inventory
    is complete and stable from the first second of training. Every read is
    redirected to the staged copy when that copy is complete (exact size
    match; writers are atomic .part + os.replace), otherwise the source file
    is read and the miss is counted on a cross-process counter, so the W2
    acceptance check ('~0 Drive reads by epoch 2') measures ALL DataLoader
    workers, not just the main process.

    Plain data only (two roots + one mp.Value), so it pickles into spawn
    workers; on start methods where mp.Value cannot be pickled the counter
    degrades to per-process counting (main-process total then undercounts,
    which is disclosed in the report).
    """

    def __init__(self, src_root: str, dst_root: str, counter=None):
        self.src_root = os.path.abspath(src_root)
        self.dst_root = os.path.abspath(dst_root)
        self._counter = counter
        self._local_reads = 0
        self._noop = same_root(self.src_root, self.dst_root)
        # Review R5d: memoize source sizes - os.path.getsize on the SOURCE
        # is a Drive round-trip and must not happen on every read.
        self._size_cache = {}

    def __getstate__(self):
        state = dict(self.__dict__)
        try:
            if mp.get_start_method(allow_none=True) != "fork":
                # mp.Value cannot cross a spawn pickle; workers count locally.
                state["_counter"] = None
        except Exception:
            state["_counter"] = None
        return state

    def resolve(self, source_path: str, source_path_alt: Optional[str] = None) -> str:
        """Return the staged copy of source_path when complete, else source."""
        path = source_path
        if self._noop:
            return path
        try:
            rel = os.path.relpath(path, self.src_root)
        except ValueError:  # different drive on Windows
            return path
        if rel.startswith(".."):
            return path
        dst = os.path.join(self.dst_root, rel)
        src_size = self._size_cache.get(path)
        if src_size is None:
            try:
                src_size = os.path.getsize(path)
            except OSError:
                src_size = -1
            self._size_cache[path] = src_size
        try:
            if (os.path.isfile(dst) and ".part-" not in os.path.basename(dst)
                    and os.path.getsize(dst) == src_size):
                return dst
        except OSError:
            pass
        with self._count_lock():
            if self._counter is not None:
                self._counter.value += 1
            else:
                self._local_reads += 1
        return path

    def _count_lock(self):
        if self._counter is not None:
            return self._counter.get_lock()
        return _NULL_LOCK

    @property
    def drive_reads(self) -> int:
        if self._counter is not None:
            return int(self._counter.value)
        return self._local_reads


class _NullLock:
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False


_NULL_LOCK = _NullLock()


class ReadResolver:
    """Chooses the staged local copy when complete, else the source file.

    A resolver is handed to the dataset layer while background staging is
    active. It never returns a path that is mid-copy (writers go through
    `<name>.part-<pid>` + os.replace), and it counts Drive/source fallbacks so
    the pipeline can verify that epoch >= 2 performs ~zero slow reads.
    """

    def __init__(self, handle: Optional["StageHandle"] = None):
        self.handle = handle
        self.drive_reads = 0
        self._lock = threading.Lock()

    def resolve(self, local_path: str, source_path: Optional[str] = None) -> str:
        if source_path is None:
            source_path = self._source_for(local_path)
        if _looks_complete(local_path):
            return local_path
        with self._lock:
            self.drive_reads += 1
            if self.handle is not None:
                self.handle.note_source_read()
        return source_path

    def _source_for(self, local_path: str) -> str:
        if self.handle is not None:
            return self.handle.source_path_for(local_path)
        return local_path


def _looks_complete(path: str) -> bool:
    """A path is readable when it exists and is not an in-flight .part file."""
    return os.path.isfile(path) and ".part-" not in os.path.basename(path)


def resolve_reader(local_path: str, source_path: str) -> str:
    """Stateless variant of ReadResolver.resolve for one-off reads."""
    return local_path if _looks_complete(local_path) else source_path


# ---------------------------------------------------------------------------
# fast archive staging (.tar / .tar.gz / .zip)
# ---------------------------------------------------------------------------
ARCHIVE_EXTS = (".tar", ".tar.gz", ".tgz", ".zip")


def find_dataset_archive(dataset: str, src_root: str) -> Optional[Path]:
    """Search for a pre-packaged dataset archive (.tar, .tar.gz, .zip).

    Streaming a single 2 GB archive from Google Drive to local NVMe takes
    ~15 seconds at 150-200 MB/s, avoiding thousands of slow per-file HTTP
    roundtrips over FUSE.

    Checks:
      1. src_root/<dataset>.tar[.gz|.zip]
      2. src_root/../<dataset>.tar[.gz|.zip]
      3. src_root/archives/<dataset>.tar[.gz|.zip]
      4. /content/drive/MyDrive/object-detection/<dataset>.tar[.gz|.zip]
    """
    src_p = Path(src_root)
    candidates: List[Path] = []
    for ext in ARCHIVE_EXTS:
        candidates.append(src_p / f"{dataset}{ext}")
        candidates.append(src_p.parent / f"{dataset}{ext}")
        candidates.append(src_p / "archives" / f"{dataset}{ext}")
        candidates.append(src_p.parent / "archives" / f"{dataset}{ext}")
        if os.path.isdir("/content/drive/MyDrive/object-detection"):
            candidates.append(Path(f"/content/drive/MyDrive/object-detection/{dataset}{ext}"))
            candidates.append(Path(f"/content/drive/MyDrive/object-detection/cleaned_data/{dataset}{ext}"))

    for c in candidates:
        try:
            if c.is_file() and c.stat().st_size > 1024:  # valid non-empty archive
                return c
        except OSError:
            continue
    return None


def extract_dataset_archive(archive_path: Path, dst_root: str, dataset: str) -> bool:
    """Fast-extract a dataset archive directly into dst_root.

    Inspects archive structure to unpack correctly whether the top-level
    folder is '<dataset>/' or directly split folders ('train/', 'val/').
    Prefers native 'tar' / 'unzip' binaries for maximum NVMe I/O speed.
    """
    dst_p = Path(dst_root)
    dst_ds = dst_p / dataset
    archive_str = str(archive_path)

    has_prefix = False
    try:
        if archive_str.endswith((".tar", ".tar.gz", ".tgz")):
            with tarfile.open(archive_path, "r:*") as tf:
                first = tf.next()
                if first and (first.name == dataset or first.name.startswith(f"{dataset}/")):
                    has_prefix = True
        elif archive_str.endswith(".zip"):
            with zipfile.ZipFile(archive_path, "r") as zf:
                nl = zf.namelist()
                first = nl[0] if nl else ""
                if first and (first == dataset or first.startswith(f"{dataset}/")):
                    has_prefix = True
    except Exception as e:
        print(f"[stage][warn] archive inspection ({archive_path.name}): {e}",
              file=sys.stderr)

    extract_dest = dst_p if has_prefix else dst_ds
    extract_dest.mkdir(parents=True, exist_ok=True)

    print(f"[stage] fast-unpacking archive {archive_path.name} -> {extract_dest}...",
          flush=True)
    t0 = time.time()
    success = False

    if archive_str.endswith((".tar", ".tar.gz", ".tgz")):
        tar_flags = "-xzf" if archive_str.endswith((".tar.gz", ".tgz")) else "-xf"
        try:
            subprocess.run(["tar", tar_flags, archive_str, "-C", str(extract_dest)],
                           check=True, capture_output=True)
            success = True
        except (subprocess.SubprocessError, FileNotFoundError):
            pass

        if not success:
            try:
                with tarfile.open(archive_path, "r:*") as tf:
                    tf.extractall(extract_dest)
                success = True
            except Exception as e:
                print(f"[stage][error] tar extraction failed ({archive_path.name}): {e}",
                      file=sys.stderr)

    elif archive_str.endswith(".zip"):
        try:
            subprocess.run(["unzip", "-q", "-o", archive_str, "-d", str(extract_dest)],
                           check=True, capture_output=True)
            success = True
        except (subprocess.SubprocessError, FileNotFoundError):
            pass

        if not success:
            try:
                with zipfile.ZipFile(archive_path, "r") as zf:
                    zf.extractall(extract_dest)
                success = True
            except Exception as e:
                print(f"[stage][error] zip extraction failed ({archive_path.name}): {e}",
                      file=sys.stderr)

    if success:
        elapsed = max(time.time() - t0, 0.1)
        sz_mb = archive_path.stat().st_size / 1e6
        print(f"[stage] successfully unpacked {archive_path.name} ({sz_mb:.1f} MB) in "
              f"{elapsed:.1f}s ({sz_mb / elapsed:.1f} MB/s)!", flush=True)
    return success


def pack_dataset_archive(src_ds_dir: str, dst_archive: Path) -> bool:
    """Pack a staged dataset directory into an uncompressed .tar on Drive/source.

    Uncompressed tar streams at full speed on Drive and extracts in seconds.
    """
    src_p = Path(src_ds_dir)
    if not src_p.is_dir():
        return False
    dst_archive = Path(dst_archive)
    dst_archive.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    ds_name = src_p.name
    parent_dir = src_p.parent

    # Try native tar first
    try:
        subprocess.run(["tar", "-cf", str(dst_archive), "-C", str(parent_dir), ds_name],
                       check=True, capture_output=True)
        sz_mb = dst_archive.stat().st_size / 1e6
        print(f"[stage] packed {dst_archive.name} ({sz_mb:.1f} MB) in "
              f"{time.time() - t0:.1f}s", flush=True)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        pass

    try:
        with tarfile.open(dst_archive, "w") as tf:
            tf.add(src_p, arcname=ds_name)
        sz_mb = dst_archive.stat().st_size / 1e6
        print(f"[stage] packed {dst_archive.name} ({sz_mb:.1f} MB) in "
              f"{time.time() - t0:.1f}s", flush=True)
        return True
    except Exception as e:
        print(f"[stage][warn] pack archive failed: {e}", file=sys.stderr)
        return False


# ---------------------------------------------------------------------------
# background staging
# ---------------------------------------------------------------------------
def _split_of(rel_path: str) -> int:
    for i, sp in enumerate(SPLITS):
        if rel_path.startswith(sp + os.sep) or rel_path.startswith(sp + "/"):
            return i
    return len(SPLITS)


class StageHandle:
    _max_attempts = 3  # B2: bounded retries per file before counting failed

    """Background Drive -> NVMe streaming stager.

    Threads copy files in priority order (train split first; the trainer can
    re-order via reprioritize()). All copies are atomic. The handle is safe
    to query from the training thread while copies are in flight.
    """

    def __init__(self, datasets: Sequence[str], src_root: Optional[str] = None,
                 dst_root: Optional[str] = None, workers: int = 8,
                 progress: bool = True):
        self.datasets = list(datasets)
        self.src_root = source_root(src_root)
        self.dst_root = stage_root(dst_root)
        self.workers = max(1, workers)
        self.progress = progress

        self._lock = threading.Lock()
        self._queue: List[Tuple[str, str, int]] = []  # (dataset, relpath, size)
        self._done = 0
        self._total = 0
        # B2: transient Drive/FUSE errors get bounded retries (class-level
        # so tests/ops can tune; instance reads it at copy time)
        self._max_attempts = type(self)._max_attempts
        # Cross-process source-read counter (W2 acceptance evidence): forked
        # DataLoader workers inherit this mp.Value, so misses in workers are
        # counted here, not only in the main process.
        self.source_reads_value = mp.Value("q", 0)
        self._errors: List[str] = []
        self._failed = 0
        self._inflight = 0
        self._drain_reported = False
        self._closed = False
        self._started_at = time.time()
        self._last_report = 0.0
        self._pool: Optional[ThreadPoolExecutor] = None
        self._noop = same_root(self.src_root, self.dst_root)

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> "StageHandle":
        if self._noop:
            # Source and destination coincide (plain Linux/Windows dev box):
            # verify inventory once and report complete.
            for ds in self.datasets:
                inv = list_dataset_files(os.path.join(self.src_root, ds))
                self._total += len(inv)
                self._done += len(inv)
            self._closed = True
            return self

        check_disk_guard(self.datasets, self.src_root, self.dst_root)
        for ds in self.datasets:
            src_ds = os.path.join(self.src_root, ds)
            dst_ds = os.path.join(self.dst_root, ds)

            # Fast archive path: if a .tar/.zip archive exists on Drive/source,
            # unpack it directly in seconds instead of slow per-file FUSE copies.
            archive = find_dataset_archive(ds, self.src_root)
            if archive:
                is_staged = False
                if os.path.isdir(dst_ds):
                    inv_chk = list_dataset_files(src_ds) if os.path.isdir(src_ds) else list_dataset_files(dst_ds)
                    if inv_chk and not pending_files(src_ds if os.path.isdir(src_ds) else dst_ds, dst_ds, inv_chk):
                        is_staged = True
                if not is_staged:
                    extract_dataset_archive(archive, self.dst_root, ds)

            if not os.path.isdir(src_ds):
                if os.path.isdir(dst_ds):
                    inv = list_dataset_files(dst_ds)
                    self._total += len(inv)
                    self._done += len(inv)
                    continue
                self._errors.append(f"source dataset missing: {src_ds}")
                continue
            # Create the destination tree eagerly so readers can resolve
            # staged paths from the first second of the run.
            os.makedirs(dst_ds, exist_ok=True)
            inv = list_dataset_files(src_ds)
            todo = pending_files(src_ds, dst_ds, inv)
            self._total += len(inv)
            self._done += len(inv) - len(todo)
            # Priority: train split first, then val, then test.
            todo.sort(key=lambda t: (_split_of(t[0]), t[0]))
            self._queue.extend((ds, rel, size) for rel, size in todo)

        if self._queue:
            print(f"[stage] background staging started: {self._done}/{self._total} "
                  f"files already cached, {len(self._queue)} to stream "
                  f"({self.src_root} -> {self.dst_root}, {self.workers} threads)",
                  flush=True)
            self._pool = ThreadPoolExecutor(max_workers=self.workers,
                                            thread_name_prefix="stage")
            for _ in range(self.workers):
                self._pool.submit(self._worker_loop)
            self._pool.shutdown(wait=False)
        else:
            print(f"[stage] all {self._total} files already staged at "
                  f"{self.dst_root} — zero source reads needed", flush=True)
            self._closed = True
        return self

    def _worker_loop(self) -> None:
        while True:
            with self._lock:
                if not self._queue:
                    return
                ds, rel, size = self._queue.pop(0)
                self._inflight += 1
            try:
                self._copy_with_retry(ds, rel)
            except Exception as e:  # never kill the trainer over a copy error
                # Review R5a: a FAILED copy must not count as done - that
                # would report staging complete while files are missing.
                # Track failures loudly; the reader falls back to source
                # for those files (per-file size check).
                with self._lock:
                    self._errors.append(f"{ds}/{rel}: {e}")
                    self._failed += 1
            else:
                with self._lock:
                    self._done += 1
            finally:
                with self._lock:
                    self._inflight -= 1
            self._maybe_report()
            with self._lock:
                drained = (not self._queue and self._inflight == 0
                           and self._failed > 0 and not self._drain_reported)
                if drained:
                    self._drain_reported = True
            if drained:
                print(f"[stage][error] {self._failed} of {self._total} files "
                      f"FAILED to stage after {self._max_attempts} attempts "
                      f"(see handle.errors); they are read from the SOURCE "
                      f"for the rest of the run and staging never reports "
                      f"complete", flush=True)

    def _copy_with_retry(self, ds: str, rel: str) -> None:
        """Copy with bounded retries/backoff (review B2): one transient
        Drive/FUSE error must not permanently demote a file to source reads,
        but a permanent failure must surface promptly, not hang waiters."""
        delay = 0.5
        last: Optional[Exception] = None
        for attempt in range(1, self._max_attempts + 1):
            try:
                self._copy_one(ds, rel)
                return
            except Exception as e:
                last = e
                if attempt < self._max_attempts:
                    time.sleep(delay)
                    delay *= 2
        raise last  # type: ignore[misc]

    def _copy_one(self, ds: str, rel: str) -> None:
        src = os.path.join(self.src_root, ds, rel)
        dst = os.path.join(self.dst_root, ds, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        tmp = f"{dst}.part-{os.getpid()}-{threading.get_ident()}"
        shutil.copy2(src, tmp)
        os.replace(tmp, dst)

    def _maybe_report(self) -> None:
        if not self.progress:
            return
        now = time.time()
        with self._lock:
            if now - self._last_report < 2.0 and self._done < self._total:
                return
            self._last_report = now
            done, total = self._done, self._total
        elapsed = max(now - self._started_at, 1e-3)
        speed = done / elapsed
        eta = (total - done) / speed if speed > 0 else 0.0
        pct = (done / total * 100.0) if total else 100.0
        sys.stderr.write(
            f"\r[stage] {done}/{total} files ({pct:.1f}%) | "
            f"{speed:.1f} files/s | ETA {eta:.0f}s | "
            f"source-read fallbacks: {int(self.source_reads_value.value)}   ")
        sys.stderr.flush()
        if done >= total:
            sys.stderr.write("\n")
            self._write_manifests()

    # -- coordination with the trainer --------------------------------------
    def reprioritize(self, ordered_rel_paths: Iterable[str],
                     dataset: Optional[str] = None) -> None:
        """Move the given relpaths (epoch consumption order) to the queue front."""
        if self._closed:
            return
        order = list(ordered_rel_paths)
        if not order:
            return
        rank = {rel: i for i, rel in enumerate(order)}
        with self._lock:
            def key(item: Tuple[str, str, int]) -> Tuple[int, int, str]:
                ds, rel, _size = item
                if dataset is not None and ds != dataset:
                    return (len(rank) + 1, _split_of(rel), rel)
                return (rank.get(rel, len(rank)), _split_of(rel), rel)
            self._queue.sort(key=key)

    def note_source_read(self) -> None:
        with self.source_reads_value.get_lock():
            self.source_reads_value.value += 1

    def source_path_for(self, local_path: str) -> str:
        """Map a staging-root path back to its source-root twin."""
        try:
            rel = os.path.relpath(local_path, self.dst_root)
        except ValueError:  # different drive on Windows
            return local_path
        if rel.startswith(".."):
            return local_path
        return os.path.join(self.src_root, rel)

    def staged_path_for(self, source_path: str) -> Optional[str]:
        """Map a source-root path to its staged twin (None when outside src)."""
        try:
            rel = os.path.relpath(source_path, self.src_root)
        except ValueError:
            return None
        if rel.startswith(".."):
            return None
        return os.path.join(self.dst_root, rel)

    def status(self) -> StagedStatus:
        with self._lock:
            return StagedStatus(self._done, self._total,
                                int(self.source_reads_value.value))

    def wait(self, timeout: Optional[float] = None,
             poll: float = 1.0) -> bool:
        """Block until staging resolves: complete (True), permanently failed
        (False), or timeout (False). Never hangs on a failed copy (B2):
        when the queue is drained with failures outstanding, returns False
        promptly instead of waiting on a `complete` that can never come."""
        deadline = None if timeout is None else time.time() + timeout
        while True:
            st = self.status()
            if st.complete:
                self._write_manifests()
                return True
            with self._lock:
                drained = not self._queue and self._inflight == 0
                failed = self._failed
            if drained:
                # Queue empty, nothing copying. done + failed == total, so
                # failed == 0 would mean complete (caught above); failures
                # mean staging can never complete - stop waiting.
                if failed:
                    return False
            if deadline is not None and time.time() >= deadline:
                return False
            time.sleep(poll)

    def _write_manifests(self) -> None:
        import json
        for ds in self.datasets:
            dst_ds = os.path.join(self.dst_root, ds)
            src_ds = os.path.join(self.src_root, ds)
            if not os.path.isdir(dst_ds):
                continue
            try:
                inv = list_dataset_files(src_ds if os.path.isdir(src_ds) else dst_ds)
                manifest = {rel: size for rel, size in inv}
                tmp = os.path.join(dst_ds, _MANIFEST_NAME + ".tmp")
                with open(tmp, "w") as f:
                    json.dump(manifest, f)
                os.replace(tmp, os.path.join(dst_ds, _MANIFEST_NAME))
            except OSError:
                pass

    def close(self) -> None:
        self._closed = True
        # Join worker threads briefly so teardown (e.g. tempdir cleanup in
        # tests) cannot strand a copy mid-flight and surface as a spurious
        # staging failure.
        pool = self._pool
        if pool is not None:
            for t in list(getattr(pool, "_threads", [])):
                t.join(timeout=10)

    @property
    def errors(self) -> List[str]:
        with self._lock:
            return list(self._errors)


def start_background_stage(datasets: Sequence[str],
                           src_root: Optional[str] = None,
                           dst_root: Optional[str] = None,
                           workers: int = 8,
                           order: Optional[Iterable[str]] = None,
                           progress: bool = True) -> StageHandle:
    """Start streaming `datasets` source -> staging root in the background.

    Returns immediately; training reads through a ReadResolver bound to the
    returned handle. Pass `order` (relpaths in first-epoch consumption order)
    to front-load exactly the files epoch 1 will need.
    """
    handle = StageHandle(datasets, src_root=src_root, dst_root=dst_root,
                         workers=workers, progress=progress)
    handle.start()
    if order:
        handle.reprioritize(order)
    return handle


def stage_datasets_blocking(datasets: Sequence[str],
                            src_root: Optional[str] = None,
                            dst_root: Optional[str] = None,
                            workers: int = 16,
                            force: bool = False,
                            progress: bool = True) -> str:
    """Fully stage datasets and return the effective (fast) data root.

    With force=True the destination is wiped first. When source == staging
    root this is a verified no-op and the source root is returned.
    """
    src = source_root(src_root)
    dst = stage_root(dst_root)
    if force and not same_root(src, dst):
        for ds in datasets:
            shutil.rmtree(os.path.join(dst, ds), ignore_errors=True)
    handle = start_background_stage(datasets, src_root=src, dst_root=dst,
                                    workers=workers, progress=progress)
    ok = handle.wait()
    for err in handle.errors:
        print(f"[stage][warn] {err}", file=sys.stderr)
    if not ok:
        # B2: blocking callers get an explicit error, never a silent hang
        # or a half-staged root presented as usable.
        raise RuntimeError(
            f"[stage] staging incomplete: {len(handle.errors)} file(s) "
            f"failed after retries: {handle.errors[:5]}")
    os.environ["VEHICLE_OBB_DATA"] = handle.dst_root
    return handle.dst_root


def effective_data_root(dataset: str, dst_root: Optional[str] = None) -> str:
    """Root a trainer should read `dataset` from right now.

    Returns the staging root when the dataset is fully staged there,
    otherwise the source root (correct but slower reads).
    """
    dst = stage_root(dst_root)
    st = stage_status(dataset, dst_root=dst)
    if st.total > 0 and st.complete:
        return dst
    return source_root()


# ---------------------------------------------------------------------------
# CLI (replaces scripts/stage_data.py as the real implementation)
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(
        description="Stage cleaned datasets from Drive to fast local disk "
                    "(exact, atomic, resumable).")
    p.add_argument("--datasets", nargs="+", required=True,
                   help="codrone / visdrone / dota / all")
    p.add_argument("--src", default=None, help="source root (default: Drive)")
    p.add_argument("--dst", default=None, help="staging root (default: /content/data)")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--force", action="store_true",
                   help="wipe the destination dataset dirs first")
    p.add_argument("--pack-archive", action="store_true",
                   help="pack staged dataset into an uncompressed <dataset>.tar on Drive/source")
    args = p.parse_args(argv)

    datasets = REAL_DATASETS if "all" in [d.lower() for d in args.datasets] \
        else [d.lower() for d in args.datasets]
    unknown = [d for d in datasets if d not in REAL_DATASETS]
    if unknown:
        p.error(f"unknown dataset(s): {unknown}")

    t0 = time.time()
    root = stage_datasets_blocking(datasets, src_root=args.src, dst_root=args.dst,
                                   workers=args.workers, force=args.force)
    print(f"[stage] done in {time.time() - t0:.1f}s — effective data root: {root}")
    for ds in datasets:
        st = stage_status(ds, src_root=args.src, dst_root=root)
        print(f"[stage] {ds}: {st.done}/{st.total} files "
              f"({'complete' if st.complete else 'INCOMPLETE'})")
        if args.pack_archive and st.complete:
            dst_ds = os.path.join(root, ds)
            tar_target = Path(source_root(args.src)) / f"{ds}.tar"
            print(f"[stage] packing archive {tar_target} for instant future staging...")
            pack_dataset_archive(dst_ds, tar_target)


if __name__ == "__main__":
    main()
