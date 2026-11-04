#!/usr/bin/env python3
# =============================================================================
# UNIFIED RUN LOGGING & TRACKING
# =============================================================================
# Every pipeline entry point (run_pipeline.py, cross_dataset_eval.py, ...) calls
# setup_run_logging() once at startup. This module then provides three layers
# of forensic visibility into what is happening at every minute of a run:
#
#   1. console.log   - a byte-for-byte mirror of EVERYTHING printed to stdout
#                      and stderr (installed via a Tee). Nothing is lost, even
#                      on Colab where scrollback disappears.
#   2. events.jsonl  - one structured JSON event per line: stage transitions,
#                      per (model, dataset) start/finish, metrics, errors,
#                      heartbeats. Machine-readable, trivially greppable.
#   3. run_config.json - an immutable snapshot of the exact configuration the
#                      run started with (args, versions, device), so any result
#                      can be traced back to how it was produced.
#
# Design constraints:
#   - stdlib only (no new dependencies)
#   - NEVER raises: if logging setup fails (read-only fs, Drive hiccup), the
#     pipeline continues silently with console-only output.
#   - zero impact on behaviour: logging is strictly additive.
#
# Quick usage:
#   from src.utils.logging_utils import setup_run_logging, get_tracker
#   run_log = setup_run_logging(run_id="my_run", results_dir=Path("results"),
#                               config=vars(args))
#   tracker = get_tracker()
#   tracker.stage_start("train")
#   tracker.job_start(model="yolo11-obb", dataset="visdrone")
#   tracker.job_end(model="yolo11-obb", dataset="visdrone", metrics={"f1": 0.31})
#   tracker.stage_end("train", status="ok")
#   tracker.run_end(status="completed")
# =============================================================================

from __future__ import annotations

import json
import os
import platform
import sys
import threading
import time
import traceback
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

#------------------------------------------------------------------------------
# Console Tee - mirrors every stdout/stderr byte into the run log file
# -----------------------------------------------------------------------------


class _Tee:
    def __init__(self, stream, log_file):
        self._stream = stream
        self._log_file = log_file
        self._lock = threading.Lock()

    def write(self, data):
        with self._lock:
            try:
                self._stream.write(data)
                self._stream.flush()
            except Exception:
                pass
            try:
                self._log_file.write(data)
                self._log_file.flush()
                os.fsync(self._log_file.fileno())
            except Exception:
                pass
        return len(data)

    def flush(self):
        try:
            self._stream.flush()
        except Exception:
            pass
        try:
            self._log_file.flush()
        except Exception:
            pass

    # Pass-throughs so anything relying on stream attributes keeps working
    def __getattr__(self, name):
        return getattr(self._stream, name)

    def isatty(self):
        try:
            return self._stream.isatty()
        except Exception:
            return False

    @property
    def encoding(self):
        return getattr(self._stream, "encoding", "utf-8")


# -----------------------------------------------------------------------------
# RunTracker - structured JSONL event log
# -----------------------------------------------------------------------------


class RunTracker:
    """Append-only structured event log (events.jsonl) + stage/job timing."""

    def __init__(self, events_path: Optional[Path] = None):
        self.events_path = events_path
        self._fh = None
        self._lock = threading.Lock()
        self._t0 = time.time()
        self._stage_t0: Dict[str, float] = {}
        # Job start-times are kept as FIFO queues keyed ONLY by (model, dataset).
        # The previous implementation built the key from all **tags, so any
        # extra kwarg passed only to job_end (e.g. weights=...) produced a
        # different key, the pop() missed, and every job logged
        # duration_s: -0.0 (time.time() - time.time()). Queue semantics also
        # tolerate interleaved jobs of the same (model, dataset).
        self._job_t0: Dict[str, "deque[float]"] = {}
        self._counters: Dict[str, int] = {}
        if events_path is not None:
            try:
                events_path.parent.mkdir(parents=True, exist_ok=True)
                self._fh = open(events_path, "a", encoding="utf-8")
            except Exception:
                self._fh = None

    # -- internals -----------------------------------------------------------
    def _write(self, event: str, **fields):
        rec = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "uptime_s": round(time.time() - self._t0, 2),
            "event": event,
        }
        rec.update({k: v for k, v in fields.items() if v is not None})
        line = json.dumps(rec, default=str, ensure_ascii=False)
        with self._lock:
            if self._fh is not None:
                try:
                    self._fh.write(line + "\n")
                    self._fh.flush()
                    os.fsync(self._fh.fileno())
                except Exception:
                    pass
        # Always echo structured events to console too (single concise line)
        try:
            preview = line if len(line) <= 240 else line[:237] + "..."
            print(f"[TRACK] {preview}", flush=True)
        except Exception:
            pass

    def bump(self, counter: str, n: int = 1) -> int:
        self._counters[counter] = self._counters.get(counter, 0) + n
        return self._counters[counter]

    # -- lifecycle -----------------------------------------------------------
    def run_start(self, run_id: str, config: Optional[Dict[str, Any]] = None):
        self._write("run_start", run_id=run_id,
                    python=platform.python_version(),
                    platform=platform.platform())
        if config:
            self._write("run_config", config=json.dumps(config, default=str)[:8000])

    def run_end(self, status: str = "completed", error: Optional[str] = None):
        self._write("run_end", status=status, error=error,
                    total_uptime_s=round(time.time() - self._t0, 2),
                    counters=dict(self._counters))
        self.close()

    def stage_start(self, stage: str, detail: Optional[str] = None):
        self._stage_t0[stage] = time.time()
        self._write("stage_start", stage=stage, detail=detail)

    def stage_end(self, stage: str, status: str = "ok", detail: Optional[str] = None):
        dur = time.time() - self._stage_t0.pop(stage, time.time())
        self._write("stage_end", stage=stage, status=status, detail=detail,
                    duration_s=round(dur, 2))

    # -- jobs (a job = one model x dataset unit of work) -----------------------
    def job_start(self, model: Optional[str] = None, dataset: Optional[str] = None,
                  **tags):
        key = self._job_key(model, dataset)
        self._job_t0.setdefault(key, deque()).append(time.time())
        self._write("job_start", model=model, dataset=dataset, **tags)

    def job_end(self, model: Optional[str] = None, dataset: Optional[str] = None,
                status: str = "ok", metrics: Optional[Dict[str, Any]] = None,
                error: Optional[str] = None, **tags):
        key = self._job_key(model, dataset)
        dq = self._job_t0.get(key)
        start_t = dq.popleft() if dq else time.time()
        if dq is not None and not dq:
            self._job_t0.pop(key, None)
        dur = time.time() - start_t
        self._write("job_end", model=model, dataset=dataset, status=status,
                    duration_s=round(dur, 2), metrics=_sanitize(metrics),
                    error=error, **tags)

    def heartbeat(self, model: Optional[str] = None, dataset: Optional[str] = None,
                  progress: Optional[str] = None, **tags):
        """Call periodically inside long loops so gaps in events.jsonl mean a
        crash, not silence."""
        self._write("heartbeat", model=model, dataset=dataset,
                    progress=progress, **tags)

    def metric(self, name: str, value: Any, model: Optional[str] = None,
               dataset: Optional[str] = None, **tags):
        self._write("metric", name=name, value=value, model=model,
                    dataset=dataset, **tags)

    def error(self, message: str, model: Optional[str] = None,
              dataset: Optional[str] = None, exc: Optional[BaseException] = None):
        tb = "".join(traceback.format_exception(exc))[-4000:] if exc else None
        self._write("error", message=message[:2000], model=model,
                    dataset=dataset, traceback=tb)

    def close(self):
        if self._fh is not None:
            try:
                self._fh.close()
            except Exception:
                pass
            self._fh = None

    @staticmethod
    def _job_key(model, dataset) -> str:
        """Timing key: (model, dataset) only — independent of extra event tags."""
        return f"{model}|{dataset}"


def _sanitize(metrics: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Make metrics JSON-safe (numpy scalars, NaN/Inf)."""
    if metrics is None:
        return None
    out = {}
    for k, v in metrics.items():
        try:
            if v is None or isinstance(v, (str, bool, int)):
                out[k] = v
            else:
                f = float(v)
                out[k] = f if (f == f and abs(f) != float("inf")) else None
        except Exception:
            out[k] = str(v)
    return out


# -----------------------------------------------------------------------------
# Module-level singleton
# -----------------------------------------------------------------------------

_tracker: Optional[RunTracker] = None
_log_dir: Optional[Path] = None


def get_tracker() -> RunTracker:
    """Return the active tracker; a no-op tracker if logging was never set up
    (so call sites never need None-checks)."""
    global _tracker
    if _tracker is None:
        _tracker = RunTracker(events_path=None)
    return _tracker


def get_log_dir() -> Optional[Path]:
    return _log_dir


def setup_run_logging(run_id: str, results_dir: Path,
                      config: Optional[Dict[str, Any]] = None,
                      log_root: Optional[Path] = None) -> Optional[Path]:
    """
    Install console tee + JSONL tracker + config snapshot for a run.

    Logs live under:  <log_root or results_dir>/run_logs/<run_id>/
        console.log      mirror of all stdout/stderr
        events.jsonl     structured events
        run_config.json  config snapshot

    Returns the run log directory, or None if setup failed (never raises).
    Safe to call multiple times: the first call wins.
    """
    global _tracker, _log_dir
    if _tracker is not None and _tracker.events_path is not None:
        return _log_dir  # already initialised

    try:
        base = Path(log_root) if log_root else Path(results_dir)
        log_dir = base / "run_logs" / str(run_id).replace("/", "_")
        log_dir.mkdir(parents=True, exist_ok=True)

        # 1. Console tee ------------------------------------------------------
        console_path = log_dir / "console.log"
        fh = open(console_path, "a", encoding="utf-8", buffering=1)
        stamp = (f"\n{'=' * 78}\n"
                 f"RUN {run_id} started {datetime.now().isoformat()}\n"
                 f"python {platform.python_version()} | {platform.platform()}\n"
                 f"{'=' * 78}\n")
        fh.write(stamp)
        fh.flush()
        sys.stdout = _Tee(sys.stdout, fh)
        sys.stderr = _Tee(sys.stderr, fh)

        # 2. Structured tracker ------------------------------------------------
        _tracker = RunTracker(events_path=log_dir / "events.jsonl")
        _log_dir = log_dir

        # 3. Config snapshot ---------------------------------------------------
        if config is not None:
            try:
                with open(log_dir / "run_config.json", "w", encoding="utf-8") as cf:
                    json.dump({"run_id": run_id,
                               "started_at": datetime.now(timezone.utc).isoformat(),
                               "config": config}, cf, indent=2, default=str)
            except Exception:
                pass

        _tracker.run_start(run_id=run_id, config=None)
        print(f"[LOG] Run logging active -> {log_dir}", flush=True)
        return log_dir
    except Exception:
        # Logging must never take the pipeline down.
        _tracker = RunTracker(events_path=None)
        try:
            print(f"[LOG] WARNING: run logging unavailable: "
                  f"{traceback.format_exc(limit=1)}", flush=True)
        except Exception:
            pass
        return None
