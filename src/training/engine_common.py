"""
Shared training-engine infrastructure (v4, W4/W7) — used by train_dense.py,
train_detr.py and train_custom.py so performance and resilience features exist
in exactly one place:

- TF32 / cudnn.benchmark / deterministic setup
- AMP selection (bf16 on A100/L4-class GPUs, fp16 + GradScaler on T4)
- optional torch.compile with graceful fallback
- EMA of weights
- atomic, fully-resumable checkpoints (model + optimizer + scaler +
  scheduler + EMA + RNG states + resolved config + staging status)
- throttled background Drive mirroring
- OOM-recovering optimizer step (empty_cache -> split batch -> retry)
- per-epoch results.csv / epochs.jsonl logging (columns kept compatible
  with training/summarize_runs.py)
- graceful SIGINT/SIGTERM stop event (finish batch, checkpoint, exit 0)
"""

from __future__ import annotations

import csv
import json
import os
import random
import shutil
import signal
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch

from src.utils.checkpoint import atomic_save_json


# ---------------------------------------------------------------------------
# performance flags
# ---------------------------------------------------------------------------
def setup_perf_flags(deterministic: bool = False) -> None:
    """TF32 + cudnn autotune (fixed-shape training), or full determinism."""
    if not torch.cuda.is_available():
        return
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception as e:  # W10 audit: never swallow silently
            print(f"[perf][warn] deterministic algorithms unavailable: {e}")
    else:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True


def seed_all(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class AmpContext:
    """Resolve AMP dtype + GradScaler for the current device."""

    def __init__(self, device: torch.device):
        self.device = device
        self.enabled = device.type == "cuda"
        self.use_bf16 = bool(
            self.enabled
            and hasattr(torch.cuda, "is_bf16_supported")
            and torch.cuda.is_bf16_supported()
        )
        self.dtype = torch.bfloat16 if self.use_bf16 else torch.float16
        self.scaler: Optional[torch.amp.GradScaler] = None
        if self.enabled and not self.use_bf16:
            try:
                self.scaler = torch.amp.GradScaler("cuda", enabled=True)
            except (AttributeError, TypeError):  # pragma: no cover - older torch
                self.scaler = torch.cuda.amp.GradScaler(enabled=True)

    def autocast(self):
        return torch.amp.autocast(device_type=self.device.type,
                                  dtype=self.dtype, enabled=self.enabled)

    def state_dict(self) -> Optional[Dict[str, Any]]:
        return self.scaler.state_dict() if self.scaler is not None else None

    def load_state_dict(self, state: Optional[Dict[str, Any]]) -> None:
        if state is not None and self.scaler is not None:
            try:
                self.scaler.load_state_dict(state)
            except Exception as e:  # W10 audit: never swallow silently
                print(f"[amp][warn] GradScaler state not restored: {e}")


def maybe_compile(net: torch.nn.Module, enabled: bool,
                  probe: Optional[Callable[[], Any]] = None) -> torch.nn.Module:
    """torch.compile behind a flag, with a graceful fallback (W4.4).

    `probe` (optional) is a zero-arg callable running one dummy forward; it
    forces lazy compilation NOW so a missing/broken backend falls back to
    eager here instead of crashing mid-epoch-1 (review E).
    """
    if not enabled:
        return net
    try:
        compiled = torch.compile(net)
        if probe is not None:
            probe()
        print("[perf] torch.compile enabled")
        return compiled
    except Exception as e:  # pragma: no cover - backend dependent
        print(f"[perf][warn] torch.compile failed, running eager: {e}")
        return net


def maybe_channels_last(net: torch.nn.Module, device: torch.device,
                        enabled: bool = True) -> torch.nn.Module:
    """channels_last for conv-dominated dense nets on CUDA (W4.3)."""
    if enabled and device.type == "cuda":
        return net.to(memory_format=torch.channels_last)
    return net


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------
class ModelEMA:
    """Exponential moving average of model weights (W4.7).

    Warmup ramp (review D): a constant 0.9999 decay keeps the EMA near the
    random init for the first thousands of steps, so validation/best.pt/
    early stopping would track near-init weights early in a run. The
    effective decay ramps up Ultralytics-style: decay * (1 - exp(-t/2000)).
    """

    def __init__(self, net: torch.nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.updates = 0
        self.ema_state = {k: v.detach().clone().float()
                          for k, v in net.state_dict().items()}

    def _effective_decay(self) -> float:
        import math
        return min(self.decay,
                   self.decay * (1.0 - math.exp(-self.updates / 2000.0))
                   if self.updates > 0 else 0.0)

    @torch.no_grad()
    def update(self, net: torch.nn.Module) -> None:
        self.updates += 1
        d = self._effective_decay()
        for k, v in net.state_dict().items():
            if v.dtype.is_floating_point:
                self.ema_state[k].mul_(d).add_(v.detach().float(),
                                               alpha=1.0 - d)
            else:
                self.ema_state[k].copy_(v)

    def state_dict(self) -> Dict[str, Any]:
        return {"decay": self.decay, "updates": self.updates,
                "ema_state": self.ema_state}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.decay = state.get("decay", self.decay)
        self.updates = state.get("updates", 0)
        self.ema_state = {k: v.float() for k, v in state["ema_state"].items()}


# ---------------------------------------------------------------------------
# checkpoints
# ---------------------------------------------------------------------------
def save_training_checkpoint(path: Path, **payload: Any) -> None:
    """Atomic checkpoint write (tmp + os.replace; W4.8)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def rng_state_dict() -> Dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def load_rng_state(state: Dict[str, Any]) -> None:
    try:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        if "cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda"])
    except Exception as e:
        print(f"[resume][warn] could not fully restore RNG states: {e}")


# ---------------------------------------------------------------------------
# Drive mirroring (background, throttled; W4.8 + W7.2)
# ---------------------------------------------------------------------------
class DriveMirror:
    """Copies files to a mirror dir from a daemon thread; never blocks the
    training loop on a Drive FUSE write."""

    def __init__(self, mirror_dir: Optional[str], every_n_epochs: int = 5,
                 run_subdir: Optional[str] = None):
        # run_subdir (review A3): every run mirrors into its OWN subfolder
        # (<dataset>_<model>/); without it every model's last.pt/best.pt/
        # results.csv overwrote each other in one shared mirror dir.
        self.mirror_dir = (Path(mirror_dir) / run_subdir
                           if mirror_dir and run_subdir
                           else (Path(mirror_dir) if mirror_dir else None))
        self.every = max(1, every_n_epochs)
        self._queue: List[Any] = []  # (src, dest_name, unlink_after)
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop = False

    @property
    def active(self) -> bool:
        return self.mirror_dir is not None

    def submit(self, path: Path, dest_name: Optional[str] = None,
               unlink_after: bool = False) -> None:
        """Queue path for async mirroring. dest_name overrides the mirrored
        basename (unique local snapshots still land as last.pt etc.);
        unlink_after deletes the local source after a successful copy
        (snapshot cleanup)."""
        if not self.active:
            return
        with self._lock:
            self._queue.append((Path(path), dest_name, unlink_after))
            if self._thread is None or not self._thread.is_alive():
                self._stop = False
                self._thread = threading.Thread(target=self._drain, daemon=True)
                self._thread.start()

    def submit_epoch(self, epoch: int, *paths: Path) -> None:
        """Mirror on the configured cadence plus the first epoch."""
        if epoch == 1 or epoch % self.every == 0:
            for p in paths:
                self.submit(p)

    def _drain(self) -> None:
        while True:
            with self._lock:
                if not self._queue or self._stop:
                    # Mark the thread gone UNDER the lock: otherwise submit()
                    # can observe is_alive()==True for a thread that is about
                    # to return and strand the queued item (review A3 race).
                    self._thread = None
                    return
                item = self._queue.pop(0)
            src, dest_name, unlink_after = item
            try:
                dst = self.mirror_dir / (dest_name or src.name)
                self.mirror_dir.mkdir(parents=True, exist_ok=True)
                tmp = dst.with_name(dst.name + ".part")
                shutil.copy2(src, tmp)
                os.replace(tmp, dst)
                if unlink_after:
                    src.unlink(missing_ok=True)
            except Exception as e:
                print(f"[mirror][warn] {src.name}: {e}")

    def flush(self) -> None:
        with self._lock:
            thread = self._thread
        if thread is not None:
            thread.join(timeout=300)


def restore_from_mirror(mirror_dir: Optional[str], run_subdir: str,
                        run_path: Path) -> List[str]:
    """Pull mirrored checkpoints back into a wiped local run dir (W7.1).

    Colab sessions die and take /content/runs with them; the Drive mirror is
    the surviving copy. Restores last.pt/best.pt/results.csv when the local
    run dir is missing them. Returns the restored file names.
    """
    if not mirror_dir:
        return []
    src_dir = Path(mirror_dir) / run_subdir
    if not src_dir.is_dir():
        return []
    restored: List[str] = []
    for name in ("last.pt", "best.pt", "results.csv", "epochs.jsonl", "train.log"):
        src = src_dir / name
        if not src.is_file():
            continue
        dst = Path(run_path) / "weights" / name if name.endswith(".pt") \
            else Path(run_path) / name
        if dst.exists():
            continue  # never clobber a local checkpoint
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.name + ".part")
        shutil.copy2(src, tmp)
        os.replace(tmp, dst)
        restored.append(name)
    if restored:
        print(f"[mirror] restored from Drive mirror: {', '.join(restored)}")
    return restored


# ---------------------------------------------------------------------------
# per-epoch logging (results.csv compatible with training/summarize_runs.py)
# ---------------------------------------------------------------------------
RESULTS_CSV_COLUMNS = [
    "epoch", "lr", "train_loss", "train_cls", "train_box", "train_ang",
    "val_loss", "metrics/mAP50(B)", "metrics/mAP50-95(B)",
    "imgs_per_sec", "gpu_mem_gb", "time", "data_wait_s",
]


class EpochLogger:
    """Append-only per-epoch results.csv + epochs.jsonl + detailed train.log in run dir."""

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.run_dir / "results.csv"
        self.jsonl_path = self.run_dir / "epochs.jsonl"
        self.log_path = self.run_dir / "train.log"
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=RESULTS_CSV_COLUMNS).writeheader()

    def log(self, row: Dict[str, Any]) -> None:
        with open(self.csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=RESULTS_CSV_COLUMNS,
                           extrasaction="ignore").writerow(row)
        with open(self.jsonl_path, "a") as f:
            f.write(json.dumps(row, default=str) + "\n")

    def detail(self, message: str) -> None:
        """Append a timestamped detail line to train.log for deep troubleshooting."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(f"[{timestamp}] {message}\n")
        except OSError:
            pass


# ---------------------------------------------------------------------------
# OOM recovery (W4.5/4.6) lives inside the training loops of train_dense.py
# and train_detr.py: retry happens OUTSIDE the except handler (so the failed
# graph is released first), halves are gradient-weighted by n_half/n, the
# accumulation group is discarded on OOM (partial .grad is unsafe), and the
# micro-batch is halved PERSISTENTLY so later batches stop OOMing. The
# earlier shared helper here was dead code with none of those properties and
# has been removed (review A4).
# ---------------------------------------------------------------------------
# graceful interruption (W7.3, ported from scripts/run_pipeline.py semantics)
# ---------------------------------------------------------------------------
class StopEvent:
    """SIGINT/SIGTERM sets this; engines finish the current batch, checkpoint
    and exit(0) so Colab disconnects/Ctrl-C never corrupt a run."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self._installed = False

    def install(self) -> None:
        if self._installed or threading.current_thread() is not threading.main_thread():
            return

        def _handler(signum, _frame):
            print(f"\n[signal] received {signal.Signals(signum).name}; finishing "
                  "current batch, then checkpointing and exiting", flush=True)
            self._event.set()

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, _handler)
            except (ValueError, OSError):  # pragma: no cover - non-main thread
                pass
        self._installed = True

    def requested(self) -> bool:
        return self._event.is_set()


STOP = StopEvent()


# ---------------------------------------------------------------------------
# misc
# ---------------------------------------------------------------------------
def gpu_mem_gb() -> float:
    if torch.cuda.is_available():
        return round(torch.cuda.max_memory_allocated() / 1e9, 2)
    return 0.0


def resolved_config_snapshot(**kwargs: Any) -> Dict[str, Any]:
    return {k: (str(v) if isinstance(v, (Path, torch.device)) else v)
            for k, v in kwargs.items()}
