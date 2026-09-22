"""
Security and robustness utilities: path sanitization, safe checkpoint deserialization,
and graceful signal handling for cloud execution runtimes.
"""

from __future__ import annotations

import os
import signal
import sys
from pathlib import Path
from typing import Any, Callable, Optional, Union

try:
    import torch
except ImportError:
    torch = None

_REGISTERED_CLEANUP_CALLBACKS = []


def safe_path_join(base_dir: Union[Path, str], *paths: str) -> Path:
    """
    Safely resolve a target path within a designated base directory.
    Guards against directory traversal attacks (e.g. `../../etc/passwd`).

    Raises
    ------
    ValueError
        If the resolved path falls outside the root base directory.
    """
    base_resolved = Path(base_dir).resolve()
    target_resolved = base_resolved.joinpath(*paths).resolve()

    try:
        # Python 3.9+ method
        target_resolved.relative_to(base_resolved)
    except ValueError:
        raise ValueError(
            f"Security Error: Attempted path traversal outside base directory. "
            f"Base: '{base_resolved}', Target: '{target_resolved}'"
        )

    return target_resolved


def is_safe_filename(filename: str) -> bool:
    """Check that a given filename has no path separators or null bytes."""
    if not filename or "\0" in filename:
        return False
    if "/" in filename or "\\" in filename or ".." in filename:
        return False
    return True


def safe_torch_load(
    filepath: Union[Path, str],
    map_location: Optional[Union[str, torch.device]] = None,
    weights_only: bool = True,
) -> Any:
    """
    Safely load a PyTorch checkpoint file using `weights_only=True` whenever possible
    to protect against arbitrary code execution via compromised pickle objects.

    Parameters
    ----------
    filepath : Path or str
        Path to the .pt or .pth checkpoint file.
    map_location : str or torch.device, optional
        Target device for tensors (default CPU).
    weights_only : bool
        If True, restrict unpickler to torch tensor weights and safe primitives.

    Returns
    -------
    Any
        Loaded state dictionary or object.
    """
    if torch is None:
        raise ImportError("PyTorch is not installed in the current environment.")

    resolved_path = Path(filepath).resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {resolved_path}")

    if map_location is None:
        map_location = torch.device("cpu")

    try:
        # PyTorch 2.4+ supports weights_only as default and parameter
        return torch.load(resolved_path, map_location=map_location, weights_only=weights_only)
    except TypeError:
        # Fallback for older PyTorch versions that do not take weights_only argument
        return torch.load(resolved_path, map_location=map_location)


def _signal_handler(signum: int, frame: Any) -> None:
    """Internal signal dispatcher for SIGINT and SIGTERM."""
    sig_name = "SIGINT (Ctrl+C)" if signum == signal.SIGINT else f"SIGTERM ({signum})"
    print(f"\n[!] Caught {sig_name}. Gracefully flushing progress before termination...", file=sys.stderr)

    for cb in _REGISTERED_CLEANUP_CALLBACKS:
        try:
            cb(signum, frame)
        except Exception as e:
            print(f"[!] Warning: Cleanup callback failed during shutdown: {e}", file=sys.stderr)

    print("[✓] State flushed. Exiting.", file=sys.stderr)
    sys.exit(128 + signum)


def setup_signal_handlers(cleanup_callback: Optional[Callable[[int, Any], None]] = None) -> None:
    """
    Register signal handlers for SIGINT (keyboard interrupt) and SIGTERM
    (cloud VM eviction or Colab timeout) to allow clean checkpoint commits.
    """
    if cleanup_callback is not None and cleanup_callback not in _REGISTERED_CLEANUP_CALLBACKS:
        _REGISTERED_CLEANUP_CALLBACKS.append(cleanup_callback)

    try:
        signal.signal(signal.SIGINT, _signal_handler)
        signal.signal(signal.SIGTERM, _signal_handler)
    except (ValueError, AttributeError):
        # In non-main threads or some restricted notebook environments
        pass
