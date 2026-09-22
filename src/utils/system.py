"""
System hardware monitoring, RAM usage verification, and memory safety guards.
Directly inspired by memory management patterns in multi-horizon-ofi.
"""

from __future__ import annotations

import gc
import logging
from typing import Dict, Any, Optional

try:
    import psutil
except ImportError:
    psutil = None

try:
    import torch
except ImportError:
    torch = None

logger = logging.getLogger("system_memory")


def get_available_ram_gb() -> float:
    """Return available system RAM in Gigabytes (GB)."""
    if psutil is None:
        return 8.0  # Safe fallback estimate if psutil is unavailable
    return psutil.virtual_memory().available / 1e9


def get_total_ram_gb() -> float:
    """Return total system RAM in Gigabytes (GB)."""
    if psutil is None:
        return 16.0
    return psutil.virtual_memory().total / 1e9


def get_ram_usage_percent() -> float:
    """Return system RAM usage percentage (0.0 to 100.0)."""
    if psutil is None:
        return 50.0
    return psutil.virtual_memory().percent


def get_gpu_memory_info() -> Dict[str, Any]:
    """Return GPU memory allocation and reservation metrics in GB."""
    if torch is None or not torch.cuda.is_available():
        return {
            "cuda_available": False,
            "device_name": "CPU",
            "allocated_gb": 0.0,
            "reserved_gb": 0.0,
            "max_allocated_gb": 0.0,
        }

    dev = torch.cuda.current_device()
    return {
        "cuda_available": True,
        "device_name": torch.cuda.get_device_name(dev),
        "allocated_gb": torch.cuda.memory_allocated(dev) / 1e9,
        "reserved_gb": torch.cuda.memory_reserved(dev) / 1e9,
        "max_allocated_gb": torch.cuda.max_memory_allocated(dev) / 1e9,
    }


def deep_cleanup_memory() -> None:
    """
    Perform deep memory garbage collection and release cached PyTorch CUDA tensors.
    Safely handles both CPU RAM and GPU VRAM cleanup.
    """
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass


def check_memory_pressure(
    critical_ram_gb: float = 1.0,
    max_usage_pct: float = 90.0,
    auto_clean: bool = True,
) -> Dict[str, Any]:
    """
    Assess system memory health. If RAM pressure is elevated, trigger deep memory
    garbage collection and advise whether batch processing should be throttled.

    Parameters
    ----------
    critical_ram_gb : float
        Threshold in GB below which available RAM is considered critically low.
    max_usage_pct : float
        Threshold percentage above which RAM is considered critically low.
    auto_clean : bool
        If True, invokes deep_cleanup_memory() automatically when pressure is detected.

    Returns
    -------
    dict
        Status summary containing 'status' ('normal', 'warning', 'critical'),
        'available_gb', 'usage_pct', and 'should_throttle'.
    """
    avail = get_available_ram_gb()
    usage = get_ram_usage_percent()

    status = "normal"
    should_throttle = False

    if avail < critical_ram_gb or usage >= max_usage_pct:
        status = "critical"
        should_throttle = True
    elif avail < (critical_ram_gb * 1.5) or usage >= (max_usage_pct - 5.0):
        status = "warning"

    if status in ("warning", "critical") and auto_clean:
        deep_cleanup_memory()
        # Re-check after cleanup
        avail = get_available_ram_gb()
        usage = get_ram_usage_percent()
        if avail >= critical_ram_gb and usage < max_usage_pct:
            status = "recovered"
            should_throttle = False

    return {
        "status": status,
        "available_gb": round(avail, 2),
        "usage_pct": round(usage, 1),
        "should_throttle": should_throttle,
    }


def format_memory_summary() -> str:
    """Format a single-line summary of current CPU and GPU memory usage."""
    avail_gb = get_available_ram_gb()
    usage_pct = get_ram_usage_percent()
    summary = f"RAM: {usage_pct:.1f}% used ({avail_gb:.2f} GB free)"

    gpu_info = get_gpu_memory_info()
    if gpu_info["cuda_available"]:
        summary += f" | VRAM: {gpu_info['allocated_gb']:.2f} GB alloc ({gpu_info['reserved_gb']:.2f} GB res) [{gpu_info['device_name']}]"

    return summary
