"""
System hardware monitoring, RAM usage verification, and memory safety guards.
Directly inspired by memory management patterns in multi-horizon-ofi.
"""

from __future__ import annotations

import os
import gc
import logging
from typing import Dict, Any, Optional, Tuple

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



def get_total_vram_gb() -> float:
    """Return total GPU VRAM in Gigabytes (GB)."""
    if torch is None or not torch.cuda.is_available():
        return 0.0
    try:
        dev = torch.cuda.current_device()
        return torch.cuda.get_device_properties(dev).total_memory / 1e9
    except Exception:
        return 0.0


def get_gpu_memory_info() -> Dict[str, Any]:
    """Return GPU memory allocation and reservation metrics in GB."""
    if torch is None or not torch.cuda.is_available():
        return {
            "cuda_available": False,
            "device_name": "CPU",
            "allocated_gb": 0.0,
            "reserved_gb": 0.0,
            "max_allocated_gb": 0.0,
            "total_gb": 0.0,
        }

    dev = torch.cuda.current_device()
    total_gb = get_total_vram_gb()
    return {
        "cuda_available": True,
        "device_name": torch.cuda.get_device_name(dev),
        "allocated_gb": torch.cuda.memory_allocated(dev) / 1e9,
        "reserved_gb": torch.cuda.memory_reserved(dev) / 1e9,
        "max_allocated_gb": torch.cuda.max_memory_allocated(dev) / 1e9,
        "total_gb": round(total_gb, 2),
    }


def auto_configure_hardware(
    device: str = "auto",
    requested_train_batch_size: Optional[int] = None,
    requested_eval_batch_size: Optional[int] = None,
    requested_workers: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Automatically detects available CPU, RAM, GPU, and VRAM to dynamically configure:
    - Optimal training batch size (to prevent CUDA OOM while saturating Tensor Cores)
    - Optimal evaluation batch size
    - Optimal DataLoader worker thread count (matching physical cores without thrashing)
    - Recommended PyTorch AMP mode (fp16 / bf16)

    Returns:
        dict containing 'device', 'train_batch_size', 'eval_batch_size', 'workers',
        'total_ram_gb', 'total_vram_gb', 'gpu_name', and 'amp_mode'.
    """
    total_ram = get_total_ram_gb()
    avail_ram = get_available_ram_gb()
    vram_gb = get_total_vram_gb()
    cpu_count = os.cpu_count() or 2

    # 1. Resolve Device
    actual_device = "cpu"
    if device in ("cuda", "auto") and torch is not None and torch.cuda.is_available():
        actual_device = "cuda"

    gpu_name = torch.cuda.get_device_name(0) if actual_device == "cuda" else "CPU"

    # 2. Worker Count Allocation (match CPU cores safely without overloading Colab 2-core container)
    if requested_workers is not None and requested_workers > 0:
        optimal_workers = requested_workers
    else:
        if actual_device == "cpu":
            optimal_workers = 0  # Avoid IPC overhead on CPU
        else:
            # Colab standard instance has 2 vCPUs; high-RAM has 4-8. Keep headroom.
            optimal_workers = min(max(cpu_count // 2, 2), 4)

    # 3. Dynamic Batch Size Configuration based on VRAM
    # VRAM scaling profiles:
    # >= 32 GB (A100, V100-32GB): train=32, eval=16
    # >= 16 GB (T4-16GB, L4-24GB, V100-16GB): train=16, eval=8
    # >= 10 GB (RTX 3080/4080, T4): train=8, eval=8
    # >= 6 GB  (RTX 3060, T4 Colab free): train=4, eval=4
    # < 6 GB or CPU: train=2, eval=2
    if actual_device == "cuda":
        if vram_gb >= 30.0:
            default_train_batch = 32
            default_eval_batch = 16
        elif vram_gb >= 14.0:
            # Tesla T4 standard in Colab (~15.0 - 15.8 GB)
            default_train_batch = 16
            default_eval_batch = 8
        elif vram_gb >= 8.0:
            default_train_batch = 8
            default_eval_batch = 4
        elif vram_gb >= 4.0:
            default_train_batch = 4
            default_eval_batch = 2
        else:
            default_train_batch = 2
            default_eval_batch = 2
    else:
        # CPU Mode
        if avail_ram >= 12.0:
            default_train_batch = 4
            default_eval_batch = 4
        else:
            default_train_batch = 2
            default_eval_batch = 2

    final_train_batch = requested_train_batch_size if (requested_train_batch_size is not None and requested_train_batch_size > 0) else default_train_batch
    final_eval_batch = requested_eval_batch_size if (requested_eval_batch_size is not None and requested_eval_batch_size > 0) else default_eval_batch

    # Determine recommended AMP dtype
    amp_mode = "fp16"
    if actual_device == "cuda" and torch.cuda.is_bf16_supported():
        amp_mode = "bf16"

    return {
        "device": actual_device,
        "gpu_name": gpu_name,
        "total_ram_gb": round(total_ram, 1),
        "avail_ram_gb": round(avail_ram, 1),
        "total_vram_gb": round(vram_gb, 1),
        "cpu_count": cpu_count,
        "train_batch_size": final_train_batch,
        "eval_batch_size": final_eval_batch,
        "workers": optimal_workers,
        "amp_mode": amp_mode,
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


def dynamic_tune_batch_size(
    current_batch_size: int,
    device: str = "cuda",
    min_batch_size: int = 1,
    max_batch_size: int = 64,
    vram_headroom_target: float = 0.80,
    ram_headroom_target: float = 0.80,
) -> Tuple[int, str]:
    """
    On-the-fly batch size tuner that dynamically scales batch size up or down
    during execution to maximize GPU/RAM utilization without risking Out-of-Memory (OOM).

    Strategy:
    - If VRAM or RAM usage is critical (> 90%), aggressively halve batch size (-50%).
    - If utilization is low (< 50% of available VRAM/RAM), scale up batch size (+25% to +50%)
      to saturate Tensor Cores and maximize throughput.
    - If utilization is optimal (50% - 85%), keep batch size stable.

    Returns:
        (new_batch_size, reason_message)
    """
    ram_usage_pct = get_ram_usage_percent()
    avail_ram = get_available_ram_gb()

    # Check RAM pressure first
    if ram_usage_pct > 90.0 or avail_ram < 1.0:
        new_bs = max(min_batch_size, current_batch_size // 2)
        if new_bs < current_batch_size:
            return new_bs, f"High RAM pressure ({ram_usage_pct:.1f}% used). Scaled down {current_batch_size} -> {new_bs}"

    # If CUDA device, inspect GPU VRAM utilization
    if device == "cuda" and torch is not None and torch.cuda.is_available():
        dev = torch.cuda.current_device()
        total_vram = torch.cuda.get_device_properties(dev).total_memory / 1e9
        allocated_vram = torch.cuda.memory_allocated(dev) / 1e9
        reserved_vram = torch.cuda.memory_reserved(dev) / 1e9

        utilization_ratio = reserved_vram / max(total_vram, 1e-3)

        # Danger zone: VRAM usage exceeds 88%
        if utilization_ratio > 0.88:
            new_bs = max(min_batch_size, int(current_batch_size * 0.75))
            if new_bs < current_batch_size:
                deep_cleanup_memory()
                return new_bs, f"VRAM saturation ({utilization_ratio*100:.1f}% of {total_vram:.1f}GB). Throttled {current_batch_size} -> {new_bs}"

        # Under-utilized zone: VRAM usage is under 45% and RAM has plenty of room
        elif utilization_ratio < 0.45 and ram_usage_pct < (ram_headroom_target * 100.0):
            # Scale up to saturate GPU Tensor Cores
            scale_factor = 2 if utilization_ratio < 0.25 else 1.5
            new_bs = min(max_batch_size, max(current_batch_size + 1, int(current_batch_size * scale_factor)))
            if new_bs > current_batch_size:
                return new_bs, f"Low VRAM utilization ({utilization_ratio*100:.1f}% of {total_vram:.1f}GB). Boosted throughput {current_batch_size} -> {new_bs}"

    # CPU Mode under-utilization
    elif device == "cpu":
        if ram_usage_pct < 45.0 and avail_ram > 8.0:
            new_bs = min(max_batch_size, current_batch_size + 2)
            if new_bs > current_batch_size:
                return new_bs, f"Ample RAM available ({avail_ram:.1f}GB free). Increased CPU batch size {current_batch_size} -> {new_bs}"

    return current_batch_size, "Optimal hardware utilization"

