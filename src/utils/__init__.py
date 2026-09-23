"""
Utility modules for environment detection, visualization, reporting,
atomic checkpointing, memory protection, and system security.
"""

from __future__ import annotations

from .env import is_colab, is_drive_mounted, get_drive_root, get_device_info, set_seed
from .system import (
    get_available_ram_gb,
    get_total_ram_gb,
    get_ram_usage_percent,
    get_gpu_memory_info,
    deep_cleanup_memory,
    check_memory_pressure,
    format_memory_summary,
)
from .security import (
    safe_path_join,
    is_safe_filename,
    safe_torch_load,
    setup_signal_handlers,
)
from .checkpoint import (
    atomic_save_json,
    atomic_save_weights,
    get_checkpoint_dir,
    is_evaluation_completed,
    load_evaluation_checkpoint,
    save_evaluation_checkpoint,
    save_batch_progress,
    load_batch_progress,
    clear_batch_progress,
    save_run_manifest,
    load_run_manifest,
)
from .visualizer import (
    plot_confusion_matrix,
    plot_angle_correlation,
    plot_benchmark_comparison,
)
from .reporter import format_metrics_table, generate_markdown_report

__all__ = [
    # Environment
    "is_colab",
    "is_drive_mounted",
    "get_drive_root",
    "get_device_info",
    "set_seed",
    # System & Memory
    "get_available_ram_gb",
    "get_total_ram_gb",
    "get_ram_usage_percent",
    "get_gpu_memory_info",
    "deep_cleanup_memory",
    "check_memory_pressure",
    "format_memory_summary",
    # Security
    "safe_path_join",
    "is_safe_filename",
    "safe_torch_load",
    "setup_signal_handlers",
    # Checkpointing
    "atomic_save_json",
    "atomic_save_weights",
    "get_checkpoint_dir",
    "is_evaluation_completed",
    "load_evaluation_checkpoint",
    "save_evaluation_checkpoint",
    "save_batch_progress",
    "load_batch_progress",
    "clear_batch_progress",
    "save_run_manifest",
    "load_run_manifest",
    # Visualization & Reporting
    "plot_confusion_matrix",
    "plot_angle_correlation",
    "plot_benchmark_comparison",
    "format_metrics_table",
    "generate_markdown_report",
]
