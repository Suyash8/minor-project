"""
Environment detection and deterministic seeding utilities.
"""

import os
import random
import sys
from typing import Dict, Any

from pathlib import Path

def is_colab() -> bool:
    """Check if code is executing inside a Google Colab notebook environment."""
    return "google.colab" in sys.modules or os.path.exists("/content")

def is_drive_mounted() -> bool:
    """Check if Google Drive is mounted at /content/drive/MyDrive."""
    drive_base = Path("/content/drive/MyDrive")
    return is_colab() and drive_base.exists()

def get_drive_root() -> Path:
    """Return the designated Google Drive project root for object-detection."""
    return Path("/content/drive/MyDrive/object-detection")


def get_device_info(requested_device: str = "auto") -> Dict[str, Any]:
    """
    Detect available computing hardware (CUDA GPU, Apple MPS, or CPU).
    """
    info = {
        "requested": requested_device,
        "device": "cpu",
        "cuda_available": False,
        "device_name": "CPU",
        "num_devices": 0,
    }

    try:
        import torch
        info["torch_version"] = torch.__version__
        if requested_device == "cpu":
            info["device"] = "cpu"
        elif requested_device == "cuda" and torch.cuda.is_available():
            info["device"] = "cuda"
            info["cuda_available"] = True
            info["device_name"] = torch.cuda.get_device_name(0)
            info["num_devices"] = torch.cuda.device_count()
        elif requested_device == "auto":
            if torch.cuda.is_available():
                info["device"] = "cuda"
                info["cuda_available"] = True
                info["device_name"] = torch.cuda.get_device_name(0)
                info["num_devices"] = torch.cuda.device_count()
            else:
                info["device"] = "cpu"
    except ImportError:
        info["torch_version"] = "not_installed"

    return info

def set_seed(seed: int = 42) -> None:
    """Ensure reproducibility across random, numpy, and torch."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            # Prevent excessive thread contention on CPU
            torch.set_num_threads(min(4, os.cpu_count() or 1))
    except ImportError:
        pass

