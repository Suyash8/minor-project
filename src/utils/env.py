"""
Environment detection and deterministic seeding utilities.
"""

import os
import random
import sys
from typing import Dict, Any

def is_colab() -> bool:
    """Check if code is executing inside a Google Colab notebook environment."""
    return "google.colab" in sys.modules or os.path.exists("/content")

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
    except ImportError:
        pass
