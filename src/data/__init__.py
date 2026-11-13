"""
Dataset loaders, download handlers, and synthetic data generation for Aerial OBB.
"""

from __future__ import annotations

from .dataset import AerialOBBDataset
from .downloader import download_dataset, verify_dataset_status

__all__ = [
    "AerialOBBDataset",
    "download_dataset",
    "verify_dataset_status",
]
