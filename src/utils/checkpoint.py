"""
Atomic Checkpointing and Resumption Engine.
Provides crash-resilient persistence for evaluations, training loops, and batch inference,
preventing data loss during online GPU disconnections or timeouts (inspired by multi-horizon-ofi).
"""

from __future__ import annotations

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import datetime

try:
    import torch
except ImportError:
    torch = None

logger = logging.getLogger("checkpoint")


def atomic_save_json(data: Any, filepath: Union[Path, str], indent: int = 2) -> None:
    """
    Atomically persist a JSON-serializable dictionary to disk.
    Writes first to a `.tmp` file and replaces the destination file atomically,
    ensuring no half-written or corrupted files occur during unexpected crashes.
    """
    target_path = Path(filepath).resolve()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target_path.with_name(f"{target_path.name}.tmp")

    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, default=str)
        f.flush()
        os.fsync(f.fileno())

    os.replace(temp_path, target_path)


def atomic_save_weights(state_dict: dict, filepath: Union[Path, str]) -> None:
    """
    Atomically persist PyTorch model weights to disk using temporary file swap.
    """
    if torch is None:
        raise ImportError("PyTorch is required to save weights.")

    target_path = Path(filepath).resolve()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target_path.with_name(f"{target_path.name}.tmp")

    torch.save(state_dict, temp_path)
    os.replace(temp_path, target_path)


def get_checkpoint_dir(results_dir: Union[Path, str], run_id: str) -> Path:
    """Return the checkpoint directory for a specific run ID."""
    ckpt_dir = Path(results_dir) / run_id / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return ckpt_dir


def _eval_checkpoint_path(results_dir: Union[Path, str], run_id: str, dataset_name: str, model_name: str) -> Path:
    return get_checkpoint_dir(results_dir, run_id) / f"{dataset_name}_{model_name}_eval.json"


def _batch_checkpoint_path(results_dir: Union[Path, str], run_id: str, dataset_name: str, model_name: str) -> Path:
    return get_checkpoint_dir(results_dir, run_id) / f"{dataset_name}_{model_name}_batch.json"


def _manifest_path(results_dir: Union[Path, str], run_id: str) -> Path:
    return Path(results_dir) / run_id / "run_manifest.json"


def is_evaluation_completed(
    results_dir: Union[Path, str],
    run_id: str,
    dataset_name: str,
    model_name: str,
) -> bool:
    """Check if a specific dataset-model evaluation was already completed."""
    ckpt_path = _eval_checkpoint_path(results_dir, run_id, dataset_name, model_name)
    if not ckpt_path.exists():
        return False
    try:
        with open(ckpt_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return bool(data.get("completed", False))
    except Exception as e:
        logger.warning(f"Failed to read checkpoint for {dataset_name}/{model_name}: {e}")
        return False


def load_evaluation_checkpoint(
    results_dir: Union[Path, str],
    run_id: str,
    dataset_name: str,
    model_name: str,
) -> Optional[Dict[str, Any]]:
    """Load evaluation checkpoint for a specific dataset-model pair."""
    ckpt_path = _eval_checkpoint_path(results_dir, run_id, dataset_name, model_name)
    if not ckpt_path.exists():
        return None
    try:
        with open(ckpt_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Error loading checkpoint {ckpt_path}: {e}")
        return None


def save_evaluation_checkpoint(
    results_dir: Union[Path, str],
    run_id: str,
    dataset_name: str,
    model_name: str,
    summary_item: Dict[str, Any],
    eval_details: Optional[Dict[str, Any]] = None,
    completed: bool = True,
) -> None:
    """Atomically persist evaluation checkpoint metadata for a model-dataset run."""
    ckpt_path = _eval_checkpoint_path(results_dir, run_id, dataset_name, model_name)
    payload = {
        "dataset": dataset_name,
        "model": model_name,
        "completed": completed,
        "timestamp": datetime.datetime.now().isoformat(),
        "summary": summary_item,
        "details": eval_details or {},
    }
    atomic_save_json(payload, ckpt_path)
    # Clear any residual mid-batch progress checkpoint once evaluation completes
    if completed:
        clear_batch_progress(results_dir, run_id, dataset_name, model_name)


def save_batch_progress(
    results_dir: Union[Path, str],
    run_id: str,
    dataset_name: str,
    model_name: str,
    batch_idx: int,
    total_batches: int,
    samples_processed: int,
    total_samples: int,
    elapsed_seconds: float,
) -> None:
    """Save in-progress batch checkpoint so progress can be monitored and resumed."""
    ckpt_path = _batch_checkpoint_path(results_dir, run_id, dataset_name, model_name)
    payload = {
        "dataset": dataset_name,
        "model": model_name,
        "batch_idx": batch_idx,
        "total_batches": total_batches,
        "samples_processed": samples_processed,
        "total_samples": total_samples,
        "progress_pct": round((samples_processed / max(1, total_samples)) * 100, 1),
        "elapsed_seconds": round(elapsed_seconds, 2),
        "updated_at": datetime.datetime.now().isoformat(),
    }
    atomic_save_json(payload, ckpt_path)


def load_batch_progress(
    results_dir: Union[Path, str],
    run_id: str,
    dataset_name: str,
    model_name: str,
) -> Optional[Dict[str, Any]]:
    """Load in-progress batch checkpoint if it exists."""
    ckpt_path = _batch_checkpoint_path(results_dir, run_id, dataset_name, model_name)
    if not ckpt_path.exists():
        return None
    try:
        with open(ckpt_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def clear_batch_progress(
    results_dir: Union[Path, str],
    run_id: str,
    dataset_name: str,
    model_name: str,
) -> None:
    """Remove mid-batch checkpoint after clean completion."""
    ckpt_path = _batch_checkpoint_path(results_dir, run_id, dataset_name, model_name)
    if ckpt_path.exists():
        try:
            ckpt_path.unlink()
        except OSError:
            pass


def save_run_manifest(
    results_dir: Union[Path, str],
    run_id: str,
    summary_rows: List[Dict[str, Any]],
    completed_pairs: List[Dict[str, str]],
    pending_pairs: List[Dict[str, str]],
    is_completed: bool = False,
) -> None:
    """Persist master run manifest tracking overall benchmark execution state."""
    manifest_p = _manifest_path(results_dir, run_id)
    payload = {
        "run_id": run_id,
        "is_completed": is_completed,
        "updated_at": datetime.datetime.now().isoformat(),
        "total_pairs": len(completed_pairs) + len(pending_pairs),
        "completed_count": len(completed_pairs),
        "pending_count": len(pending_pairs),
        "completed_pairs": completed_pairs,
        "pending_pairs": pending_pairs,
        "summaries": summary_rows,
    }
    atomic_save_json(payload, manifest_p)


def load_run_manifest(results_dir: Union[Path, str], run_id: str) -> Optional[Dict[str, Any]]:
    """Load existing master run manifest."""
    manifest_p = _manifest_path(results_dir, run_id)
    if not manifest_p.exists():
        return None
    try:
        with open(manifest_p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to read manifest {manifest_p}: {e}")
        return None
