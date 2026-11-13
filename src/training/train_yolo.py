"""
Ultralytics YOLO-OBB PyTorch Training Engine
============================================
Official Reference & Implementation:
    Ultralytics YOLO Oriented Object Detection Engine
    Official Repository: https://github.com/ultralytics/ultralytics
    Documentation: https://docs.ultralytics.com/modes/train/

Supported Models:
    yolov8n-obb, yolov8s-obb, yolov8m-obb, yolov8l-obb, yolov8x-obb,
    yolo11n-obb, yolo11s-obb, yolo11m-obb, yolo11l-obb, yolo11x-obb.

Features:
    - GPU acceleration with Tensor Core FP16 Automatic Mixed Precision (AMP).
    - Multi-worker parallel data loading with automatic mosaic and affine augmentations.
    - Continuous validation tracking and automated checkpoint persistence to Google Drive.
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Dict, Any, Optional
import torch

from src.config import DEFAULT_WEIGHTS_DIR, resolve_pipeline_paths
from src.data.yolo_converter import prepare_dataset_for_yolo_training
from src.utils.env import is_colab, is_drive_mounted, get_drive_root
from src.utils.system import auto_configure_hardware


def _make_staged_trainer(staging):
    """Build an OBBTrainer subclass whose dataset resolves every image read
    through the staged-or-source resolver (W2 overlap for the YOLO family,
    review A2).

    The dataset inventory comes from the SOURCE root (complete and stable),
    so Ultralytics never sees a half-copied directory; each training read
    lands on the staged copy once that copy is size-complete, and the
    im_files entry is then rewritten so later epochs read local only.
    """
    from training.staging import SourceResolver
    from ultralytics.models.yolo.obb import OBBTrainer
    from ultralytics.data.dataset import YOLODataset
    from ultralytics.data.utils import get_split_fraction
    from ultralytics.utils.torch_utils import unwrap_model

    resolver = SourceResolver(staging.src_root, staging.dst_root,
                              staging.source_reads_value)

    class StagedYOLODataset(YOLODataset):
        def load_image(self, i, rect_mode=True):
            f = self.im_files[i]
            rf = resolver.resolve(f)
            if rf != f:
                self.im_files[i] = rf  # stick to the landed local copy
            return super().load_image(i, rect_mode)

    class StagedOBBTrainer(OBBTrainer):
        def build_dataset(self, img_path, mode="train", batch=None):
            gs = max(int(unwrap_model(self.model).stride.max()), 32)
            return StagedYOLODataset(
                img_path=img_path,
                imgsz=self.args.imgsz,
                batch_size=batch,
                augment=mode == "train",
                hyp=self.args,
                rect=self.args.rect or mode == "val",
                cache=self.args.cache or None,
                single_cls=self.args.single_cls or False,
                stride=int(gs),
                pad=0.0 if mode == "train" else 0.5,
                prefix=f"{mode}: ",
                task="obb",
                classes=self.args.classes,
                data=self.data,
                fraction=get_split_fraction(
                    self.args.fraction,
                    "train" if mode == "train" else self.args.split),
            )

    return StagedOBBTrainer


def train_yolo_obb(
    model_name: str = "yolov8n-obb",
    dataset_name: str = "visdrone",
    epochs: int = 20,
    batch_size: int = 16,
    imgsz: int = 640,
    device: str = "auto",
    workers: int = 4,
    lr0: float = 0.01,
    cache: Optional[str] = None,
    pretrained_weights: Optional[str] = None,
    resume: bool = True,
    project_dir: Optional[str] = None,
    data_dir: Optional[str] = None,
    patience: Optional[int] = None,
    cos_lr: bool = True,
    close_mosaic: int = 10,
    # ---- v4 additions (W1/W7; accepted from the unified pipeline) ---------
    run_dir: Optional[str] = None,
    staging=None,
    seed: int = 42,
    deterministic: bool = False,
    mirror_dir: Optional[str] = None,
    mirror_every: int = 5,
    test_mode: bool = False,
    max_batches: Optional[int] = None,
    # accepted for interface parity with the dense/DETR engines; Ultralytics
    # owns these behaviors internally (its own EMA, augmentation, letterbox):
    micro_batch: Optional[int] = None,
    ema: Optional[bool] = None,
    augment: Optional[bool] = None,
    resize_mode: Optional[str] = None,
    compile_model: Optional[bool] = None,
    weight_decay: Optional[float] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Train a YOLO-OBB model on a given dataset with full GPU acceleration and epoch checkpointing.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("Ultralytics is required for YOLO training. Run: pip install ultralytics")

    # 1. Resolve Computing Device & Auto Hardware Tuning
    hw_config = auto_configure_hardware(
        device=device,
        requested_train_batch_size=batch_size,
        requested_workers=workers,
    )
    if device == "auto":
        dev = "0" if hw_config["device"] == "cuda" else "cpu"
    elif device.startswith("cuda"):
        dev = "0"
    else:
        dev = "cpu"

    batch_size = hw_config["train_batch_size"]
    workers = hw_config["workers"]

    print(f"\n{'=' * 70}")
    print(f" Starting GPU Training: {model_name} on {dataset_name.upper()}")
    print(f"{'=' * 70}")
    print(f"[*] Target Hardware : {'NVIDIA CUDA GPU (device 0)' if dev != 'cpu' else 'CPU'}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"[*] GPU Name        : {gpu_name} ({gpu_mem:.2f} GB VRAM)")
    print(f"[*] Training Config : {epochs} Epochs | Batch Size: {batch_size} | Image Size: {imgsz}px | Workers: {workers}")

    # 2. Prepare Dataset YAML. W2 fix (review A2): with staging active the
    # YAML points at the SOURCE root (complete, stable listing from second 0;
    # run.py passes staging.src_root as data_dir). Training READS are
    # redirected to staged copies as they land by the StagedOBBTrainer below,
    # so YOLO overlaps staging like the other families instead of scanning a
    # half-copied directory. Review R4b: the Ultralytics labels.cache
    # under the SOURCE root is KEPT (it lives on the user's Drive when the
    # source is Drive); deleting it every run forces the full label scan to
    # re-read every image on every run.
    effective_data_dir = data_dir
    if staging is not None:
        effective_data_dir = staging.src_root
        print(f"[*] Dataset inventory from source root: {effective_data_dir} "
              f"(reads overlap staging via the resolver)")
    print(f"[*] Verifying YOLO-OBB dataset format for '{dataset_name}'...")
    yaml_path = prepare_dataset_for_yolo_training(dataset_name, data_dir=Path(effective_data_dir) if effective_data_dir else None)
    print(f"[✓] Dataset YAML ready at: {yaml_path}")

    # W2 overlap for YOLO (review A2): a custom Ultralytics trainer whose
    # dataset lists from the source but resolves every image read through
    # the staged-or-source resolver (staged copy once size-complete, else a
    # counted source read). Once a staged copy lands, the im_files entry is
    # rewritten so later epochs read local only. The one-time Ultralytics
    # label-scan header reads still hit the source for un-staged files
    # (headers only, once, then labels.cache) - disclosed in the report.
    staged_trainer_cls = None
    if staging is not None:
        try:
            staged_trainer_cls = _make_staged_trainer(staging)
            print("[*] YOLO staging overlap ENABLED (staged-or-source reads)")
        except Exception as e:
            print(f"[!] YOLO staging overlap unavailable ({e}); falling back "
                  f"to blocking staging for correctness (overlap NOT achieved "
                  f"for this family - disclosed spec gap)", flush=True)
            if not staging.wait():
                # B2: explicit error, never a silent hang on failed copies.
                raise RuntimeError(
                    f"staging incomplete for the blocking YOLO fallback: "
                    f"{staging.errors[:5]}")
            effective_data_dir = staging.dst_root
            yaml_path = prepare_dataset_for_yolo_training(
                dataset_name, data_dir=Path(effective_data_dir))

    # 3. Setup Project & Weights Directory (v4: run_dir wins so the unified
    # pipeline's runs/train/<dataset>_<model> contract holds, §4.4)
    paths = resolve_pipeline_paths()
    if run_dir is not None:
        save_dir = Path(run_dir).resolve()
        weights_dir = save_dir.parent.resolve()
        save_dir.mkdir(parents=True, exist_ok=True)
    else:
        weights_dir = Path(project_dir or paths["weights_dir"]).resolve()
        weights_dir.mkdir(parents=True, exist_ok=True)
        save_dir = (weights_dir / f"{model_name}_{dataset_name}").resolve()

    run_name = save_dir.name
    last_pt = save_dir / "weights" / "last.pt"
    best_pt = save_dir / "weights" / "best.pt"

    # W7.1 (review R4c): pull surviving checkpoints back from the Drive
    # mirror BEFORE the resume probe - a wiped Colab session leaves only
    # the mirror copy.
    if mirror_dir:
        try:
            from src.training.engine_common import restore_from_mirror
            restore_from_mirror(mirror_dir, f"{dataset_name}_{model_name}",
                                save_dir)
        except Exception as e:
            print(f"[mirror][warn] restore-from-mirror failed ({e}); "
                  f"continuing with local files", flush=True)

    # Check for resumption
    from src.training.yolo_resume import extension_trainer, checkpoint_progress
    can_resume = False
    completed_before = 0
    stripped_extension = False
    if resume and last_pt.exists():
        try:
            # Check if checkpoint retains optimizer/epoch training state.
            # weights_only=False (review A2): torch>=2.6 defaults to True and
            # Ultralytics checkpoints are NOT plain-tensor dicts, so the
            # default load fails and resume was silently disabled.
            ckpt_obj = torch.load(str(last_pt), map_location="cpu",
                                  weights_only=False)
            completed_before = checkpoint_progress(ckpt_obj, save_dir)
            if completed_before >= epochs:
                return {"model_name": model_name, "dataset_name": dataset_name, "run_dir": str(save_dir),
                        "best_weights": str(best_pt if best_pt.exists() else last_pt), "last_weights": str(last_pt),
                        "epochs_trained": completed_before, "epochs_completed": completed_before}
            if int(ckpt_obj.get("epoch", -1)) >= 0 and ckpt_obj.get("optimizer") is not None:
                can_resume = True
                print(f"[*] Found resumable training checkpoint at: {last_pt}. Resuming training seamlessly...")
                model = YOLO(str(last_pt))
            else:
                print(f"[resume] stripped legacy checkpoint: continue at global epoch {completed_before}; optimizer moments cannot be recovered")
                # Keep the original untouched. Native stripped files reset epoch=-1;
                # recover global progress from CSV rather than restarting counters.
                ckpt_obj["epoch"] = completed_before-1
                recovered = save_dir / "resume_input.pt"
                torch.save(ckpt_obj, recovered)
                model = YOLO(str(recovered))
                can_resume = completed_before > 0
        except Exception as e:  # W10 audit: never swallow silently
            print(f"[resume][warn] checkpoint probe failed ({e}); loading "
                  f"weights without optimizer state", flush=True)
            can_resume = False
            model = YOLO(str(last_pt))
    else:
        init_weights = pretrained_weights
        if not init_weights:
            candidate_weights = weights_dir / f"{model_name}.pt"
            if candidate_weights.exists():
                init_weights = str(candidate_weights)
            elif test_mode:
                # smoke mode: skip pretrained-weight downloads entirely and
                # build the architecture from its bundled YAML (offline-safe)
                init_weights = ("yolo11n-obb.yaml" if model_name == "yolo11-obb"
                                else f"{model_name}.yaml")
            elif model_name == "yolo11-obb":
                init_weights = "yolo11n-obb.pt"
            else:
                init_weights = f"{model_name}.pt"
        print(f"[*] Initializing model with base weights: {init_weights}")
        try:
            model = YOLO(init_weights)
        except Exception as e:
            # v4 fallback: pretrained .pt unavailable (offline / not yet
            # released) -> build the architecture from its YAML and train
            # from scratch instead of crashing the suite (W1 failure isolation).
            yaml_cfg = init_weights.replace(".pt", ".yaml")
            print(f"[!] Could not load '{init_weights}' ({e}); "
                  f"falling back to from-scratch config '{yaml_cfg}'")
            model = YOLO(yaml_cfg)

    # 4. Execute Native Ultralytics Training
    start_time = time.time()
    train_args = {
        "data": str(yaml_path),
        "epochs": epochs-completed_before if stripped_extension else epochs,
        "batch": batch_size,
        "imgsz": imgsz,
        "device": dev,
        "workers": workers if dev != "cpu" else 0,
        "lr0": lr0,
        "save_period": 1,        # Save checkpoint after EVERY single epoch
        "save": True,
        "project": str(weights_dir),
        "name": run_name,
        "exist_ok": True,
        "verbose": True,
        "resume": can_resume,
        "cos_lr": cos_lr,            # cosine LR decay — best on cleaned 1280px data
        "close_mosaic": close_mosaic, # disable mosaic near the end so small-box
                                      # regression settles (10 epochs default)
        "seed": seed,
        "deterministic": deterministic,
    }
    if patience is not None:
        train_args["patience"] = patience
    if test_mode:
        # smoke mode: skip plot rendering (large transient memory spike on
        # small-CI boxes) — training/val metrics are unaffected
        train_args["plots"] = False
    if weight_decay is not None:
        train_args["weight_decay"] = weight_decay
    if cache and str(cache).lower() in ("ram", "disk", "true", "1"):
        train_args["cache"] = str(cache).lower() if str(cache).lower() in ("ram", "disk") else True

    def _record_last_progress(trainer):
        import hashlib, json
        path = Path(trainer.last)
        if not path.is_file():
            return
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024*1024), b""):
                digest.update(block)
        from src.utils.checkpoint import atomic_save_json
        atomic_save_json({"epoch": int(trainer.epoch)+1, "checkpoint_size": path.stat().st_size,
                          "sha256": digest.hexdigest()}, path.with_name(path.name+".progress.json"))
    model.add_callback("on_model_save", _record_last_progress)

    # W7.2 (review R4c + B-round): mirror ASYNCHRONOUSLY via DriveMirror,
    # INCLUDING best.pt. Callback order verified on installed Ultralytics
    # 8.4.211 (engine/trainer.py L679-693): save_model() -> on_model_save
    # -> on_fit_epoch_end. We hook on_model_save so the checkpoint exists
    # when we read it, and SNAPSHOT to a local temp name first: the async
    # DriveMirror thread must never be reading last.pt while the NEXT
    # epoch's save_model rewrites it (Ultralytics writes in place, so an
    # unsnapshotted async copy can be torn).
    _drive_mirror = None
    if mirror_dir:
        from src.training.engine_common import DriveMirror
        _drive_mirror = DriveMirror(mirror_dir, every_n_epochs=1,
                                    run_subdir=f"{dataset_name}_{model_name}")
        _snap_dir = save_dir / ".mirror_snapshot"

        def _mirror_on_save(_trainer):
            try:
                _snap_dir.mkdir(parents=True, exist_ok=True)
                epoch = getattr(_trainer, "epoch", 0) or 0
                if (epoch+1) % max(1, mirror_every) and epoch+1 < _trainer.epochs:
                    return
                for f in (last_pt, best_pt, save_dir / "results.csv", save_dir / "train.log", save_dir / "args.yaml"):
                    if f.exists():
                        # B-round note 2: UNIQUE snapshot name per save -
                        # a fixed name could be overwritten by the next
                        # epoch while the async thread is still reading it
                        # (torn mirror). The mirrored file keeps the
                        # canonical basename via dest_name; the snapshot is
                        # unlinked after a successful copy.
                        if f.name == "last.pt":
                            import json, hashlib
                            hasher = hashlib.sha256()
                            with f.open("rb") as stream:
                                for block in iter(lambda: stream.read(1024*1024), b""):
                                    hasher.update(block)
                            digest = hasher.hexdigest()
                            progress = _snap_dir / f"{f.stem}_e{epoch}.progress.json"
                            progress.write_text(json.dumps({"epoch": epoch+1, "checkpoint_size": f.stat().st_size, "sha256": digest}))
                            _drive_mirror.submit(progress, dest_name=f.name+".progress.json", unlink_after=True)
                        snap = _snap_dir / f"{f.stem}_e{epoch}{f.suffix}"
                        shutil.copy2(f, snap)
                        _drive_mirror.submit(snap, dest_name=f.name,
                                             unlink_after=True)
            except Exception as e:
                print(f"[mirror][warn] snapshot failed: {e}")

        model.add_callback("on_model_save", _mirror_on_save)

    from ultralytics.models.yolo.obb import OBBTrainer
    train_args["trainer"] = extension_trainer(staged_trainer_cls or OBBTrainer, train_args["epochs"])

    results = model.train(**train_args)
    elapsed_minutes = (time.time() - start_time) / 60.0

    print(f"\n[✓] Training completed in {elapsed_minutes:.2f} minutes!")
    mirror_errors = []
    if _drive_mirror is not None:
        mirror_errors = _drive_mirror.finish()  # drain the async mirror before end-of-run work

    # 5. Persist Best and Last Checkpoints
    final_best = save_dir / "weights" / "best.pt"
    final_last = save_dir / "weights" / "last.pt"

    target_best = weights_dir / f"{model_name}_{dataset_name}_best.pt"
    if final_best.exists() and final_best.resolve() != target_best.resolve():
        shutil.copy2(final_best, target_best)
        print(f"[✓] Saved best model checkpoint to: {target_best}")

    # Synchronize to Google Drive if active
    if is_drive_mounted():
        drive_weights = get_drive_root() / "weights"
        drive_weights.mkdir(parents=True, exist_ok=True)
        if target_best.exists():
            dest = drive_weights / target_best.name
            if target_best.resolve() != dest.resolve():
                shutil.copy2(target_best, dest)
                print(f"[✓] Mirrored checkpoint to Google Drive: {dest}")

    # v4 (W7.2): explicit mirror directory from the unified pipeline —
    # mirror the whole run directory (weights + results.csv + args).
    if mirror_dir:
        try:
            mirror_root = Path(mirror_dir) / save_dir.name
            mirror_root.mkdir(parents=True, exist_ok=True)
            for item in save_dir.rglob("*"):
                if item.is_file() and ".mirror_snapshot" not in item.parts:
                    rel = item.relative_to(save_dir)
                    dest = mirror_root / rel
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(item, dest)
            print(f"[✓] Mirrored run directory to: {mirror_root}")
        except Exception as e:
            print(f"[!] Mirror to {mirror_dir} failed (training unaffected): {e}")

    # Surface final metrics for the pipeline manifest (W7.1) when available
    best_map50 = None
    try:
        rd = getattr(results, "results_dict", None) or {}
        for k, v in rd.items():
            if "mAP50" in str(k) and "95" not in str(k):
                best_map50 = float(v)
    except Exception as e:  # W10 audit: never swallow silently
        print(f"[metrics][warn] could not parse Ultralytics results_dict: {e}")
        best_map50 = None

    completed_actual = int(getattr(model.trainer, "epoch", -1))+1
    if stripped_extension:
        completed_actual += completed_before
    return {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "best_weights": str(target_best if target_best.exists() else final_best),
        "last_weights": str(final_last),
        "elapsed_minutes": elapsed_minutes,
        "mirror_errors": mirror_errors,
        "epochs_completed": completed_actual,
        "epochs_trained": completed_actual,
        "best_map50": best_map50,
        "best_val_loss": None,
        "oom_events": 0,
        "run_dir": str(save_dir),
    }
