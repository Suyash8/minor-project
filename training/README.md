# Training Pipeline — YOLO-OBB suite on the cleaned vehicle datasets

Trains the Ultralytics **YOLO-OBB suite** (YOLOv8 / YOLO11 / YOLO26, n→m
scales; DOTAv1-pretrained, task-specific OBB head with refined angle
decoding and STAL small-target label assignment) on the cleaned 8-class
datasets produced by [`data_cleaning/`](../data_cleaning/). The research
architectures (dense/DETR engines under `src/training/`) train on the same
cleaned data — see [docs/MODEL_TUNING_GUIDE.md](../docs/MODEL_TUNING_GUIDE.md)
for their per-model tuned settings.

Classes (identical in every dataset):
`0 pedestrian · 1 bicycle · 2 car · 3 van · 4 truck · 5 bus · 6 tricycle · 7 motorcyclist`

```
training/
├── common.py          # dataset registry, yaml generation, Drive->local staging, manifests
├── train.py           # main trainer CLI (model x dataset, fully configurable)
├── evaluate.py        # val/test evaluation with per-class table + JSON
├── cross_eval.py      # cross-dataset matrix (all runs x all datasets)
├── summarize_runs.py  # aggregate all runs into summary.csv/.md
└── colab_quickstart.py# one-cell Colab bootstrap (mount, check, install, train)
configs/experiments/   # ready-made experiment presets
```

## Quick start (Colab, GPU runtime)

```python
!git clone https://github.com/Suyash8/minor-project.git /content/minor-project
%cd /content/minor-project
!python training/colab_quickstart.py --cfg configs/experiments/visdrone_yolo26s.yaml
```

`colab_quickstart.py` mounts Drive, checks the cleaned datasets are complete,
installs `ultralytics>=8.4.189`, prints the GPU, and starts training.

Equivalent direct call:

```bash
python training/train.py --datasets visdrone --models yolo26s-obb
```

## Why these defaults (what's tuned for THIS data)

| Setting | Value | Why |
|---|---|---|
| `imgsz` | **1280** | the cleaner pre-resizes every image to 1280 px long side — training at native resolution keeps the small-vehicle detail (no downscale loss) |
| `model` | **yolo26x-obb** (DOTAv1-pretrained) | YOLO26's OBB head has refined angle decoding (+3.4 mAP over YOLO11 on DOTA-v1.0) and STAL small-target label assignment — exactly our regime |
| `epochs` | 150 (DOTA 120) | small objects converge slower; patience-based early stopping decides the real end |
| `patience` | 30 | stops when val mAP50-95 stops improving |
| `batch=-1` | autobatch | picks the largest batch that fits T4/L4/A100 — same command, any GPU |
| `cos_lr` | true | smoother final convergence for small-box regression |
| `close_mosaic=10` | last 10 epochs mosaic-free | lets small-box regression settle on unaugmented images |
| `amp` | true | mixed precision (safe on T4+) |
| staging | Drive -> local disk | Drive FUSE reads dominate epoch time; the trainer copies the dataset to `/content` once, then epochs run at full local-disk speed |

To override anything (full list: Ultralytics train args):

```bash
python training/train.py --datasets dota --models yolo26m-obb \
    --epochs 100 --set lr0=0.002 degrees=10.0 mosaic=1.0
```

`degrees>0` is safe for OBB (Ultralytics rotates the polygon labels too) —
use it for extra rotation robustness if you like.

## CLI reference — train.py

```
--datasets codrone|visdrone|dota|all     (multiple allowed)
--models   yolo26n-obb|yolo26s-obb|yolo26m-obb|yolo26l-obb|yolo26x-obb|yolo11*-obb|yolov8*-obb (multiple allowed)
--cfg      configs/experiments/<preset>.yaml     (CLI args override the file;
                                                  yaml may list several models
                                                  under `models:` for a sweep)
--epochs --imgsz --batch --patience --workers --device
--runs-dir /content/drive/MyDrive/object-detection/runs   (default in Colab: Drive)
--data-root <cleaned_data root>                          (default: auto)
--name     custom run name
--set      key=value ...                       (arbitrary Ultralytics kwargs)
--resume   runs/train/<run>/last.pt
--no-stage-local / --no-eval-test / --dry-run
```

Each run writes: `weights (best.pt/last.pt)`, `results.csv`, plots, a
`run_manifest.json` (model/dataset/git commit/args) and — after training —
`eval_<dataset>_test.json` (best.pt on the test split).

## After training

```bash
# per-class test metrics for one run
python training/evaluate.py --run runs/train/<run>

# cross-dataset matrix: every run x every dataset (test split)
python training/cross_eval.py

# one table of all runs (val best epoch + test metrics + params)
python training/summarize_runs.py
```

Outputs land in the runs dir: `cross_eval.csv/.md/.json`, `summary.csv/.md`.

## Recommended experiment plan

| # | Config | Purpose | Est. time (T4) |
|---|---|---|---|
| 0 | `quick_smoke.yaml` | 2-epoch pipeline validation | ~2 min |
| 1 | `visdrone_yolo26s.yaml` | primary per-dataset model | ~4-6 h |
| 2 | `codrone_yolo26s.yaml` | primary per-dataset model | ~5-7 h |
| 3 | `dota_yolo26s.yaml` | primary per-dataset model | ~2-3 h |
| 4 | `all_yolo26s.yaml` | merged-data model | ~6-8 h |
| 5 | `all_yolo26m.yaml` | scaled-up merged model | ~10-14 h |
| 6 | `visdrone_yolo_family.yaml` | YOLOv8 vs YOLO11 vs YOLO26 (s) | ~12-18 h |
| 7 | `codrone_yolo_family.yaml` / `dota_yolo_family.yaml` | family comparison on the other datasets | ~15 h / ~6 h |
| 8 | `all_yolo_family.yaml` | family comparison + yolo26m on merged data | ~30-40 h |
| 9 | `visdrone_yolo26_scale.yaml` | n/s/m scale sweep | ~10-14 h |

Research architectures (LSKNet, STD, RVSA, Swin-OBB, Custom-OBB, ARS-DETR,
AO2-DETR, RiO-DETR, RHINO, OrientedFormer) run through
`src/training/dispatcher.py` with the same cleaned data; their tuned
schedules/LRs/resolutions are documented in
[docs/MODEL_TUNING_GUIDE.md](../docs/MODEL_TUNING_GUIDE.md).

Tips: keep the Colab tab open; runs auto-save to Drive so a disconnect never
loses more than the current epoch. Resume with
`--resume runs/train/<run>/last.pt`. L4/A100 runtimes are ~1.5-3x faster than
T4 — prefer them when available (high-RAM helps the staging copy, not the
training itself).
