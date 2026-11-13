# Per-Model Tuning Guide - Cleaned v4.1 Datasets

> **v4 update:** the tuning mechanism is unchanged and still lives ONLY in
> `src/training/dispatcher.py::MODEL_TUNING` (explicit CLI flags override it).
> v4 adds gradient accumulation, so `batch_size` in the table is the
> EFFECTIVE batch; the engines split it into micro-batches automatically when
> VRAM requires. W5.2 note: v4 augmentation landed after this grid was
> measured; no re-tuned numbers are claimed - any future grid change must be
> justified by measured short experiments and documented here, never invented.
> DETR query-collapse guard: warmup epochs stay mandatory, micro-batch >= 2
> on GPU, and the engine warns when matched queries hit 0 in an epoch.


Every architecture in this repo is now tuned for the **cleaned v4.1 datasets**
(8-class unified space, images pre-resized to 1280 px long side, v4.1 split
fix; see `data_cleaning/CLEANING_FULL_RUN_REPORT.md`):

| Dataset | Images | Boxes | Notes |
|---|---|---|---|
| CODrone | 9,894 | 416,727 | dense scenes, 40+ objects/image of one class |
| VisDrone | 8,633 | 446,585 | all 8 classes healthy |
| DOTA | 1,233 | 193,073 | aerial, higher altitude, dense small boxes |
| merged `all` | 19,760 | 1,056,385 | via `dataset_all.yaml` |

Two things changed to make "best possible result everywhere" the default:

1. **All engines now read the cleaned data directly.** The dense/DETR engines
   (`src/training/train_dense.py`, `train_detr.py`) and the YOLO converter
   (`src/data/yolo_converter.py`) detect the cleaned layout
   (`{split}/images` + `{split}/labels` in YOLO-OBB format) and use it as-is
   in the unified 8-class space. Previously the custom models trained on raw
   Drive layouts with mismatched class spaces (10–15 classes), which is a
   large part of why their benchmark mAP was near zero.
2. **Per-model optimized defaults** (`MODEL_TUNING` in
   `src/training/dispatcher.py`) replace the old flat 15-epoch/640 px stub
   for every architecture. Any argument you pass explicitly still overrides
   the tuned default.

---

## 1. Tuned settings per model

### Dense grid-based detectors (`src/training/train_dense.py`)

| Model | epochs | LR (AdamW) | weight decay | batch* | img size | patience |
|---|---|---|---|---|---|---|
| LSKNet | 60 | 2e-3 | 5e-2 | 8 | 1024 | 12 |
| STD | 60 | 2e-3 | 5e-2 | 8 | 1024 | 12 |
| RVSA | 60 | 1.5e-3 | 5e-2 | 6 | 1024 | 12 |
| Swin-OBB | 60 | 1.5e-3 | 5e-2 | 6 | 1024 | 12 |
| Custom-OBB | 60 | 2e-3 | 5e-2 | 8 | 1024 | 12 |
| YOLO11-OBB (native impl) | 60 | 2e-3 | 5e-2 | 8 | 1024 | 12 |

*batch is the request — `auto_configure_hardware()` still scales it down to
fit the actual GPU (e.g. 4 on a Colab T4).

Why these values:

- **LR 1.5–2e-3** (up from 1e-3): with the pos-weighted objectness BCE and
  the recovered ~127k CODrone boxes, the previous 1e-3 plateaued early; the
  higher rate with the existing cosine schedule reaches a clearly lower
  validation loss within the 60-epoch budget.
- **img 1024** (up from 640): the cleaned vehicle boxes are 8–20 px objects
  at native resolution; at 640 px training the stride-8 grid cells coarsen
  them beyond recall. 1280 exceeds 16 GB VRAM for Swin/ViT variants at
  batch ≥ 4, so 1024 is the safe optimum for the whole family.
- **weight decay 5e-2**: standard for AdamW + large-ish LR on detection
  heads (the DETR-family 1e-4 would under-regularize the dense heads).
- **60 epochs + patience 12**: the loss on ~10–20k images converges between
  epoch 35 and 50; 60 with early stop is the best cost/quality point.

### DETR-family detectors (`src/training/train_detr.py`)

| Model | epochs | LR (AdamW) | weight decay | batch* | img size | patience | warmup |
|---|---|---|---|---|---|---|---|
| ARS-DETR | 80 | 2.5e-4 | 1e-4 | 4 | 768 | 15 | 5 |
| AO2-DETR | 80 | 2.5e-4 | 1e-4 | 4 | 768 | 15 | 5 |
| RiO-DETR | 80 | 2.5e-4 | 1e-4 | 4 | 768 | 15 | 5 |
| RHINO | 80 | 2.5e-4 | 1e-4 | 4 | 768 | 15 | 5 |
| OrientedFormer | 80 | 2.5e-4 | 1e-4 | 4 | 768 | 15 | 5 |

Why these values:

- **LR 2.5e-4 + 5-epoch linear warmup** (up from flat 1e-4): set-prediction
  matching on the 8-class aerial data collapsed to empty predictions at
  1e-4 (observed near-0 mAP with ~0 boxes/image). The warmup stabilizes the
  Hungarian assignment in the first epochs, which is when DETR-style models
  decide whether queries produce detections at all.
- **img 768**: ViT-based DETR memory grows quadratically; 768 keeps
  batch ≥ 4 on a T4 while preserving small-vehicle recall far better
  than 640.
- **80 epochs**: DETR-family schedules are longer by design (the matching
  needs the full cosine tail); patience 15 guards the budget.

### Ultralytics YOLO-OBB suite (`training/train.py`)

| Model | epochs | imgsz | batch | patience | schedule |
|---|---|---|---|---|---|
| YOLOv8 n/s/m/l-OBB | 150 (120 merged) | 1280 | -1 (autobatch) | 30 | cosine LR, close_mosaic 10 |
| YOLO11 n/s/m/l-OBB | 150 (120 merged) | 1280 | -1 | 30 | cosine LR, close_mosaic 10 |
| YOLO26 n/s/m/l-OBB | 150 (120 merged) | 1280 | -1 | 30 | cosine LR, close_mosaic 10 |

- 1280 px = native cleaned resolution (no up/down-sampling loss).
- `close_mosaic=10` disables mosaic for the last 10 epochs so small-box
  regression settles on un-augmented statistics.
- DOTA presets use 120 epochs (fewer images → converges faster).
- Ready-made presets: `configs/experiments/*.yaml` (one per dataset, plus
  `_yolo_family` variants that benchmark all three generations in one run).

---

## 2. How to run each family on the cleaned data

### Ultralytics suite (primary benchmark)

```bash
# single model on one dataset
python training/train.py --datasets visdrone --models yolo26s-obb

# all three YOLO generations at s-scale on the primary dataset
python training/train.py --cfg configs/experiments/visdrone_yolo_family.yaml

# scale sweep on the merged dataset
python training/train.py --cfg configs/experiments/all_yolo_family.yaml
```

Each run writes `run_manifest.json` + `eval_<ds>_test.json` automatically,
then `python training/summarize_runs.py` builds the results matrix and
`python training/cross_eval.py` runs the cross-dataset matrix.

### Dense / DETR architectures (secondary benchmark)

```bash
python -c "from src.training.dispatcher import train_model; \
           train_model('lsknet', dataset_name='codrone')"

# or several in one go
python - <<'EOF'
from src.training.dispatcher import train_model
for m in ["lsknet", "std", "rvsa", "swin-obb", "custom-obb",
          "ars-detr", "ao2-detr", "rio-detr", "rhino", "oriented-former"]:
    train_model(m, dataset_name="codrone")   # tuned defaults applied per model
EOF
```

`train_model` applies the table above automatically; pass `epochs=`,
`img_size=`, `lr=` etc. only when you deliberately want to deviate.

Data location: set `VEHICLE_OBB_DATA=/path/to/cleaned_data` (or rely on the
Colab Drive default `/content/drive/MyDrive/object-detection/cleaned_data`).
The engines fall back to that root automatically when the requested
`data_dir` does not contain the dataset.

---

## 3. Fixes that ship with this tuning

- **`src/models/custom_obb.py`**: top-level `forward` now returns
  `(cls_logits, reg_preds, (feat_h, feat_w))` like every other dense model —
  it previously returned 2 values and crashed the dense training loop.
- **`src/data/dataset.py`**: static-path entries are now priority lists
  (cleaned layout first, raw Drive layouts as fallback); the unified 8-class
  names are applied automatically when the cleaned layout is detected.
- **`src/data/yolo_converter.py`**: cleaned datasets pass through untouched
  (no re-conversion, no bundled-zip extraction); raw layouts still convert
  as before.
- **Early stopping** (val-loss patience) added to the dense and DETR
  engines; **linear LR warmup** added to the DETR engine.
- **`training/train.py`**: experiment YAMLs now accept a `models:` list in
  addition to a single `model:`, so one config can sweep a whole family.

## 4. Notes and expectations

- On equal data, the Ultralytics YOLO-OBB suite will lead in mAP (native
  pretrained OBB heads, mosaic, TAL assignment); the dense/DETR engines are
  from-scratch implementations kept for the academic comparison — the tuning
  above is what makes them converge at all (they previously scored ~0 mAP
  due to the class-space mismatch and undertrained schedules).
- Per-class top-k caps during NMS were raised 25→100 and the objectness BCE
  is pos-weighted (upstream commits `a447499`, `4b514c0`) — both are load
  bearing for the dense CODrone scenes; do not revert them.
- CODrone `van` has 7 source instances; expect that class to stay near zero
  everywhere. It is source-limited, not a tuning issue.
