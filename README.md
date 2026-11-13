# v4.2 verification status

This revision implements the convergence/optimization changes but does not
certify real-data convergence or CUDA timing. Read [FINAL_REPORT_V4.md](FINAL_REPORT_V4.md),
[operations](docs/V42_OPERATIONS.md), and the unresolved
[class schema conflict](docs/CLASS_SCHEMA_CONFLICT.md) before a real run.
No labels/checkpoints were silently remapped. The actual master specification
is preserved in [PROMPT.md](PROMPT.md).

# Aerial OBB Detection & Benchmark Suite (v4)

One pipeline trains, evaluates and compares **23 oriented-object-detection
models** on the cleaned CoDrone / VisDrone / DOTA datasets (8 frozen classes:
pedestrian, bicycle, car, van, truck, bus, tricycle, motorcyclist).

- **12 Ultralytics YOLO-OBB**: yolov8 / yolo11 / yolo26 x n,s,m,l
- **6 dense anchor-free PyTorch**: lsknet, std, rvsa, swin-obb, custom-obb, yolo11-obb (native)
- **5 DETR-family**: ars-detr, ao2-detr, rio-detr, rhino, oriented-former

## Quick start (Colab)

```python
# one cell: mount Drive, install pinned deps, verify data, launch
!python training/colab_quickstart.py            # bootstrap + mock smoke test
!python training/colab_quickstart.py --full     # the whole 23-model benchmark
```

Equivalent local flow:

```bash
python install.py                          # pinned deps (idempotent)
python training/run.py --env               # environment report for bug reports
python training/run.py --models all --datasets mock \
    --steps stage,train,eval,summary --test   # no-GPU/no-data smoke gate
python training/run.py --models lsknet --datasets codrone   # tuned defaults, no flags needed
python training/run.py --models all --datasets all --mode full
```

## The unified pipeline

`training/run.py` is the single entry point. Steps are composable:

```
--steps stage,train,eval,cross,summary,export,report
--mode  full | train | eval        # presets
```

- `--models all` or any mix of names + aliases: `yolo`, `dense`, `detr`,
  `yolov8`, `yolo11`, `yolo26`, `nano`, `small`, `medium`, `large`.
- `--datasets codrone|visdrone|dota|all|mock`.
- Datasets are **staged once** to fast local disk in the background while
  training already reads them (atomic copies, byte-verified, W2).
- Jobs run **small models first** (`--order bigfirst` inverts) so a dying
  Colab session banks results early.
- **One model crashing never kills the suite**: failures are caught, logged
  with tracebacks into `runs/pipeline_state.json` + the run's
  `run_manifest.json`, and the run continues.
- **Auto-resume** (default): a relaunch replays only unfinished jobs and each
  engine resumes from `last.pt` (optimizer/scaler/EMA/RNG included).
- `--mirror-dir` mirrors manifests, results.csv and weights to Drive every
  `--mirror-every` epochs (default 5).
- `--test` smoke mode: 1 epoch, 2 batches, imgsz 160, workers 0; with
  `--datasets mock` it needs no GPU and no data.
- `--version` / `--env`: version, git commit, torch/CUDA/GPU, dataset
  availability — the header for every bug report.

Per-model tuned defaults (epochs, LR, resolution, batch, patience, warmup)
live **only** in `src/training/dispatcher.py::MODEL_TUNING`; any explicit CLI
flag overrides them. See `docs/MODEL_TUNING_GUIDE.md`.

## Legacy entry points (still work, print a deprecation notice)

`training/train.py`, `scripts/run_pipeline.py`, `scripts/train.py`,
`scripts/run_evaluation.py`, `scripts/cross_dataset_eval.py`,
`scripts/stage_data.py` are thin shims over the unified pipeline.

## Outputs

Each run writes `runs/train/<dataset>_<model>/` with `weights/best.pt`,
`weights/last.pt`, `results.csv` (per-epoch losses, val mAP50/mAP50-95,
imgs/s, GPU mem), `run_manifest.json`, and `eval_<dataset>_<split>.json`
after evaluation. Suite-level: `runs/summary.csv|md`,
`results/master_summary.csv`, `runs/train/cross_eval.csv|md|json` +
heatmap. See `docs/RESULTS_AND_LOGGING.md`.

## Repo map

```
training/run.py          unified pipeline (start here)
training/staging.py      background Drive->local staging (python -m training.staging)
training/evaluate.py     one evaluator for all families
training/cross_eval.py   cross-dataset matrix + heatmap
training/summarize_runs.py  master summary table
src/config.py            canonical 23-model registry + dataset config
src/training/            dispatcher + yolo/dense/detr engines
src/metrics/             rotated IoU, mAP, NMS (the only metric code)
docs/                    pipeline, tuning, results/logging guides
tests/test_v4_units.py   CPU unit tests (registry, decode, mAP, augmentation)
```
