# Pipeline Guide (v4)

`training/run.py` runs any subset of `stage -> train -> eval -> cross ->
summary -> export -> report` in one process, for any subset of the 23
registered models x the cleaned datasets.

## Steps

| step | what it does | implementation |
|---|---|---|
| stage | background copy Drive -> local NVMe, atomic (.part + rename), byte-verified, `.stage_manifest.json` receipts; training reads through a race-safe resolver so epoch 1 starts before staging finishes | `training/staging.py` |
| train | per (dataset, model) job; family engine (yolo/dense/detr); tuned defaults; failure isolation | `src/training/dispatcher.py` |
| eval | one protocol for all families: mAP50/50-95/75, per-class AP, sm/md/lg, confusion matrix, speed; writes `eval_<ds>_<split>.json` | `training/evaluate.py` |
| cross | train-on-A/test-on-B matrix, staged data reused, CSV + MD + heatmap | `training/cross_eval.py` |
| summary | master table dataset x model -> mAP/params/latency | `training/summarize_runs.py` |
| export | Excel export | `scripts/export_results.py` |
| report | markdown report | `scripts/generate_report.py` |

## Job queue and resume

The pipeline writes `runs/pipeline_state.json` (job plan + status). On
relaunch with `--auto-resume` (default), completed jobs are skipped and the
interrupted job's engine resumes from `weights/last.pt`, which contains model
+ optimizer + scaler + scheduler + EMA + RNG states + resolved config +
staging status. `run_manifest.json` without `final_metrics` marks an
unfinished run.

Ordering: small/fast models first (`--order smallfirst`, default) so a
time-limited Colab session banks the most results; `--order bigfirst`
inverts.

## Engines (v4, W4)

- bf16/fp16 AMP (train + val), TF32, cudnn.benchmark, optional
  `--compile` (torch.compile) and `--deterministic`.
- `--batch` means EFFECTIVE batch; `--micro-batch` overrides the per-step
  batch; gradient accumulation bridges them (DETR needs effective >= 8).
- VRAM-aware micro-batch scaling + OOM catch -> empty_cache -> split/retry.
- EMA of weights (`--ema` default on, `--no-ema`), validated with EMA weights.
- Early stopping on val mAP50 (not loss) with `MODEL_TUNING` patience.
- Graceful SIGINT/SIGTERM: finish batch, checkpoint, exit 0.

## Data

Cleaned v4.1 layout per dataset: `<ds>/{train,val,test}/{images,labels}` with
YOLO-OBB 8-value normalized corner labels and the frozen 8-class order
(pedestrian, bicycle, car, van, truck, bus, tricycle, motorcyclist).
`--datasets mock` generates a tiny synthetic set for CI (`--test`).

## Examples

```bash
python training/run.py --models yolo26 --datasets visdrone --steps stage,train
python training/run.py --models nano --datasets all --mode full --mirror-dir /content/drive/MyDrive/object-detection/runs_v4
python training/run.py --env     # paste into bug reports
```
