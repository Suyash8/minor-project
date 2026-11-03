# Logging, Run Tracking, and Result Exports

This document describes the observability and reporting infrastructure of the
benchmark suite: what gets written where during a run, and how to turn raw
results into the final Excel/report tables.

---

## 1. What every run writes (automatic)

Both entry points — `scripts/run_pipeline.py` and
`scripts/cross_dataset_eval.py` — install **run logging** at startup
(`src/utils/logging_utils.py`). Given a `run_id`, three files are created under
`<results_dir>/run_logs/<run_id>/`:

| File | What it contains |
|---|---|
| `console.log` | Byte-for-byte mirror of **everything** printed to stdout/stderr, with a header stamp (run id, timestamp, Python version, platform). Nothing printed is ever lost — even on Colab where scrollback disappears. |
| `events.jsonl` | One JSON object per line, in event order: `run_start`, `run_config`, `stage_start`/`stage_end` (per pipeline phase), `job_start`/`job_end` (per model×dataset pair, with full metric payload and duration), `heartbeat`, `metric`, `error`, `run_end` (with total uptime and counters). |
| `run_config.json` | Immutable snapshot of the exact CLI arguments + device the run started with. Any result can be traced back to how it was produced. |

### Event semantics

- `job_start` / `job_end` pair up for every unit of work (one model on one
  dataset in one phase). A `job_start` without a matching `job_end` means the
  job crashed or the run was killed — gaps are diagnostic, silence is never
  ambiguous.
- `job_end.status` is one of `ok`, `failed`, `load_failed`,
  `dataset_unavailable`, `skipped_cached`.
- `run_end.status` is one of `completed`, `failed`, `interrupted`, `exit`.
- Durations are seconds; timestamps are UTC ISO-8601 with milliseconds.

### Quick recipes

```bash
# What happened in a run, in order?
cat results/run_logs/<run_id>/events.jsonl | python3 -c \
  "import sys,json; [print(json.loads(l)['ts'], json.loads(l)['event'],
   json.loads(l).get('model',''), json.loads(l).get('status','')) for l in sys.stdin]"

# All failures across all runs?
grep -h '"event": "error"' results/run_logs/*/events.jsonl

# Why is a result what it is?
cat results/run_logs/<run_id>/run_config.json
```

`events.jsonl` is also trivially loadable with
`pandas.read_json(..., lines=True)` for plotting time-per-pair, throughput,
or failure timelines.

---

## 2. Cross-dataset evaluation (trained on A, tested on B)

`scripts/cross_dataset_eval.py` evaluates every **trained** checkpoint on
**every** dataset — in-distribution and out-of-distribution — with correct
class-space remapping (a VisDrone-trained model keeps its 10-class head and
predictions are mapped to CODrone/DOTA classes via a synonym table). It stores
results per (source, model, eval) pair, so global/generalization numbers can
be computed afterwards:

```
<results-dir>/<run-name>/
├── checkpoints/<source>__<model>__<eval>.json    # resumable per-pair results
├── per_class/<source>__<model>__<eval>.csv       # per-class AP/TP/FP/FN/P/R/F1
├── run_logs/<run-name>/                          # the three logging files
├── cross_dataset_summary.csv                     # one row per (model, source, eval)
└── global_summary.csv                            # macro-average per (model, source)
```

Run it after training (no training happens here):

```bash
python scripts/cross_dataset_eval.py \
    --data-dir    /content/drive/MyDrive/object-detection/data \
    --weights-dir /content/drive/MyDrive/object-detection/weights \
    --results-dir /content/drive/MyDrive/object-detection/results \
    --run-name    cross_eval_20ep \
    --device      cuda \
    --match-mode  common
```

`--match-mode common` scores only classes shared between the training and eval
datasets (fair literature-style comparison); `--match-mode all` scores all eval
classes with unmapped ones at 0 (honest out-of-distribution number). Completed
pairs are checkpointed and skipped on re-run; add `--force` to redo.

---

## 3. Exporting everything for the Excel / final report

`scripts/export_results.py` scans all run folders (repo + any extra folder,
e.g. a Drive cross-eval output) and consolidates every metric ever produced
into `saves/excel_export/`:

| File | Feeds |
|---|---|
| `summary_all_runs.csv` | audit trail: every (run, phase, model, dataset) with all 21 metrics |
| `summary_latest.csv` | **Summary sheet**: latest run per (phase, model, dataset) |
| `summary_matrix_post_train.csv` | **Summary sheet**: models as columns, metrics as rows (direct paste) |
| `class_wise_all.csv` | **class_wise_results sheet**: per-class AP50/AP75/TP/FP/FN/P/R/F1 |
| `cross_dataset_pairs.csv` | cross-dataset section: one row per (model, source, eval) |
| `cross_dataset_global.csv` | cross-dataset global table: macro-averages |
| `EXPORT_README.md` | column dictionary + provenance |

```bash
# repo-local runs only
python scripts/export_results.py

# include the Colab cross-dataset run folder
python scripts/export_results.py \
    --extra-dirs /content/drive/MyDrive/object-detection/results/cross_eval_20ep
```

Metric columns (all summary tables): `map50, map75, map50_95, ap_small,
ap_medium, ap_large, precision, recall, f1, accuracy, specificity,
angle_mae, angle_rmse, pearson_r, spearman_rho, r2_score,
center_offset_mae, center_offset_rmse, aspect_ratio_r2, fps, mean_latency_ms`.

---

## 4. Keeping the repository small

Plot PNGs and disposable smoke/test run outputs are the only things that ever
blew up the repo size. Policy:

- Plots are **regenerable** (`--save-plots`) and gitignored
  (`results/**/plots/`, `saves/results/**/plots/`); metric JSON/CSV files are
  always kept.
- Old run folders can be compressed losslessly with
  `scripts/archive_results.py` (tar.gz into `archives/`, verified, optionally
  keeping metric essentials in `archives/essentials/`, then removing the
  original). Restore anytime with `--restore`.

```bash
python scripts/archive_results.py --dry-run          # preview
python scripts/archive_results.py --keep-essentials  # archive test/smoke runs
python scripts/archive_results.py --restore archives/<name>.tar.gz
```
