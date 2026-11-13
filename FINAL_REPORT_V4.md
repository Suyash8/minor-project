# v4.2 hand-off report

Date: 13 November 2026

## Status

The C1-C8 source overhaul and CPU software checks are complete. This is NOT
certification of real-data convergence, NVIDIA L4 latency, or the new prompt's
class-order requirement. Those acceptance conditions remain open. No paid
compute was provisioned. Existing numeric class meanings are preserved.

The original attached master specification is PROMPT.md. CHANGELOG-v4.md
records the implementation and self-discovery fixes with source references.
The operations guide is docs/V42_OPERATIONS.md.

## What changed

- Exact tensor rectangle IoU/NMS path plus vectorized Shapely CPU path,
  stable matching, original-image-coordinate validation and versioned metrics.
- DETR sigmoid focal/Hungarian matching, stabilized Gaussian KL, periodic
  angle loss and intermediate auxiliary supervision. RiO now uses image
  memory. Repeated nonfinite main/aux batches stop training instead of
  hiding divergence. Legacy head shapes stay loadable, with semantic warnings.
- Absolute-epoch cosine remapping preserves optimizer state for custom
  engines. Native YOLO full-state extension and explicitly disclosed
  stripped-checkpoint recovery are tested.
- Immutable per-file asynchronous mirroring, verified checkpoint progress,
  unique backups, conservative recovery and target-aware queue replay.
  Unknown local checkpoint authority never loses to a mirror's newer mtime.
- Cross-dataset matrices/coverage, evidence-only LaTeX and four plot types.
  Evaluation JSONs carry size/SHA256 weight fingerprints; cache reuse and
  report loading reject missing/mismatched fingerprints. Rerun old evals.
- Speed evidence stays separated by device/hardware/timing basis. Combined
  tables omit incomparable speed values; Pareto plots use separate groups.
- Schema-name validation, read-only split/hash audit, cache race/budget fixes,
  repaired unit-test entry point, missing data-module recovery, and removal
  of GT-conditioned fake inference from production model behavior.

## Actual verification

Environment: Python 3.10.12, torch 2.9.1+cpu, torchvision 0.24.1+cpu,
Ultralytics 8.4.211, NumPy 2.2.6, SciPy 1.15.3, OpenCV 5.0.0.93,
Shapely 2.1.2. CUDA unavailable. Dependency pins were tested on this CPU
installation, not a clean Colab GPU runtime.

| Check | Actual result |
|---|---|
| compileall/import | Passed |
| tests/test_v4_units.py | All 26 baseline tests actually executed, then 18 new regression groups passed |
| tests/test_c6_c8_units.py | 9/9 passed |
| tests/test_resume_integration.py --native | Dense and DETR real optimizer/checkpoint 2 to 5 epochs; native YOLO 2 to 5, then deliberately stripped checkpoint to global epoch 6 passed |
| Mandated mock smoke, lsknet/ao2-detr/yolo11-obb | Passed; yolo11-obb is native dense, so this checks two dense models and one DETR |
| Actual three-engine smoke, lsknet/ao2-detr/yolo11n-obb | Train/eval/summary/report passed, including CPU native Ultralytics |
| Broader all23 CPU mock attempt | 22 trained/evaluated; yolov8l-obb killed during checkpoint saving twice. No checkpoint or pass for that model |
| Report outputs | LaTeX, Pareto, size/accuracy, radar and qualitative PDF/SVG/PNG exercised. Mock pixels inspected; not real benchmark evidence |
| Mirror recovery invariants | Higher local verified epoch wins despite newer mirror mtime; higher verified mirror replaces with unique backups; sidecar placement, selected-local report, no-op preservation, legacy history and stale-eval handling tested |

The supplied test runner defined 26 tests but previously called only the first
15. The entry point now runs every test. Earlier inherited pass counts were
not used as proof for this revision.

## Acceptance still open

1. Class schema conflict: existing cleaned order is pedestrian, bicycle,
   car, van, truck, bus, tricycle, motorcyclist. PROMPT.md instead names
   pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle.
   This is not a permutation: the existing cleaner merged subclasses.
   Adoption requires an owner decision and likely raw-data rebuild/retraining.
   No labels or checkpoints were renamed to fake compliance.
2. CUDA 10,000-pair timing, validation/NMS timing targets, L4 epoch speed,
   memory headroom, epoch25 foreground probability and all-model real-data
   convergence are unrun. Device NMS still has O(N) small kernel launches;
   SHA sidecars add streaming I/O. These are measured-performance risks.
3. Real 3x3 transfer evaluation, split leakage audit, full real-data report
   and actual Colab preemption/restart remain unrun. Colab was signed out;
   connected Drive access was unavailable. Local mirror tests are not proof
   of maximum-loss guarantees across Drive failures.
4. Cosine horizon remapping raises LR from an old completed-run floor as
   the requested formula dictates. This is not an absolute no-jump LR or
   smooth real-loss-curve claim. Stripped YOLO files cannot recover lost
   optimizer moments; the warning/fresh warmup is deliberate.
5. Mirrors are async per file, not transactional bundles. Even a verified best checkpoint cannot prove that a later best upload was
   not interrupted. All engines conservatively fall back to selected last
   after mirror checkpoint replacement, backing up/removing stale best.
   Legacy manifest epoch history is retained but does not prove completion.

## Reproduce and continue

```bash
python -m compileall -q .
python -c "import src, training.run, src.training.dispatcher"
python tests/test_v4_units.py
python tests/test_c6_c8_units.py
python tests/test_resume_integration.py --native
python training/run.py --models lsknet ao2-detr yolo11n-obb --datasets mock --steps stage,train,eval,summary,report --test --device cpu
```

For real runs, settle docs/CLASS_SCHEMA_CONFLICT.md first, verify CUDA and
actual cleaned-data paths, run scripts/audit_dataset.py --hash-images,
then follow PROMPT.md and docs/V42_OPERATIONS.md. Old metric JSONs and old
JSONs without weight fingerprints need reevaluation. No mock numbers belong
in the publication benchmark table.

## Package

minor-project-code-v4-latest.zip contains one minor-project/ root with source,
configs, tests, docs, scripts, notebooks, requirements, README, this report,
CHANGELOG-v4.md, original PROMPT.md and assets/dota_labels_all.zip. It excludes
raw datasets, generated outputs, checkpoints, venv/git/cache/scratch folders,
old patch dumps and literature papers. The original required labels asset
is the only intentional nested data ZIP.
