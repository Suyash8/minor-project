# v4.2 operations and evidence boundaries

## Environment

The CPU verification used Python 3.10.12, torch 2.9.1+cpu,
torchvision 0.24.1+cpu, Ultralytics 8.4.211, NumPy 2.2.6,
SciPy 1.15.3, OpenCV 5.0.0.93 and Shapely 2.1.2.
CUDA was unavailable. Colab home was signed out and no Drive account was
connected. No GPU runtime, paid compute or training credits were provisioned.

On Colab, keep the installed CUDA torch/torchvision pair rather than blindly
replacing it with CPU wheels. The requirements pins were not tested on a
fresh Colab GPU. Install remaining requirements, verify `torch.cuda.is_available()`
and run the smoke gate before long jobs. `install.py` remains compatible with
the existing requirements workflow; it does not guarantee CUDA compatibility.

## Verification commands

```bash
python -m compileall -q .
python -c "import src, training.run, src.training.dispatcher"
python tests/test_v4_units.py
python tests/test_c6_c8_units.py
python tests/test_resume_integration.py --native
python tests/benchmark_geometry.py --device cuda --pairs 10000
python training/run.py --models lsknet ao2-detr yolo11-obb --datasets mock --steps stage,train,eval,summary --test
python training/run.py --models lsknet ao2-detr yolo11n-obb --datasets mock --steps stage,train,eval,summary --test
python training/run.py --steps summary,report --test
```

`yolo11-obb` is the native **dense** architecture, not Ultralytics.
`yolo11n-obb` is the actual native Ultralytics engine. The first smoke command
therefore checks two dense architectures and one DETR; the second covers all
three engines. Mock datasets and scene generators are confined to `tests/`.
No mock metrics are publication benchmark results.

The original unit script defined 26 tests but invoked only its first 15.
The repaired entry point enumerates all 26, then all 18 v4.2 regressions.
C6-C8 has nine further tests. The resume integration test is separate and
uses real optimizers/checkpoint files with synthetic images, not fake states.

## C1 geometry and metrics

Exact convex rectangle intersections use contained corners plus all edge
crossings, with angular sorting and shoelace area. Inputs/outputs remain
cx,cy,w,h,angle_deg at the public metrics boundary. Ground-truth label files
remain normalized four-corner polygons; metrics use raw original-image pixels.
CUDA tensor clipping is FP64 internally for coincident/very thin/high-aspect
boxes and returns the original floating dtype. CPU uses Shapely 2 vectorized
intersections with AABB filtering; Shapely 1 and tensor fallback remain exact.
NMS uses exact overlaps and deterministic tie order. CUDA suppression avoids
per-candidate host boolean reads, but still launches O(N) small device ops.
CUDA pairwise geometry uses AABB rejection and bounded chunks. Device transfers
in the existing NumPy metric API remain per-image/per-class. These design
choices are NOT evidence that the latency targets have been met.

The v4.2 matching correction tries the best unmatched ground-truth box in
confidence order. Older metric JSONs must be rerun. Downstream consumers require
both original-image coord_space and `metric_version=v4.2-greedy-unmatched`.

## C2 DETR

Foreground channels use sigmoid focal loss (alpha .25, gamma 2). The last
legacy K+1 head row is unused, retained to keep old parameter and Adam-state
shapes loadable. Matching weights: focal class 2, L1 box 5, symmetric log1p
Gaussian KL 2, plus periodic angle cost 1. Angle loss uses Smooth L1 of
sin(2 theta)/cos(2 theta). KL uses a regularized elementwise 2x2 inverse.
Auxiliary losses share heads, average over intermediate outputs, then add once
to final loss. RiO-DETR now consumes image features; old RiO checkpoints need
reconvergence. Legacy softmax checkpoints produce a calibration warning.
Three consecutive nonfinite main/aux batches abort instead of masking divergence
indefinitely. An isolated degraded batch warns and sanitizes its nonfinite
predictions. This is resilience, not evidence of successful convergence.

## C3/C4 schedule and resume

Completed epoch is the absolute custom-engine epoch. Resuming to a new horizon
uses eta_min + .5*(base-eta_min)*(1+cos(pi*completed/target)), preserving optimizer,
EMA, scaler and RNG state. Warmup multiplies the early part only.

An old run at its 20-epoch floor will **rise** when remapped to an 80-epoch
horizon. This is required by the supplied formula, and is not absolute LR
continuity with the old schedule. Subsequent LR is closed-form monotone cosine
to the new floor; no stale T_max recursion/oscillation. `--warm-restart` instead
starts cosine from base LR at the resumed epoch (custom engines only).
No smooth real-data loss-curve claim follows from finite mock losses.

YOLO extension changes target epochs before native scheduler/resume setup and
uses the requested cosine horizon even if an old run was linear. It logs this
schedule-shape/LR change. Full last.pt is preserved around native final optimizer
stripping. Old already-stripped files cannot recover lost moments: global epoch
is recovered from results.csv, an explicit warning is emitted, and the fresh
optimizer warms over its first resumed epoch. This is not momentum continuity.

## C5-C7 execution, staging and recovery

Use the exact full-suite command in PROMPT.md after settling class schema and
verifying real dataset paths. Defaults retain 60 dense, 80 DETR and 150 YOLO
epoch targets, with configured early stopping. Those targets are not guaranteed
convergence. Staging enumerates one dataset inventory, shares its handle across
models, and serves local copies after completion. Source enumeration still
reads metadata; zero source reads is verified only for staged image consumers.
Outside recovery, cross evaluation reuses matching current-protocol, current-resolution,
newer-than-checkpoint JSONs. Recovery invalidates restored eval JSONs or evals affected by a checkpoint
replacement, never healthy local evals on no-op recovery. Evaluation JSONs
now carry size/SHA256 weight bindings, checked by cache reuse and reports. Mtime alone is not a safe binding
after files move between runtimes. --force re-evaluates. Missing cells stay missing.

Drive mirrors take immutable snapshots and use atomic replacement, bounded
retries and explicit error reporting. Local final checkpoints are retained even
when a backup fails; inspect mirror_errors in manifests. Queue state mirrors
under `_suite`, each job has its own directory. Quickstart restores manifests,
last/best/logs/eval JSONs and invocation settings before resuming. Local checkpoints are replaced only when both sides have verified epoch
sidecars and the mirror epoch is strictly ahead. Legacy/unknown local
checkpoint bytes are preserved conservatively. Replacements keep unique
.bak-* copies. When last advances, best from a per-file mirror cannot prove the latest best upload landed.
Recovery therefore uses selected last, for all engines. Async same-folder files are not an atomic bundle.
Otherwise stale local best is backed up/removed, and evaluation falls back
to selected last. Legacy unverified manifest history is preserved, but cannot
prove done status. Metadata uses mtimes with
backups, but evals become non-comparable and manifest epochs are clamped to
the selected verified last checkpoint. Queue state is advisory, not proof of
checkpoint completion. Checkpoint selection does not use mtime. Checkpoint progress is reported only when a size/SHA256 sidecar
matches last.pt, separate from log-line progress. A real Colab kill/restart was not tested; local
mirror/restore with real mock-trained weights was tested. The maximum-loss
checkpoint-interval claim is conditional on successful, completed mirror writes,
not guaranteed during outages or disk errors.

## C8 artifacts

```bash
python scripts/generate_report.py --runs-dir /content/minor-project/runs --data-root /path/to/cleaned_data
python training/cross_eval.py --runs-dir /content/minor-project/runs --datasets visdrone codrone dota --models lsknet ao2-detr yolo11n-obb
```

Produces LaTeX, Pareto, size/accuracy, per-class radar and qualitative outputs
in PDF/SVG/PNG, plus coverage JSON. Every result is traced to a current-protocol
eval JSON. Missing data is skipped/reported; no stand-in benchmark values.
Hardware labels must agree with recorded measured hardware. Mixed device,
hardware or timing-basis groups get separate Pareto plots; the combined
LaTeX table omits speed rather than selecting an arbitrary finite value. Qualitative panels
show GT and at most the top 20 predictions above the visualization threshold;
shown/total counts disclose this display-only filter, evaluation is unchanged.
All four plot types were exercised and visually inspected with mock CPU output.
Publication-grade real figures for all 23 models require their real results.
