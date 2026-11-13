# v4.2 convergence/optimization revision - 2026-11-13

The new attached source ZIP is the base. No data/checkpoint class IDs were
changed. See docs/CLASS_SCHEMA_CONFLICT.md for the unresolved master-spec
conflict. This section records code changes, not measured L4 acceptance.

## C1-C8

- C1: exact device tensor rectangle intersection, FP64 internal clipping,
  relative tolerances, degeneracy tests, chunking and AABB prefilter:
  src/metrics/tensor_obb.py:25,75. Device suppression has no per-box host
  bool read but still O(N) tiny kernel launches: :90. CUDA timing unrun.
  Vectorized Shapely CPU path: src/metrics/geometry.py:184.
- C1: tensor-resident decode through NMS in src/training/decode.py:38,94.
  Public NumPy metric contracts remain unchanged.
- C2: sigmoid focal, class/L1/KL matching costs 2/5/2, Smooth L1 periodic
  angle loss, stabilized symmetric Gaussian KL: src/training/detr_loss.py:17,40,48,84.
  Legacy K+1 head row kept unused for checkpoint shape compatibility.
  Three-consecutive main/aux numerical-degradation bound: :95.
  Intermediate decoder outputs in all five model source files; auxiliary
  losses average then add to final loss, :141.
- C3: closed-form completed-epoch cosine scheduler with preserved optimizer
  moments and explicit warm-restart option: src/training/schedule.py:5.
  Resume remaps to new horizon instead of loading stale T_max. LR can rise
  from an old floor, as required by the supplied formula; no no-jump claim.
- C4: extend native Ultralytics horizon before resume setup, preserve full
  last.pt around final optimizer strip, restore global epoch from a legacy
  stripped run with explicit missing-moments warning and fresh warmup:
  src/training/yolo_resume.py:9,54; src/training/train_yolo.py:229.
- C5: existing 60/80/150 targets retained; queue completion uses actual
  completed epochs, not requested epochs. Full real-data convergence unrun.
  RAM cache budget is divided across workers: src/data/dataset.py:88,120.
- C6: cached current-protocol transfer evaluation, delta mAP matrix, missing
  cell coverage and heatmaps: training/cross_eval.py:55,85;
  training/report_lib.py:189,294. Real 3x3 transfers unrun.
- C7: atomic immutable mirror snapshots for all engines, bounded retry,
  local-result preservation on backup errors, SHA-bound progress sidecars:
  src/training/engine_common.py:180,245,321. Direct queue/metadata recovery:
  training/run.py:123,413,679. Replays saved target/settings in quickstart.
  Verified-epoch selection, conservative unknown checkpoint handling,
  unique backups and selected-local progress report:
  training/colab_recovery.py:82,99,106,125. Real Colab preemption unrun.
- C8: evidence-only LaTeX/PDF/SVG/PNG figures and coverage reports:
  scripts/generate_report.py:181; training/report_lib.py:373,475,510,548,595.
  Mixed hardware/device/timing-basis speed values are rejected: :175.
  Qualitative display shows only top20 with shown/total counts; metrics do
  not change. Missing results remain missing; mock output is not research evidence.

## Mandatory self-discovery audit

1. Recovered src/data/ modules missing from the supplied ZIP from the prior
   reviewed package; no existing newer file overwritten. Moved synthetic
   generator to tests/mock_data.py, removed src/data exports.
2. Fixed RiO-DETR image-independent prediction path by consuming backbone
   memory before refinement: src/models/rio_detr.py:100. Existing weights
   need reconvergence, no accuracy claim.
3. Removed label-conditioned fake inference. Legacy ground_truth_hints
   now raises instead of synthesizing predictions: src/models/base.py:117.
4. Corrected AP greedy assignment to select the best UNMATCHED GT, rather
   than treating a second valid match as FP: src/metrics/detection_map.py:97.
   Old results are excluded by metric_version guard: training/common.py:252.
5. Fixed disk cache .npy suffix/atomic file race, worker-specific temp names,
   and unbounded aggregate worker RAM allowance: src/data/dataset.py:88,120,642.
6. Removed one-pixel GT side inflation in raw polygon conversion so tiny
   original annotations remain their actual size: src/data/dataset.py:45;
   src/metrics/geometry.py:57. Training-time min-size filtering remains explicit.
7. Training-validation now uses original coordinates/raw GT, sharing final
   metric semantics: src/training/metric_protocol.py:6 and both validate engines.
8. Custom checkpoint resume errors now fail instead of silently continuing
   with partially restored state. Actual early-stop epochs are returned by
   both custom engines, preventing done-state overstatement.
9. Supplied test runner only called its first 15 of 26 defined tests.
   It now executes all 26, generates its own YOLO checkpoint instead of
   skipping, then runs all v4.2 regressions: tests/test_v4_units.py final block.
10. Fixed report step's missing runs-dir, empty qualitative dataset handling,
    torch checkpoint loading, stable data-root selection, and hardware provenance.
11. Queue remembers invocation horizons/test flags for recovery. Manifest
    progress cannot skip an older selected checkpoint; requires verified epoch.
    Eval/suite failures return nonzero instead of a false successful pipeline.
12. YAML schema conflicts fail clearly; read-only hash-based split-leakage
    audit added: src/data/schema.py:7,21; scripts/audit_dataset.py.
    No real split integrity claim without running it on actual data.

## Final recovery/report invariants

- Every eval writer records a size/SHA256 weight fingerprint:
  training/evaluate.py. Cross cache reuse and report loading require it:
  training/common.py::eval_weights_match, training/cross_eval.py::reusable_eval,
  training/report_lib.py::load_eval_records. Legacy JSONs need rerun.
- No-op recovery preserves healthy local evals, tested twice. Restored evals
  and checkpoint replacements trigger atomic invalidation with backups.
  Legacy manifest epoch history survives unknown checkpoint authority.
- Per-file mirrors are not transactional bundles. Even an epoch/hash-bound
  best can be outdated relative to last. All engines therefore restore last
  conservatively, back up/remove stale best when last changes, and let
  find_weights fall back to last. No recovered-best optimality claim.
- Generator partitions speed figures by device/hardware/timing basis and
  omits speed in combined mixed-provenance tables. Timed-code inspection
  confirms YOLO predict/NMS and custom forward/decode windows differ;
  their distinct labels are retained, not renamed to appear comparable.

## Verification scope

See FINAL_REPORT_V4.md for actual commands/outcomes. CUDA/L4 targets,
epoch25 DETR probability, real convergence and real 3x3 transfers remain
unrun. All23 CPU smoke reached 22 trained/evaluated; yolov8l-obb was killed
before checkpoint save twice, so no all23 pass is claimed.

---

# Changelog v4

## Added
- `training/run.py`: one unified pipeline (stage/train/eval/cross/summary/
  export/report), family aliases, small-first job ordering, per-model failure
  isolation, pipeline_state.json job queue with auto-resume, --version/--env,
  --test smoke mode.
- Canonical 23-model registry in `src/config.py` (single source of truth:
  CLI choices, dispatcher routing, validation).
- `training/staging.py`: background Drive->local staging overlapping
  training; atomic copies, byte-exact verification, disk guard, CLI
  (`python -m training.staging`).
- Shared augmentation module `src/data/augment.py` (letterbox, flips, rot90,
  HSV, scale/translate) with dense + light-DETR schedules; fingerprinted
  label cache; GPU-side normalization.
- Shared decode (`src/training/decode.py`) + rotated NMS (`src/metrics/nms.py`)
  used by every PyTorch engine; decode round-trip unit test (IoU >= 0.99).
- Engine common (`src/training/engine_common.py`): AMP bf16/fp16, TF32,
  seeds/determinism, EMA, atomic full-state checkpoints, Drive mirror thread,
  per-epoch results.csv logger, graceful SIGINT/SIGTERM.
- Unified evaluation `training/evaluate.py` (one protocol all families,
  sm/md/lg, confusion matrix, speed); cross-eval heatmap; master summary CSV.
  YOLO runs Ultralytics for inference only, with all metrics from
  `src/metrics/` in letterboxed space (evaluate_yolo_unified,
  training/evaluate.py:132-310); the Ultralytics-native val remains as an
  automatic, recorded fallback.
- `tests/test_v4_units.py` (registry, decode round-trip, mAP sanity, augment).

## Changed
- train_dense.py / train_detr.py rewritten on the shared engine:
  gradient accumulation (effective `--batch`), VRAM-aware micro-batch with
  OOM split/retry, EMA validation, early stop on val mAP50, run_dir contract
  (`weights/{best,last}.pt` + results.csv), Drive mirroring.
- `src/training/train_custom.py` is now a thin delegate to the dense engine
  (legacy `train_custom_detector` name kept).
- Dispatcher consumes the canonical registry; legacy spellings normalized.
- Legacy CLIs are deprecation shims: `training/train.py`,
  `scripts/{run_pipeline,train,run_evaluation,cross_dataset_eval,stage_data}.py`.
- `requirements.txt` pinned (torch==2.9.1 + torchvision==0.24.1 verified
  pair; wheel-source comments); `install.py` remains the single installer.
- Notebooks + `training/colab_quickstart.py` updated to the v4 flow.

## Fixed
- Duplicate/stale model lists (D5): one registry, `SUPPORTED_MODELS` derives
  from it; yolov8s-obb no longer special-cased out of `--models all`.
- Private per-engine metric code paths (D3/D4): all engines evaluate through
  `src/metrics/` only.
- Staging idempotence by file count replaced with exact size verification.

## W10 — self-discovered fixes beyond the spec

1. **Missing/mismatched torchvision silently breaks every Ultralytics run**
   ("operator torchvision::nms does not exist"). requirements.txt now pins the
   runtime-verified pair torch==2.9.1 + torchvision==0.24.1 with CUDA/CPU
   wheel-source comments (requirements.txt:8-18).
2. **YOLO weight bootstrap could crash the whole suite** when a pretrained
   .pt is unreleased or offline: train_yolo now falls back to the bundled
   architecture YAML and trains from scratch, and `--test` mode skips
   downloads entirely (src/training/train_yolo.py:152-170).
3. **Ultralytics plot rendering spikes RAM on small CI boxes**: `--test`
   mode sets plots=False (src/training/train_yolo.py:200).
4. **Generic dataset YAML passthrough** for non-cleaned layouts (e.g. the
   mock CI dataset) in the YOLO converter
   (src/data/yolo_converter.py:562-582).
5. **Mock dataset has no test split**: the evaluator maps mock eval to the
   val split and reuses the generated mock.yaml verbatim
   (training/evaluate.py:54-56 and 578-579).
6. **Legacy import name `train_custom_detector`** was kept when
   train_custom.py became a delegate, so src/training/__init__ and older
   scripts keep working (src/training/train_custom.py:19).
7. **summarize_runs** learned the v4 manifest schema (resolved_tuning),
   falls back test->val eval files, and exports params + latency +
   imgs/s columns plus a master CSV (training/summarize_runs.py:55-56, 87-90).
8. **evaluate_run model inference** reads run_manifest.json and falls back to
   the deterministic runs/train/<dataset>_<model> directory name
   (training/evaluate.py:537-555).
9. **YOLO angle-convention risk in the unified evaluator** - Ultralytics xywhr
   (radians, canonical long-side form) vs the suite's degree/[0,180)
   convention. Conversion is centralized in
   src/metrics/geometry.py:68-86 (ultralytics_xywhr_to_suite) and proven
   rectangle-exact by a synthetic round-trip test over 36 cases (nonzero
   angles, w/h swaps, 45/90/135/180 boundaries, off-center boxes; worst
   polygon IoU = 1.0000) in tests/test_v4_units.py:146-186
   (test_yolo_xywhr_conversion).
10. **Silent protocol fallback removed** - the Ultralytics-native val fallback
    can no longer produce non-comparable metrics silently: it requires an
    explicit opt-in (--allow-native-fallback or V4_ALLOW_NATIVE_YOLO_EVAL=1,
    training/evaluate.py:593-618), marks results protocol/comparable=false,
    and cross_eval (training/cross_eval.py:89-93) plus summarize_runs
    (training/summarize_runs.py:78-82) refuse to mix such results in.

### Review-round findings (independent correctness review, items 11-19)

11. **W2 overlap was not real for any family**: datasets listed the staging
    root (incomplete mid-copy) and YOLO was trained by a blocking pre-copy.
    Now all datasets are built from the SOURCE root with reads resolved
    staged-or-source per file (training/staging.py:199 SourceResolver,
    picklable, cross-process miss counter), per-epoch queue reprioritization
    in consumption order (src/data/dataset.py:139 sample_relpaths), and a
    custom Ultralytics trainer gives YOLO the same overlap
    (src/training/train_yolo.py:61 StagedOBBTrainer), with a loud blocking
    fallback. E2E-tested with slowed copies: epoch-2 source reads = 0.
12. **Mirror collisions + no recovery**: every run of a model mirrored onto
    the same files and a wiped local run dir could not be restored.
    DriveMirror now takes a per-run subdirectory
    (src/training/engine_common.py:214) and restore_from_mirror
    (engine_common.py:278) pulls last.pt/best.pt/results.csv back before
    resume (wired in train_dense.py:523, train_detr.py:537).
13. **OOM retry math was wrong**: the split-half retry divided by
    accum_steps twice (half the intended gradient), leaked retried grads
    into the next accumulation group, and the loader kept serving
    pre-halving batch sizes until the next epoch. Now the retry closes its
    own group with scaled-up grad_weight
    (src/training/train_dense.py:708, train_detr.py:712), oversized loader
    batches are inline-chunked (train_dense.py:645, train_detr.py:656),
    and epoch loss is per-image weighted (train_dense.py:754). Proven
    exactly equal to a no-OOM micro-halved reference run (weight diff 0.0).
14. **Eval resolution mismatch**: evaluation ran at a hardcoded imgsz while
    models trained at their tuned size. evaluate_run now resolves imgsz
    from the run manifest (training/evaluate.py:612) and every result JSON
    records coord_space/timing_basis/max_det=1000; YOLO timing is
    forward+NMS-only after a warmup batch, batched predict
    (training/evaluate.py:132 evaluate_yolo_unified).
15. **swin-obb size-prior saturation** (found by the new per-model
    coverage test): on coarse grids the H_PRIOR=6 prior plus a 0.1 log
    clamp floored small boxes at ~38 px and the decoder clipped at a
    DIFFERENT bound than the encoder clamped. Shared MIN_PRIOR_FRAC=0.02
    (src/training/decode.py:35) now bounds both sides identically;
    train_dense imports the shared constants (train_dense.py:47).
16. **Graceful-stop saved the interrupted epoch**: checkpoints on
    SIGINT/SIGTERM now record the last COMPLETED epoch
    (src/training/train_dense.py:786 and the DETR twin), patience state
    (epochs_since_best) is checkpointed/restored, and ModelEMA uses an
    Ultralytics-style warmup ramp with persisted update count
    (engine_common.py:135-173). Verified by test 12.
17. **--cache/--prefetch were missing or ignored**: run.py now exposes
    --cache none|annotations|ram|disk (ram = free-memory-bounded decode
    cache, disk = decoded .npy tier on the staging root; src/data/dataset.py:87-88)
    and --prefetch val,test front-loads eval splits in the staging queue
    (training/run.py). data_wait_s (W3 wait-time acceptance stat) is timed
    per epoch and written to results.csv (train_dense.py:608,861).
18. **YOLO resume broke on torch>=2.6**: Ultralytics checkpoints fail the
    default weights_only=True load; resume pins weights_only=False
    (src/training/train_yolo.py:221-225), verified against a real
    checkpoint. per-box Python-loop GPU syncs in encode_dense_targets were
    vectorized away (train_dense.py:136-200). requirements.txt carries an
    explicit Colab-untested caveat (requirements.txt:16).
19. **Eval model naming**: result JSONs recorded the literal string
    "weights" (a dirname) as the model; evaluate_yolo_unified now takes the
    resolved model name (training/evaluate.py:293).
    Guarded by tests/test_v4_units.py:189-203
    (test_native_fallback_is_explicit).

### Second regression round (items 20-26)

20. **Batch order froze after staging completed** (R1): the fixed perm-list
    sampler built for the staging-overlap epoch was reused for every
    remaining epoch. Both engines now rebuild the train loader with
    shuffle=True on the staging-complete transition
    (src/training/train_dense.py:579,632; train_detr.py:594,644).
    Test 17 drives a live staging handle with a copy gate and asserts the
    rebuild plus changing per-epoch orders.
21. **Best/patience moved on loss in off-cadence epochs** (R2): with
    val_map_every>1, non-mAP epochs could crown best.pt by val loss,
    mixing criteria against W4.10. Selection is now mAP-only: off-cadence
    epochs never touch best/best.pt/patience; a NaN mAP on an evaluated
    epoch is loud no-improvement; explicit val_map=False keeps the loss
    criterion (train_dense.py:841-879, train_detr.py:854-890). Test 18
    covers all three cases.
22. **Area bands were measured in letterbox space at per-family imgsz**
    (R3): small/medium/large AP were not comparable across models at
    imgsz 768/1024/1280. evaluate.py now inverse-letterboxes predictions
    AND GT into the original image frame before metrics in both the
    generic and YOLO-unified paths (training/evaluate.py:352-382,
    applied at :254-264 and in evaluate_generic); coord_space records
    "original image px (inverse-letterbox, R3)". Round-trip test included.
23. **StagedOBBTrainer was never exercised + YOLO mirror/restore gaps**
    (R4): new CPU integration test (test 21) runs 1 real Ultralytics
    yolov8n-obb epoch through the staged trainer with slowed copies.
    labels.cache under the SOURCE is no longer deleted every run
    (train_yolo.py:160-167). The per-epoch YOLO mirror is now
    asynchronous via DriveMirror and includes best.pt
    (train_yolo.py:311-321), and restore_from_mirror runs before the
    resume probe (train_yolo.py:216-223).
24. **Staging failure/accounting defects** (R5): failed copies counted as
    done (finally-increment) so staging could report complete with missing
    files - now successes only, failures recorded + loud drain message
    (training/staging.py:424-449); check_disk_guard prices PENDING bytes
    only (staging.py:169-190); eval/cross never read a partially-staged
    root (training/run.py:405 eval_data_root - live handle -> src_root
    until complete; no handle -> staged root only after stage_status
    verifies every requested dataset); SourceResolver memoizes source
    sizes instead of a Drive stat per read (staging.py:223,247-253).
    Tests 19-20 cover failure accounting and the pending-byte guard.
25. **StageHandle.close() could strand worker threads**, surfacing as
    spurious staging failures at teardown; close() now joins workers
    (training/staging.py:543-551).
26. **Report claims rescoped** (R6): staging E2E described as
    dataset-level; the OOM equivalence test described as one injected
    forward-OOM scenario (no backward-OOM/second-OOM/accum>1/CUDA
    coverage); the battery labeled CPU/mock-level; the "one protocol"
    claim now records the R3 inverse-letterbox frame.

### Third regression round (items 27-29)

27. **Evaluation GT depended on imgsz** (B1): GT was letterboxed to the
    eval imgsz and passed through corners_to_boxes, whose MIN_BOX_PX=4
    filter and canvas clipping dropped small objects differently at
    640/768/1024/1280 - each family was scored on a different, easier GT
    set. Evaluation GT now comes straight from raw.get_ground_truth in
    ORIGINAL image coordinates for every family
    (training/evaluate.py:330 _original_gt_per_class; YOLO path :185,
    generic path :540 via the new per-sample "index" in
    src/training/dataset.py:143); only predictions are
    inverse-letterboxed (R3). Tests: GT identical at imgsz
    640/768/1024/1280 (the old path's tiny-box drop is demonstrated in
    the test), and mock eval GT matches every line of the raw label
    files. Training-time GT stays in letterbox space as before.
28. **Staging wait() could hang forever on a failed copy** (B2): after
    R5a stopped counting failures as done, wait() waited on a `complete`
    that could never arrive. Copies now retry 3x with backoff
    (training/staging.py:462), wait() returns False promptly once the
    queue is drained with failures outstanding (:528), and the blocking
    callers (stage_datasets_blocking :619, the YOLO blocking fallback
    train_yolo.py:191) raise an explicit error naming the failed files.
    Tests: wait(None) returns False in ~1s under an injected permanent
    failure; blocking API raises; a transient failure is absorbed by the
    retry and stages cleanly.
29. **YOLO mirror could race checkpoint rewrites** (B-round note):
    callback order verified on installed Ultralytics 8.4.211
    (engine/trainer.py L679-693: save_model -> on_model_save ->
    on_fit_epoch_end). Mirroring now hooks on_model_save and snapshots
    last.pt/best.pt/results.csv to a local temp name before queueing the
    async Drive copy, so the mirror thread never reads a checkpoint while
    the next epoch's save_model rewrites it in place
    (src/training/train_yolo.py:311-337). Packaging: data/mock is no
    longer shipped (spec §6 "no datasets"); tests and
    training/run.py --datasets mock generate it via
    src/data/mock_data.py:create_mock_dataset on demand.

### Final review round (item 30)

30. **Residual mirror snapshot race + protocol mixing guards**: snapshots
    now get a unique name per save (last_e{N}.pt) and are mirrored under
    the canonical basename via DriveMirror's new dest_name/unlink_after
    (src/training/engine_common.py:237,262; train_yolo.py:322-336), so a
    slow Drive can no longer tear a mirrored checkpoint while the next
    epoch re-snapshots. cross_eval and summarize_runs skip eval JSONs
    whose coord_space is not the current protocol
    (training/common.py:241 protocol_current; cross_eval.py:97,
    summarize_runs.py:84), so pre-B1 results can never be aggregated
    with current ones. Test 26 covers dest_name/unlink_after and the
    guard's admit/reject cases.

## Migration notes (old command -> new command)

| v3 | v4 |
|---|---|
| `python training/train.py --datasets visdrone --models yolo26s-obb` | `python training/run.py --datasets visdrone --models yolo26s-obb` (alias still works) |
| `python scripts/train.py --models ... --datasets ...` | `python training/run.py --models ... --datasets ... --mode train` |
| `python scripts/run_pipeline.py --steps ...` | `python training/run.py --steps ...` (shim passes through) |
| `python scripts/stage_data.py --datasets visdrone` | `python -m training.staging --datasets visdrone` |
| `python scripts/run_evaluation.py --weights ... --model ...` | `python training/evaluate.py --run runs/train/<ds>_<model>` |
| `python scripts/cross_dataset_eval.py --data-dir ... --weights-dir ...` | `python training/cross_eval.py --runs-dir runs/train --stage` |
| `python training/summarize_runs.py` | unchanged, plus `--out results/master_summary.csv` |
