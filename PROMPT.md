# MASTER OPTIMIZATION & CONVERGENCE PROMPT — "minor-project" v4.2 (Production Benchmark Suite)

> **Who this is for:** An advanced AI coding/research agent (Claude 3.7 / GPT-4o / Gemini / Cursor) tasked
> with executing the **final optimization, full convergence, and publication-ready benchmarking** of this
> aerial Oriented Bounding Box (OBB) research framework.
>
> Read this entire document **completely and strictly** before touching or modifying any file. This document
> reflects hard-won lessons, empirical profiling, and runtime debugging on real Google Colab NVIDIA L4 GPUs
> on the cleaned v4.1 datasets. Where this document and existing code or comments disagree, trust this document.

---

## 0. Operating Principles & Workflow Rules

1. **Empirical Rigor over Assumptions:** Never guess why a model failed or stalled. Trace logs, measure
   GPU memory, inspect tensor shapes, and profile execution times.
2. **Preserve Compatibility & Architectural Contracts (§4):** Label formats, class indices (0–7),
   coordinate spaces, evaluation metrics, and CLI flags must remain 100% backward-compatible.
3. **No Mocks in Production Logic:** Mock datasets and synthetic tests belong strictly in `tests/`.
   Training engines, metric evaluation, and dataset loaders must operate on real data with production precision.
4. **Never Leave Throwaway Scripts in Git:** Diagnostic or single-use verification scripts must be placed
   in `scratch/` (shadowed by `.gitignore`) or deleted immediately after execution.
5. **Verify Against the Acceptance Gates (§5):** Every change must be validated against the 8 acceptance
   gates before packaging.

---

## 1. Mission & Objectives

The v4 overhaul successfully established the **unified pipeline** (`training/run.py`), registering all
**23 models** across 3 distinct engine families (Ultralytics YOLO-OBB, custom dense anchor-free CNNs,
and arbitrary-oriented DETRs) with background staging and unified evaluation.

However, real-world deployment on Google Colab (NVIDIA L4 24GB GPU, 53GB system RAM) on the cleaned
VisDrone dataset surfaced **deep training dynamics, mathematical scheduling traps, and performance bottlenecks**
that must now be systematically resolved in **v4.2**:

1. **Scale Models to Full Convergence:** Extend training beyond early exploratory stubs (20 epochs) to their
   optimal convergence targets (60 epochs for Dense/YOLO, 80 epochs for DETRs) so models reach high mAP.
2. **GPU-Accelerated Vectorized Polygon NMS:** Eliminate CPU Shapely polygon intersection bottlenecks during
   validation by introducing a GPU-accelerated tensor OBB IoU and NMS kernel.
3. **Stabilize DETR Query Learning:** Resolve multi-class softmax probability dilution and Hungarian bipartite
   query collapse warnings so Transformer architectures learn dense aerial targets effectively.
4. **Resilient Schedule Continuation:** Enable seamless training resumption from existing 20-epoch checkpoints
   without suffering from decaying learning rate stalls or pipeline skipping.
5. **Automated Multi-Dataset Benchmarking & Cross-Evaluation:** Execute the full 3-dataset suite
   (VisDrone, CODrone, DOTA) and generate publication-grade cross-evaluation heatmaps and LaTeX tables.

---

## 2. Complete System Context & Lessons from Real GPU Profiling

### 2.1 The 23 Registered Models & 3 Engine Families

The framework unifies 23 architectures under `src/config.py::CANONICAL_MODELS`:

| Family | Count | Models | Core Engine File | Convergence Target |
|---|---|---|---|---|
| **YOLO-OBB** | 12 | `yolov8n-obb` .. `yolov8l-obb`, `yolo11n-obb` .. `yolo11l-obb`, `yolo26n-obb` .. `yolo26l-obb` | `src/training/train_yolo.py` | 60–150 epochs |
| **Dense Grid** | 6 | `custom-obb`, `lsknet`, `std`, `rvsa`, `swin-obb`, `yolo11-obb` (native) | `src/training/train_dense.py` | 60 epochs |
| **DETR** | 5 | `ao2-detr`, `ars-detr`, `rio-detr`, `rhino`, `oriented-former` | `src/training/train_detr.py` | 80 epochs |

### 2.2 Datasets & The Unified 8-Class Space

All training and evaluation operate on the cleaned v4.1 datasets pre-processed to 1280 px long side:
- **Datasets:** VisDrone2019-DET, CODrone (custom drone aerial), DOTA v1.0/v1.5.
- **Unified 8 Classes (Frozen Order):**
  `0: pedestrian`, `1: people`, `2: bicycle`, `3: car`, `4: van`, `5: truck`, `6: tricycle`, `7: awning-tricycle`.
- **Label Format:** YOLO-OBB text format (`cls x1 y1 x2 y2 x3 y3 x4 y4`, normalized coordinates in [0, 1], clockwise).

### 2.3 Empirical Findings from Real Google Colab Execution (L4 GPU)

Real-world training on NVIDIA L4 (24GB VRAM) revealed the following critical insights:

1. **The Shapely Validation Bottleneck:**
   - In early epochs, unsuppressed candidate boxes (~2,000 per image across 10 classes) generated millions
     of pairwise polygon intersection checks. Standard Shapely Python loops caused epoch 1 validation to hang
     for 15+ minutes.
   - Vectorized corner precomputation and strict axis-aligned bounding box (AABB) geometric filtering solved
     the hang, but pure Python Shapely polygon intersections still take 10–12 ms per image. A pure tensor
     implementation is required for instant evaluation.
2. **DETR "Cold Start" vs. CNN Inductive Bias:**
   - At epoch 3, CNNs (`custom-obb`) achieved mAP50 ~0.0302 and climbed to 0.0566 by epoch 15.
   - At epoch 3, `ao2-detr` achieved mAP50 ~0.0004 and only 0.0015 by epoch 9.
   - *Root Cause:* CNNs have spatial convolutions and anchor priors. DETRs start with 300 random queries that
     have zero spatial awareness. The Hungarian matcher spends the first 25–35 epochs learning query
     specialization. At 20 epochs, DETR is prematurely cut off. It requires 80 epochs.
3. **DETR Multi-Class Softmax Dilution:**
   - DETR heads use softmax across $(K + 1)$ classes (8 foreground + 1 background).
   - Even when a query correctly detects a vehicle, the probability mass is split among similar classes
     (`car`, `van`, `truck`), resulting in peak foreground probabilities around 0.15–0.30.
   - A standard confidence filter of `conf_thresh = 0.05` or `0.10` dropped over 80% of true positive queries.
     Calibrating `conf_thresh = 0.01` restored recall, but transitioning to multi-label sigmoid with Focal Loss
     is the mathematically superior solution.
4. **The CosineAnnealingLR Resumption Trap:**
   - Resuming a run that finished 20 epochs with `CosineAnnealingLR` restores `T_max = 20`.
   - The learning rate has already decayed to the floor (~$10^{-5}$). If training resumes to epoch 80 without
     re-scaling $T_{max}$ and continuous LR, the model stays at $10^{-5}$ or oscillates wildly.
5. **Multiprocessing RAM Cache Crash:**
   - Using `--cache ram` with 8 DataLoader workers caused Linux kernel OOM kills (`Killed by signal 9`) because
     each worker attempted to allocate 25% of RAM ($8 \times 25\% = 200\%$).
   - The verified optimal configuration is `--cache annotations` combined with background local NVMe staging,
     delivering 75–80+ imgs/s with near-zero RAM overhead.
6. **PipelineState Skip Guard:**
   - In `training/run.py`, when a job completed its initial target (e.g. 20 epochs), `pipeline_state.json`
     marked `status: "done"`. A subsequent run wanting 80 epochs skipped the model unless the pipeline checked
     `already_trained < target_epochs`.

---

## 3. The Optimization Work Items (C1 to C8)

### C1 — GPU Tensor-Accelerated Rotated NMS & Fast Evaluator
**Objective:** Replace CPU-bound polygon operations with GPU-accelerated tensor routines.
- Implement vectorized rotated IoU calculation entirely in PyTorch tensors on CUDA (using differentiable
  polygon clipping via Sutherland-Hodgman or separating axis theorem on CUDA).
- Fallback gracefully to vectorized Shapely with AABB pre-filtering when running on CPU.
- Ensure evaluation of 500+ validation images across all 10 IoU thresholds (0.50:0.05:0.95) completes in
  under 5 seconds per epoch.
- **Acceptance Criteria:** Batched IoU computation on 10,000 box pairs on CUDA runs in < 25 ms; mAP output
  matches existing Shapely reference within $10^{-4}$.

### C2 — DETR Loss & Query Matching Overhaul
**Objective:** Prevent query collapse and accelerate DETR convergence on dense aerial targets.
- Replace multi-class softmax in DETR prediction heads with independent **binary sigmoid focal loss**
  ($\alpha=0.25, \gamma=2.0$) per class.
- Re-balance Hungarian matching costs:
  - Class cost: 2.0
  - L1 box cost: 5.0
  - Rotated GIoU/KLD cost: 2.0
  - Angle loss: Smooth L1 on $\sin(2\theta), \cos(2\theta)$ or Gaussian Wasserstein Distance (GWD).
- Add auxiliary decoding loss to intermediate transformer decoder layers to ensure gradient flow.
- **Acceptance Criteria:** In DETR training, matched query mean foreground probability exceeds 0.50 by
  epoch 25; zero query collapse warnings.

### C3 — Continuous Cosine Scheduling & Multi-Stage Resumption
**Objective:** Allow seamless extension of any model from early checkpoints to full convergence.
- When resuming from an existing checkpoint where `ckpt_epoch < target_epochs`:
  - Automatically update the scheduler's $T_{max}$ to `target_epochs`.
  - Recalculate and set the continuous learning rate:
    $$\text{LR} = \eta_{\min} + \frac{1}{2}(\text{base\_lr} - \eta_{\min})\left(1 + \cos\left(\frac{\text{ckpt\_epoch}}{\text{target\_epochs}}\pi\right)\right)$$
  - Ensure AdamW first and second momentum states are properly loaded without discontinuity.
- Support warm restart fine-tuning if resuming from an already decayed run.
- **Acceptance Criteria:** Resuming a 20-epoch checkpoint to 80 epochs continues with continuous LR decay
  down to $\eta_{\min}$ at epoch 80 with continuous loss curves.

### C4 — Ultralytics YOLO Fine-Tuning & Resume Parity
**Objective:** Ensure YOLO models can be cleanly extended beyond initial checkpoint epochs.
- Detect when a YOLO `last.pt` checkpoint has completed its internal `epochs`.
- When `target_epochs > ckpt_epochs`, update the training arguments or load weights into a fresh trainer
  instance so Ultralytics does not terminate with an early exit.
- Ensure evaluation passes through `src/metrics/` to maintain strict metric comparability with Dense and
  DETR models.
- **Acceptance Criteria:** Running `yolo11-obb` with resume on a 20-epoch checkpoint continues training to
  60 epochs; metrics generated match unified suite format.

### C5 — Full Multi-Dataset Convergence Suite Execution
**Objective:** Train and evaluate the benchmark models across all 3 cleaned datasets.
- Execute full convergence runs across:
  - `visdrone` (6,831 train / 733 val)
  - `codrone` (custom aerial drone imagery)
  - `dota` (14,119 train / 4,587 val)
- Ensure background staging streams images Drive $\to$ NVMe once per dataset, avoiding redundant re-staging.
- **Acceptance Criteria:** All target models achieve convergence; checkpoints and logs mirror to Drive every
  5 epochs; final metrics record finite mAP50 and mAP50-95.

### C6 — Cross-Dataset Generalization & Transfer Matrix
**Objective:** Evaluate cross-domain robustness without manual scripting.
- Implement automated execution of the $3 \times 3$ domain transfer evaluation:
  - Train on VisDrone $\to$ Test on CODrone, DOTA
  - Train on CODrone $\to$ Test on VisDrone, DOTA
  - Train on DOTA $\to$ Test on VisDrone, CODrone
- Output `results/cross_dataset_matrix.csv` and a styled Seaborn heatmap plot.
- **Acceptance Criteria:** `training/cross_eval.py` executes end-to-end for selected models across all 3
  datasets without re-staging; summary table outputs clean domain shift drop metrics ($\Delta\text{mAP}$).

### C7 — Headless Colab Auto-Recovery & Mirror Sync
**Objective:** Guarantee that Google Colab preemptions cost at most 1 checkpoint interval.
- Enhance `training/colab_quickstart.py` with an automated sync function:
  - On Colab boot, inspect `/content/drive/MyDrive/object-detection/runs_v4`.
  - Restore `pipeline_state.json` and all surviving `weights/last.pt` into `/content/minor-project/runs`.
  - Automatically resume the queue at the exact unfinished job and epoch.
- **Acceptance Criteria:** Killing a Colab runtime and re-running the quickstart cell seamlessly picks up
  from the last completed epoch without re-running finished models.

### C8 — Publication-Grade Artifact & Report Generation
**Objective:** Produce final camera-ready tables, figures, and summaries for the project report.
- Generate high-resolution vector figures in `results/figures/`:
  1. **Speed vs. Accuracy Pareto Curve:** FPS (imgs/s) on L4 GPU vs. mAP50 for all 23 models.
  2. **Model Size vs. Accuracy Scatter:** Parameter count (M) vs. mAP50-95.
  3. **Per-Class Radar Chart:** Comparing top YOLO, Dense, and DETR models across the 8 vehicle classes.
  4. **Qualitative Visualizations:** Ground truth vs. predicted oriented bounding boxes on sample test images
     with color-coded class labels.
- Output camera-ready LaTeX table code (`results/benchmark_table.tex`) ready for paper inclusion.
- **Acceptance Criteria:** Running `python scripts/generate_report.py` outputs all 4 publication figures and
  the complete LaTeX table.

---

## 4. Hard Constraints (Do NOT Break)

1. **Class Order & Indices:** Class indices `0` through `7` are strictly frozen per §2.2.
2. **Normalized Corner Representation:** Ground truth and prediction coordinates must strictly follow the
   YOLO-OBB clockwise normalized 4-corner format ($x_1, y_1, x_2, y_2, x_3, y_3, x_4, y_4 \in [0, 1]$).
3. **Metric Standardization:** All models must be evaluated using `src/metrics/detection_map.py` using
   un-letterboxed original-image pixel coordinates for ground truth.
4. **Lightweight Core:** No heavy orchestration frameworks (no Ray, MLflow, or Lightning). Keep dependencies
   to PyTorch, Ultralytics, OpenCV, NumPy, and standard libraries.
5. **No Data Leakage:** Stratified train/val/test splits must never overlap.

---

## 5. Verification & Acceptance Gates

Before certifying any changes, execute and pass all gates:

1. **Compilation Gate:** `python -m compileall -q .` $\to$ Exit 0.
2. **Import Gate:** `python -c "import src, training.run, src.training.dispatcher"` $\to$ Clean exit.
3. **Unit Test Gate:** `python tests/test_v4_units.py` $\to$ 100% of tests pass.
4. **Smoke Gate (Mock Data):**
   `python training/run.py --models lsknet ao2-detr yolo11-obb --datasets mock --steps stage,train,eval,summary --test`
   $\to$ Completes end-to-end with valid loss, mAP, and manifest generation.
5. **Scheduler Continuity Gate:** Run a 2-epoch test, interrupt, resume to 5 epochs, and verify learning rate
   and loss continue smoothly without jump or freeze.
6. **Report Generation Gate:** `python training/run.py --steps summary,report --test` $\to$ Generates summary
   CSV and markdown reports.

---

## 6. Deliverable: `minor-project-code-v4-latest.zip`

Package the codebase:
- **Zip Archive:** `minor-project-code-v4-latest.zip` (must be $< 25$ MB).
- **Structure:** Root contains `minor-project/`. Exclude `.git`, `venv`, `data/`, `saves/`, `weights/`,
  `*.pyc`, `*.pt`, `scratch/`, and `literature survey/papers/`.
- **Must Include:** All source files, `training/`, `src/`, `configs/`, `tests/`, `docs/`, `scripts/`,
  `assets/dota_labels_all.zip`, `requirements.txt`, `README.md`, and this `PROMPT.md`.

---

## 7. Execution Commands for Colab GPU Convergence

To launch the full convergence suite on Google Colab:

```bash
# 1. Mount Drive & Install Dependencies
from google.colab import drive
drive.mount('/content/drive')
!pip install -q -r /content/minor-project/requirements.txt

# 2. Run Full Convergence Across Key Architectures
!python /content/minor-project/training/run.py \
  --models custom-obb lsknet std rvsa ao2-detr ars-detr yolo11n-obb yolo11s-obb yolo26s-obb \
  --datasets visdrone codrone dota \
  --steps stage,train,eval,cross,summary,report \
  --device cuda \
  --workers 8 \
  --cache annotations \
  --runs-dir /content/minor-project/runs \
  --mirror-dir /content/drive/MyDrive/object-detection/runs_v4 \
  --mirror-every 5
```
