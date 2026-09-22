# Aerial OBB Object Detection & Benchmark Suite

A modular, reproducible research framework for evaluating and analyzing Oriented Bounding Box (OBB) object detection in challenging aerial drone conditions (e.g. 30° oblique tilt, altitude drift 30m–100m, sub-15px targets, dense traffic queues, and road shadows).

---

## Directory Structure

```
minor-project/
├── requirements.txt            # Core dependencies (torch, torchvision, ultralytics, scipy, etc.)
├── install.py                  # Automated installer (prefers uv, falls back to pip)
├── colab_runner.ipynb          # 1-Click Google Colab GPU runner
├── README.md                   # Comprehensive documentation and usage guide
├── scripts/
│   ├── run_pipeline.py         # Master CLI runner (download -> eval -> metrics -> plots -> report)
│   ├── download_datasets.py    # Dedicated dataset downloader & manual guide
│   ├── run_evaluation.py       # Standalone single-model evaluation runner
│   └── generate_report.py      # Standalone report and plot generator
├── src/
│   ├── config.py               # Dataset metadata, classes, thresholds, and paths
│   ├── data/
│   │   ├── downloader.py       # Automated dataset downloads & manual fallback guides
│   │   ├── dataset.py          # Unified OBB dataset reader (YOLO-OBB, DOTA, VisDrone)
│   │   └── mock_data.py        # Synthetic small-object aerial OBB generator for fast CPU testing
│   ├── models/
│   │   ├── base.py             # Base detector interface & latency benchmark
│   │   ├── yolo_obb.py         # Ultralytics YOLOv8-OBB / YOLO11-OBB wrapper
│   │   └── custom_obb.py       # Pure PyTorch oriented detector baseline (no C++ extensions)
│   ├── metrics/
│   │   ├── detection_map.py    # mAP50, mAP75, mAP50-95, AP_s, AP_m, AP_l (101-point COCO style)
│   │   ├── confusion_matrix.py # Multi-class confusion matrix with Background FP/FN
│   │   ├── classification.py   # Accuracy, Precision, Recall, F1, Specificity
│   │   ├── regression.py       # Angle MAE/RMSE, Pearson correlation r, Spearman rho, R²
│   │   └── geometry.py         # Rotated polygon IoU, Gaussian GWD & KLD
│   └── utils/
│       ├── env.py              # Hardware detection (CPU/CUDA) & seeding
│       ├── visualizer.py       # Confusion matrix heatmaps, Pearson scatter plots, bar charts
│       └── reporter.py         # ASCII terminal tables & Markdown report generation
└── results/                    # Generated output directory (CSVs, JSONs, plots, reports)
```

---

## Installation

Run the automated installer:
```bash
python install.py
```
Or install via `uv` or `pip`:
```bash
uv pip install -r requirements.txt
# or
pip install -r requirements.txt
```

---

## Quickstart: Laptop CPU Smoke Test (`--test`)

To test the entire pipeline on a laptop with **no GPU** and **no multi-gigabyte downloads**:
```bash
python scripts/run_pipeline.py --test --save-plots
```
This runs synthetic aerial scenes through the models, calculates all detection, classification, and regression metrics, generates confusion matrix heatmaps and Pearson scatter plots, and writes a markdown summary table in ~5–10 seconds.

---

## Full Evaluation on Datasets

### 1. Check or Download Datasets
```bash
# Check dataset status and view manual instructions:
python scripts/download_datasets.py --check-only

# Download automated datasets (e.g. VisDrone):
python scripts/download_datasets.py --datasets visdrone
```

> **Manual Downloads**:
> - **CODrone (2025)**: Hosted on Google Drive / Baidu via [GitHub](https://github.com/AHideoKuzeA/CODrone). Place extracted files under `data/codrone/`.
> - **DOTA**: Hosted on Baidu / Google Drive via [DOTA Official Site](https://captain-whu.github.io/DOTA/dataset.html). Place extracted files under `data/dota/`.

### 2. Run Comprehensive Multi-Model Benchmark
```bash
python scripts/run_pipeline.py \
    --datasets codrone visdrone \
    --models yolov8n-obb custom-obb \
    --device auto \
    --save-plots
```

### 3. Run on Google Colab with GPU
Open [colab_runner.ipynb](file:///home/illionar/Projects/minor-project/colab_runner.ipynb) in Google Colab, select a **T4 GPU** runtime, and run all cells.

---

## Metrics Computed

| Category | Metric | Description |
|---|---|---|
| **Detection** | `mAP@0.50` | Standard detection accuracy at 0.50 IoU. |
| | `mAP@0.75` | Strict localization accuracy measuring angular alignment. |
| | `mAP@0.50:0.95` | Mean AP averaged across 10 IoU thresholds (0.50 to 0.95). |
| | `AP_s`, `AP_m`, `AP_l` | Scale-specific AP for small ($<32^2$), medium ($32^2–96^2$), and large ($>96^2$) objects. |
| **Classification** | `Confusion Matrix` | $(K+1) \times (K+1)$ matrix including explicit **Background** false alarms and misses. |
| | `Precision` & `Recall` | True Positive rate and precision per category and macro-average. |
| | `F1-Score` | Harmonic mean of Precision and Recall. |
| | `Accuracy` | Overall classification accuracy over matched targets. |
| | `Specificity` | True Negative rate across categories. |
| **Regression** | `Angle MAE` & `RMSE` | Mean and root-mean-squared angular error (°) modulo $180^\circ$. |
| | `Pearson r` | Linear correlation between ground-truth and predicted orientation angles. |
| | `Spearman ρ` | Monotonic rank correlation of orientation. |
| | `R² Score` | Coefficient of determination for heading angle regression. |
| | `Center Offset Error` | Euclidean pixel offset between ground truth and predicted center coordinates. |
| | `Aspect Ratio Fit` | $R^2$ and Pearson correlation for width-to-height ratio preservation. |
| **Operational** | `FPS` & `Latency` | Inference throughput and per-image latency (ms). |
| | `Parameters (M)` | Model size in millions of trainable weights. |
