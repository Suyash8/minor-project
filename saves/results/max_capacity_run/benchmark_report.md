# Aerial OBB Object Detection Benchmark & Comparative Analysis Report

## Executive Summary
- **Best Performing Model**: `yolov8m-obb`
- **Best Evaluated Dataset**: `visdrone`

## Pre-Training vs Post-Training Performance Progression
This table illustrates empirical model gains from zero-shot / base pretraining to full fine-tuning on the aerial datasets.

| Model | Dataset | Pre-Train mAP50 | Post-Train mAP50 | mAP Gain (Δ) | Pre F1 | Post F1 | Pre Angle MAE | Post Angle MAE |
|---|---|---|---|---|---|---|---|---|
| yolov8m-obb | visdrone | 0.0000 | 0.0000 | **+0.0%** | 0.0000 | 0.0000 | 0.00° | 0.00° |
| yolo11m-obb | visdrone | 0.0000 | 0.0000 | **+0.0%** | 0.0000 | 0.0000 | 0.00° | 0.00° |
| custom-obb | visdrone | 0.0161 | 0.0000 | **-1.6%** | 0.0071 | 0.0000 | 84.21° | 0.00° |
| yolov8m-obb | codrone | 0.0000 | 0.0000 | **+0.0%** | 0.0000 | 0.0000 | 0.00° | 0.00° |
| yolo11m-obb | codrone | 0.0000 | 0.0000 | **+0.0%** | 0.0000 | 0.0000 | 0.00° | 0.00° |
| custom-obb | codrone | 0.0006 | 0.0000 | **-0.1%** | 0.0019 | 0.0000 | 58.06° | 0.00° |
| yolov8m-obb | dota | 0.0000 | 0.0000 | **+0.0%** | 0.0000 | 0.0000 | 0.00° | 0.00° |
| yolo11m-obb | dota | 0.0000 | 0.0000 | **+0.0%** | 0.0000 | 0.0000 | 0.00° | 0.00° |
| custom-obb | dota | 0.0000 | 0.0000 | **+0.0%** | 0.0000 | 0.0000 | 0.00° | 0.00° |

## Complete Performance Metrics Across Models & Datasets

| Model | Dataset | mAP@0.50 | mAP@0.75 | mAP@0.50:0.95 | Precision | Recall | F1-Score | Accuracy | Angle MAE | Pearson r | R² Score | FPS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| yolov8m-obb | visdrone | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00° | 0.0000 | 0.0000 | 50.3 |
| yolo11m-obb | visdrone | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00° | 0.0000 | 0.0000 | 48.4 |
| custom-obb | visdrone | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00° | 0.0000 | 0.0000 | 229.4 |
| yolov8m-obb | codrone | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00° | 0.0000 | 0.0000 | 63.2 |
| yolo11m-obb | codrone | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00° | 0.0000 | 0.0000 | 56.2 |
| custom-obb | codrone | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00° | 0.0000 | 0.0000 | 47.1 |
| yolov8m-obb | dota | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00° | 0.0000 | 0.0000 | 45.3 |
| yolo11m-obb | dota | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00° | 0.0000 | 0.0000 | 41.3 |
| custom-obb | dota | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00° | 0.0000 | 0.0000 | 61.1 |

## Metrics Definitions & Interpretation
1. **Detection Metrics (mAP50, mAP75, mAP50-95)**: Area under precision-recall curve. mAP75 measures strict spatial and angular alignment critical for aerial vehicles.
2. **Classification Metrics (Precision, Recall, F1, Accuracy)**: Evaluates whether matched oriented detections correctly distinguish vehicle types and road users without background false alarms.
3. **Orientation Regression Metrics (Angle MAE, Pearson r, R²)**: Continuous goodness-of-fit for heading angles. Pearson r close to 1.0 indicates strong directional fidelity.
4. **Operational Metrics (FPS, Latency)**: Real-time suitability for UAV edge deployment (target: >30 FPS).
