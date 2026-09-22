# Aerial OBB Object Detection Benchmark & Comparative Analysis Report

## Executive Summary
- **Best Performing Model**: `yolov8n-obb`
- **Best Evaluated Dataset**: `dota`

## Complete Performance Metrics Across Models & Datasets

| Model | Dataset | mAP@0.50 | mAP@0.75 | mAP@0.50:0.95 | Precision | Recall | F1-Score | Accuracy | Angle MAE | Pearson r | R² Score | FPS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| yolov8n-obb | dota | 0.8865 | 0.8288 | 0.8105 | 0.2667 | 0.2201 | 0.2410 | 0.8345 | 2.49° | 0.6814 | 0.3574 | 7708.8 |

## Metrics Definitions & Interpretation
1. **Detection Metrics (mAP50, mAP75, mAP50-95)**: Area under precision-recall curve. mAP75 measures strict spatial and angular alignment critical for aerial vehicles.
2. **Classification Metrics (Precision, Recall, F1, Accuracy)**: Evaluates whether matched oriented detections correctly distinguish vehicle types and road users without background false alarms.
3. **Orientation Regression Metrics (Angle MAE, Pearson r, R²)**: Continuous goodness-of-fit for heading angles. Pearson r close to 1.0 indicates strong directional fidelity.
4. **Operational Metrics (FPS, Latency)**: Real-time suitability for UAV edge deployment (target: >30 FPS).
