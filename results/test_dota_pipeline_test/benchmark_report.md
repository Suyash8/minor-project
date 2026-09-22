# Aerial OBB Object Detection Benchmark & Comparative Analysis Report

## Executive Summary
- **Best Performing Model**: `yolov8n-obb`
- **Best Evaluated Dataset**: `dota`

## Complete Performance Metrics Across Models & Datasets

| Model | Dataset | mAP@0.50 | mAP@0.75 | mAP@0.50:0.95 | Precision | Recall | F1-Score | Accuracy | Angle MAE | Pearson r | R² Score | FPS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| yolov8n-obb | dota | 0.8858 | 0.8247 | 0.8091 | 0.2667 | 0.2197 | 0.2408 | 0.8371 | 2.44° | 0.6711 | 0.3352 | 7047.6 |
| custom-obb | dota | 0.8346 | 0.7225 | 0.7444 | 0.2552 | 0.1720 | 0.2053 | 0.6489 | 4.53° | 0.5341 | -0.0010 | 16.2 |

## Metrics Definitions & Interpretation
1. **Detection Metrics (mAP50, mAP75, mAP50-95)**: Area under precision-recall curve. mAP75 measures strict spatial and angular alignment critical for aerial vehicles.
2. **Classification Metrics (Precision, Recall, F1, Accuracy)**: Evaluates whether matched oriented detections correctly distinguish vehicle types and road users without background false alarms.
3. **Orientation Regression Metrics (Angle MAE, Pearson r, R²)**: Continuous goodness-of-fit for heading angles. Pearson r close to 1.0 indicates strong directional fidelity.
4. **Operational Metrics (FPS, Latency)**: Real-time suitability for UAV edge deployment (target: >30 FPS).
