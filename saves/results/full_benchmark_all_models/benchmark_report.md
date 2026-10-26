# Aerial OBB Object Detection Benchmark & Comparative Analysis Report

## Executive Summary
- **Best Performing Model**: `custom-obb`
- **Best Evaluated Dataset**: `dota`

## Complete Performance Metrics Across Models & Datasets

| Model | Dataset | mAP@0.50 | mAP@0.75 | mAP@0.50:0.95 | Precision | Recall | F1-Score | Accuracy | Angle MAE | Pearson r | R² Score | FPS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| yolov8n-obb | codrone | 0.0177 | 0.0177 | 0.0134 | 0.0312 | 0.0278 | 0.0294 | 0.0462 | 0.73° | -0.8307 | -932.6750 | 67.5 |
| yolo11n-obb | codrone | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00° | 0.0000 | 0.0000 | 68.2 |
| custom-obb | codrone | 0.6014 | 0.1080 | 0.2766 | 0.6999 | 0.5919 | 0.5944 | 0.4615 | 5.42° | 0.4508 | -0.0911 | 97.4 |
| yolov8n-obb | visdrone | 0.0157 | 0.0037 | 0.0034 | 0.0222 | 0.0667 | 0.0333 | 0.0351 | 1.03° | 1.0000 | 0.9999 | 69.9 |
| yolo11n-obb | visdrone | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00° | 0.0000 | 0.0000 | 60.7 |
| custom-obb | visdrone | 0.5297 | 0.1891 | 0.2471 | 0.8833 | 0.5279 | 0.6399 | 0.5400 | 4.79° | 0.9016 | 0.7875 | 53.4 |
| yolov8n-obb | dota | 0.0002 | 0.0001 | 0.0001 | 0.0005 | 0.0444 | 0.0011 | 0.0007 | 3.57° | 0.9626 | 0.5560 | 62.0 |
| yolo11n-obb | dota | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00° | 0.0000 | 0.0000 | 52.2 |
| custom-obb | dota | 0.7581 | 0.2940 | 0.3833 | 0.3892 | 0.7733 | 0.4873 | 0.3388 | 5.06° | 0.6642 | 0.3166 | 66.0 |

## Metrics Definitions & Interpretation
1. **Detection Metrics (mAP50, mAP75, mAP50-95)**: Area under precision-recall curve. mAP75 measures strict spatial and angular alignment critical for aerial vehicles.
2. **Classification Metrics (Precision, Recall, F1, Accuracy)**: Evaluates whether matched oriented detections correctly distinguish vehicle types and road users without background false alarms.
3. **Orientation Regression Metrics (Angle MAE, Pearson r, R²)**: Continuous goodness-of-fit for heading angles. Pearson r close to 1.0 indicates strong directional fidelity.
4. **Operational Metrics (FPS, Latency)**: Real-time suitability for UAV edge deployment (target: >30 FPS).
