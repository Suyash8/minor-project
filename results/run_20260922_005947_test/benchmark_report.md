# Aerial OBB Object Detection Benchmark & Comparative Analysis Report

## Executive Summary
- **Best Performing Model**: `yolov8n-obb`
- **Best Evaluated Dataset**: `codrone`

## Complete Performance Metrics Across Models & Datasets

| Model | Dataset | mAP@0.50 | mAP@0.75 | mAP@0.50:0.95 | Precision | Recall | F1-Score | Accuracy | Angle MAE | Pearson r | R² Score | FPS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| yolov8n-obb | codrone | 0.8878 | 0.5973 | 0.5433 | 0.8796 | 0.8754 | 0.8680 | 0.7586 | 2.42° | 0.6007 | 0.2102 | 7452.9 |
| custom-obb | codrone | 0.7607 | 0.3218 | 0.3703 | 0.9306 | 0.7391 | 0.7870 | 0.6364 | 5.61° | 0.5315 | 0.0347 | 16.6 |

## Metrics Definitions & Interpretation
1. **Detection Metrics (mAP50, mAP75, mAP50-95)**: Area under precision-recall curve. mAP75 measures strict spatial and angular alignment critical for aerial vehicles.
2. **Classification Metrics (Precision, Recall, F1, Accuracy)**: Evaluates whether matched oriented detections correctly distinguish vehicle types and road users without background false alarms.
3. **Orientation Regression Metrics (Angle MAE, Pearson r, R²)**: Continuous goodness-of-fit for heading angles. Pearson r close to 1.0 indicates strong directional fidelity.
4. **Operational Metrics (FPS, Latency)**: Real-time suitability for UAV edge deployment (target: >30 FPS).
