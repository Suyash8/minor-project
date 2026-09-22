# Aerial OBB Object Detection Benchmark & Comparative Analysis Report

## Executive Summary
- **Best Performing Model**: `yolov8n-obb`
- **Best Evaluated Dataset**: `visdrone`

## Complete Performance Metrics Across Models & Datasets

| Model | Dataset | mAP@0.50 | mAP@0.75 | mAP@0.50:0.95 | Precision | Recall | F1-Score | Accuracy | Angle MAE | Pearson r | R² Score | FPS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| yolov8n-obb | codrone | 0.7720 | 0.6507 | 0.5950 | 0.5264 | 0.4525 | 0.4847 | 0.6917 | 2.48° | 0.7834 | 0.5646 | 7869.6 |
| custom-obb | codrone | 0.6853 | 0.5178 | 0.5230 | 0.4611 | 0.3974 | 0.4228 | 0.4836 | 4.60° | 0.6626 | 0.3266 | 15.3 |
| yolov8n-obb | visdrone | 0.8781 | 0.6583 | 0.6132 | 0.8673 | 0.7777 | 0.8186 | 0.7984 | 2.56° | 0.0000 | 0.0000 | 7500.7 |
| custom-obb | visdrone | 0.6986 | 0.3107 | 0.4015 | 0.7780 | 0.6317 | 0.6900 | 0.5739 | 4.56° | 0.0000 | 0.0000 | 15.6 |

## Metrics Definitions & Interpretation
1. **Detection Metrics (mAP50, mAP75, mAP50-95)**: Area under precision-recall curve. mAP75 measures strict spatial and angular alignment critical for aerial vehicles.
2. **Classification Metrics (Precision, Recall, F1, Accuracy)**: Evaluates whether matched oriented detections correctly distinguish vehicle types and road users without background false alarms.
3. **Orientation Regression Metrics (Angle MAE, Pearson r, R²)**: Continuous goodness-of-fit for heading angles. Pearson r close to 1.0 indicates strong directional fidelity.
4. **Operational Metrics (FPS, Latency)**: Real-time suitability for UAV edge deployment (target: >30 FPS).
