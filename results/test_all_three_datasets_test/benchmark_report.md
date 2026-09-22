# Aerial OBB Object Detection Benchmark & Comparative Analysis Report

## Executive Summary
- **Best Performing Model**: `yolov8n-obb`
- **Best Evaluated Dataset**: `dota`

## Complete Performance Metrics Across Models & Datasets

| Model | Dataset | mAP@0.50 | mAP@0.75 | mAP@0.50:0.95 | Precision | Recall | F1-Score | Accuracy | Angle MAE | Pearson r | R² Score | FPS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| yolov8n-obb | codrone | 0.7646 | 0.6463 | 0.5879 | 0.5192 | 0.4452 | 0.4774 | 0.6886 | 2.49° | 0.7824 | 0.5626 | 7087.1 |
| custom-obb | codrone | 0.7144 | 0.4705 | 0.4913 | 0.5297 | 0.4104 | 0.4572 | 0.4372 | 4.86° | 0.6695 | 0.3354 | 12.2 |
| yolov8n-obb | visdrone | 0.7695 | 0.5535 | 0.5015 | 0.8300 | 0.7781 | 0.7855 | 0.8179 | 2.45° | 0.0000 | 0.0000 | 5331.5 |
| custom-obb | visdrone | 0.6276 | 0.2436 | 0.3386 | 0.6814 | 0.5314 | 0.5899 | 0.5072 | 4.72° | 0.0000 | 0.0000 | 12.2 |
| yolov8n-obb | dota | 0.8791 | 0.8267 | 0.8051 | 0.2663 | 0.2132 | 0.2355 | 0.8500 | 2.56° | 0.6439 | 0.2750 | 6929.5 |
| custom-obb | dota | 0.7799 | 0.6408 | 0.6777 | 0.2570 | 0.1834 | 0.2137 | 0.6256 | 4.17° | 0.4813 | -0.0448 | 15.3 |

## Metrics Definitions & Interpretation
1. **Detection Metrics (mAP50, mAP75, mAP50-95)**: Area under precision-recall curve. mAP75 measures strict spatial and angular alignment critical for aerial vehicles.
2. **Classification Metrics (Precision, Recall, F1, Accuracy)**: Evaluates whether matched oriented detections correctly distinguish vehicle types and road users without background false alarms.
3. **Orientation Regression Metrics (Angle MAE, Pearson r, R²)**: Continuous goodness-of-fit for heading angles. Pearson r close to 1.0 indicates strong directional fidelity.
4. **Operational Metrics (FPS, Latency)**: Real-time suitability for UAV edge deployment (target: >30 FPS).
