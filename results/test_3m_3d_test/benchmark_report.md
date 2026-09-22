# Aerial OBB Object Detection Benchmark & Comparative Analysis Report

## Executive Summary
- **Best Performing Model**: `yolov8n-obb`
- **Best Evaluated Dataset**: `dota`

## Complete Performance Metrics Across Models & Datasets

| Model | Dataset | mAP@0.50 | mAP@0.75 | mAP@0.50:0.95 | Precision | Recall | F1-Score | Accuracy | Angle MAE | Pearson r | R² Score | FPS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| yolov8n-obb | codrone | 0.7646 | 0.6463 | 0.5879 | 0.5192 | 0.4452 | 0.4774 | 0.6886 | 2.49° | 0.7824 | 0.5626 | 6740.0 |
| yolo11n-obb | codrone | 0.7473 | 0.6486 | 0.6214 | 0.5133 | 0.4282 | 0.4646 | 0.6851 | 2.53° | 0.6855 | 0.3623 | 6761.4 |
| custom-obb | codrone | 0.6729 | 0.4824 | 0.5089 | 0.4879 | 0.3763 | 0.4198 | 0.4590 | 4.42° | 0.6233 | 0.2503 | 12.1 |
| yolov8n-obb | visdrone | 0.8315 | 0.6563 | 0.5863 | 0.8306 | 0.7357 | 0.7751 | 0.7784 | 2.55° | 0.0000 | 0.0000 | 9608.4 |
| yolo11n-obb | visdrone | 0.8893 | 0.6997 | 0.5864 | 0.8517 | 0.7931 | 0.8176 | 0.8462 | 2.62° | 0.0000 | 0.0000 | 8567.2 |
| custom-obb | visdrone | 0.6281 | 0.2733 | 0.3458 | 0.7111 | 0.5448 | 0.6146 | 0.5707 | 4.66° | 0.0000 | 0.0000 | 14.1 |
| yolov8n-obb | dota | 0.9512 | 0.8900 | 0.8686 | 0.2653 | 0.2192 | 0.2400 | 0.8204 | 2.38° | 0.6681 | 0.3368 | 8188.7 |
| yolo11n-obb | dota | 0.8957 | 0.8376 | 0.8103 | 0.2663 | 0.2298 | 0.2462 | 0.8250 | 2.58° | 0.6733 | 0.3208 | 9903.4 |
| custom-obb | dota | 0.9099 | 0.8097 | 0.8274 | 0.2573 | 0.1808 | 0.2115 | 0.6463 | 4.45° | 0.5009 | -0.0263 | 13.1 |

## Metrics Definitions & Interpretation
1. **Detection Metrics (mAP50, mAP75, mAP50-95)**: Area under precision-recall curve. mAP75 measures strict spatial and angular alignment critical for aerial vehicles.
2. **Classification Metrics (Precision, Recall, F1, Accuracy)**: Evaluates whether matched oriented detections correctly distinguish vehicle types and road users without background false alarms.
3. **Orientation Regression Metrics (Angle MAE, Pearson r, R²)**: Continuous goodness-of-fit for heading angles. Pearson r close to 1.0 indicates strong directional fidelity.
4. **Operational Metrics (FPS, Latency)**: Real-time suitability for UAV edge deployment (target: >30 FPS).
