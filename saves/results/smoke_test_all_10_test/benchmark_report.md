# Aerial OBB Object Detection Benchmark & Comparative Analysis Report

## Executive Summary
- **Best Performing Model**: `ao2-detr`
- **Best Evaluated Dataset**: `visdrone`

## Pre-Training vs Post-Training Performance Progression
This table illustrates empirical model gains from zero-shot / base pretraining to full fine-tuning on the aerial datasets.

| Model | Dataset | Pre-Train mAP50 | Post-Train mAP50 | mAP Gain (Δ) | Pre F1 | Post F1 | Pre Angle MAE | Post Angle MAE |
|---|---|---|---|---|---|---|---|---|
| std | visdrone | 0.0000 | 0.8331 | **+83.3%** | 0.0000 | 0.8095 | nan° | 2.54° |
| rvsa | visdrone | 0.0000 | 0.8687 | **+86.9%** | 0.0000 | 0.8253 | nan° | 2.72° |
| ars-detr | visdrone | 0.0000 | 0.7391 | **+73.9%** | 0.0000 | 0.7182 | nan° | 2.73° |
| oriented-former | visdrone | 0.0000 | 0.8066 | **+80.7%** | 0.0000 | 0.7885 | nan° | 2.28° |
| rio-detr | visdrone | 0.0000 | 0.8779 | **+87.8%** | 0.0000 | 0.8299 | nan° | 2.18° |
| rhino | visdrone | 0.0000 | 0.8256 | **+82.6%** | 0.0000 | 0.7978 | nan° | 2.18° |
| ao2-detr | visdrone | 0.0000 | 0.8829 | **+88.3%** | 0.0000 | 0.8213 | nan° | 2.70° |
| swin-obb | visdrone | 0.0000 | 0.8291 | **+82.9%** | 0.0000 | 0.7944 | nan° | 2.07° |
| lsknet | visdrone | 0.0000 | 0.7847 | **+78.5%** | 0.0000 | 0.7303 | nan° | 2.07° |
| yolo11-obb | visdrone | 0.0000 | 0.0033 | **+0.3%** | 0.0000 | 0.0051 | nan° | 1.57° |

## Complete Performance Metrics Across Models & Datasets

| Model | Dataset | mAP@0.50 | mAP@0.75 | mAP@0.50:0.95 | Precision | Recall | F1-Score | Accuracy | Angle MAE | Pearson r | R² Score | FPS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| std | visdrone | 0.8331 | 0.6381 | 0.5585 | 0.8740 | 0.7563 | 0.8095 | 0.8074 | 2.54° | 0.0208 | -1.7908 | 28.2 |
| rvsa | visdrone | 0.8687 | 0.6447 | 0.5612 | 0.8630 | 0.7928 | 0.8253 | 0.7907 | 2.72° | -0.0165 | -1.9562 | 12.7 |
| ars-detr | visdrone | 0.7391 | 0.4787 | 0.4470 | 0.7547 | 0.6868 | 0.7182 | 0.7775 | 2.73° | -0.0312 | -1.9699 | 31.6 |
| oriented-former | visdrone | 0.8066 | 0.5738 | 0.5267 | 0.8468 | 0.7416 | 0.7885 | 0.7943 | 2.28° | 0.2390 | -1.0505 | 37.6 |
| rio-detr | visdrone | 0.8779 | 0.6329 | 0.5805 | 0.8697 | 0.7971 | 0.8299 | 0.8161 | 2.18° | 0.1383 | -1.3696 | 41.7 |
| rhino | visdrone | 0.8256 | 0.6123 | 0.5495 | 0.8577 | 0.7585 | 0.7978 | 0.8373 | 2.18° | 0.0028 | -1.9142 | 29.4 |
| ao2-detr | visdrone | 0.8829 | 0.6383 | 0.5780 | 0.8553 | 0.7961 | 0.8213 | 0.7758 | 2.70° | -0.0729 | -2.1375 | 25.4 |
| swin-obb | visdrone | 0.8291 | 0.6602 | 0.5832 | 0.8461 | 0.7514 | 0.7944 | 0.7829 | 2.07° | 0.1239 | -1.4488 | 34.5 |
| lsknet | visdrone | 0.7847 | 0.5768 | 0.5221 | 0.7670 | 0.7002 | 0.7303 | 0.8556 | 2.07° | 0.2288 | -1.0550 | 28.3 |
| yolo11-obb | visdrone | 0.0033 | 0.0000 | 0.0013 | 0.1000 | 0.0026 | 0.0051 | 0.0027 | 1.57° | 0.0000 | 0.0000 | 60.3 |

## Metrics Definitions & Interpretation
1. **Detection Metrics (mAP50, mAP75, mAP50-95)**: Area under precision-recall curve. mAP75 measures strict spatial and angular alignment critical for aerial vehicles.
2. **Classification Metrics (Precision, Recall, F1, Accuracy)**: Evaluates whether matched oriented detections correctly distinguish vehicle types and road users without background false alarms.
3. **Orientation Regression Metrics (Angle MAE, Pearson r, R²)**: Continuous goodness-of-fit for heading angles. Pearson r close to 1.0 indicates strong directional fidelity.
4. **Operational Metrics (FPS, Latency)**: Real-time suitability for UAV edge deployment (target: >30 FPS).
