# Excel Export - generated 2026-11-03T19:49:13

Source run folders scanned: 10

| File | Feeds | Notes |
|---|---|---|
| `summary_all_runs.csv` | appendix / audit | every (run, phase, model, dataset) with all metrics |
| `summary_latest.csv` | xlsx **Summary** sheet | latest run per (phase, model, dataset) |
| `summary_matrix_post_train.csv` | xlsx **Summary** sheet | models as columns, metrics as rows - direct paste |
| `class_wise_all.csv` | xlsx **class_wise_results** sheet | per-class AP50/AP75/TP/FP/FN/P/R/F1 |
| `cross_dataset_pairs.csv` | cross-dataset section | one row per (model, source, eval) pair |
| `cross_dataset_global.csv` | cross-dataset global table | macro-average per (model, source) |

## Column dictionary (summary tables)
- IDs: `run`, `run_dir`, `run_timestamp` (UTC ISO), `phase` (pre_train / post_train / eval / cross_eval),
  `model`, `dataset` (eval dataset; for cross_eval this is the target dataset),
  `source_dataset` (training dataset for cross_eval rows), `total_gt_boxes`
- Detection: `map50`, `map75`, `map50_95`, `ap_small`, `ap_medium`, `ap_large`
- Classification: `precision`, `recall`, `f1`, `accuracy`, `specificity` (macro-averaged)
- Angle regression (matched boxes): `angle_mae` (deg), `angle_rmse` (deg), `pearson_r`,
  `spearman_rho`, `r2_score`, `center_offset_mae`/`rmse` (px), `aspect_ratio_r2`
- Efficiency: `fps`, `mean_latency_ms`

## Reproduce
```
python scripts/export_results.py \
    --extra-dirs /path/to/cross_eval_run_folder
```
