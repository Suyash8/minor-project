"""
Classification metrics derived from detection confusion matrix:
Precision, Recall, F1-Score, Specificity, and Detection Accuracy.
"""

from __future__ import annotations

from typing import Dict, Any, List
import numpy as np

def compute_classification_metrics(cm: np.ndarray, class_names: List[str]) -> Dict[str, Any]:
    """
    Extract multi-class classification and detection metrics from the (K+1)x(K+1) confusion matrix.
    """
    k = len(class_names)
    tp_per_class = np.zeros(k, dtype=float)
    fp_per_class = np.zeros(k, dtype=float)
    fn_per_class = np.zeros(k, dtype=float)
    tn_per_class = np.zeros(k, dtype=float)

    total_instances = np.sum(cm)

    for i in range(k):
        tp = float(cm[i, i])
        fp = float(np.sum(cm[:, i]) - tp)
        fn = float(np.sum(cm[i, :]) - tp)
        tn = float(total_instances - tp - fp - fn)

        tp_per_class[i] = tp
        fp_per_class[i] = fp
        fn_per_class[i] = fn
        tn_per_class[i] = tn

    # Safe divisions without warnings
    precision_per_class = np.divide(
        tp_per_class, tp_per_class + fp_per_class,
        out=np.zeros_like(tp_per_class),
        where=(tp_per_class + fp_per_class > 0)
    )
    recall_per_class = np.divide(
        tp_per_class, tp_per_class + fn_per_class,
        out=np.zeros_like(tp_per_class),
        where=(tp_per_class + fn_per_class > 0)
    )
    denom_f1 = precision_per_class + recall_per_class
    f1_per_class = np.divide(
        2.0 * precision_per_class * recall_per_class, denom_f1,
        out=np.zeros_like(denom_f1),
        where=(denom_f1 > 0)
    )
    specificity_per_class = np.divide(
        tn_per_class, tn_per_class + fp_per_class,
        out=np.zeros_like(tn_per_class),
        where=(tn_per_class + fp_per_class > 0)
    )

    total_tp = float(np.sum(tp_per_class))
    total_fp = float(np.sum(fp_per_class))
    total_fn = float(np.sum(fn_per_class))

    # Overall Detection Accuracy (Jaccard-like index over all detections and targets)
    accuracy = total_tp / max(total_tp + total_fp + total_fn, 1e-12)

    return {
        "accuracy": float(accuracy),
        "macro_precision": float(np.mean(precision_per_class)),
        "macro_recall": float(np.mean(recall_per_class)),
        "macro_f1": float(np.mean(f1_per_class)),
        "macro_specificity": float(np.mean(specificity_per_class)),
        "per_class": {
            class_names[i]: {
                "precision": float(precision_per_class[i]),
                "recall": float(recall_per_class[i]),
                "f1": float(f1_per_class[i]),
                "specificity": float(specificity_per_class[i]),
                "tp": int(tp_per_class[i]),
                "fp": int(fp_per_class[i]),
                "fn": int(fn_per_class[i]),
            }
            for i in range(k)
        },
    }
