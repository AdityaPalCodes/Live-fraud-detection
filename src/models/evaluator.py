"""
Model evaluation module for highly imbalanced fraud detection.

Computes metrics that matter in production fraud systems:
- Precision, Recall, F1
- Precision-Recall AUC (PR-AUC / Average Precision)
- ROC-AUC
- False Positive Rate (FPR), False Negative Rate (FNR)
- Brier Score (probability calibration error)
- Confusion Matrix
- Financial Cost
"""

import logging
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    brier_score_loss,
    precision_recall_curve,
    roc_curve
)

logger = logging.getLogger(__name__)


def evaluate_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    fp_cost: float = 25.0,
    fn_cost: float = 250.0
) -> Dict[str, Any]:
    """
    Compute comprehensive fraud evaluation metrics for given ground truth and predicted probabilities.

    Args:
        y_true: Binary ground-truth labels (0 or 1).
        y_prob: Predicted probabilities for the positive class (fraud).
        threshold: Classification decision threshold.
        fp_cost: Financial/friction cost per False Positive ($).
        fn_cost: Financial loss per False Negative ($).

    Returns:
        Dictionary of metrics.
    """
    y_pred = (y_prob >= threshold).astype(int)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Core Imbalanced Metrics
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    pr_auc = average_precision_score(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)

    # Rates
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    # Calibration error
    brier = brier_score_loss(y_true, y_prob)

    # Financial Cost
    total_cost = (fp * fp_cost) + (fn * fn_cost)
    cost_per_transaction = total_cost / len(y_true) if len(y_true) > 0 else 0.0

    return {
        "threshold": round(threshold, 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "pr_auc": round(float(pr_auc), 4),
        "roc_auc": round(float(roc_auc), 4),
        "brier_score": round(float(brier), 5),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "fpr": round(float(fpr), 5),
        "fnr": round(float(fnr), 5),
        "total_cost": round(float(total_cost), 2),
        "cost_per_transaction": round(float(cost_per_transaction), 4)
    }


def print_evaluation_summary(metrics: Dict[str, Any], model_name: str = "Model") -> None:
    """Pretty prints evaluation metrics to console."""
    print(f"\n{'='*20} {model_name} Evaluation Summary {'='*20}")
    print(f"Threshold:           {metrics['threshold']:.2f}")
    print(f"PR-AUC (Primary):    {metrics['pr_auc']:.4f}")
    print(f"ROC-AUC:             {metrics['roc_auc']:.4f}")
    print(f"Recall (Sensitivity):{metrics['recall']*100:.2f}% (Found {metrics['tp']} of {metrics['tp']+metrics['fn']} frauds)")
    print(f"Precision:           {metrics['precision']*100:.2f}% ({metrics['fp']} false alarms)")
    print(f"F1-Score:            {metrics['f1']:.4f}")
    print(f"Brier Score:         {metrics['brier_score']:.5f}")
    print(f"Confusion Matrix:    [TN: {metrics['tn']:,} | FP: {metrics['fp']:,}]")
    print(f"                     [FN: {metrics['fn']:,} | TP: {metrics['tp']:,}]")
    print(f"Total Business Cost: ${metrics['total_cost']:,.2f} (${metrics['cost_per_transaction']:.3f}/tx)")
    print(f"{'='*60}\n")
