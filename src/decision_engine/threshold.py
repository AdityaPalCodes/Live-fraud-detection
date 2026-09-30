"""
Threshold optimization module for fraud detection.

Evaluates trade-offs across:
- Precision vs Recall curve
- F1-Score maximization
- Cost-sensitive loss minimization
- Tri-state operating thresholds (Approve, Review, Decline)
"""

import logging
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, f1_score

from src.decision_engine.cost_model import CostModel

logger = logging.getLogger(__name__)


def optimize_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    fp_cost: float = 25.0,
    fn_cost: float = 250.0,
    min_review_recall: float = 0.90,
    min_decline_precision: float = 0.85
) -> Dict[str, Any]:
    """
    Search and recommend operational thresholds based on validation empirical distributions:
    1. cost_optimal_threshold: Minimizes total financial loss ($)
    2. f1_optimal_threshold: Maximizes harmonic mean of Precision and Recall
    3. review_threshold: Catch >= min_review_recall of frauds (high sensitivity for human review queue)
    4. decline_threshold: Ensure >= min_decline_precision (high confidence for automatic hard decline)

    Args:
        y_true: Ground-truth binary labels.
        y_prob: Predicted risk probabilities.
        fp_cost: Cost of False Positive ($).
        fn_cost: Cost of False Negative ($).
        min_review_recall: Target recall for human review stage.
        min_decline_precision: Target precision for hard decline stage.

    Returns:
        Dictionary containing threshold recommendations and full analysis table.
    """
    threshold_grid = np.linspace(0.01, 0.99, 197)
    cost_model = CostModel(fp_cost=fp_cost, fn_cost=fn_cost)

    records = []
    for t in threshold_grid:
        y_pred = (y_prob >= t).astype(int)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        tn = np.sum((y_true == 0) & (y_pred == 0))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        cost_dict = cost_model.calculate_cost(y_true, y_pred)

        records.append({
            "threshold": round(float(t), 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
            "fpr": round(float(fpr), 5),
            "fnr": round(float(fnr), 5),
            "total_cost": cost_dict["total_loss"],
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn)
        })

    df_eval = pd.DataFrame(records)

    # 1. Cost Optimal
    cost_opt_row = df_eval.loc[df_eval["total_cost"].idxmin()]
    cost_optimal_thresh = float(cost_opt_row["threshold"])

    # 2. F1 Optimal
    f1_opt_row = df_eval.loc[df_eval["f1"].idxmax()]
    f1_optimal_thresh = float(f1_opt_row["threshold"])

    # 3. Review Threshold (find highest threshold that achieves >= min_review_recall)
    eligible_review = df_eval[df_eval["recall"] >= min_review_recall]
    if not eligible_review.empty:
        review_thresh = float(eligible_review["threshold"].max())
    else:
        review_thresh = 0.20

    # 4. Decline Threshold (find lowest threshold that achieves >= min_decline_precision)
    eligible_decline = df_eval[df_eval["precision"] >= min_decline_precision]
    if not eligible_decline.empty:
        decline_thresh = float(eligible_decline["threshold"].min())
    else:
        decline_thresh = 0.65

    # Ensure review_thresh < decline_thresh
    if review_thresh >= decline_thresh:
        review_thresh = max(0.05, round(decline_thresh * 0.5, 2))

    recommendations = {
        "cost_optimal_threshold": cost_optimal_thresh,
        "cost_optimal_total_loss": float(cost_opt_row["total_cost"]),
        "f1_optimal_threshold": f1_optimal_thresh,
        "f1_optimal_score": float(f1_opt_row["f1"]),
        "recommended_review_threshold": review_thresh,
        "recommended_decline_threshold": decline_thresh
    }

    logger.info(
        f"Threshold Optimization Results:\n"
        f"  Cost-Optimal Threshold: {cost_optimal_thresh:.2f} (Total Loss: ${cost_opt_row['total_cost']:,.2f})\n"
        f"  F1-Optimal Threshold:   {f1_optimal_thresh:.2f} (Max F1: {f1_opt_row['f1']:.4f})\n"
        f"  Operational Tri-State:  Review >= {review_thresh:.2f}, Decline >= {decline_thresh:.2f}"
    )

    return {
        "recommendations": recommendations,
        "threshold_curve": df_eval
    }
