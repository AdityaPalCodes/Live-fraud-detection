"""
Cost-sensitive decision model for financial fraud detection.

Business Cost Formulation:
Total Financial Loss = (False Positives * FP_Cost) + (False Negatives * FN_Cost)

Where:
- FP_Cost represents customer friction, manual review analyst time, or customer churn.
- FN_Cost represents direct unrecovered chargeback loss and merchant fraud penalties.
"""

import logging
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

logger = logging.getLogger(__name__)


class CostModel:
    """
    Configurable asymmetric cost optimization model.
    """

    def __init__(self, fp_cost: float = 25.0, fn_cost: float = 250.0):
        self.fp_cost = fp_cost
        self.fn_cost = fn_cost

    def calculate_cost(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Compute financial cost given binary true and predicted values."""
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        fp_loss = fp * self.fp_cost
        fn_loss = fn * self.fn_cost
        total_loss = fp_loss + fn_loss
        cost_per_tx = total_loss / len(y_true) if len(y_true) > 0 else 0.0

        return {
            "fp_count": int(fp),
            "fn_count": int(fn),
            "tp_count": int(tp),
            "tn_count": int(tn),
            "fp_loss": round(float(fp_loss), 2),
            "fn_loss": round(float(fn_loss), 2),
            "total_loss": round(float(total_loss), 2),
            "cost_per_transaction": round(float(cost_per_tx), 4)
        }

    def evaluate_cost_curve(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        thresholds: np.ndarray = np.linspace(0.01, 0.99, 99)
    ) -> pd.DataFrame:
        """
        Evaluate business loss across a range of potential classification thresholds.
        """
        records = []
        for t in thresholds:
            y_pred = (y_prob >= t).astype(int)
            res = self.calculate_cost(y_true, y_pred)
            res["threshold"] = round(float(t), 4)
            records.append(res)

        df_curve = pd.DataFrame(records)
        optimal_row = df_curve.loc[df_curve["total_loss"].idxmin()]
        logger.info(
            f"Cost Curve Evaluation: Minimum cost ${optimal_row['total_loss']:,.2f} "
            f"achieved at optimal threshold = {optimal_row['threshold']:.2f}"
        )
        return df_curve
