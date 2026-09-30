"""
Probability calibration module for fraud classifiers.

Why Calibration Matters in Fraud Detection:
- Raw probabilities from Tree Ensembles (e.g. Random Forest, LightGBM) are frequently uncalibrated:
  leaf averaging compresses probabilities away from 0 and 1, while boosting can create overconfidence.
- If a risk engine expects 0.70 to mean "70% empirical probability of fraud", an uncalibrated 0.70
  might correspond to only 20% empirical fraud rate, causing excessive false declines.
- Calibrated probabilities ensure risk scores are meaningful, interpretable, and mathematically
  admissible for Bayesian cost optimization.
"""

import logging
from typing import Dict, Any, Tuple
import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss

logger = logging.getLogger(__name__)


def calibrate_classifier(
    base_model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    method: str = "isotonic",
    cv: int = 3
) -> CalibratedClassifierCV:
    """
    Train and calibrate a classifier using either Isotonic regression or Platt scaling (Sigmoid).

    Args:
        base_model: Uncalibrated Scikit-Learn / LightGBM / XGBoost classifier.
        X_train: Training feature matrix.
        y_train: Training binary labels.
        method: 'isotonic' (non-parametric, best for large datasets) or
                'sigmoid' (Platt scaling, parametric logistic fit).
        cv: Cross-validation folds for calibration to prevent overfitting.

    Returns:
        Fitted CalibratedClassifierCV model.
    """
    logger.info(f"Calibrating model using method='{method}' with {cv}-fold CV...")
    calibrated_model = CalibratedClassifierCV(
        estimator=base_model,
        method=method,
        cv=cv
    )
    calibrated_model.fit(X_train, y_train)
    return calibrated_model


def compare_calibration(
    y_true: np.ndarray,
    uncalibrated_probs: np.ndarray,
    calibrated_probs: np.ndarray,
    n_bins: int = 10
) -> Dict[str, Any]:
    """
    Compare probability calibration metrics before and after calibration.
    Computes Brier Score (mean squared error of probability predictions)
    and reliability curve data points.

    Returns:
        Dictionary containing Brier scores and calibration curve coordinates.
    """
    brier_uncalibrated = brier_score_loss(y_true, uncalibrated_probs)
    brier_calibrated = brier_score_loss(y_true, calibrated_probs)

    frac_pos_uncal, mean_prob_uncal = calibration_curve(y_true, uncalibrated_probs, n_bins=n_bins)
    frac_pos_cal, mean_prob_cal = calibration_curve(y_true, calibrated_probs, n_bins=n_bins)

    improvement_pct = ((brier_uncalibrated - brier_calibrated) / brier_uncalibrated) * 100.0

    logger.info(
        f"Calibration Results:\n"
        f"  Uncalibrated Brier Score: {brier_uncalibrated:.6f}\n"
        f"  Calibrated Brier Score:   {brier_calibrated:.6f} "
        f"({improvement_pct:+.2f}% improvement)"
    )

    return {
        "brier_uncalibrated": round(float(brier_uncalibrated), 6),
        "brier_calibrated": round(float(brier_calibrated), 6),
        "brier_improvement_pct": round(float(improvement_pct), 2),
        "uncalibrated_curve": {
            "fraction_of_positives": [round(float(x), 4) for x in frac_pos_uncal],
            "mean_predicted_probability": [round(float(x), 4) for x in mean_prob_uncal]
        },
        "calibrated_curve": {
            "fraction_of_positives": [round(float(x), 4) for x in frac_pos_cal],
            "mean_predicted_probability": [round(float(x), 4) for x in mean_prob_cal]
        }
    }
