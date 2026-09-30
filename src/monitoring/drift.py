"""
Data and feature drift detection module.

Implements:
- Population Stability Index (PSI): Industry standard quantitative drift measure
- Two-Sample Kolmogorov-Smirnov (KS) Test: Non-parametric distribution test
"""

import logging
from typing import Dict, List, Any, Union, Optional
import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def calculate_psi(
    reference: np.ndarray,
    current: np.ndarray,
    num_bins: int = 10,
    epsilon: float = 1e-4
) -> float:
    """
    Compute Population Stability Index (PSI) between reference baseline and current production data.

    Interpretation:
    - PSI < 0.10: Stable / No significant drift
    - 0.10 <= PSI < 0.25: Moderate drift (warning, monitor closely)
    - PSI >= 0.25: Significant drift (alert, retrain model)
    """
    # Clean non-finite values
    ref = reference[np.isfinite(reference)]
    cur = current[np.isfinite(current)]

    if len(ref) == 0 or len(cur) == 0:
        return 0.0

    # Determine quantile bin edges based on reference distribution
    quantiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(ref, quantiles)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    # Ensure unique edges
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) <= 2:
        return 0.0

    # Count frequencies
    ref_counts, _ = np.histogram(ref, bins=bin_edges)
    cur_counts, _ = np.histogram(cur, bins=bin_edges)

    # Convert to proportions with smoothing epsilon
    ref_props = (ref_counts / len(ref)) + epsilon
    cur_props = (cur_counts / len(cur)) + epsilon

    # Re-normalize
    ref_props /= np.sum(ref_props)
    cur_props /= np.sum(cur_props)

    # PSI formula: sum((actual% - expected%) * ln(actual% / expected%))
    psi_value = np.sum((cur_props - ref_props) * np.log(cur_props / ref_props))
    return float(max(0.0, psi_value))


def calculate_ks_test(reference: np.ndarray, current: np.ndarray) -> Dict[str, float]:
    """
    Compute two-sample Kolmogorov-Smirnov test to detect distribution shifts.
    """
    ref = reference[np.isfinite(reference)]
    cur = current[np.isfinite(current)]

    if len(ref) == 0 or len(cur) == 0:
        return {"statistic": 0.0, "p_value": 1.0}

    stat, p_val = stats.ks_2samp(ref, cur)
    return {
        "statistic": round(float(stat), 4),
        "p_value": round(float(p_val), 6)
    }


class DriftMonitor:
    """
    Monitors feature distributions between reference training set and production stream.
    """

    def __init__(self, reference_df: pd.DataFrame, features_to_track: Optional[List[str]] = None):
        self.features_to_track = features_to_track or ["Amount", "V14", "V10", "V4", "V12"]
        self.reference_data = {
            f: reference_df[f].dropna().values
            for f in self.features_to_track if f in reference_df.columns
        }

    def evaluate_drift(self, current_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluate drift metrics across tracked features.
        """
        drift_report = {}
        for feat in self.features_to_track:
            if feat not in self.reference_data or feat not in current_df.columns:
                continue

            ref_vals = self.reference_data[feat]
            cur_vals = current_df[feat].dropna().values

            psi = calculate_psi(ref_vals, cur_vals)
            ks = calculate_ks_test(ref_vals, cur_vals)

            if psi >= 0.25:
                status = "ALERT_SIGNIFICANT_DRIFT"
            elif psi >= 0.10:
                status = "WARNING_MODERATE_DRIFT"
            else:
                status = "STABLE"

            drift_report[feat] = {
                "psi": round(psi, 4),
                "ks_statistic": ks["statistic"],
                "ks_p_value": ks["p_value"],
                "status": status
            }

        return drift_report
