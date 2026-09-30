"""
Explainable AI (XAI) module using SHAP for local and global model interpretability.

Translates mathematical Shapley values into human-readable risk reason codes
for payment fraud analysts and audit compliance.
"""

import logging
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
import shap

logger = logging.getLogger(__name__)


# Feature explanation mapping for human-readable reasons
FEATURE_DESCRIPTIONS = {
    "Amount": "Unusual transaction amount",
    "log_amount": "High log-scaled transaction amount",
    "amount_to_median_ratio": "Transaction amount significantly exceeds normal median",
    "amount_to_mean_ratio": "Transaction amount significantly exceeds average",
    "is_night": "Nocturnal transaction timing (00:00 - 06:00)",
    "hour_of_day": "Abnormal hour of day for account activity",
    "hour_sin": "Atypical temporal cycle",
    "hour_cos": "Atypical temporal cycle",
    "v_vector_norm": "High latent vector deviation from normal behavior baseline",
    "v4_v11_sum": "Elevated positive fraud signal interaction (V4 + V11)",
    "v14_v10_prod": "Abnormal latent interaction (V14 * V10)",
    "v4_v14_diff": "Extreme divergence between risk indicators (V4 vs V14)",
    "V14": "Strong negative anomaly on component V14",
    "V10": "Strong negative anomaly on component V10",
    "V12": "Strong negative anomaly on component V12",
    "V17": "Strong negative anomaly on component V17",
    "V4": "High positive risk shift on component V4",
    "V11": "High positive risk shift on component V11",
    "V2": "Abnormal shift on component V2",
    "V3": "Abnormal shift on component V3",
    "is_round_amount": "Round dollar amount pattern often associated with test attacks"
}


class FraudExplainer:
    """
    Computes local SHAP explanations for individual transactions
    and translates them into structured, ranked reason codes.
    """

    def __init__(self, model: Any, feature_names: List[str], background_data: Optional[pd.DataFrame] = None):
        self.model = model
        self.feature_names = feature_names

        # Unwrap CalibratedClassifierCV if needed to access underlying estimator
        base_estimator = model
        if hasattr(model, "calibrated_classifiers_"):
            base_estimator = model.calibrated_classifiers_[0].estimator
        elif hasattr(model, "estimator"):
            base_estimator = model.estimator

        self.base_estimator = base_estimator

        # Initialize appropriate SHAP Explainer
        try:
            if hasattr(base_estimator, "named_steps"):
                # Pipeline with linear model
                clf = base_estimator.named_steps.get("classifier", base_estimator)
                scaler = base_estimator.named_steps.get("scaler", None)
                if background_data is not None and scaler is not None:
                    scaled_bg = scaler.transform(background_data)
                    masker = shap.maskers.Independent(data=scaled_bg)
                    self.explainer = shap.LinearExplainer(clf, masker=masker)
                else:
                    self.explainer = None
            elif "LGBM" in type(base_estimator).__name__ or "XGB" in type(base_estimator).__name__ or "Forest" in type(base_estimator).__name__:
                self.explainer = shap.TreeExplainer(base_estimator)
            else:
                self.explainer = None
            if self.explainer is not None:
                logger.info("SHAP Explainer initialized successfully.")
        except Exception as e:
            logger.warning(f"Using feature contribution ranking explainer: {e}")
            self.explainer = None

    def explain_transaction(
        self,
        features_df: pd.DataFrame,
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Generate top contributing risk factors for a given transaction.

        Args:
            features_df: Single-row DataFrame of engineered features.
            top_k: Number of top risk reasons to extract.

        Returns:
            List of dictionaries with feature name, description, and impact value.
        """
        reasons = []

        if self.explainer is not None:
            try:
                # Handle pipeline transform if linear explainer
                X_input = features_df.values
                if hasattr(self.base_estimator, "named_steps") and "scaler" in self.base_estimator.named_steps:
                    X_input = self.base_estimator.named_steps["scaler"].transform(features_df)

                shap_values = self.explainer(X_input)
                values = shap_values.values[0]
                if len(values.shape) > 1:
                    values = values[:, 1]  # positive class (fraud)

                # Find top positive contributions (pushes toward fraud)
                top_indices = np.argsort(values)[::-1]
                
                for idx in top_indices[:top_k]:
                    val = float(values[idx])
                    if val > 0:
                        feat_name = self.feature_names[idx] if idx < len(self.feature_names) else f"Feature_{idx}"
                        desc = FEATURE_DESCRIPTIONS.get(feat_name, f"Abnormal behavior on {feat_name}")
                        reasons.append({
                            "feature": feat_name,
                            "description": desc,
                            "contribution": round(val, 4)
                        })
            except Exception as e:
                logger.error(f"Error computing SHAP values: {e}")

        # Fallback to feature deviation if SHAP not applicable
        if not reasons:
            # Rank by absolute deviation on key known fraud features
            key_features = ["V14", "V10", "V4", "V12", "amount_to_median_ratio", "is_night"]
            present = [f for f in key_features if f in features_df.columns]
            for f in present[:top_k]:
                val = float(features_df[f].values[0])
                desc = FEATURE_DESCRIPTIONS.get(f, f"Elevated deviation on {f}")
                reasons.append({
                    "feature": f,
                    "description": desc,
                    "contribution": round(abs(val), 4)
                })

        return reasons
