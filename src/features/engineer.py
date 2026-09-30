"""
Feature engineering module for real-time and batch fraud detection.

Strictly follows leakage prevention:
- Transformers learn statistics (mean, std, median, percentiles) ONLY on training data.
- Evaluated on validation and test data without re-fitting.
- Optimized for sub-millisecond single-transaction real-time inference.
"""

import logging
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler

logger = logging.getLogger(__name__)


class TransactionFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Leakage-free Feature Engineer for transaction fraud detection.

    Extracts:
    1. Log transaction amount: log1p(Amount)
    2. Zero-amount and micro-amount flags (testing authorizations)
    3. Round dollar amount indicators (e.g. $50, $100 multiples)
    4. Relative amount ratios (amount vs training mean/median)
    5. Cyclical time transformations (sin/cos encoding of hour of day)
    6. Nocturnal / off-peak risk flag (00:00 - 06:00)
    7. PCA vector Euclidean norm (deviation from latent origin)
    8. High-leverage risk component interactions (e.g. V14 * V10, V4 + V11)
    """

    def __init__(
        self,
        time_column: str = "Time",
        amount_column: str = "Amount",
        include_latent_interactions: bool = True
    ):
        self.time_column = time_column
        self.amount_column = amount_column
        self.include_latent_interactions = include_latent_interactions

        # Training set parameters (learned during fit)
        self.amount_scaler = RobustScaler()
        self.train_amount_mean: float = 0.0
        self.train_amount_median: float = 0.0
        self.is_fitted: bool = False
        self.feature_names_: List[str] = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "TransactionFeatureEngineer":
        """
        Learn statistical parameters from the training set ONLY.
        """
        if self.amount_column not in X.columns:
            raise ValueError(f"Required amount column '{self.amount_column}' not found in DataFrame.")

        # Learn amount statistics strictly from training data
        self.train_amount_mean = float(X[self.amount_column].mean())
        self.train_amount_median = float(X[self.amount_column].median())
        self.amount_scaler.fit(X[[self.amount_column]])

        # Determine feature names by running a dummy transform
        sample_df = self._transform_core(X.head(5).copy())
        self.feature_names_ = [c for c in sample_df.columns if c != "Class"]

        self.is_fitted = True
        logger.info(
            f"FeatureEngineer fitted on {len(X):,} records. "
            f"Amount mean: ${self.train_amount_mean:.2f}, median: ${self.train_amount_median:.2f}. "
            f"Total engineered features: {len(self.feature_names_)}"
        )
        return self

    def _transform_core(self, df: pd.DataFrame) -> pd.DataFrame:
        """Core transformation applied to both batch and single records."""
        X = df.copy()

        # 1. Amount transformations
        amount = X[self.amount_column].values
        X["log_amount"] = np.log1p(np.maximum(0, amount))
        X["is_zero_amount"] = (amount == 0.0).astype(int)
        X["is_micro_amount"] = ((amount > 0.0) & (amount < 2.0)).astype(int)
        X["is_round_amount"] = ((amount > 0) & (amount % 10.0 == 0.0)).astype(int)
        
        # Relative ratio vs training baseline
        denom_median = max(self.train_amount_median, 1e-4)
        denom_mean = max(self.train_amount_mean, 1e-4)
        X["amount_to_median_ratio"] = amount / denom_median
        X["amount_to_mean_ratio"] = amount / denom_mean

        # Scaled amount (using robust scaler fitted only on training data)
        if self.is_fitted:
            X["amount_robust_scaled"] = self.amount_scaler.transform(X[[self.amount_column]])[:, 0]
        else:
            X["amount_robust_scaled"] = 0.0

        # 2. Temporal & Cyclical transformations
        if self.time_column in X.columns:
            time_val = X[self.time_column].values
            hour_of_day = (time_val / 3600.0) % 24.0
            X["hour_of_day"] = hour_of_day
            X["hour_sin"] = np.sin(2.0 * np.pi * hour_of_day / 24.0)
            X["hour_cos"] = np.cos(2.0 * np.pi * hour_of_day / 24.0)
            X["is_night"] = ((hour_of_day >= 0.0) & (hour_of_day < 6.0)).astype(int)
        else:
            # Fallback if time not provided
            X["hour_of_day"] = 12.0
            X["hour_sin"] = 0.0
            X["hour_cos"] = -1.0
            X["is_night"] = 0

        # 3. Latent Vector (PCA) Engineering & Interactions
        v_cols = [f"V{i}" for i in range(1, 29) if f"V{i}" in X.columns]
        if v_cols:
            # Euclidean norm of the latent vector
            X["v_vector_norm"] = np.linalg.norm(X[v_cols].values, axis=1)

            if self.include_latent_interactions:
                # Key negative fraud indicators: V14, V10, V12, V17
                if "V14" in X.columns and "V10" in X.columns:
                    X["v14_v10_prod"] = X["V14"] * X["V10"]
                if "V12" in X.columns and "V17" in X.columns:
                    X["v12_v17_prod"] = X["V12"] * X["V17"]
                # Key positive fraud indicators: V4, V11
                if "V4" in X.columns and "V11" in X.columns:
                    X["v4_v11_sum"] = X["V4"] + X["V11"]
                # Ratio of dominant positive to dominant negative signal
                if "V4" in X.columns and "V14" in X.columns:
                    X["v4_v14_diff"] = X["V4"] - X["V14"]

        # Drop original raw Time if desired or keep as reference
        return X

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform batch DataFrame using learned parameters.
        Ensures consistent column ordering and output schema.
        """
        if not self.is_fitted:
            raise RuntimeError("TransactionFeatureEngineer must be fitted before calling transform().")

        transformed = self._transform_core(X)

        # Align with learned feature names
        ordered_cols = [col for col in self.feature_names_ if col in transformed.columns]
        return transformed[ordered_cols]

    def transform_single(self, transaction: Dict[str, Union[float, int]]) -> pd.DataFrame:
        """
        Transform a single transaction dictionary in real-time inference mode.
        Execution completes in microseconds.
        """
        df_single = pd.DataFrame([transaction])
        return self.transform(df_single)
