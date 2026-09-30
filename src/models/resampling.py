"""
Resampling and imbalance handling techniques for training fraud models.

Strict Leakage Rule:
- Resampling is applied EXCLUSIVELY to the training split.
- Validation and test sets remain un-resampled with their natural real-world class distribution.
"""

import logging
from typing import Tuple, Optional
import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

logger = logging.getLogger(__name__)


def apply_random_undersampling(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sampling_strategy: float = 0.1,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Apply Random Under-Sampling to the majority legitimate class on training data.
    E.g. sampling_strategy=0.1 means 1 fraud per 10 legitimate transactions in training.

    Preserves all minority fraud examples while speeding up training and re-balancing gradients.
    """
    rus = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=random_state)
    X_res, y_res = rus.fit_resample(X_train, y_train)
    logger.info(
        f"Random Under-Sampling applied: Train size changed from {len(X_train):,} "
        f"to {len(X_res):,} (Frauds: {sum(y_res):,}, Legit: {len(y_res) - sum(y_res):,})"
    )
    return X_res, y_res


def apply_smote(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sampling_strategy: float = 0.05,
    k_neighbors: int = 5,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Apply Synthetic Minority Over-sampling Technique (SMOTE) strictly on training data.
    Interpolates synthetic fraud transactions along line segments of k-nearest fraud neighbors.
    """
    # Guard if number of fraud cases is smaller than k_neighbors
    n_pos = sum(y_train)
    actual_k = min(k_neighbors, max(1, n_pos - 1))
    
    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        k_neighbors=actual_k,
        random_state=random_state
    )
    X_res, y_res = smote.fit_resample(X_train, y_train)
    logger.info(
        f"SMOTE applied: Train size changed from {len(X_train):,} "
        f"to {len(X_res):,} (Frauds: {sum(y_res):,}, Legit: {len(y_res) - sum(y_res):,})"
    )
    return X_res, y_res
