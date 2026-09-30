"""
Tree-based model implementations for fraud detection:
- Random Forest Classifier (bagging, robust to outliers)
- LightGBM / XGBoost (gradient boosting, state-of-the-art tabular performance)
"""

import logging
from typing import Optional, Union, Dict, Any
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb

logger = logging.getLogger(__name__)


def build_random_forest_model(
    n_estimators: int = 150,
    max_depth: int = 12,
    min_samples_split: int = 5,
    min_samples_leaf: int = 2,
    class_weight: str = "balanced_subsample",
    random_state: int = 42,
    n_jobs: int = -1
) -> RandomForestClassifier:
    """
    Construct Random Forest Classifier with balanced subsample weighting.
    """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=n_jobs
    )


def build_lightgbm_model(
    n_estimators: int = 150,
    max_depth: int = 6,
    learning_rate: float = 0.05,
    num_leaves: int = 31,
    scale_pos_weight: float = 100.0,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    random_state: int = 42,
    n_jobs: int = -1
) -> lgb.LGBMClassifier:
    """
    Construct LightGBM Classifier optimized for fast tabular gradient boosting
    under extreme class imbalance.
    """
    return lgb.LGBMClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        scale_pos_weight=scale_pos_weight,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=-1
    )


def build_xgboost_model(
    n_estimators: int = 150,
    max_depth: int = 5,
    learning_rate: float = 0.05,
    scale_pos_weight: float = 100.0,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    random_state: int = 42,
    n_jobs: int = -1
) -> xgb.XGBClassifier:
    """
    Construct XGBoost Classifier with scale_pos_weight for imbalanced classification.
    """
    return xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        scale_pos_weight=scale_pos_weight,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=random_state,
        n_jobs=n_jobs,
        eval_metric="logloss"
    )
