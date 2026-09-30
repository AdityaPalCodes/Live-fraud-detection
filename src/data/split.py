"""
Data splitting module designed to prevent data leakage in fraud detection.

Strict methodology:
- Raw Data -> Split -> Preprocessing fitted ONLY on train -> Model training -> Val evaluation -> Test evaluation.
- Supports both Time-Based (chronological Out-Of-Time) splitting and Stratified splitting.
"""

import logging
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def time_based_split(
    df: pd.DataFrame,
    time_column: str = "Time",
    val_size: float = 0.15,
    test_size: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split transaction dataset chronologically to reflect real-world production deployment:
    - Train on past transactions
    - Validate on intermediate transactions (for threshold tuning and model selection)
    - Test on strictly future transactions (unseen out-of-time evaluation)

    Prevents look-ahead bias and simulates true temporal production conditions.

    Args:
        df: Input DataFrame with transaction records.
        time_column: Name of the timestamp or elapsed seconds column.
        val_size: Fraction of transactions reserved for validation.
        test_size: Fraction of transactions reserved for test.

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    if not (0 < val_size + test_size < 1.0):
        raise ValueError("Sum of val_size and test_size must be strictly between 0 and 1.")

    df_sorted = df.sort_values(time_column).reset_index(drop=True)
    n = len(df_sorted)

    train_end = int(n * (1.0 - val_size - test_size))
    val_end = int(n * (1.0 - test_size))

    train_df = df_sorted.iloc[:train_end].copy()
    val_df = df_sorted.iloc[train_end:val_end].copy()
    test_df = df_sorted.iloc[val_end:].copy()

    logger.info(
        f"Time-based split completed:\n"
        f"  Train: {len(train_df):,} records ({train_df['Class'].sum():,} frauds, "
        f"{train_df['Class'].mean()*100:.3f}%)\n"
        f"  Val:   {len(val_df):,} records ({val_df['Class'].sum():,} frauds, "
        f"{val_df['Class'].mean()*100:.3f}%)\n"
        f"  Test:  {len(test_df):,} records ({test_df['Class'].sum():,} frauds, "
        f"{test_df['Class'].mean()*100:.3f}%)"
    )

    return train_df, val_df, test_df


def stratified_split(
    df: pd.DataFrame,
    target_column: str = "Class",
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split transaction dataset using stratified sampling to preserve the exact
    minority fraud class ratio across Train, Validation, and Test sets.

    Args:
        df: Input DataFrame.
        target_column: Name of the binary target label ('Class').
        val_size: Proportion for validation set.
        test_size: Proportion for test set.
        random_state: Reproducibility seed.

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    total_holdout = val_size + test_size
    train_df, holdout_df = train_test_split(
        df,
        test_size=total_holdout,
        stratify=df[target_column],
        random_state=random_state
    )

    test_relative_ratio = test_size / total_holdout
    val_df, test_df = train_test_split(
        holdout_df,
        test_size=test_relative_ratio,
        stratify=holdout_df[target_column],
        random_state=random_state
    )

    logger.info(
        f"Stratified split completed:\n"
        f"  Train: {len(train_df):,} records ({train_df[target_column].sum():,} frauds)\n"
        f"  Val:   {len(val_df):,} records ({val_df[target_column].sum():,} frauds)\n"
        f"  Test:  {len(test_df):,} records ({test_df[target_column].sum():,} frauds)"
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True)
    )
