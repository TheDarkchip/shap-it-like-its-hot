"""Utilities for resampling training folds to target class ratios."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.utils import resample


@dataclass(frozen=True)
class ResampleResult:
    """Container for resampled data and class counts."""

    X: pd.DataFrame
    y: pd.Series
    positive_count: int
    negative_count: int


def _validate_ratio(ratio: float) -> None:
    if not 0.0 < ratio < 1.0:
        raise ValueError("target_positive_ratio must be between 0 and 1 (exclusive)")


def _validate_target(y: pd.Series) -> None:
    unique = set(y.unique())
    if not unique.issubset({0, 1}):
        raise ValueError("y must be binary with values 0/1")


def resample_train_fold(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    target_positive_ratio: float,
    random_state: int | None = None,
) -> ResampleResult:
    """Resample the training fold to a target positive ratio.

    The total number of samples is preserved. Sampling is performed with
    replacement when a class needs to be oversampled.
    """
    _validate_ratio(target_positive_ratio)

    y_series = pd.Series(y).reset_index(drop=True)
    X_frame = pd.DataFrame(X).reset_index(drop=True)

    if len(X_frame) != len(y_series):
        raise ValueError("X and y must have the same length")

    _validate_target(y_series)

    n_total = len(y_series)
    n_pos = int(round(n_total * target_positive_ratio))
    n_pos = max(1, min(n_total - 1, n_pos))
    n_neg = n_total - n_pos

    pos_idx = y_series.index[y_series == 1].to_numpy()
    neg_idx = y_series.index[y_series == 0].to_numpy()

    if len(pos_idx) == 0 or len(neg_idx) == 0:
        raise ValueError("Both classes must be present to resample")

    pos_sample = resample(
        pos_idx,
        replace=n_pos > len(pos_idx),
        n_samples=n_pos,
        random_state=random_state,
    )
    neg_sample = resample(
        neg_idx,
        replace=n_neg > len(neg_idx),
        n_samples=n_neg,
        random_state=None if random_state is None else random_state + 1,
    )

    rng = np.random.default_rng(random_state)
    combined = np.concatenate([pos_sample, neg_sample])
    rng.shuffle(combined)

    X_resampled = X_frame.iloc[combined].reset_index(drop=True)
    y_resampled = y_series.iloc[combined].reset_index(drop=True)

    return ResampleResult(
        X=X_resampled,
        y=y_resampled,
        positive_count=n_pos,
        negative_count=n_neg,
    )
