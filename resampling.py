"""Utilities for resampling training folds to target class ratios."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd


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
    n_pos = max(0, min(n_total, n_pos))
    n_neg = n_total - n_pos

    rng = np.random.default_rng(random_state)

    pos_idx = y_series.index[y_series == 1].to_numpy()
    neg_idx = y_series.index[y_series == 0].to_numpy()

    if len(pos_idx) == 0 or len(neg_idx) == 0:
        raise ValueError("Both classes must be present to resample")

    pos_sample = rng.choice(pos_idx, size=n_pos, replace=n_pos > len(pos_idx))
    neg_sample = rng.choice(neg_idx, size=n_neg, replace=n_neg > len(neg_idx))

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
