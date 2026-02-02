"""Protocol sanity checks for leakage and determinism."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .nested_cv import iter_outer_folds
from .resampling import resample_train_fold


@dataclass(frozen=True)
class LeakageReport:
    overlaps: int
    total_folds: int

    @property
    def ok(self) -> bool:
        return self.overlaps == 0


def check_no_leakage(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    outer_folds: int,
    outer_repeats: int,
    seed: int,
) -> LeakageReport:
    """Check that train/test indices do not overlap across outer folds."""
    overlaps = 0
    total = 0
    for fold in iter_outer_folds(
        X,
        y,
        outer_folds=outer_folds,
        outer_repeats=outer_repeats,
        seed=seed,
    ):
        total += 1
        train_idx = set(fold.train_idx.tolist())
        test_idx = set(fold.test_idx.tolist())
        if train_idx.intersection(test_idx):
            overlaps += 1
    return LeakageReport(overlaps=overlaps, total_folds=total)


def check_resampling_determinism(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    target_positive_ratio: float,
    seed: int,
) -> bool:
    """Verify resampling returns identical indices for same seed."""
    first = resample_train_fold(
        X,
        y,
        target_positive_ratio=target_positive_ratio,
        random_state=seed,
    )
    second = resample_train_fold(
        X,
        y,
        target_positive_ratio=target_positive_ratio,
        random_state=seed,
    )
    return first.X.equals(second.X) and first.y.equals(second.y)


def check_metric_sanity(scores: dict[str, float]) -> bool:
    """Ensure metric outputs are finite or NaN when allowed."""
    for name, value in scores.items():
        if value is None:
            return False
        if isinstance(value, float) and np.isnan(value):
            continue
        if not np.isfinite(value):
            return False
    return True
