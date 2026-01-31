"""Nested cross-validation harness skeleton."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


@dataclass(frozen=True)
class OuterFold:
    repeat_id: int
    fold_id: int
    train_idx: np.ndarray
    test_idx: np.ndarray
    seed: int


def iter_outer_folds(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    outer_folds: int,
    outer_repeats: int,
    seed: int,
) -> Iterator[OuterFold]:
    """Yield outer folds for nested CV with repeat metadata."""
    for repeat_id in range(outer_repeats):
        repeat_seed = seed + repeat_id
        splitter = StratifiedKFold(
            n_splits=outer_folds,
            shuffle=True,
            random_state=repeat_seed,
        )
        for fold_id, (train_idx, test_idx) in enumerate(splitter.split(X, y)):
            yield OuterFold(
                repeat_id=repeat_id,
                fold_id=fold_id,
                train_idx=train_idx,
                test_idx=test_idx,
                seed=repeat_seed,
            )
