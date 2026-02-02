"""Nested cross-validation harness skeleton."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .modeling.hpo_utils import HPOResult, tune_and_train


@dataclass(frozen=True)
class OuterFold:
    repeat_id: int
    fold_id: int
    train_idx: np.ndarray
    test_idx: np.ndarray
    seed: int


@dataclass(frozen=True)
class OuterFoldResult:
    repeat_id: int
    fold_id: int
    seed: int
    best_params: dict[str, Any]
    best_score: float
    model: object


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


def run_inner_hpo_for_outer_folds(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    outer_folds: int,
    outer_repeats: int,
    inner_folds: int,
    seed: int,
    metric_name: str,
    param_grid: dict[str, Iterable[Any]],
    base_params: dict[str, Any] | None = None,
) -> Iterator[OuterFoldResult]:
    """Run inner CV HPO for each outer fold and refit on the full train split."""
    for outer in iter_outer_folds(
        X,
        y,
        outer_folds=outer_folds,
        outer_repeats=outer_repeats,
        seed=seed,
    ):
        X_train = X.iloc[outer.train_idx]
        y_train = y.iloc[outer.train_idx]
        hpo_result: HPOResult = tune_and_train(
            X_train,
            y_train,
            param_grid=param_grid,
            metric_name=metric_name,
            inner_folds=inner_folds,
            seed=outer.seed,
            base_params=base_params,
        )

        yield OuterFoldResult(
            repeat_id=outer.repeat_id,
            fold_id=outer.fold_id,
            seed=outer.seed,
            best_params=hpo_result.best_params,
            best_score=hpo_result.best_score,
            model=hpo_result.model,
        )
