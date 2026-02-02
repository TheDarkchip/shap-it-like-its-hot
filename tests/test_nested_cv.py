from __future__ import annotations

import numpy as np
import pandas as pd

from shap_stability.nested_cv import iter_outer_folds


def test_iter_outer_folds_counts_and_shapes() -> None:
    X = pd.DataFrame({"a": np.arange(12), "b": np.arange(12)})
    y = pd.Series([0, 1] * 6)

    folds = list(iter_outer_folds(X, y, outer_folds=3, outer_repeats=2, seed=10))

    assert len(folds) == 6
    for fold in folds:
        assert fold.train_idx.ndim == 1
        assert fold.test_idx.ndim == 1
        assert set(fold.train_idx).isdisjoint(set(fold.test_idx))
        assert len(fold.train_idx) + len(fold.test_idx) == len(X)


def test_iter_outer_folds_repeat_seeds_change() -> None:
    X = pd.DataFrame({"a": np.arange(10), "b": np.arange(10)})
    y = pd.Series([0, 1] * 5)

    folds = list(iter_outer_folds(X, y, outer_folds=5, outer_repeats=2, seed=5))

    assert {fold.seed for fold in folds} == {5, 6}
    assert {fold.repeat_id for fold in folds} == {0, 1}
