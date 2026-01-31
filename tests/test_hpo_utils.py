from __future__ import annotations

import numpy as np
import pandas as pd

from hpo_utils import select_best_params, tune_and_train
from nested_cv import run_inner_hpo_for_outer_folds


def _toy_data(n: int = 60) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(123)
    X = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
        }
    )
    y = pd.Series((X["x1"] > X["x2"]).astype(int))
    return X, y


def test_select_best_params_single_candidate() -> None:
    X, y = _toy_data()
    best_params, best_score = select_best_params(
        X,
        y,
        param_grid={"learning_rate": [0.1]},
        metric_name="roc_auc",
        inner_folds=3,
        seed=7,
        base_params={"n_estimators": 5, "max_depth": 2},
    )

    assert best_params["learning_rate"] == 0.1
    assert best_params["n_estimators"] == 5
    assert best_params["max_depth"] == 2
    assert 0.0 <= best_score <= 1.0


def test_tune_and_train_returns_model() -> None:
    X, y = _toy_data()
    result = tune_and_train(
        X,
        y,
        param_grid={"learning_rate": [0.1, 0.2]},
        metric_name="roc_auc",
        inner_folds=3,
        seed=5,
        base_params={"n_estimators": 5, "max_depth": 2},
    )

    proba = result.model.predict_proba(X)
    assert proba.shape[0] == len(X)
    assert "learning_rate" in result.best_params


def test_run_inner_hpo_for_outer_folds_iterates() -> None:
    X, y = _toy_data(40)
    results = list(
        run_inner_hpo_for_outer_folds(
            X,
            y,
            outer_folds=2,
            outer_repeats=1,
            inner_folds=2,
            seed=3,
            metric_name="roc_auc",
            param_grid={"learning_rate": [0.1]},
            base_params={"n_estimators": 5, "max_depth": 2},
        )
    )

    assert len(results) == 2
    for result in results:
        assert result.best_params["learning_rate"] == 0.1
        assert hasattr(result.model, "predict_proba")
