from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from xgboost_wrapper import predict_proba, train_xgb_classifier


def _toy_data(n: int = 50) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(0)
    X = pd.DataFrame({
        "x1": rng.normal(size=n),
        "x2": rng.normal(size=n),
    })
    y = pd.Series((X["x1"] + X["x2"] > 0).astype(int))
    return X, y


def test_train_xgb_classifier_returns_model() -> None:
    X, y = _toy_data(80)
    result = train_xgb_classifier(
        X,
        y,
        params={"n_estimators": 5, "max_depth": 2, "learning_rate": 0.3},
        random_state=0,
    )

    assert result.model is not None
    assert result.feature_names == ["x1", "x2"]


def test_predict_proba_shape() -> None:
    X, y = _toy_data(60)
    result = train_xgb_classifier(
        X,
        y,
        params={"n_estimators": 5, "max_depth": 2, "learning_rate": 0.3},
        random_state=1,
    )

    proba = predict_proba(result.model, X)
    assert proba.shape == (len(X),)
    assert np.all((proba >= 0) & (proba <= 1))


def test_train_rejects_non_binary_target() -> None:
    X, _ = _toy_data(20)
    y = pd.Series([0, 1, 2] * 7)[: len(X)]

    with pytest.raises(ValueError):
        train_xgb_classifier(X, y)
