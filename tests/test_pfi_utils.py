from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from shap_stability.explain.pfi_utils import compute_pfi_importance


def _toy_data() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(42)
    n_samples = 200
    x_signal = rng.normal(size=n_samples)
    x_noise = rng.normal(size=n_samples)
    logits = 2.0 * x_signal + 0.1 * rng.normal(size=n_samples)
    probs = 1 / (1 + np.exp(-logits))
    y = (probs > 0.5).astype(int)
    X = pd.DataFrame({"signal": x_signal, "noise": x_noise})
    return X, pd.Series(y)


def test_compute_pfi_importance_prefers_signal() -> None:
    X, y = _toy_data()
    model = LogisticRegression(max_iter=200)
    model.fit(X, y)

    result = compute_pfi_importance(
        model,
        X,
        y,
        metric_name="roc_auc",
        n_repeats=5,
        random_state=0,
    )

    assert set(result.mean.index) == {"signal", "noise"}
    assert set(result.std.index) == {"signal", "noise"}
    assert result.mean["signal"] > result.mean["noise"]


def test_compute_pfi_importance_supports_log_loss() -> None:
    X, y = _toy_data()
    model = LogisticRegression(max_iter=200)
    model.fit(X, y)

    result = compute_pfi_importance(
        model,
        X,
        y,
        metric_name="log_loss",
        n_repeats=3,
        random_state=0,
    )

    assert set(result.mean.index) == {"signal", "noise"}
    assert result.mean.notna().all()


def test_compute_pfi_importance_handles_single_class() -> None:
    X = pd.DataFrame({"a": [0.1, 0.2, 0.3], "b": [0.4, 0.5, 0.6]})
    y = pd.Series([1, 1, 1])
    model = DummyClassifier(strategy="most_frequent")
    model.fit(X, y)

    result = compute_pfi_importance(
        model,
        X,
        y,
        metric_name="roc_auc",
        n_repeats=3,
        random_state=0,
    )

    assert np.isnan(result.mean).all()
    assert np.isnan(result.std).all()


def test_compute_pfi_importance_rejects_unknown_metric() -> None:
    X, y = _toy_data()
    model = LogisticRegression(max_iter=200)
    model.fit(X, y)

    with pytest.raises(ValueError, match="Unsupported metric"):
        compute_pfi_importance(model, X, y, metric_name="unknown")
