from __future__ import annotations

import numpy as np
import pandas as pd

from metrics_utils import parse_metrics_config, score_metrics
from protocol_checks import (
    check_metric_sanity,
    check_no_leakage,
    check_resampling_determinism,
)


def _toy_data(n: int = 40) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(123)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = pd.Series((X["a"] > X["b"]).astype(int))
    return X, y


def test_check_no_leakage_reports_ok() -> None:
    X, y = _toy_data(40)
    report = check_no_leakage(X, y, outer_folds=4, outer_repeats=2, seed=7)

    assert report.ok
    assert report.total_folds == 8


def test_check_resampling_determinism_true() -> None:
    X, y = _toy_data(30)
    assert check_resampling_determinism(
        X,
        y,
        target_positive_ratio=0.3,
        seed=11,
    )


def test_check_metric_sanity_allows_nan() -> None:
    y_true = np.array([1, 1, 1, 1])
    y_score = np.array([0.8, 0.7, 0.9, 0.6])
    cfg = parse_metrics_config({"primary": "roc_auc", "additional": ["log_loss"]})
    scores = score_metrics(cfg, y_true, y_score)

    assert check_metric_sanity(scores)
