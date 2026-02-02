"""Permutation feature importance utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import log_loss

from ..metrics.metrics_utils import SUPPORTED_METRICS, MetricsConfigError


@dataclass(frozen=True)
class PFIResult:
    mean: pd.Series
    std: pd.Series


def _predict_scores(estimator: object, X: pd.DataFrame) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(X)
        return np.asarray(proba)[:, 1]
    raise MetricsConfigError("Estimator must support predict_proba for PFI scoring")


def _scorer(metric_name: str) -> Callable[[object, pd.DataFrame, np.ndarray], float]:
    if metric_name not in SUPPORTED_METRICS:
        raise MetricsConfigError(f"Unsupported metric: {metric_name}")

    if metric_name == "log_loss":
        def score(estimator: object, X: pd.DataFrame, y: np.ndarray) -> float:
            y_score = _predict_scores(estimator, X)
            return -float(log_loss(y, y_score, labels=[0, 1]))

        return score

    metric = SUPPORTED_METRICS[metric_name]

    def score(estimator: object, X: pd.DataFrame, y: np.ndarray) -> float:
        y_score = _predict_scores(estimator, X)
        return metric.scorer(y, y_score)

    return score


def compute_pfi_importance(
    estimator: object,
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    *,
    metric_name: str,
    n_repeats: int = 5,
    random_state: int | None = None,
    n_jobs: int | None = None,
) -> PFIResult:
    """Compute permutation feature importances.

    Returns a series indexed by feature name with mean importances.
    """
    y_array = np.asarray(y)
    feature_names = list(pd.DataFrame(X).columns)
    if np.unique(y_array).size < 2:
        nan_series = pd.Series(
            np.full(X.shape[1], np.nan, dtype=float),
            index=feature_names,
            name="pfi_importance",
        )
        return PFIResult(mean=nan_series, std=nan_series.rename("pfi_importance_std"))

    scorer = _scorer(metric_name)
    result = permutation_importance(
        estimator,
        X,
        y_array,
        scoring=scorer,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=n_jobs,
    )

    mean = pd.Series(result.importances_mean, index=feature_names, name="pfi_importance")
    std = pd.Series(result.importances_std, index=feature_names, name="pfi_importance_std")
    return PFIResult(mean=mean, std=std)
