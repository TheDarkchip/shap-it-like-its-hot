"""Inner CV hyperparameter search utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid, StratifiedKFold

from metrics_utils import MetricsConfig, MetricsConfigError, score_metrics
from xgboost_wrapper import predict_proba, train_xgb_classifier


@dataclass(frozen=True)
class HPOResult:
    best_params: dict[str, Any]
    best_score: float
    model: object


def _metric_greater_is_better(metric_name: str) -> bool:
    if metric_name == "log_loss":
        return False
    return True


def _validate_grid(param_grid: dict[str, Iterable[Any]]) -> None:
    if not param_grid:
        raise MetricsConfigError("param_grid must be non-empty")
    if not list(ParameterGrid(param_grid)):
        raise MetricsConfigError("param_grid must yield at least one candidate")


def _evaluate_params(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    params: dict[str, Any],
    metric_name: str,
    inner_folds: int,
    seed: int,
) -> float:
    splitter = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)
    scores: list[float] = []
    cfg = MetricsConfig(primary=metric_name, additional=())

    for fold_id, (train_idx, test_idx) in enumerate(splitter.split(X, y)):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]

        model_result = train_xgb_classifier(
            X_train,
            y_train,
            params=params,
            random_state=seed + fold_id,
        )
        proba = predict_proba(model_result.model, X_test)
        metrics = score_metrics(cfg, np.asarray(y_test), proba)
        scores.append(metrics[metric_name])

    return float(np.nanmean(scores))


def select_best_params(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    param_grid: dict[str, Iterable[Any]],
    metric_name: str,
    inner_folds: int,
    seed: int,
    base_params: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], float]:
    _validate_grid(param_grid)

    candidates = list(ParameterGrid(param_grid))
    best_score: float | None = None
    best_params: dict[str, Any] | None = None
    greater_is_better = _metric_greater_is_better(metric_name)

    for candidate in candidates:
        merged = dict(base_params or {})
        merged.update(candidate)
        score = _evaluate_params(
            X,
            y,
            params=merged,
            metric_name=metric_name,
            inner_folds=inner_folds,
            seed=seed,
        )

        if best_score is None:
            best_score = score
            best_params = merged
            continue

        if greater_is_better:
            if score > best_score:
                best_score = score
                best_params = merged
        else:
            if score < best_score:
                best_score = score
                best_params = merged

    if best_score is None or best_params is None:
        raise MetricsConfigError("No valid hyperparameter candidates")

    return best_params, float(best_score)


def tune_and_train(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    param_grid: dict[str, Iterable[Any]],
    metric_name: str,
    inner_folds: int,
    seed: int,
    base_params: dict[str, Any] | None = None,
) -> HPOResult:
    best_params, best_score = select_best_params(
        X,
        y,
        param_grid=param_grid,
        metric_name=metric_name,
        inner_folds=inner_folds,
        seed=seed,
        base_params=base_params,
    )

    model_result = train_xgb_classifier(
        X,
        y,
        params=best_params,
        random_state=seed,
    )

    return HPOResult(
        best_params=best_params,
        best_score=best_score,
        model=model_result.model,
    )
