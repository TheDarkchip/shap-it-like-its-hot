from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid, StratifiedKFold

from ..metrics.metrics_utils import MetricsConfig, MetricsConfigError, score_metrics
from .xgboost_wrapper import predict_proba, train_xgb_classifier
from .hpo_sobol import suggest_configs as sobol_suggest_configs


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
    resample_fn: Callable[[pd.DataFrame, pd.Series, int], tuple[pd.DataFrame, pd.Series]]
    | None = None,
    preprocess_fn: Callable[[pd.DataFrame, pd.DataFrame], tuple[pd.DataFrame, pd.DataFrame]]
    | None = None,
) -> float:
    splitter = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)
    scores: list[float] = []
    cfg = MetricsConfig(primary=metric_name, additional=())

    for fold_id, (train_idx, test_idx) in enumerate(splitter.split(X, y)):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        if resample_fn is not None:
            X_train, y_train = resample_fn(X_train, y_train, seed + fold_id)
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]
        if preprocess_fn is not None:
            X_train, X_test = preprocess_fn(X_train, X_test)

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
    optimizer: str = "grid",
    budget: int | None = None,
    param_space: dict[str, dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], float]:
    """
    Select best hyperparameters via inner CV.

    optimizer:
      - "grid": iterate ParameterGrid(param_grid)
      - "sobol": sample candidates from param_space using Sobol (budget required)
      - "smac": SMAC ask/tell optimization (budget + param_space required)
    """
    optimizer = optimizer.lower().strip()
    greater_is_better = _metric_greater_is_better(metric_name)

    if optimizer == "smac":
        if budget is None:
            raise MetricsConfigError("budget must be provided for optimizer='smac'")
        if not param_space:
            raise MetricsConfigError("param_space must be provided for optimizer='smac'")
        from .hpo_smac import select_best_params_smac 

        return select_best_params_smac(
            X,
            y,
            param_space=param_space,
            metric_name=metric_name,
            inner_folds=inner_folds,
            seed=seed,
            budget=budget,
            base_params=base_params,
        )

    if optimizer == "grid":
        _validate_grid(param_grid)
        candidates: list[dict[str, Any]] = list(ParameterGrid(param_grid))

    elif optimizer == "sobol":
        if budget is None:
            raise MetricsConfigError("budget must be provided for optimizer='sobol'")
        if not param_space:
            raise MetricsConfigError("param_space must be provided for optimizer='sobol'")
        candidates = sobol_suggest_configs(param_space, budget=budget, seed=seed)

    else:
        raise MetricsConfigError(f"Unknown optimizer: {optimizer}")

    best_score: float | None = None
    best_params: dict[str, Any] | None = None

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
            resample_fn=resample_fn,
            preprocess_fn=preprocess_fn,
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
    optimizer: str = "grid",
    budget: int | None = None,
    param_space: dict[str, dict[str, Any]] | None = None,
) -> HPOResult:
    best_params, best_score = select_best_params(
        X,
        y,
        param_grid=param_grid,
        metric_name=metric_name,
        inner_folds=inner_folds,
        seed=seed,
        base_params=base_params,
        optimizer=optimizer,
        budget=budget,
        param_space=param_space,
    )

    if preprocess_fn is not None:
        X_train, _ = preprocess_fn(X, X)
    else:
        X_train = X

    model_result = train_xgb_classifier(
        X_train,
        y,
        params=best_params,
        random_state=seed,
    )

    return HPOResult(
        best_params=best_params,
        best_score=best_score,
        model=model_result.model,
    )