"""Thin XGBoost training wrapper for experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from xgboost import XGBClassifier


@dataclass(frozen=True)
class XGBTrainResult:
    """Container for trained model and metadata."""

    model: XGBClassifier
    feature_names: list[str]


def train_xgb_classifier(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    params: dict[str, Any] | None = None,
    random_state: int | None = None,
) -> XGBTrainResult:
    """Train an XGBoost classifier and return the fitted model.

    Parameters
    ----------
    X:
        Feature matrix as a pandas DataFrame.
    y:
        Binary target series (0/1).
    params:
        Optional XGBoost parameter overrides.
    random_state:
        Seed for model reproducibility.
    """
    if len(X) != len(y):
        raise ValueError("X and y must have the same length")

    y_series = pd.Series(y).reset_index(drop=True)
    if not set(y_series.unique()).issubset({0, 1}):
        raise ValueError("y must be binary with values 0/1")

    X_frame = pd.DataFrame(X).reset_index(drop=True)

    base_params: dict[str, Any] = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "n_estimators": 200,
        "max_depth": 4,
        "learning_rate": 0.1,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "n_jobs": 1,
        "tree_method": "hist",
        "random_state": random_state,
        "verbosity": 0,
    }

    if params:
        base_params.update(params)

    model = XGBClassifier(**base_params)
    model.fit(X_frame, y_series)

    return XGBTrainResult(model=model, feature_names=list(X_frame.columns))


def predict_proba(model: XGBClassifier, X: pd.DataFrame) -> np.ndarray:
    """Predict positive-class probabilities."""
    proba = model.predict_proba(pd.DataFrame(X))
    return proba[:, 1]
