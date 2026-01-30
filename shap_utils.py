"""Compute SHAP values and global importance summaries."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap
from xgboost import XGBClassifier


@dataclass(frozen=True)
class ShapResult:
    """Container for SHAP values and global importance."""

    values: np.ndarray
    base_values: np.ndarray
    feature_names: list[str]
    global_importance: pd.Series


def compute_tree_shap(
    model: XGBClassifier,
    X: pd.DataFrame,
) -> ShapResult:
    """Compute per-instance SHAP values and global mean(|SHAP|) importance."""
    X_frame = pd.DataFrame(X)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_frame)
    base_values = explainer.expected_value

    if isinstance(shap_values, list):
        if len(shap_values) != 1:
            raise ValueError("Expected binary classification SHAP output")
        shap_values = shap_values[0]

    values = np.asarray(shap_values)
    if values.ndim != 2:
        raise ValueError("SHAP values must be 2D (n_samples, n_features)")

    feature_names = list(X_frame.columns)
    global_importance = pd.Series(
        np.mean(np.abs(values), axis=0),
        index=feature_names,
        name="mean_abs_shap",
    )

    return ShapResult(
        values=values,
        base_values=np.asarray(base_values),
        feature_names=feature_names,
        global_importance=global_importance,
    )
