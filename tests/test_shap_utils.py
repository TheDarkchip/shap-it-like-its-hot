from __future__ import annotations

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from shap_utils import compute_tree_shap


def _toy_data(n: int = 40) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(0)
    X = pd.DataFrame({
        "x1": rng.normal(size=n),
        "x2": rng.normal(size=n),
    })
    y = pd.Series((X["x1"] - X["x2"] > 0).astype(int))
    return X, y


def test_compute_tree_shap_returns_global_importance() -> None:
    X, y = _toy_data(80)
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=10,
        max_depth=2,
        learning_rate=0.3,
        base_score=0.5,
        n_jobs=1,
        tree_method="hist",
        random_state=0,
        verbosity=0,
    )
    model.fit(X, y)

    result = compute_tree_shap(model, X)

    assert result.values.shape[0] == len(X)
    assert result.values.shape[1] == X.shape[1]
    assert list(result.global_importance.index) == ["x1", "x2"]
    assert (result.global_importance >= 0).all()
