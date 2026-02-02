from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from shap_stability.single_run import run_single_experiment


def _config() -> dict:
    return {
        "experiment": {"name": "unit", "random_seed": 7},
        "cv": {"outer_folds": 2, "outer_repeats": 1, "inner_folds": 2},
        "resampling": {"target_positive_ratios": [0.5]},
        "model": {"name": "xgboost", "params": {"n_estimators": 10, "max_depth": 2}},
        "metrics": {"primary": "roc_auc"},
    }


def _toy_data() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(123)
    n_samples = 40
    X = pd.DataFrame(
        {
            "feature_a": rng.normal(size=n_samples),
            "feature_b": rng.normal(size=n_samples),
        }
    )
    y = pd.Series(rng.integers(0, 2, size=n_samples))
    return X, y


def test_run_single_experiment_writes_artifacts(tmp_path: Path) -> None:
    artifacts = run_single_experiment(
        _config(),
        output_dir=tmp_path,
        data=_toy_data(),
    )

    assert artifacts.results_path.exists()
    assert artifacts.metadata_path.exists()
    assert artifacts.log_path.exists()

    results = pd.read_csv(artifacts.results_path)
    assert "metric_accuracy" in results.columns
    assert "metric_roc_auc" in results.columns
    assert "shap_feature_a" in results.columns
    assert "shap_feature_b" in results.columns
    assert "pfi_feature_a" in results.columns
    assert "pfi_feature_b" in results.columns
    assert "pfi_std_feature_a" in results.columns
    assert "pfi_std_feature_b" in results.columns
    assert len(results) == 2  # 1 repeat x 2 folds x 1 ratio


def test_run_single_experiment_requires_ratios(tmp_path: Path) -> None:
    cfg = _config()
    cfg["resampling"]["target_positive_ratios"] = []

    with pytest.raises(ValueError, match="target_positive_ratios must be non-empty"):
        run_single_experiment(cfg, output_dir=tmp_path, data=_toy_data())


def test_run_single_experiment_requires_xgboost(tmp_path: Path) -> None:
    cfg = _config()
    cfg["model"]["name"] = "random-forest"

    with pytest.raises(ValueError, match="model.name='xgboost'"):
        run_single_experiment(cfg, output_dir=tmp_path, data=_toy_data())
