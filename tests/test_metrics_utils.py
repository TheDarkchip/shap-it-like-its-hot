from __future__ import annotations

import numpy as np
import pytest

from metrics_utils import MetricsConfigError, parse_metrics_config, score_metrics


def test_parse_metrics_config_accepts_supported() -> None:
    cfg = parse_metrics_config({"primary": "roc_auc", "additional": ["pr_auc"]})
    assert cfg.primary == "roc_auc"
    assert cfg.additional == ("pr_auc",)
    assert cfg.all_metrics() == ("roc_auc", "pr_auc")


def test_parse_metrics_config_rejects_unknown() -> None:
    with pytest.raises(MetricsConfigError, match="Unsupported metric"):
        parse_metrics_config({"primary": "unknown"})


def test_score_metrics_returns_values() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])
    cfg = parse_metrics_config(
        {"primary": "roc_auc", "additional": ["pr_auc", "log_loss"]}
    )

    scores = score_metrics(cfg, y_true, y_score)

    assert set(scores.keys()) == {"roc_auc", "pr_auc", "log_loss"}
    assert scores["roc_auc"] > 0.9
    assert scores["pr_auc"] > 0.9
    assert scores["log_loss"] > 0.0


def test_score_metrics_handles_single_class() -> None:
    y_true = np.array([1, 1, 1, 1])
    y_score = np.array([0.9, 0.8, 0.7, 0.6])
    cfg = parse_metrics_config({"primary": "roc_auc", "additional": ["pr_auc", "log_loss"]})

    scores = score_metrics(cfg, y_true, y_score)

    assert np.isnan(scores["roc_auc"])
    assert np.isnan(scores["pr_auc"])
    assert scores["log_loss"] > 0.0
