"""Centralized metric definitions and scoring helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    log_loss,
    roc_auc_score,
)


MetricFn = Callable[[np.ndarray, np.ndarray], float]


@dataclass(frozen=True)
class MetricDefinition:
    name: str
    display_name: str
    scorer: MetricFn
    requires_probabilities: bool = True


SUPPORTED_METRICS: Dict[str, MetricDefinition] = {
    "roc_auc": MetricDefinition(
        name="roc_auc",
        display_name="ROC AUC",
        scorer=lambda y_true, y_score: float(roc_auc_score(y_true, y_score)),
    ),
    "pr_auc": MetricDefinition(
        name="pr_auc",
        display_name="PR AUC",
        scorer=lambda y_true, y_score: float(average_precision_score(y_true, y_score)),
    ),
    "log_loss": MetricDefinition(
        name="log_loss",
        display_name="Log Loss",
        scorer=lambda y_true, y_score: float(log_loss(y_true, y_score)),
    ),
}


@dataclass(frozen=True)
class MetricsConfig:
    primary: str
    additional: tuple[str, ...]

    def all_metrics(self) -> tuple[str, ...]:
        extras = tuple(name for name in self.additional if name != self.primary)
        return (self.primary,) + extras


class MetricsConfigError(ValueError):
    """Raised when the metrics configuration is invalid."""


def parse_metrics_config(raw: dict) -> MetricsConfig:
    if "primary" not in raw:
        raise MetricsConfigError("metrics.primary is required")

    primary = str(raw["primary"])
    additional_raw = raw.get("additional", [])
    if not isinstance(additional_raw, list):
        raise MetricsConfigError("metrics.additional must be a list")
    additional = tuple(str(name) for name in additional_raw)

    cfg = MetricsConfig(primary=primary, additional=additional)
    _validate_metrics(cfg)
    return cfg


def _validate_metrics(cfg: MetricsConfig) -> None:
    for name in cfg.all_metrics():
        if name not in SUPPORTED_METRICS:
            raise MetricsConfigError(f"Unsupported metric: {name}")


def score_metrics(
    cfg: MetricsConfig,
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> dict[str, float]:
    unique_labels = np.unique(y_true)
    scores: dict[str, float] = {}
    for name in cfg.all_metrics():
        metric = SUPPORTED_METRICS[name]
        if name in {"roc_auc", "pr_auc"} and unique_labels.size < 2:
            scores[name] = float("nan")
            continue
        if name == "log_loss" and unique_labels.size < 2:
            scores[name] = float(log_loss(y_true, y_score, labels=[0, 1]))
            continue
        scores[name] = metric.scorer(y_true, y_score)
    return scores
