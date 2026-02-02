"""Compute stability metrics for SHAP and PFI importances."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StabilitySummary:
    ratio: float
    method: str
    mean_rank_corr: float
    mean_magnitude_var: float
    mean_dispersion: float
    n_folds: int


def _rank_vector(values: pd.Series) -> pd.Series:
    return values.rank(ascending=False, method="average")


def _pairwise_corr(ranks: pd.DataFrame) -> list[float]:
    cols = list(ranks.columns)
    corrs: list[float] = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            corr = ranks[cols[i]].corr(ranks[cols[j]], method="spearman")
            if corr is not None and not np.isnan(corr):
                corrs.append(float(corr))
    return corrs


def _magnitude_variance(values: pd.DataFrame) -> float:
    return float(values.var(axis=1, ddof=1).mean())


def _dispersion(values: pd.Series) -> float:
    total = values.sum()
    if total == 0:
        return float("nan")
    proportions = values / total
    return float(1 - np.square(proportions).sum())


def _collect_importances(
    frame: pd.DataFrame,
    *,
    prefix: str,
    ratio: float,
) -> pd.DataFrame:
    subset = frame[frame["class_ratio"] == ratio]
    cols = [col for col in subset.columns if col.startswith(prefix)]
    if not cols:
        raise ValueError(f"No columns found for prefix {prefix}")
    matrices: list[pd.Series] = []
    for idx, row in subset.iterrows():
        series = row[cols]
        series.index = [c.replace(prefix, "") for c in cols]
        matrices.append(series.astype(float))
    return pd.concat(matrices, axis=1)


def summarize_stability(
    frame: pd.DataFrame,
    *,
    ratios: Iterable[float],
    method: str,
) -> list[StabilitySummary]:
    if method not in {"shap", "pfi"}:
        raise ValueError("method must be 'shap' or 'pfi'")
    prefix = f"{method}_"
    summaries: list[StabilitySummary] = []

    for ratio in ratios:
        values = _collect_importances(frame, prefix=prefix, ratio=ratio)
        ranks = values.apply(_rank_vector, axis=0)
        corrs = _pairwise_corr(ranks)
        mean_corr = float(np.mean(corrs)) if corrs else float("nan")
        magnitude_var = _magnitude_variance(values)
        dispersions = [_dispersion(values[col]) for col in values.columns]
        mean_dispersion = float(np.nanmean(dispersions))

        summaries.append(
            StabilitySummary(
                ratio=float(ratio),
                method=method,
                mean_rank_corr=mean_corr,
                mean_magnitude_var=magnitude_var,
                mean_dispersion=mean_dispersion,
                n_folds=values.shape[1],
            )
        )

    return summaries


def write_stability_summary(
    frame: pd.DataFrame,
    *,
    ratios: Iterable[float],
    output_path: str | Path,
) -> pd.DataFrame:
    summaries = []
    for method in ("shap", "pfi"):
        summaries.extend(summarize_stability(frame, ratios=ratios, method=method))

    out_frame = pd.DataFrame([summary.__dict__ for summary in summaries])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_frame.to_csv(output_path, index=False)
    return out_frame
