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
    variant: str
    mean_rank_corr: float
    mean_topk_overlap: float
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


def _topk_overlap(a: pd.Series, b: pd.Series, k: int) -> float:
    a_clean = a.dropna()
    b_clean = b.dropna()
    top_a = set(a_clean.nlargest(k).index)
    top_b = set(b_clean.nlargest(k).index)
    denom = min(k, len(a_clean), len(b_clean))
    if denom == 0:
        return float("nan")
    return len(top_a.intersection(top_b)) / float(denom)


def _pairwise_topk_overlap(values: pd.DataFrame, k: int, *, use_abs: bool) -> list[float]:
    cols = list(values.columns)
    overlaps: list[float] = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            left = values[cols[i]].abs() if use_abs else values[cols[i]]
            right = values[cols[j]].abs() if use_abs else values[cols[j]]
            overlaps.append(_topk_overlap(left, right, k))
    return overlaps


def _normalize_columns(values: pd.DataFrame) -> pd.DataFrame:
    sums = values.sum(axis=0)
    normalized = values.copy()
    for col in values.columns:
        total = sums[col]
        if total == 0:
            normalized[col] = np.nan
        else:
            normalized[col] = values[col] / total
    return normalized


def _magnitude_variance(values: pd.DataFrame) -> float:
    normalized = _normalize_columns(values)
    return float(normalized.var(axis=1, ddof=1).mean())


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
        series.index = [c.removeprefix(prefix) for c in cols]
        matrices.append(series.astype(float))
    return pd.concat(matrices, axis=1)


def summarize_stability(
    frame: pd.DataFrame,
    *,
    ratios: Iterable[float],
    method: str,
    variant: str = "magnitude",
    top_k: int = 5,
) -> list[StabilitySummary]:
    if method not in {"shap", "pfi"}:
        raise ValueError("method must be 'shap' or 'pfi'")
    if variant not in {"directional", "magnitude"}:
        raise ValueError("variant must be 'directional' or 'magnitude'")
    prefix = f"{method}_"
    summaries: list[StabilitySummary] = []

    for ratio in ratios:
        values = _collect_importances(frame, prefix=prefix, ratio=ratio)
        variant_values = values.abs() if variant == "magnitude" else values
        ranks = variant_values.apply(_rank_vector, axis=0)
        corrs = _pairwise_corr(ranks)
        mean_corr = float(np.mean(corrs)) if corrs else float("nan")
        overlaps = _pairwise_topk_overlap(
            variant_values, top_k, use_abs=(variant == "directional")
        )
        mean_topk = float(np.nanmean(overlaps)) if overlaps else float("nan")
        magnitude_var = _magnitude_variance(variant_values)
        dispersions = [_dispersion(variant_values[col]) for col in values.columns]
        mean_dispersion = float(np.nanmean(dispersions))

        summaries.append(
            StabilitySummary(
                ratio=float(ratio),
                method=method,
                variant=variant,
                mean_rank_corr=mean_corr,
                mean_topk_overlap=mean_topk,
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
    variants: Iterable[str] = ("magnitude", "directional"),
    top_k: int = 5,
) -> pd.DataFrame:
    summaries = []
    for method in ("shap", "pfi"):
        for variant in variants:
            summaries.extend(
                summarize_stability(
                    frame,
                    ratios=ratios,
                    method=method,
                    variant=variant,
                    top_k=top_k,
                )
            )

    out_frame = pd.DataFrame([summary.__dict__ for summary in summaries])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_frame.to_csv(output_path, index=False)
    return out_frame
