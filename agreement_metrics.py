"""Compute SHAP vs PFI agreement metrics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AgreementSummary:
    ratio: float
    mean_spearman: float
    mean_topk_overlap: float
    mean_cosine: float
    n_folds: int


def _load_importance(
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
    for _, row in subset.iterrows():
        series = row[cols]
        series.index = [c.removeprefix(prefix) for c in cols]
        matrices.append(series.astype(float))
    return pd.concat(matrices, axis=1)


def _topk_overlap(a: pd.Series, b: pd.Series, k: int) -> float:
    top_a = set(a.nlargest(k).index)
    top_b = set(b.nlargest(k).index)
    denom = min(k, len(a), len(b))
    if denom == 0:
        return float("nan")
    return len(top_a.intersection(top_b)) / float(denom)


def _cosine_similarity(a: pd.Series, b: pd.Series) -> float:
    a_vec = a.to_numpy(dtype=float)
    b_vec = b.to_numpy(dtype=float)
    denom = np.linalg.norm(a_vec) * np.linalg.norm(b_vec)
    if denom == 0:
        return float("nan")
    return float(np.dot(a_vec, b_vec) / denom)


def summarize_agreement(
    frame: pd.DataFrame,
    *,
    ratios: Iterable[float],
    top_k: int = 5,
) -> list[AgreementSummary]:
    summaries: list[AgreementSummary] = []

    for ratio in ratios:
        shap_values = _load_importance(frame, prefix="shap_", ratio=ratio)
        pfi_values = _load_importance(frame, prefix="pfi_", ratio=ratio)

        corrs: list[float] = []
        overlaps: list[float] = []
        cosines: list[float] = []

        for col in shap_values.columns:
            shap_vec = shap_values[col]
            pfi_vec = pfi_values[col]
            corr = shap_vec.corr(pfi_vec, method="spearman")
            if corr is not None and not np.isnan(corr):
                corrs.append(float(corr))
            overlaps.append(_topk_overlap(shap_vec, pfi_vec, top_k))
            cosines.append(_cosine_similarity(shap_vec.abs(), pfi_vec.abs()))

        summaries.append(
            AgreementSummary(
                ratio=float(ratio),
                mean_spearman=float(np.nanmean(corrs)),
                mean_topk_overlap=float(np.nanmean(overlaps)),
                mean_cosine=float(np.nanmean(cosines)),
                n_folds=shap_values.shape[1],
            )
        )

    return summaries


def write_agreement_summary(
    frame: pd.DataFrame,
    *,
    ratios: Iterable[float],
    output_path: str | Path,
    top_k: int = 5,
) -> pd.DataFrame:
    summaries = summarize_agreement(frame, ratios=ratios, top_k=top_k)
    out_frame = pd.DataFrame([summary.__dict__ for summary in summaries])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_frame.to_csv(output_path, index=False)
    return out_frame
