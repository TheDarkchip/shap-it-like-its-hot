from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from shap_stability.metrics.stability import summarize_stability, write_stability_summary


def _toy_results() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "class_ratio": [0.1, 0.1, 0.1],
            "shap_a": [0.5, 0.4, 0.6],
            "shap_b": [0.2, 0.3, 0.1],
            "shap_pfi_score": [0.1, 0.2, 0.3],
            "pfi_a": [0.05, 0.04, 0.06],
            "pfi_b": [0.02, 0.03, 0.01],
        }
    )


def test_summarize_stability_returns_expected_fields() -> None:
    frame = _toy_results()
    summaries = summarize_stability(frame, ratios=[0.1], method="shap", variant="magnitude")

    assert len(summaries) == 1
    summary = summaries[0]
    assert summary.method == "shap"
    assert summary.ratio == 0.1
    assert summary.n_folds == 3
    assert summary.variant == "magnitude"
    assert np.isfinite(summary.mean_rank_corr)
    assert 0.0 <= summary.mean_topk_overlap <= 1.0
    assert summary.mean_magnitude_var >= 0


def test_summarize_stability_requires_method() -> None:
    frame = _toy_results()
    try:
        summarize_stability(frame, ratios=[0.1], method="other")
    except ValueError as exc:
        assert "method" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid method")


def test_write_stability_summary_writes_csv(tmp_path: Path) -> None:
    frame = _toy_results()
    out_path = tmp_path / "summary.csv"

    out_frame = write_stability_summary(frame, ratios=[0.1], output_path=out_path)

    assert out_path.exists()
    assert set(out_frame["method"]) == {"shap", "pfi"}
    assert set(out_frame["variant"]) == {"magnitude", "directional"}
    assert len(out_frame) == 4


def test_dispersion_nan_for_zero_importance() -> None:
    frame = pd.DataFrame(
        {
            "class_ratio": [0.1, 0.1],
            "shap_a": [0.0, 0.0],
            "shap_b": [0.0, 0.0],
        }
    )
    # Expect runtime warnings from constant-input correlations / NaN means.
    with pytest.warns((RuntimeWarning, UserWarning)):
        summaries = summarize_stability(frame, ratios=[0.1], method="shap", variant="magnitude")
    assert np.isnan(summaries[0].mean_dispersion)


def test_pfi_dispersion_uses_absolute_values() -> None:
    frame = pd.DataFrame(
        {
            "class_ratio": [0.1, 0.1],
            "pfi_a": [0.1, -0.1],
            "pfi_b": [0.2, -0.2],
        }
    )
    summaries = summarize_stability(frame, ratios=[0.1], method="pfi", variant="magnitude")
    assert np.isfinite(summaries[0].mean_dispersion)


def test_directional_vs_magnitude_rank_corr_differs() -> None:
    frame = pd.DataFrame(
        {
            "class_ratio": [0.1, 0.1],
            "pfi_a": [0.2, -0.2],
            "pfi_b": [0.1, -0.1],
        }
    )
    directional = summarize_stability(frame, ratios=[0.1], method="pfi", variant="directional")
    magnitude = summarize_stability(frame, ratios=[0.1], method="pfi", variant="magnitude")

    assert directional[0].mean_rank_corr != magnitude[0].mean_rank_corr


def test_magnitude_variance_uses_normalized_values() -> None:
    frame = pd.DataFrame(
        {
            "class_ratio": [0.1, 0.1],
            "shap_a": [1.0, 2.0],
            "shap_b": [2.0, 4.0],
        }
    )
    summaries = summarize_stability(frame, ratios=[0.1], method="shap", variant="magnitude")
    assert summaries[0].mean_magnitude_var == 0.0


def test_prefix_removal_only_strips_leading() -> None:
    frame = pd.DataFrame(
        {
            "class_ratio": [0.1],
            "shap_pfi_score": [0.2],
            "shap_other": [0.1],
        }
    )
    summaries = summarize_stability(frame, ratios=[0.1], method="shap", variant="magnitude")
    assert summaries[0].n_folds == 1
