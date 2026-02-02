from __future__ import annotations

import numpy as np
import pandas as pd

from stability_metrics import summarize_stability


def _toy_results() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "class_ratio": [0.1, 0.1, 0.1],
            "shap_a": [0.5, 0.4, 0.6],
            "shap_b": [0.2, 0.3, 0.1],
            "pfi_a": [0.05, 0.04, 0.06],
            "pfi_b": [0.02, 0.03, 0.01],
        }
    )


def test_summarize_stability_returns_expected_fields() -> None:
    frame = _toy_results()
    summaries = summarize_stability(frame, ratios=[0.1], method="shap")

    assert len(summaries) == 1
    summary = summaries[0]
    assert summary.method == "shap"
    assert summary.ratio == 0.1
    assert summary.n_folds == 3
    assert np.isfinite(summary.mean_rank_corr)
    assert summary.mean_magnitude_var >= 0


def test_summarize_stability_requires_method() -> None:
    frame = _toy_results()
    try:
        summarize_stability(frame, ratios=[0.1], method="other")
    except ValueError as exc:
        assert "method" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid method")
