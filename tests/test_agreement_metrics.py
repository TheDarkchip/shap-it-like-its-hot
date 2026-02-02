from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from agreement_metrics import summarize_agreement, write_agreement_summary


def _toy_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "class_ratio": [0.1, 0.1],
            "shap_a": [0.5, 0.6],
            "shap_b": [0.2, 0.1],
            "pfi_a": [0.4, 0.5],
            "pfi_b": [0.3, 0.2],
        }
    )


def test_summarize_agreement_outputs_metrics() -> None:
    frame = _toy_frame()
    summaries = summarize_agreement(frame, ratios=[0.1], top_k=1)

    assert len(summaries) == 1
    summary = summaries[0]
    assert summary.ratio == 0.1
    assert summary.n_folds == 2
    assert np.isfinite(summary.mean_spearman)
    assert 0.0 <= summary.mean_topk_overlap <= 1.0
    assert np.isfinite(summary.mean_cosine)


def test_write_agreement_summary_writes_csv(tmp_path: Path) -> None:
    frame = _toy_frame()
    out_path = tmp_path / "agreement.csv"

    out_frame = write_agreement_summary(frame, ratios=[0.1], output_path=out_path, top_k=1)

    assert out_path.exists()
    assert len(out_frame) == 1
