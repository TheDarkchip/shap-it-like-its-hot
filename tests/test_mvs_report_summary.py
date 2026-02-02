from __future__ import annotations

from pathlib import Path

import pandas as pd

from importlib.util import module_from_spec, spec_from_file_location


def _load_render_summary():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "generate_mvs_report.py"
    spec = spec_from_file_location("generate_mvs_report", module_path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module._render_summary


def test_render_summary_contains_core_sections(tmp_path: Path) -> None:
    results = pd.DataFrame(
        [
            {
                "fold_id": 0,
                "repeat_id": 0,
                "seed": 42,
                "model_name": "xgboost",
                "class_ratio": 0.1,
                "metric_accuracy": 0.70,
                "metric_roc_auc": 0.75,
                "shap_f1": 0.2,
                "pfi_f1": 0.1,
            },
            {
                "fold_id": 1,
                "repeat_id": 0,
                "seed": 42,
                "model_name": "xgboost",
                "class_ratio": 0.3,
                "metric_accuracy": 0.72,
                "metric_roc_auc": 0.78,
                "shap_f1": 0.25,
                "pfi_f1": 0.12,
            },
        ]
    )
    stability = pd.DataFrame(
        [
            {
                "ratio": 0.1,
                "method": "shap",
                "variant": "magnitude",
                "mean_rank_corr": 0.8,
                "mean_magnitude_var": 1.0e-4,
            },
            {
                "ratio": 0.3,
                "method": "shap",
                "variant": "magnitude",
                "mean_rank_corr": 0.85,
                "mean_magnitude_var": 8.0e-5,
            },
            {
                "ratio": 0.1,
                "method": "pfi",
                "variant": "magnitude",
                "mean_rank_corr": 0.3,
                "mean_magnitude_var": 3.0e-4,
            },
            {
                "ratio": 0.3,
                "method": "pfi",
                "variant": "magnitude",
                "mean_rank_corr": 0.4,
                "mean_magnitude_var": 2.0e-4,
            },
        ]
    )
    agreement = pd.DataFrame(
        [
            {
                "ratio": 0.1,
                "variant": "magnitude",
                "mean_spearman": 0.7,
                "mean_topk_overlap": 0.6,
                "mean_cosine": 0.8,
            },
            {
                "ratio": 0.3,
                "variant": "magnitude",
                "mean_spearman": 0.75,
                "mean_topk_overlap": 0.65,
                "mean_cosine": 0.82,
            },
        ]
    )
    paired = pd.DataFrame(
        [
            {
                "ratio_high": 0.3,
                "ratio_low": 0.1,
                "metric": "metric_roc_auc",
                "median_diff": 0.02,
                "iqr_diff": 0.01,
                "n_pairs": 2,
            }
        ]
    )
    metadata = {
        "run_id": "mvs-demo",
        "outer_folds": 5,
        "outer_repeats": 2,
        "inner_folds": 3,
        "pfi_repeats": 10,
        "param_grid": {"n_estimators": [100, 200], "max_depth": [3]},
    }

    render_summary = _load_render_summary()
    summary = render_summary(
        results,
        stability_table=stability,
        agreement_table=agreement,
        paired_table=paired,
        metadata=metadata,
        results_dir=tmp_path,
        ratios=[0.1, 0.3],
        top_k=5,
    )

    assert "Run ID: `mvs-demo`" in summary
    assert "## Performance (mean across folds)" in summary
    assert "| 0.1 | 0.700 | 0.750 |" in summary
    assert "## Stability (within-method)" in summary
    assert "## Agreement (SHAP vs PFI)" in summary
    assert "## Paired ratio differences (outer-fold paired)" in summary
    assert "| 0.3 - 0.1 |" in summary
    assert "Directional variant preserves sign for correlation/cosine metrics" in summary
    assert "Agreement/stability top-k uses k=5" in summary
