# MVS results summary (baseline HPO run)

Run ID: `mvs-hpo-20260202-031158-8bf30d5a`  
Data source: `results/mvs-hpo-20260202-031158-8bf30d5a/`

## Setup

- Dataset: Statlog German Credit
- Model: XGBoost + TreeSHAP
- Outer CV: 5 folds × 5 repeats (25 folds total)
- Inner CV: 3 folds with 32‑config grid
- Train-only resampling ratios: 0.1, 0.3, 0.5
- Importance methods: mean(|SHAP|) and PFI
- PFI repeats: 10 per fold

## Performance (mean across folds)

| Class ratio | Accuracy | ROC AUC |
| --- | --- | --- |
| 0.1 | 0.727 | 0.757 |
| 0.3 | 0.755 | 0.790 |
| 0.5 | 0.741 | 0.779 |

## Stability (within-method)

**SHAP**
- Rank stability is highest at 0.3 (mean Spearman ≈ 0.84), lower at 0.1 (~0.66) and 0.5 (~0.78).
- Magnitude variance is low across ratios (~5e-5 to 1.7e-4), indicating relatively stable importance magnitudes.

**PFI**
- Rank stability is lower than SHAP (mean Spearman ≈ 0.28–0.40).
- Magnitude variance is higher than SHAP (≈ 1.6e-4 to 4.2e-4), indicating greater fold-to-fold variability.

## Agreement (SHAP vs PFI)

- Spearman agreement is moderate‑to‑high (≈ 0.76–0.85), with slightly higher agreement at 0.1.
- Top‑k overlap is moderate (≈ 0.70–0.78), indicating partial agreement on the most important features.
- Cosine similarity of normalized importance vectors is high (≈ 0.87–0.90), indicating broadly aligned magnitude patterns.

## Notes / limitations

- Rank-stability and agreement plots now show fold-level distributions; tables still report fold-averaged means.
- Magnitude-variance plots show mean with bootstrap SD across folds (dispersion, not inferential CIs).
- Agreement metrics compare absolute importances to align with mean(|SHAP|).
- These are baseline MVS results; optional stretches (e.g., metric sensitivity, signed SHAP) may alter conclusions.

## Files generated

- `results.csv`, `stability_summary.csv`, `agreement_summary.csv`
- `stability_table.csv`, `agreement_table.csv`
- plots under `results/.../plots/`
