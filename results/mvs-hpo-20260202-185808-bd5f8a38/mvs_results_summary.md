# MVS results summary

Run ID: `mvs-hpo-20260202-185808-bd5f8a38`
Data source: `results/mvs-hpo-20260202-185808-bd5f8a38/`

## Setup

- Dataset: Statlog German Credit
- Model: XGBoost + TreeSHAP
- Outer CV: 5 folds × 5 repeats (25 folds total)
- Inner CV: 3 folds with 32-config grid
- Train-only resampling ratios: 0.1, 0.3, 0.5
- Importance methods: mean(|SHAP|) and PFI
- PFI repeats: 10 per fold

## Performance (mean across folds)

| Class ratio | Accuracy | ROC AUC |
| --- | --- | --- |
| 0.1 | 0.719 | 0.754 |
| 0.3 | 0.757 | 0.791 |
| 0.5 | 0.726 | 0.781 |

## Stability (within-method)

### Directional

**SHAP**
- Rank stability (mean Spearman): 0.1: 0.65, 0.3: 0.84, 0.5: 0.71.
- Magnitude variance: 0.1: 1.90e-04, 0.3: 5.65e-05, 0.5: 1.08e-04.

**PFI**
- Rank stability (mean Spearman): 0.1: 0.51, 0.3: 0.62, 0.5: 0.54.
- Magnitude variance: 0.1: 1.62e-04, 0.3: 6.94e-05, 0.5: 1.09e-04.

### Magnitude

**SHAP**
- Rank stability (mean Spearman): 0.1: 0.65, 0.3: 0.84, 0.5: 0.71.
- Magnitude variance: 0.1: 1.90e-04, 0.3: 5.65e-05, 0.5: 1.08e-04.

**PFI**
- Rank stability (mean Spearman): 0.1: 0.57, 0.3: 0.70, 0.5: 0.60.
- Magnitude variance: 0.1: 9.66e-05, 0.3: 4.37e-05, 0.5: 7.50e-05.

## Agreement (SHAP vs PFI)

### Directional
- Spearman agreement: 0.1: 0.41, 0.3: 0.42, 0.5: 0.41.
- Top-k overlap: 0.1: 0.63, 0.3: 0.62, 0.5: 0.60.
- Cosine similarity: 0.1: 0.79, 0.3: 0.84, 0.5: 0.84.

### Magnitude
- Spearman agreement: 0.1: 0.86, 0.3: 0.80, 0.5: 0.82.
- Top-k overlap: 0.1: 0.63, 0.3: 0.62, 0.5: 0.60.
- Cosine similarity: 0.1: 0.87, 0.3: 0.89, 0.5: 0.90.

## Paired ratio differences (outer-fold paired)

Median difference with IQR in parentheses; positive values mean higher metric at higher ratio.

| Metric | 0.3 - 0.1 | 0.5 - 0.1 | 0.5 - 0.3 |
| --- | --- | --- | --- |
| accuracy | 0.040 (0.020) | 0.015 (0.030) | -0.025 (0.030) |
| roc_auc | 0.035 (0.040) | 0.029 (0.048) | -0.009 (0.024) |

## Within-fold PFI permutation uncertainty

Mean per-feature permutation std by ratio: 0.1: 2.822e-03, 0.3: 2.762e-03, 0.5: 2.713e-03.

## Notes / limitations
- Rank-stability and agreement plots show fold-level distributions; tables include mean plus median/IQR summaries.
- Magnitude-variance plots show mean with bootstrap SD across folds (dispersion, not inferential CIs).
- Directional variant preserves sign for correlation/cosine metrics; top-k overlap uses magnitudes to track important-feature membership.
- Agreement/stability top-k uses k=5 from run metadata.

## Files generated

- `results.csv`, `stability_summary.csv`, `agreement_summary.csv`
- `stability_table.csv`, `agreement_table.csv`
- plots under `results/.../plots/`
