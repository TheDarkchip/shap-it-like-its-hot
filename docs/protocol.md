# Protocol: Nested CV Evaluation

This document defines the evaluation protocol for the SHAP stability study and the
outputs expected from each run. It is intentionally minimal and aligned with the
current evaluation harness utilities.

## Goals

- Compare SHAP and permutation feature importance (PFI) under training-imbalance
  changes while keeping evaluation folds untouched.
- Ensure protocol is leakage-safe and reproducible across random seeds.

## Data and preprocessing

- Dataset: Statlog German Credit.
- Target: binary label with "bad" as positive class (1).
- Categorical features are one-hot encoded **within each fold**:
  fit on the training split and applied to the matching test split with
  aligned columns.

## Splits and resampling

- Use repeated stratified K-fold on the **outer** split.
- Within each outer fold, resample only the training data to target positive
  class ratios (e.g., 0.1, 0.3, 0.5). The test fold is never resampled.

## Nested CV flow (pseudocode)

```
for repeat in 1..outer_repeats:
  for fold in 1..outer_folds:
    split train/test indices
    for ratio in target_positive_ratios:
      if HPO enabled:
        run inner CV on outer training fold:
          for each inner split:
            resample inner training fold to ratio
            encode inner train; encode inner validation to same columns
            fit/evaluate; aggregate inner metric
        choose best params
      resample full outer training fold to ratio
      encode outer train; encode outer test to same columns
      retrain model on full resampled train fold
      evaluate on untouched test fold
      compute SHAP + PFI on test fold
      write results row + metadata
```

## Inner CV (hyperparameter search)

- Inner CV uses stratified K-fold on the **original outer training fold**.
- For each inner split, the **inner training fold is resampled** to the target
  ratio, then encoded, and the inner validation fold is encoded to the same columns.
- Metrics for selection come from the metrics config (primary metric).
- Best params are selected, then the model is retrained on the full resampled
  outer training fold (with fold-fitted encoding) before final evaluation.
- The standalone single-run script uses fixed model params; HPO is part of the
  nested CV harness utilities.

## Metrics

- Primary metric: ROC AUC (default).
- Additional metrics: PR AUC and log loss (configurable).
- PFI uses the **primary** metric for permutation scoring.

## Artifacts

Each run produces a run directory with:

- `results.csv` with fold-level metrics, SHAP importances, and PFI importances.
- `run_metadata.json` capturing run id, seed, and environment.
- `run.log` with structured logging.

## Sanity checks

- Leakage: train/test indices never overlap within outer folds.
- Determinism: resampling is repeatable given a fixed seed.
- Metric sanity: metrics are finite or NaN for single-class folds.

## Randomness and reproducibility

- All stochastic components (outer split, inner split, resampling, PFI permutations)
  are seeded from a base seed deterministically as a function of repeat/fold/ratio.
