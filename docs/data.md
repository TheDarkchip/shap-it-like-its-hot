# Data source and preprocessing plan

## Source

- Dataset: Statlog (German Credit Data) from the UCI Machine Learning Repository.
- DOI: 10.24432/C5NC77
- Raw file: `german.data` (space-delimited, 1000 rows, 20 features + target).
- Download URL used by the repo:
  - https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data
- License: Creative Commons Attribution 4.0 International (CC BY 4.0) as listed on the UCI dataset page.
- Data is not committed to the repository. The download utility caches to `data/raw/`.

## Ethics and sensitivity notes

- The dataset includes sensitive attributes (for example: `personal_status_sex`, `age_years`, and `foreign_worker`).
- Use this dataset strictly for research and evaluation; do not deploy resulting models in real decision-making without additional review and safeguards.
- Any reporting should acknowledge potential bias in historical credit decisions.

## Preprocessing plan

- Column naming: apply a fixed set of 20 feature names plus `target`.
- Types:
  - Numeric columns cast to integers.
  - Categorical columns stored as pandas `category` dtype.
- Target mapping:
  - Raw values are `1` (good) and `2` (bad).
  - Loader default encodes `bad` as 1 and `good` as 0; this is configurable via `target_positive`.
- No feature engineering or one-hot encoding in the loader; that will be handled later in the modeling pipeline.

## Local cache

- Environment variable `SHAP_IT_DATA_DIR` can override the default cache root.
- Default cache location: `./data/raw/german.data`.
