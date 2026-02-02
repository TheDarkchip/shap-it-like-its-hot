"""Download and load the Statlog German Credit dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable
import os
import urllib.request

import pandas as pd

GERMAN_CREDIT_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"

GERMAN_CREDIT_COLUMNS = [
    "checking_status",
    "duration_months",
    "credit_history",
    "purpose",
    "credit_amount",
    "savings_status",
    "employment_since",
    "installment_rate",
    "personal_status_sex",
    "other_debtors",
    "residence_since",
    "property",
    "age_years",
    "other_installment_plans",
    "housing",
    "existing_credits",
    "job",
    "people_liable",
    "telephone",
    "foreign_worker",
    "target",
]

_NUMERIC_COLUMNS = {
    "duration_months",
    "credit_amount",
    "installment_rate",
    "residence_since",
    "age_years",
    "existing_credits",
    "people_liable",
}

_CATEGORICAL_COLUMNS = [
    col for col in GERMAN_CREDIT_COLUMNS if col not in _NUMERIC_COLUMNS | {"target"}
]


def _default_data_dir() -> Path:
    env_dir = os.getenv("SHAP_IT_DATA_DIR")
    if env_dir:
        return Path(env_dir)
    return Path.cwd() / "data"


def download_german_credit(
    cache_dir: Path | None = None,
    *,
    url: str = GERMAN_CREDIT_URL,
    filename: str = "german.data",
    fetcher: Callable[[str, str | os.PathLike[str]], object] | None = None,
) -> Path:
    """Download the dataset into a local cache and return the file path.

    If the file already exists, it is reused. Set SHAP_IT_DATA_DIR to override
    the default data directory.
    """
    base_dir = Path(cache_dir) if cache_dir else _default_data_dir()
    raw_dir = base_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / filename

    if dest.exists():
        return dest

    tmp_path = raw_dir / f"{filename}.tmp"
    if fetcher is None:
        fetcher = urllib.request.urlretrieve

    fetcher(url, tmp_path)
    tmp_path.replace(dest)
    return dest


def load_german_credit(
    path: Path | str | None = None,
    *,
    target_positive: str | int = "bad",
) -> tuple[pd.DataFrame, pd.Series]:
    """Load the German Credit dataset and return (X, y).

    Parameters
    ----------
    path:
        Optional path to a local german.data file. If not provided, the dataset
        is downloaded via ``download_german_credit``.
    target_positive:
        Which class should be encoded as 1 in the output target. Use "bad"
        (default) or "good", or pass raw values 1 or 2.
    """
    data_path = Path(path) if path is not None else download_german_credit()

    df = pd.read_csv(
        data_path,
        sep=r"\s+",
        header=None,
        names=GERMAN_CREDIT_COLUMNS,
        dtype=str,
    )

    for col in _NUMERIC_COLUMNS:
        df[col] = df[col].astype(int)
    for col in _CATEGORICAL_COLUMNS:
        df[col] = df[col].astype("category")

    raw_target = df.pop("target").astype(int)
    if target_positive in ("good", "bad"):
        positive_value = 1 if target_positive == "good" else 2
    elif target_positive in (1, 2):
        positive_value = int(target_positive)
    else:
        raise ValueError("target_positive must be 'good', 'bad', 1, or 2")

    target = (raw_target == positive_value).astype(int)
    target.name = "target"

    return df, target


def iter_german_credit_columns(*, numeric_only: bool = False) -> Iterable[str]:
    """Yield column names, optionally limited to numeric columns."""
    if numeric_only:
        return iter(sorted(_NUMERIC_COLUMNS))
    return iter(GERMAN_CREDIT_COLUMNS)


def one_hot_encode_train_test(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit one-hot encoding on train and align test columns."""
    train_encoded = pd.get_dummies(X_train, drop_first=False)
    test_encoded = pd.get_dummies(X_test, drop_first=False)
    train_encoded = train_encoded.reindex(sorted(train_encoded.columns), axis=1)
    test_encoded = test_encoded.reindex(columns=train_encoded.columns, fill_value=0.0)
    return train_encoded, test_encoded


if __name__ == "__main__":
    dataset_path = download_german_credit()
    print(dataset_path)
