from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from shap_it_like_its_hot.data.german_credit import (
    GERMAN_CREDIT_COLUMNS,
    download_german_credit,
    iter_german_credit_columns,
    load_german_credit,
)

SAMPLE_DATA = """\
A11 6 A34 A43 1169 A65 A75 4 A93 A101 4 A121 67 A143 A152 2 A173 1 A192 A201 1
A12 48 A32 A43 5951 A61 A73 2 A92 A101 2 A121 22 A143 A152 1 A173 1 A191 A201 2
"""

NUMERIC_COLUMNS = {
    "duration_months",
    "credit_amount",
    "installment_rate",
    "residence_since",
    "age_years",
    "existing_credits",
    "people_liable",
}


def _write_sample(path: Path) -> None:
    path.write_text(SAMPLE_DATA, encoding="utf-8")


def test_download_uses_cache(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True)
    dest = raw_dir / "german.data"
    _write_sample(dest)

    def fetcher(_url: str, _dest: str) -> None:
        raise AssertionError("Fetcher should not be called when cached")

    path = download_german_credit(cache_dir=tmp_path, fetcher=fetcher)
    assert path == dest


def test_download_writes_file(tmp_path: Path) -> None:
    def fetcher(_url: str, dest: str | Path) -> None:
        Path(dest).write_text(SAMPLE_DATA, encoding="utf-8")

    path = download_german_credit(cache_dir=tmp_path, fetcher=fetcher)
    assert path.exists()
    assert path.read_text(encoding="utf-8") == SAMPLE_DATA


def test_load_parses_data(tmp_path: Path) -> None:
    data_path = tmp_path / "german.data"
    _write_sample(data_path)

    X, y = load_german_credit(data_path)

    assert list(X.columns) == GERMAN_CREDIT_COLUMNS[:-1]
    assert X.shape == (2, 20)
    assert y.tolist() == [0, 1]

    for col in NUMERIC_COLUMNS:
        assert pd.api.types.is_integer_dtype(X[col])
    for col in X.columns:
        if col not in NUMERIC_COLUMNS:
            assert isinstance(X[col].dtype, pd.CategoricalDtype)


def test_target_positive_good(tmp_path: Path) -> None:
    data_path = tmp_path / "german.data"
    _write_sample(data_path)

    _, y = load_german_credit(data_path, target_positive="good")
    assert y.tolist() == [1, 0]


def test_target_positive_validation(tmp_path: Path) -> None:
    data_path = tmp_path / "german.data"
    _write_sample(data_path)

    with pytest.raises(ValueError):
        load_german_credit(data_path, target_positive="unknown")


def test_iter_german_credit_columns_all() -> None:
    """Test iter_german_credit_columns returns all columns in order."""
    columns = list(iter_german_credit_columns(numeric_only=False))
    assert columns == GERMAN_CREDIT_COLUMNS


def test_iter_german_credit_columns_numeric_only() -> None:
    """Test iter_german_credit_columns returns only numeric columns when numeric_only=True."""
    columns = list(iter_german_credit_columns(numeric_only=True))

    # Should return only numeric columns
    assert set(columns) == NUMERIC_COLUMNS

    # Should be sorted alphabetically
    assert columns == sorted(columns)
