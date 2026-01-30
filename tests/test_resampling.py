from __future__ import annotations

import pandas as pd
import pytest

from resampling import ResampleResult, resample_train_fold


def _toy_data(n_pos: int, n_neg: int) -> tuple[pd.DataFrame, pd.Series]:
    X = pd.DataFrame({"feature": list(range(n_pos + n_neg))})
    y = pd.Series([1] * n_pos + [0] * n_neg)
    return X, y


def test_resample_preserves_size_and_ratio() -> None:
    X, y = _toy_data(30, 70)
    result = resample_train_fold(X, y, target_positive_ratio=0.3, random_state=1)

    assert isinstance(result, ResampleResult)
    assert len(result.X) == len(X)
    assert len(result.y) == len(y)
    assert result.positive_count == 30
    assert result.negative_count == 70


def test_resample_oversamples_minority() -> None:
    X, y = _toy_data(2, 8)
    result = resample_train_fold(X, y, target_positive_ratio=0.5, random_state=0)

    assert result.positive_count == 5
    assert result.negative_count == 5
    assert result.y.sum() == 5


def test_resample_is_reproducible() -> None:
    X, y = _toy_data(10, 10)
    first = resample_train_fold(X, y, target_positive_ratio=0.4, random_state=42)
    second = resample_train_fold(X, y, target_positive_ratio=0.4, random_state=42)

    pd.testing.assert_frame_equal(first.X, second.X)
    pd.testing.assert_series_equal(first.y, second.y)


def test_resample_keeps_both_classes_for_small_ratio() -> None:
    X, y = _toy_data(1, 9)
    result = resample_train_fold(X, y, target_positive_ratio=0.01, random_state=7)

    assert result.positive_count == 1
    assert result.negative_count == 9
    assert result.y.sum() == 1


def test_invalid_ratio_raises() -> None:
    X, y = _toy_data(2, 8)
    with pytest.raises(ValueError):
        resample_train_fold(X, y, target_positive_ratio=1.0)


def test_requires_both_classes() -> None:
    X = pd.DataFrame({"feature": [1, 2, 3]})
    y = pd.Series([1, 1, 1])

    with pytest.raises(ValueError):
        resample_train_fold(X, y, target_positive_ratio=0.5)
