from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from results_io import (
    ResultRecord,
    append_record_csv,
    record_to_frame,
    write_records_parquet,
)


def _record() -> ResultRecord:
    return ResultRecord(
        fold_id=0,
        repeat_id=1,
        seed=42,
        model_name="xgboost",
        class_ratio=0.3,
        metrics={"roc_auc": 0.71},
        shap_importance={"f1": 0.2, "f2": 0.1},
        pfi_importance={"f1": 0.15, "f2": 0.05},
    )


def test_record_to_frame_columns() -> None:
    frame = record_to_frame(_record())
    assert frame.shape == (1, 10)
    assert set(frame.columns) == {
        "fold_id",
        "repeat_id",
        "seed",
        "model_name",
        "class_ratio",
        "metric_roc_auc",
        "shap_f1",
        "shap_f2",
        "pfi_f1",
        "pfi_f2",
    }


def test_append_record_csv(tmp_path: Path) -> None:
    path = tmp_path / "results.csv"
    append_record_csv(path, _record())
    append_record_csv(path, _record())

    frame = pd.read_csv(path)
    assert frame.shape[0] == 2


def test_append_record_csv_stable_column_order(tmp_path: Path) -> None:
    path = tmp_path / "results.csv"
    append_record_csv(path, _record())

    swapped = ResultRecord(
        fold_id=0,
        repeat_id=1,
        seed=42,
        model_name="xgboost",
        class_ratio=0.3,
        metrics={"roc_auc": 0.71},
        shap_importance={"f2": 0.1, "f1": 0.2},
        pfi_importance={"f2": 0.05, "f1": 0.15},
    )
    append_record_csv(path, swapped)

    frame = pd.read_csv(path)
    assert frame.loc[1, "shap_f1"] == 0.2
    assert frame.loc[1, "shap_f2"] == 0.1
    assert frame.loc[1, "pfi_f1"] == 0.15
    assert frame.loc[1, "pfi_f2"] == 0.05


def test_append_record_csv_expands_schema(tmp_path: Path) -> None:
    path = tmp_path / "results.csv"
    append_record_csv(path, _record())

    expanded = ResultRecord(
        fold_id=0,
        repeat_id=1,
        seed=42,
        model_name="xgboost",
        class_ratio=0.3,
        metrics={"roc_auc": 0.71, "pr_auc": 0.44},
        shap_importance={"f1": 0.2, "f2": 0.1},
        pfi_importance={"f1": 0.15, "f2": 0.05},
    )
    append_record_csv(path, expanded)

    frame = pd.read_csv(path)
    assert "metric_pr_auc" in frame.columns
    assert pd.isna(frame.loc[0, "metric_pr_auc"])
    assert frame.loc[1, "metric_pr_auc"] == 0.44


def test_write_records_parquet(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    path = tmp_path / "results.parquet"
    write_records_parquet(path, [_record()])

    frame = pd.read_parquet(path)
    assert frame.shape[0] == 1


def test_write_records_parquet_empty_no_file(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    path = tmp_path / "empty.parquet"
    write_records_parquet(path, [])
    assert not path.exists()


def test_validate_ratio_rejects_invalid() -> None:
    record = ResultRecord(
        fold_id=0,
        repeat_id=0,
        seed=0,
        model_name="xgboost",
        class_ratio=1.0,
        metrics={},
        shap_importance={},
        pfi_importance={},
    )
    with pytest.raises(ValueError):
        record_to_frame(record)
