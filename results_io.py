"""Schema and writer utilities for experiment outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class ResultRecord:
    """Single fold-level result record."""

    fold_id: int
    repeat_id: int
    seed: int
    model_name: str
    class_ratio: float
    metrics: dict[str, float]
    shap_importance: dict[str, float]
    pfi_importance: dict[str, float]


def _validate_record(record: ResultRecord) -> None:
    if record.fold_id < 0 or record.repeat_id < 0:
        raise ValueError("fold_id and repeat_id must be non-negative")
    if not 0.0 < record.class_ratio < 1.0:
        raise ValueError("class_ratio must be between 0 and 1 (exclusive)")


def record_to_frame(record: ResultRecord) -> pd.DataFrame:
    """Convert a record into a normalized single-row DataFrame."""
    _validate_record(record)

    row: dict[str, Any] = {
        "fold_id": record.fold_id,
        "repeat_id": record.repeat_id,
        "seed": record.seed,
        "model_name": record.model_name,
        "class_ratio": record.class_ratio,
    }

    for key in sorted(record.metrics.keys()):
        value = record.metrics[key]
        row[f"metric_{key}"] = value

    for key in sorted(record.shap_importance.keys()):
        value = record.shap_importance[key]
        row[f"shap_{key}"] = value

    for key in sorted(record.pfi_importance.keys()):
        value = record.pfi_importance[key]
        row[f"pfi_{key}"] = value

    return pd.DataFrame([row])


def append_record_csv(path: str | Path, record: ResultRecord) -> None:
    """Append a record to a CSV file (creates file if missing)."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame = record_to_frame(record)
    if output_path.exists():
        existing_cols = pd.read_csv(output_path, nrows=0).columns
        frame = frame.reindex(columns=existing_cols, fill_value=pd.NA)
        frame.to_csv(output_path, mode="a", index=False, header=False)
    else:
        frame.to_csv(output_path, index=False)


def write_records_parquet(path: str | Path, records: list[ResultRecord]) -> None:
    """Write a collection of records to a parquet file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frames = [record_to_frame(record) for record in records]
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    combined.to_parquet(output_path, index=False)
