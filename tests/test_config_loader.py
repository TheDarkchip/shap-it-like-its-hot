from __future__ import annotations

from pathlib import Path

import pytest

from config_loader import ConfigError, load_config, validate_config


def test_load_config_missing(tmp_path: Path) -> None:
    missing = tmp_path / "missing.yaml"
    with pytest.raises(FileNotFoundError):
        load_config(missing)


def test_validate_config_missing_sections() -> None:
    with pytest.raises(ConfigError):
        validate_config({"experiment": {}})


def test_validate_config_success(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        """
experiment:
  name: test
cv:
  outer_folds: 2
  outer_repeats: 1
  inner_folds: 2
resampling:
  target_positive_ratios: [0.3]
model:
  name: xgboost
  params: {}
metrics:
  primary: roc_auc
""",
        encoding="utf-8",
    )

    cfg = load_config(cfg_path)
    validate_config(cfg)


def test_validate_config_resampling_not_mapping() -> None:
    cfg = {
        "experiment": {},
        "cv": {},
        "resampling": [],
        "model": {},
        "metrics": {"primary": "roc_auc"},
    }
    with pytest.raises(ConfigError, match="resampling must be a mapping"):
        validate_config(cfg)


def test_validate_config_metrics_not_mapping() -> None:
    cfg = {
        "experiment": {},
        "cv": {},
        "resampling": {"target_positive_ratios": [0.3]},
        "model": {},
        "metrics": [],
    }
    with pytest.raises(ConfigError, match="metrics must be a mapping"):
        validate_config(cfg)
