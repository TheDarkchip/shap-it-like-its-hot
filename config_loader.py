"""YAML config loader for experiment settings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when configuration is invalid or missing required fields."""


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file from disk."""
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if not isinstance(data, dict):
        raise ConfigError("Config root must be a mapping")

    return data


def validate_config(cfg: dict[str, Any]) -> None:
    """Validate required config sections and basic types."""
    required_sections = ("experiment", "cv", "resampling", "model", "metrics")
    missing = [section for section in required_sections if section not in cfg]
    if missing:
        raise ConfigError(f"Missing required sections: {', '.join(missing)}")

    if not isinstance(cfg["resampling"], dict):
        raise ConfigError("resampling must be a mapping")

    if not isinstance(cfg["metrics"], dict):
        raise ConfigError("metrics must be a mapping")

    if not isinstance(cfg["resampling"].get("target_positive_ratios"), list):
        raise ConfigError("resampling.target_positive_ratios must be a list")

    if "primary" not in cfg["metrics"]:
        raise ConfigError("metrics.primary is required")

    from metrics_utils import MetricsConfigError, parse_metrics_config

    try:
        parse_metrics_config(cfg["metrics"])
    except MetricsConfigError as exc:
        raise ConfigError(str(exc)) from exc


if __name__ == "__main__":
    import sys

    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/mvs.yaml"
    config = load_config(config_path)
    validate_config(config)
    print(f"Loaded config: {config_path}")
