"""Run entry point that loads and validates a config file."""

from __future__ import annotations

import argparse
from pathlib import Path

from shap_stability.config_loader import load_config, validate_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate experiment config")
    parser.add_argument("config", help="Path to YAML config file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    validate_config(cfg)
    print(f"Config OK: {args.config}")


if __name__ == "__main__":
    main()
