"""CLI wrapper for single-run experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from single_run import run_from_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single experiment")
    parser.add_argument("config", help="Path to YAML config file")
    parser.add_argument("--output-dir", help="Directory for artifacts")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = run_from_config(args.config, output_dir=args.output_dir)
    print(f"Wrote results to {artifacts.results_path}")


if __name__ == "__main__":
    main()
