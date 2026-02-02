"""Run a single end-to-end experiment from a config file."""

from __future__ import annotations

import argparse

from shap_stability.single_run import run_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single experiment")
    parser.add_argument("config", help="Path to YAML config file")
    parser.add_argument("--output-dir", help="Directory for artifacts")
    args = parser.parse_args()

    artifacts = run_from_config(args.config, output_dir=args.output_dir)
    print(f"Wrote results to {artifacts.results_path}")


if __name__ == "__main__":
    main()
