"""Compute stability metrics from MVS results."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from shap_stability.metrics.stability import write_stability_summary  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute stability metrics")
    parser.add_argument("results", help="Path to results.csv")
    parser.add_argument("--output", help="Path to write summary CSV")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_path = Path(args.results)
    out_path = (
        Path(args.output)
        if args.output
        else results_path.parent / "stability_summary.csv"
    )

    frame = pd.read_csv(results_path)
    ratios = sorted(frame["class_ratio"].unique())
    write_stability_summary(frame, ratios=ratios, output_path=out_path)
    print(f"Wrote stability summary to {out_path}")


if __name__ == "__main__":
    main()
