"""Compute agreement metrics from MVS results."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shap_stability.metrics.agreement import write_agreement_summary  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute agreement metrics")
    parser.add_argument("results", help="Path to results.csv")
    parser.add_argument("--output", help="Path to write summary CSV")
    parser.add_argument("--top-k", type=int, default=5, help="Top-k overlap")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_path = Path(args.results)
    out_path = (
        Path(args.output)
        if args.output
        else results_path.parent / "agreement_summary.csv"
    )

    frame = pd.read_csv(results_path)
    ratios = sorted(frame["class_ratio"].unique())
    write_agreement_summary(frame, ratios=ratios, output_path=out_path, top_k=args.top_k)
    print(f"Wrote agreement summary to {out_path}")


if __name__ == "__main__":
    main()
