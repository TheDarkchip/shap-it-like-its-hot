"""Generate MVS plots and summary tables."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _plot_metric(
    frame: pd.DataFrame,
    *,
    x: str,
    y: str,
    title: str,
    output: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(frame[x], frame[y], marker="o")
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def generate_report(results_dir: Path) -> None:
    stability_path = results_dir / "stability_summary.csv"
    agreement_path = results_dir / "agreement_summary.csv"

    stability = pd.read_csv(stability_path)
    agreement = pd.read_csv(agreement_path)

    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    for method in stability["method"].unique():
        subset = stability[stability["method"] == method].sort_values("ratio")
        _plot_metric(
            subset,
            x="ratio",
            y="mean_rank_corr",
            title=f"{method.upper()} rank stability",
            output=plots_dir / f"{method}_rank_stability.png",
        )
        _plot_metric(
            subset,
            x="ratio",
            y="mean_magnitude_var",
            title=f"{method.upper()} magnitude variance",
            output=plots_dir / f"{method}_magnitude_variance.png",
        )

    agreement_sorted = agreement.sort_values("ratio")
    _plot_metric(
        agreement_sorted,
        x="ratio",
        y="mean_spearman",
        title="SHAP vs PFI Spearman agreement",
        output=plots_dir / "agreement_spearman.png",
    )
    _plot_metric(
        agreement_sorted,
        x="ratio",
        y="mean_topk_overlap",
        title="SHAP vs PFI top-k overlap",
        output=plots_dir / "agreement_topk_overlap.png",
    )
    _plot_metric(
        agreement_sorted,
        x="ratio",
        y="mean_cosine",
        title="SHAP vs PFI cosine agreement",
        output=plots_dir / "agreement_cosine.png",
    )

    stability.to_csv(results_dir / "stability_table.csv", index=False)
    agreement.to_csv(results_dir / "agreement_table.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate MVS plots/tables")
    parser.add_argument("results_dir", help="Path to MVS results directory")
    args = parser.parse_args()
    generate_report(Path(args.results_dir))
