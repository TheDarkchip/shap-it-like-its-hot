"""Generate MVS plots and summary tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
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


def _plot_metric_with_error(
    frame: pd.DataFrame,
    *,
    x: str,
    y: str,
    yerr: pd.Series,
    title: str,
    x_label: str,
    y_label: str,
    note: str | None,
    reference_lines: list[float] | None,
    use_scientific: bool = False,
    y_limits: tuple[float, float] | None = None,
    x_ticks: list[float] | None = None,
    output: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(frame[x], frame[y], yerr=yerr, marker="o", capsize=4, linestyle="none")
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if x_ticks is not None:
        ax.set_xticks(x_ticks, [f"{value:.2f}" for value in x_ticks])
    if reference_lines:
        for value in reference_lines:
            ax.axhline(value, color="gray", linestyle="--", linewidth=1, alpha=0.4)
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    ax.grid(True, alpha=0.3)
    if use_scientific:
        ax.ticklabel_format(axis="y", style="plain")
    if note:
        fig.text(0.5, 0.01, note, ha="center", va="bottom", fontsize=9)
        fig.tight_layout(rect=[0, 0.04, 1, 1])
    else:
        fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def _plot_distribution(
    frame: pd.DataFrame,
    *,
    x: str,
    y: str,
    mean_frame: pd.DataFrame,
    mean_x: str,
    mean_y: str,
    title: str,
    x_label: str,
    y_label: str,
    note: str | None,
    reference_lines: list[float] | None,
    y_limits: tuple[float, float] | None = None,
    output: Path,
) -> None:
    ratios = sorted(frame[x].unique())
    fig, ax = plt.subplots(figsize=(6, 4))
    data = [frame.loc[frame[x] == ratio, y].dropna() for ratio in ratios]
    box = ax.boxplot(
        data,
        positions=range(len(ratios)),
        widths=0.6,
        patch_artist=True,
        boxprops={"facecolor": "#9ecae1", "alpha": 0.6},
        medianprops={"color": "black"},
        whiskerprops={"color": "#555555"},
        capprops={"color": "#555555"},
    )
    means = [
        float(mean_frame.loc[mean_frame[mean_x] == ratio, mean_y].iloc[0])
        for ratio in ratios
    ]
    ax.scatter(range(len(ratios)), means, color="black", zorder=3, label="mean")
    ax.set_xticks(range(len(ratios)), [f"{ratio:.2f}" for ratio in ratios])
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    if reference_lines:
        for value in reference_lines:
            ax.axhline(value, color="gray", linestyle="--", linewidth=1, alpha=0.4)
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    ax.grid(True, axis="y", alpha=0.3)
    median_handle = plt.Line2D([0], [0], color="black", linewidth=2, label="median")
    handles = [median_handle, ax.collections[0]]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5))
    if note:
        fig.text(0.5, 0.01, note, ha="center", va="bottom", fontsize=9)
        fig.tight_layout(rect=[0, 0.04, 1, 1])
    else:
        fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def _collect_importances(frame: pd.DataFrame, *, prefix: str, ratio: float) -> pd.DataFrame:
    subset = frame[frame["class_ratio"] == ratio]
    cols = [col for col in subset.columns if col.startswith(prefix)]
    if not cols:
        raise ValueError(f"No columns found for prefix {prefix}")
    matrices: list[pd.Series] = []
    for _, row in subset.iterrows():
        series = row[cols]
        series.index = [c.removeprefix(prefix) for c in cols]
        matrices.append(series.astype(float))
    return pd.concat(matrices, axis=1)


def _rank_vector(values: pd.Series) -> pd.Series:
    return values.rank(ascending=False, method="average")


def _mean_rank_corr_per_fold(values: pd.DataFrame) -> list[float]:
    ranks = values.apply(_rank_vector, axis=0)
    corr = ranks.corr(method="spearman")
    per_fold: list[float] = []
    for col in corr.columns:
        others = corr[col].drop(index=col)
        per_fold.append(float(others.mean()))
    return per_fold


def _normalize_columns(values: pd.DataFrame) -> pd.DataFrame:
    sums = values.sum(axis=0)
    normalized = values.copy()
    for idx, col in enumerate(values.columns):
        total = sums.iloc[idx]
        if total == 0:
            normalized.iloc[:, idx] = np.nan
        else:
            normalized.iloc[:, idx] = values.iloc[:, idx] / total
    return normalized


def _magnitude_variance(values: pd.DataFrame) -> float:
    normalized = _normalize_columns(values)
    return float(normalized.var(axis=1, ddof=1).mean())


def _bootstrap_magnitude_variance(
    values: pd.DataFrame,
    *,
    n_boot: int = 200,
    seed: int = 0,
) -> float:
    if values.shape[1] < 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    cols = list(values.columns)
    stats = []
    for _ in range(n_boot):
        sampled = rng.choice(cols, size=len(cols), replace=True)
        stats.append(_magnitude_variance(values[sampled]))
    return float(np.std(stats, ddof=1))


def _topk_overlap(a: pd.Series, b: pd.Series, k: int) -> float:
    a_clean = a.dropna()
    b_clean = b.dropna()
    top_a = set(a_clean.nlargest(k).index)
    top_b = set(b_clean.nlargest(k).index)
    denom = min(k, len(a_clean), len(b_clean))
    if denom == 0:
        return float("nan")
    return len(top_a.intersection(top_b)) / float(denom)


def _cosine_similarity(a: pd.Series, b: pd.Series) -> float:
    common = a.index.intersection(b.index)
    if common.empty:
        return float("nan")
    a_vec = a.loc[common].to_numpy(dtype=float)
    b_vec = b.loc[common].to_numpy(dtype=float)
    denom = np.linalg.norm(a_vec) * np.linalg.norm(b_vec)
    if denom == 0:
        return float("nan")
    return float(np.dot(a_vec, b_vec) / denom)


def _agreement_per_fold(
    shap_values: pd.DataFrame,
    pfi_values: pd.DataFrame,
    *,
    top_k: int = 5,
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for col in shap_values.columns:
        shap_vec = shap_values[col].abs()
        pfi_vec = pfi_values[col].abs()
        corr = shap_vec.corr(pfi_vec, method="spearman")
        rows.append(
            {
                "spearman": float(corr) if corr is not None else float("nan"),
                "topk_overlap": _topk_overlap(shap_vec, pfi_vec, top_k),
                "cosine": _cosine_similarity(shap_vec, pfi_vec),
            }
        )
    return pd.DataFrame(rows)


def generate_report(results_dir: Path) -> None:
    stability_path = results_dir / "stability_summary.csv"
    agreement_path = results_dir / "agreement_summary.csv"
    results_path = results_dir / "results.csv"
    metadata_path = results_dir / "run_metadata.json"

    stability = pd.read_csv(stability_path)
    agreement = pd.read_csv(agreement_path)
    results = pd.read_csv(results_path)
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing run metadata at {metadata_path}; cannot determine top-k."
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if "agreement_top_k" not in metadata:
        raise KeyError(
            "run_metadata.json missing 'agreement_top_k'; regenerate run metadata before plotting."
        )
    top_k = int(metadata["agreement_top_k"])

    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    ratios = sorted(results["class_ratio"].unique())
    ratio_label = "Target positive class ratio (train resampling, fraction)"
    fold_note = "Box=IQR, line=median, dot=mean"
    ratio_ticks = [float(ratio) for ratio in ratios]

    for method in stability["method"].unique():
        subset = stability[stability["method"] == method].sort_values("ratio")
        rank_rows = []
        n_folds = None
        for ratio in ratios:
            values = _collect_importances(results, prefix=f"{method}_", ratio=ratio)
            n_folds = values.shape[1]
            per_fold = _mean_rank_corr_per_fold(values)
            rank_rows.extend(
                {"ratio": ratio, "mean_rank_corr": value} for value in per_fold
            )
        rank_frame = pd.DataFrame(rank_rows)
        _plot_distribution(
            rank_frame,
            x="ratio",
            y="mean_rank_corr",
            mean_frame=subset,
            mean_x="ratio",
            mean_y="mean_rank_corr",
            title=f"{method.upper()} rank stability (fold variability, n={n_folds})",
            x_label=ratio_label,
            y_label="Fold-to-fold rank correlation (Spearman)",
            note=fold_note,
            reference_lines=None,
            output=plots_dir / f"{method}_rank_stability.png",
        )
        mag_rows = []
        for ratio in ratios:
            values = _collect_importances(results, prefix=f"{method}_", ratio=ratio)
            magnitude_values = values.abs() if method == "pfi" else values
            mag_rows.append(
                {
                    "ratio": ratio,
                    "sd_magnitude_var": _bootstrap_magnitude_variance(magnitude_values),
                }
            )
        mag_frame = pd.DataFrame(mag_rows)
        _plot_metric_with_error(
            subset,
            x="ratio",
            y="mean_magnitude_var",
            yerr=mag_frame.sort_values("ratio")["sd_magnitude_var"],
            title=f"{method.upper()} magnitude variance (mean ± bootstrap SD)",
            x_label=ratio_label,
            y_label="Mean magnitude variance",
            note="Point = mean over folds; error bars = bootstrap SD (pooled across folds)",
            reference_lines=None,
            use_scientific=True,
            x_ticks=ratio_ticks,
            output=plots_dir / f"{method}_magnitude_variance.png",
        )

    agreement_sorted = agreement.sort_values("ratio")
    agreement_rows = []
    n_folds = None
    for ratio in ratios:
        shap_values = _collect_importances(results, prefix="shap_", ratio=ratio)
        pfi_values = _collect_importances(results, prefix="pfi_", ratio=ratio)
        n_folds = shap_values.shape[1]
        per_fold = _agreement_per_fold(shap_values, pfi_values, top_k=top_k)
        per_fold["ratio"] = ratio
        agreement_rows.append(per_fold)
    agreement_frame = pd.concat(agreement_rows, ignore_index=True)
    _plot_distribution(
        agreement_frame,
        x="ratio",
        y="spearman",
        mean_frame=agreement_sorted,
        mean_x="ratio",
        mean_y="mean_spearman",
        title=f"SHAP vs PFI Spearman agreement (fold variability, n={n_folds})",
        x_label=ratio_label,
        y_label="Spearman correlation",
        note=fold_note,
        reference_lines=None,
        output=plots_dir / "agreement_spearman.png",
    )
    _plot_distribution(
        agreement_frame,
        x="ratio",
        y="topk_overlap",
        mean_frame=agreement_sorted,
        mean_x="ratio",
        mean_y="mean_topk_overlap",
        title=f"SHAP vs PFI top-k overlap (k={top_k}, n={n_folds})",
        x_label=ratio_label,
        y_label="Top-k overlap (|intersection|/k)",
        note=fold_note,
        reference_lines=None,
        output=plots_dir / "agreement_topk_overlap.png",
    )
    _plot_distribution(
        agreement_frame,
        x="ratio",
        y="cosine",
        mean_frame=agreement_sorted,
        mean_x="ratio",
        mean_y="mean_cosine",
        title=f"SHAP vs PFI cosine agreement (fold variability, n={n_folds})",
        x_label=ratio_label,
        y_label="Cosine similarity",
        note=fold_note,
        reference_lines=None,
        output=plots_dir / "agreement_cosine.png",
    )

    stability.to_csv(results_dir / "stability_table.csv", index=False)
    agreement.to_csv(results_dir / "agreement_table.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate MVS plots/tables")
    parser.add_argument("results_dir", help="Path to MVS results directory")
    args = parser.parse_args()
    generate_report(Path(args.results_dir))
