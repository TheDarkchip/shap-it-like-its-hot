# step5_make_report_artifacts.py
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Logging
# -----------------------------
def log(msg: str) -> None:
    print(msg, flush=True)


# -----------------------------
# IO helpers
# -----------------------------
def read_json(path: str | os.PathLike[str]) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | os.PathLike[str], obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def ensure_dir(path: str | os.PathLike[str]) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def require_columns(df: pd.DataFrame, required: set[str], where: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{where} is missing required columns: {sorted(missing)}")


def pick_first_existing(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def fmt_mean_std(mean: float, std: float, ndigits: int = 3) -> str:
    return f"{mean:.{ndigits}f} ± {std:.{ndigits}f}"


# -----------------------------
# Plot helpers (matplotlib only)
# -----------------------------
def bar_with_error(
    df: pd.DataFrame,
    mean_col: str,
    std_col: str,
    *,
    title: str,
    ylabel: str,
    out_path: str,
    baseline: Optional[float] = None,
    baseline_label: Optional[str] = None,
) -> None:
    labels = df["label"].tolist()
    means = df[mean_col].to_numpy(dtype=float)
    stds = df[std_col].to_numpy(dtype=float)

    x = np.arange(len(labels))

    plt.figure(figsize=(10, 4.8))
    plt.bar(x, means, yerr=stds, capsize=4)
    plt.xticks(x, labels, rotation=0)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)

    if baseline is not None:
        plt.axhline(baseline, linestyle="--", linewidth=1)
        if baseline_label:
            ymax = max(float(np.max(means + stds)), baseline)
            plt.text(
                len(labels) - 0.5,
                baseline,
                f"  {baseline_label}={baseline:.3f}",
                va="bottom",
                ha="left",
                fontsize=9,
            )
            plt.ylim(bottom=min(0.0, float(np.min(means - stds)) - 0.02), top=ymax + 0.05)

    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def topk_hbar(
    df: pd.DataFrame,
    value_col: str,
    *,
    title: str,
    xlabel: str,
    out_path: str,
    k: int = 20,
) -> pd.DataFrame:
    work = df[["feature", value_col]].copy()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce").fillna(0.0)
    work = work.sort_values(value_col, ascending=False).head(k)

    # reverse for nicer plot (top at top)
    work_plot = work.iloc[::-1]

    plt.figure(figsize=(10, 6.2))
    plt.barh(work_plot["feature"].tolist(), work_plot[value_col].to_numpy(dtype=float))
    plt.xlabel(xlabel)
    plt.title(title)
    plt.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()

    return work


# -----------------------------
# Report writer
# -----------------------------
def write_markdown_report(
    *,
    out_path: str,
    merged: pd.DataFrame,
    best_config: dict,
    holdout: dict,
    importance_csv: str,
    plots: list[str],
) -> None:
    # Best config (Step4_best_config.json is usually {"ratio":..., "model":...})
    best_ratio = best_config.get("ratio", best_config.get("best_ratio"))
    best_model = best_config.get("model", best_config.get("best_model"))

    # Holdout metrics (handle both naming styles)
    auc = holdout.get("auc", holdout.get("holdout_auc"))
    pr_auc = holdout.get("pr_auc", holdout.get("holdout_pr_auc"))
    acc = holdout.get("accuracy", holdout.get("holdout_accuracy"))

    prev = holdout.get("test_prevalence", holdout.get("prevalence", None))
    pr_base = holdout.get("pr_auc_baseline", (float(prev) if prev is not None else None))
    acc_base = holdout.get("accuracy_baseline", (max(float(prev), 1.0 - float(prev)) if prev is not None else None))

    spearman_holdout = holdout.get("shap_pfi_spearman_holdout", holdout.get("spearman_shap_pfi_holdout", None))

    def md_table(df: pd.DataFrame) -> str:
        return df.to_markdown(index=False)

    # choose formatted columns if available
    perf_cols = [c for c in ["auc_fmt", "pr_auc_fmt", "acc_fmt"] if c in merged.columns]
    explain_cols = [c for c in ["spearman_fmt", "top5_fmt"] if c in merged.columns]

    perf_tbl = merged[["ratio", "model"] + perf_cols].copy() if perf_cols else pd.DataFrame()
    exp_tbl = merged[["ratio", "model"] + explain_cols].copy() if explain_cols else pd.DataFrame()

    lines: list[str] = []
    lines.append("# Step 5 Report Artifacts")
    lines.append("")
    lines.append("## Selected configuration")
    lines.append(f"- Best config: **ratio={best_ratio}**, **model={best_model}**")
    lines.append("")

    lines.append("## Holdout performance (final model)")
    if auc is not None:
        lines.append(f"- AUC: **{float(auc):.4f}** (baseline 0.5000)")
    if pr_auc is not None:
        if pr_base is not None:
            lines.append(f"- PR-AUC: **{float(pr_auc):.4f}** (baseline prevalence {float(pr_base):.4f})")
        else:
            lines.append(f"- PR-AUC: **{float(pr_auc):.4f}**")
    if acc is not None:
        if acc_base is not None:
            lines.append(f"- Accuracy: **{float(acc):.4f}** (baseline {float(acc_base):.4f})")
        else:
            lines.append(f"- Accuracy: **{float(acc):.4f}**")
    if prev is not None:
        lines.append(f"- Holdout prevalence (positive rate): {float(prev):.4f}")
    if spearman_holdout is not None:
        lines.append(f"- Spearman(SHAP vs PFI) on holdout: {float(spearman_holdout):.4f}")
    lines.append("")

    lines.append("## Cross-validated summary (mean ± std across 25 folds)")
    lines.append("")
    if not perf_tbl.empty:
        lines.append("### Predictive performance")
        lines.append(md_table(perf_tbl))
        lines.append("")
    if not exp_tbl.empty:
        lines.append("### Explainability agreement")
        lines.append(md_table(exp_tbl))
        lines.append("")

    lines.append("## Final feature importance")
    lines.append(f"- Source: `{importance_csv}`")
    lines.append("")
    lines.append("## Generated plots")
    for p in plots:
        lines.append(f"- `{p}`")
    lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Step 5: generate report-ready tables + plots from Step 4 outputs.")
    ap.add_argument("--out_dir", default="./results")
    ap.add_argument("--merged_summary", default="./results/step4_merged_summary.csv")
    ap.add_argument("--best_config", default="./results/step4_best_config.json")
    ap.add_argument("--holdout_metrics", default="./results/step4_final_holdout_metrics.json")
    ap.add_argument("--feature_importance", default="./results/step4_final_feature_importance.csv")
    ap.add_argument("--topk", type=int, default=20)
    args = ap.parse_args()

    out_dir = args.out_dir
    ensure_dir(out_dir)

    log("[start] Step 5: generate report artifacts")

    # --- Load inputs
    if not os.path.exists(args.merged_summary):
        raise FileNotFoundError(f"Missing merged summary: {args.merged_summary}")
    if not os.path.exists(args.best_config):
        raise FileNotFoundError(f"Missing best config json: {args.best_config}")
    if not os.path.exists(args.holdout_metrics):
        raise FileNotFoundError(f"Missing holdout metrics json: {args.holdout_metrics}")
    if not os.path.exists(args.feature_importance):
        raise FileNotFoundError(f"Missing feature importance csv: {args.feature_importance}")

    merged = pd.read_csv(args.merged_summary)
    best_cfg = read_json(args.best_config)
    holdout = read_json(args.holdout_metrics)
    imp = pd.read_csv(args.feature_importance)

    require_columns(merged, {"ratio", "model"}, where=args.merged_summary)
    require_columns(imp, {"feature"}, where=args.feature_importance)

    # --- Ensure mean/std columns exist (fill std=0 if not present)
    expected_pairs = [
        ("auc_mean", "auc_std"),
        ("pr_auc_mean", "pr_auc_std"),
        ("acc_mean", "acc_std"),
        ("spearman_mean", "spearman_std"),
        ("top5_mean", "top5_std"),
    ]
    for mean_c, std_c in expected_pairs:
        if mean_c in merged.columns and std_c not in merged.columns:
            merged[std_c] = 0.0

    # --- Stable ordering + labels
    merged = merged.copy()
    merged["ratio"] = pd.to_numeric(merged["ratio"], errors="coerce")
    merged["model"] = merged["model"].astype(str)
    merged = merged.sort_values(["ratio", "model"]).reset_index(drop=True)
    merged["label"] = merged.apply(lambda r: f"r={r['ratio']:.1f}\n{r['model']}", axis=1)

    # --- Baselines
    prev = holdout.get("test_prevalence", holdout.get("prevalence", None))
    try:
        prevalence_f = float(prev) if prev is not None else 0.30
    except Exception:
        prevalence_f = 0.30

    pr_auc_baseline = float(holdout.get("pr_auc_baseline", prevalence_f))
    acc_baseline = float(holdout.get("accuracy_baseline", max(prevalence_f, 1.0 - prevalence_f)))

    log(f"[info] Using baselines: PR-AUC={pr_auc_baseline:.3f}, Accuracy={acc_baseline:.3f}, AUC=0.500")

    # --- Add formatted columns for tables
    if "auc_mean" in merged.columns:
        merged["auc_fmt"] = [
            fmt_mean_std(float(m), float(s))
            for m, s in zip(merged["auc_mean"], merged.get("auc_std", 0.0))
        ]
    if "pr_auc_mean" in merged.columns:
        merged["pr_auc_fmt"] = [
            fmt_mean_std(float(m), float(s))
            for m, s in zip(merged["pr_auc_mean"], merged.get("pr_auc_std", 0.0))
        ]
    if "acc_mean" in merged.columns:
        merged["acc_fmt"] = [
            fmt_mean_std(float(m), float(s))
            for m, s in zip(merged["acc_mean"], merged.get("acc_std", 0.0))
        ]
    if "spearman_mean" in merged.columns:
        merged["spearman_fmt"] = [
            fmt_mean_std(float(m), float(s))
            for m, s in zip(merged["spearman_mean"], merged.get("spearman_std", 0.0))
        ]
    if "top5_mean" in merged.columns:
        merged["top5_fmt"] = [
            fmt_mean_std(float(m), float(s))
            for m, s in zip(merged["top5_mean"], merged.get("top5_std", 0.0))
        ]

    # --- Save a “pretty” table
    pretty_out = os.path.join(out_dir, "step5_report_summary_table.csv")
    merged.to_csv(pretty_out, index=False)
    log(f"[ok] Saved {pretty_out}")

    # --- Plots
    plots: list[str] = []

    def plot_if_available(mean_c: str, std_c: str, title: str, ylabel: str, filename: str, baseline=None, blabel=None):
        if mean_c not in merged.columns:
            log(f"[skip] plot {filename}: missing column {mean_c} in merged summary")
            return
        if std_c not in merged.columns:
            merged[std_c] = 0.0

        out_path = os.path.join(out_dir, filename)
        bar_with_error(
            merged,
            mean_col=mean_c,
            std_col=std_c,
            title=title,
            ylabel=ylabel,
            out_path=out_path,
            baseline=baseline,
            baseline_label=blabel,
        )
        plots.append(out_path)
        log(f"[ok] Saved {out_path}")

    plot_if_available(
        "auc_mean", "auc_std",
        title="Cross-validated ROC-AUC (mean ± std across 25 folds)",
        ylabel="ROC-AUC",
        filename="step5_cv_auc_mean_std.png",
        baseline=0.5,
        blabel="baseline",
    )
    plot_if_available(
        "pr_auc_mean", "pr_auc_std",
        title="Cross-validated PR-AUC (mean ± std across 25 folds)",
        ylabel="PR-AUC",
        filename="step5_cv_pr_auc_mean_std.png",
        baseline=pr_auc_baseline,
        blabel="baseline",
    )
    plot_if_available(
        "acc_mean", "acc_std",
        title="Cross-validated Accuracy (mean ± std across 25 folds)",
        ylabel="Accuracy",
        filename="step5_cv_accuracy_mean_std.png",
        baseline=acc_baseline,
        blabel="baseline",
    )
    plot_if_available(
        "spearman_mean", "spearman_std",
        title="SHAP vs PFI Spearman agreement (mean ± std across 25 folds)",
        ylabel="Spearman correlation",
        filename="step5_cv_spearman_mean_std.png",
        baseline=None,
        blabel=None,
    )
    plot_if_available(
        "top5_mean", "top5_std",
        title="Top-5 overlap (SHAP vs PFI) (mean ± std across 25 folds)",
        ylabel="Top-5 overlap fraction",
        filename="step5_cv_top5_overlap_mean_std.png",
        baseline=None,
        blabel=None,
    )

    # --- Final feature importance plots (detect columns robustly)
    # Step 4 exports: shap_mean_abs, pfi_auc_drop
    shap_col = pick_first_existing(imp, ["shap_importance", "shap_mean_abs", "shap_mean", "shap"])
    pfi_col = pick_first_existing(imp, ["pfi_auc_drop", "pfi_importance", "pfi_mean", "pfi"])

    topk = int(args.topk)
    top_features_rows = []

    if shap_col is not None:
        out_path = os.path.join(out_dir, "step5_final_top_shap.png")
        top_shap = topk_hbar(
            imp, shap_col,
            title=f"Final model: Top-{topk} SHAP-like global importance (mean |contrib|)",
            xlabel="Importance",
            out_path=out_path,
            k=topk,
        )
        plots.append(out_path)
        log(f"[ok] Saved {out_path}")
        top_features_rows.append(top_shap.rename(columns={shap_col: "value"}).assign(method="shap"))
    else:
        log("[warn] Could not find a SHAP importance column in step4_final_feature_importance.csv")

    if pfi_col is not None:
        out_path = os.path.join(out_dir, "step5_final_top_pfi.png")
        top_pfi = topk_hbar(
            imp, pfi_col,
            title=f"Final model: Top-{topk} Permutation Feature Importance (ROC-AUC drop)",
            xlabel="Importance",
            out_path=out_path,
            k=topk,
        )
        plots.append(out_path)
        log(f"[ok] Saved {out_path}")
        top_features_rows.append(top_pfi.rename(columns={pfi_col: "value"}).assign(method="pfi"))
    else:
        log("[warn] Could not find a PFI importance column in step4_final_feature_importance.csv")

    if top_features_rows:
        top_feat = pd.concat(top_features_rows, ignore_index=True)
        top_feat_out = os.path.join(out_dir, "step5_final_top_features_long.csv")
        top_feat.to_csv(top_feat_out, index=False)
        log(f"[ok] Saved {top_feat_out}")

    # --- Markdown report
    md_out = os.path.join(out_dir, "step5_report_tables.md")
    write_markdown_report(
        out_path=md_out,
        merged=merged,
        best_config=best_cfg,
        holdout=holdout,
        importance_csv=args.feature_importance,
        plots=plots,
    )
    log(f"[ok] Saved {md_out}")

    # --- Key numbers JSON (for quick copy/paste)
    key_numbers = {
        "best_config": {
            "ratio": best_cfg.get("ratio", best_cfg.get("best_ratio")),
            "model": best_cfg.get("model", best_cfg.get("best_model")),
        },
        "holdout": {
            "auc": holdout.get("auc", holdout.get("holdout_auc")),
            "pr_auc": holdout.get("pr_auc", holdout.get("holdout_pr_auc")),
            "accuracy": holdout.get("accuracy", holdout.get("holdout_accuracy")),
            "prevalence": holdout.get("test_prevalence", holdout.get("prevalence")),
            "baselines": {
                "auc": 0.5,
                "pr_auc": pr_auc_baseline,
                "accuracy": acc_baseline,
            },
        },
    }
    key_out = os.path.join(out_dir, "step5_key_numbers.json")
    write_json(key_out, key_numbers)
    log(f"[ok] Saved {key_out}")

    log("[done] Step 5 complete")
    log("\nGenerated artifacts:")
    log(f" - {pretty_out}")
    log(f" - {md_out}")
    log(f" - {key_out}")
    for p in plots:
        log(f" - {p}")


if __name__ == "__main__":
    main()