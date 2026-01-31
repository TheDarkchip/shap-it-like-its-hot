from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from sklearn.preprocessing import StandardScaler

import xgboost as xgb

from german_credit import load_german_credit


ModelType = Literal["tree", "linear"]

OUT_DIR = "./results"
PER_FOLD_CSV = os.path.join(OUT_DIR, "step2_per_fold.csv")
SUMMARY_CSV = os.path.join(OUT_DIR, "step2_predictive_summary.csv")


# -----------------------------
# Logging helpers
# -----------------------------
def log(msg: str) -> None:
    print(msg, flush=True)


def fmt_secs(s: float) -> str:
    if s < 60:
        return f"{s:.1f}s"
    m = int(s // 60)
    r = s - 60 * m
    return f"{m}m{r:.0f}s"


# -----------------------------
# 1) Data Load (RAW, no one-hot here)
# -----------------------------
def load_german_raw(path: str | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """
    Loads german.data via german_credit.py.
    Returns:
      X_raw: DataFrame with categorical + numeric
      y: int Series, bad=1 (positive), good=0
    """
    log("[stage] Loading dataset (load_german_credit)...")
    X_raw, y = load_german_credit(path, target_positive="bad")

    log(f"[check] X_raw shape: {X_raw.shape}")
    pos = float(y.mean())
    log(f"[check] y positive rate (bad=1): {pos:.3f} ({int(y.sum())}/{len(y)})")
    return X_raw, y


def onehot_align_train_test(
    X_train_raw: pd.DataFrame,
    X_test_raw: pd.DataFrame,
    *,
    drop_first: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    One-hot encode *within a fold*:
      - get_dummies on train and test separately
      - align test columns to train columns
      - fill unseen categories in test with 0
      - drop categories that only appear in test (since train doesn't know them)

    This avoids the "encode once before CV" leakage and prevents column mismatch.
    """
    X_train_oh = pd.get_dummies(X_train_raw, drop_first=drop_first)
    X_test_oh = pd.get_dummies(X_test_raw, drop_first=drop_first)

    # Align columns: test must match training columns exactly
    X_test_oh = X_test_oh.reindex(columns=X_train_oh.columns, fill_value=0)

    # Make sure both are float for xgboost
    X_train_oh = X_train_oh.astype(float)
    X_test_oh = X_test_oh.astype(float)

    return X_train_oh, X_test_oh


# -----------------------------
# 2) Resample TRAIN only to ratio
# -----------------------------
def resample_train_to_ratio(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    pos_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Resamples the training fold to have approx pos_ratio positives, keeping
    total size fixed at len(y_train). Uses replacement only if needed.
    """
    if not (0.0 < pos_ratio < 1.0):
        raise ValueError("pos_ratio must be in (0,1).")

    rng = np.random.default_rng(seed)
    yv = y_train.to_numpy()

    idx_pos = np.flatnonzero(yv == 1)
    idx_neg = np.flatnonzero(yv == 0)

    n_total = len(y_train)
    n_pos = int(round(pos_ratio * n_total))
    n_neg = n_total - n_pos

    replace_pos = n_pos > len(idx_pos)
    replace_neg = n_neg > len(idx_neg)

    samp_pos = rng.choice(idx_pos, size=n_pos, replace=replace_pos)
    samp_neg = rng.choice(idx_neg, size=n_neg, replace=replace_neg)

    idx = np.concatenate([samp_pos, samp_neg])
    rng.shuffle(idx)

    return (
        X_train.iloc[idx].reset_index(drop=True),
        y_train.iloc[idx].reset_index(drop=True),
    )


# -----------------------------
# 3) Model + HPO
# -----------------------------
def make_xgb_params(model_type: ModelType, params: dict, random_seed: int) -> dict:
    base = dict(
        objective="binary:logistic",
        eval_metric="auc",
        random_state=random_seed,
        n_jobs=-1,
        verbosity=0,
    )
    if model_type == "tree":
        base.update(dict(booster="gbtree"))
    elif model_type == "linear":
        base.update(dict(booster="gblinear"))
    else:
        raise ValueError("model_type must be 'tree' or 'linear'.")
    base.update(params)
    return base


def sample_param_candidates(model_type: ModelType, n_samples: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)

    if model_type == "tree":
        grid = {
            "n_estimators": [200, 400, 600],
            "max_depth": [2, 3, 4, 5],
            "learning_rate": [0.03, 0.05, 0.1],
            "subsample": [0.7, 0.85, 1.0],
            "colsample_bytree": [0.7, 0.85, 1.0],
            "min_child_weight": [1, 3, 5],
            "reg_lambda": [0.5, 1.0, 2.0],
        }
    elif model_type == "linear":
        grid = {
            "n_estimators": [200, 400, 800],
            "learning_rate": [0.03, 0.05, 0.1],
            "reg_lambda": [0.5, 1.0, 2.0, 5.0],
            "reg_alpha": [0.0, 0.01, 0.1],
        }
    else:
        raise ValueError("model_type must be 'tree' or 'linear'.")

    keys = list(grid.keys())
    out: list[dict] = []
    for _ in range(n_samples):
        out.append({k: rng.choice(grid[k]).item() for k in keys})
    return out


def mean_auc_inner_3fold(
    X: np.ndarray,
    y: np.ndarray,
    model_type: ModelType,
    params: dict,
    seed: int,
) -> float:
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    aucs: list[float] = []

    for tr, va in skf.split(X, y):
        X_tr, X_va = X[tr], X[va]
        y_tr, y_va = y[tr], y[va]

        if model_type == "linear":
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_va = scaler.transform(X_va)

        clf = xgb.XGBClassifier(**make_xgb_params(model_type, params, random_seed=seed))
        clf.fit(X_tr, y_tr)

        p = clf.predict_proba(X_va)[:, 1]
        aucs.append(float(roc_auc_score(y_va, p)))

    return float(np.mean(aucs))


def tune_hpo_inner(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: ModelType,
    n_candidates: int,
    seed: int,
    *,
    verbose: bool = True,
) -> dict:
    Xn = X_train.to_numpy()
    yn = y_train.to_numpy()

    cands = sample_param_candidates(model_type, n_samples=n_candidates, seed=seed)
    best_score = -np.inf
    best_params: dict = {}

    t0 = time.time()
    for i, params in enumerate(cands, start=1):
        score = mean_auc_inner_3fold(Xn, yn, model_type=model_type, params=params, seed=seed)
        if score > best_score:
            best_score = score
            best_params = params
        if verbose:
            log(f"    [hpo] {model_type} cand {i:02d}/{n_candidates} AUC={score:.4f} best={best_score:.4f}")

    if verbose:
        log(f"    [hpo] done in {fmt_secs(time.time() - t0)}; best={best_score:.4f}")

    return best_params


# -----------------------------
# 4) Step 2 protocol runner
# -----------------------------
@dataclass(frozen=True)
class FoldRow:
    ratio: float
    model: str
    repeat: int
    fold: int
    split_id: int
    n_train: int
    n_test: int
    train_pos_ratio: float
    auc: float
    pr_auc: float
    accuracy: float
    best_params: str
    seconds: float


def run_step2_protocol(
    *,
    data_path: str | None = None,
    ratios: list[float] = [0.10, 0.30, 0.50],
    model_types: list[ModelType] = ["tree", "linear"],
    n_splits: int = 5,
    n_repeats: int = 5,
    hpo_candidates: int = 10,
    seed: int = 42,
    progress_every_split: int = 1,
) -> tuple[pd.DataFrame, float]:
    """
    Returns:
      per_fold_df: per-fold results (150 rows = 25 splits × 3 ratios × 2 models)
      prevalence: overall dataset prevalence (bad=1) used for plot baselines
    """
    t_all = time.time()

    # Load RAW X (no one-hot here)
    X_raw, y = load_german_raw(data_path)
    prevalence = float(y.mean())

    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    outer_splits = list(rskf.split(X_raw, y))
    total_splits = len(outer_splits)

    total_jobs = total_splits * len(ratios) * len(model_types)
    log(f"[stage] Outer CV: {n_splits} folds × {n_repeats} repeats = {total_splits} splits")
    log(f"[stage] Total runs (split × ratio × model): {total_jobs}")

    rows: list[FoldRow] = []

    for split_id, (tr_idx, te_idx) in enumerate(outer_splits, start=1):
        t_split = time.time()

        repeat = (split_id - 1) // n_splits + 1
        fold = (split_id - 1) % n_splits + 1

        # Raw folds
        X_tr0_raw = X_raw.iloc[tr_idx].reset_index(drop=True)
        y_tr0 = y.iloc[tr_idx].reset_index(drop=True)
        X_te_raw = X_raw.iloc[te_idx].reset_index(drop=True)
        y_te = y.iloc[te_idx].reset_index(drop=True)

        # Fold-safe one-hot encoding + alignment
        X_tr0, X_te = onehot_align_train_test(X_tr0_raw, X_te_raw, drop_first=False)

        if split_id % progress_every_split == 1 or split_id == 1:
            log(
                f"\n[outer] split {split_id}/{total_splits} (repeat={repeat}, fold={fold}) "
                f"train={len(y_tr0)} test={len(y_te)} X_train_cols={X_tr0.shape[1]}"
            )

        for ratio in ratios:
            X_tr, y_tr = resample_train_to_ratio(
                X_tr0,
                y_tr0,
                pos_ratio=ratio,
                seed=seed + 10_000 * split_id + int(ratio * 1000),
            )
            log(f"  [ratio] target={ratio:.2f} achieved_train_pos={y_tr.mean():.3f}")

            for model_type in model_types:
                log(f"  [model] {model_type} -> HPO ({hpo_candidates} candidates, inner 3-fold AUC)")

                best_params = tune_hpo_inner(
                    X_tr,
                    y_tr,
                    model_type=model_type,
                    n_candidates=hpo_candidates,
                    seed=seed + 20_000 * split_id + int(ratio * 1000) + (1 if model_type == "tree" else 2),
                    verbose=True,
                )

                X_tr_np = X_tr.to_numpy()
                y_tr_np = y_tr.to_numpy()
                X_te_np = X_te.to_numpy()
                y_te_np = y_te.to_numpy()

                if model_type == "linear":
                    log("    [fit] standardizing (linear only)")
                    scaler = StandardScaler()
                    X_tr_np = scaler.fit_transform(X_tr_np)
                    X_te_np = scaler.transform(X_te_np)

                log("    [fit] training final model...")
                clf = xgb.XGBClassifier(**make_xgb_params(model_type, best_params, random_seed=seed))
                clf.fit(X_tr_np, y_tr_np)

                log("    [eval] computing metrics on test fold...")
                p = clf.predict_proba(X_te_np)[:, 1]
                yhat = (p >= 0.5).astype(int)

                auc = float(roc_auc_score(y_te_np, p))
                pr_auc = float(average_precision_score(y_te_np, p))
                acc = float(accuracy_score(y_te_np, yhat))
                log(f"    [done] AUC={auc:.4f} PR-AUC={pr_auc:.4f} Acc={acc:.4f}")

                rows.append(
                    FoldRow(
                        ratio=float(ratio),
                        model=str(model_type),
                        repeat=int(repeat),
                        fold=int(fold),
                        split_id=int(split_id),
                        n_train=int(len(y_tr)),
                        n_test=int(len(y_te)),
                        train_pos_ratio=float(y_tr.mean()),
                        auc=auc,
                        pr_auc=pr_auc,
                        accuracy=acc,
                        best_params=json.dumps(best_params, sort_keys=True),
                        seconds=float(time.time() - t_split),
                    )
                )

        log(f"[outer] split {split_id}/{total_splits} finished in {fmt_secs(time.time() - t_split)}")

    log(f"\n[stage] All done in {fmt_secs(time.time() - t_all)}")
    per_fold_df = pd.DataFrame([r.__dict__ for r in rows])
    return per_fold_df, prevalence


# -----------------------------
# 5) Aggregate + Plot
# -----------------------------
def aggregate_predictive_performance(df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df.groupby(["ratio", "model"], as_index=False)
        .agg(
            auc_mean=("auc", "mean"),
            auc_std=("auc", "std"),
            pr_auc_mean=("pr_auc", "mean"),
            pr_auc_std=("pr_auc", "std"),
            acc_mean=("accuracy", "mean"),
            acc_std=("accuracy", "std"),
            n=("auc", "count"),
        )
    )
    return agg.fillna(0.0)


def _boxplot_by_ratio_model(
    df: pd.DataFrame,
    metric: str,
    title: str,
    out_path: str,
    *,
    baseline: float | None = None,
    baseline_label: str | None = None,
) -> None:
    ratios = sorted(df["ratio"].unique().tolist())
    models = ["tree", "linear"]

    data: list[np.ndarray] = []
    labels: list[str] = []
    positions: list[float] = []
    pos = 1.0

    for r in ratios:
        for m in models:
            vals = df.loc[(df["ratio"] == r) & (df["model"] == m), metric].values
            data.append(vals)
            labels.append(f"{r:.1f}\n{m}")
            positions.append(pos)
            pos += 1.0
        pos += 0.75

    plt.figure(figsize=(11, 5))
    plt.boxplot(data, positions=positions, widths=0.6, showfliers=True)
    plt.xticks(positions, labels)
    plt.ylabel(metric)
    plt.title(title)
    plt.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)

    if metric == "auc":
        plt.axhline(0.5, linestyle="--", linewidth=1)

    if baseline is not None:
        plt.axhline(baseline, linestyle="--", linewidth=1)
        if baseline_label:
            plt.text(0.5, baseline, f"  {baseline_label}", va="bottom")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_step2_results(df: pd.DataFrame, prevalence: float) -> None:
    acc_baseline = max(prevalence, 1.0 - prevalence)

    _boxplot_by_ratio_model(
        df,
        metric="auc",
        title="Test ROC-AUC across outer folds (by ratio & model)",
        out_path=os.path.join(OUT_DIR, "step2_auc_boxplot.png"),
    )
    _boxplot_by_ratio_model(
        df,
        metric="pr_auc",
        title="Test PR-AUC across outer folds (by ratio & model)",
        out_path=os.path.join(OUT_DIR, "step2_pr_auc_boxplot.png"),
        baseline=prevalence,
        baseline_label=f"PR baseline = prevalence = {prevalence:.2f}",
    )
    _boxplot_by_ratio_model(
        df,
        metric="accuracy",
        title="Test Accuracy across outer folds (by ratio & model)",
        out_path=os.path.join(OUT_DIR, "step2_accuracy_boxplot.png"),
        baseline=acc_baseline,
        baseline_label=f"Acc baseline = max(p,1-p) = {acc_baseline:.2f}",
    )


# -----------------------------
# 6) Main
# -----------------------------
OUT_DIR = "./results"
def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    log("[start] Running Step 2 protocol...")
    per_fold_df, prevalence = run_step2_protocol(
        data_path=None,
        ratios=[0.10, 0.30, 0.50],
        model_types=["tree", "linear"],
        n_splits=5,
        n_repeats=5,
        hpo_candidates=10,
        seed=42,
        progress_every_split=1,
    )

    log(f"[stage] Writing per-fold results -> {PER_FOLD_CSV}")
    per_fold_df.to_csv(PER_FOLD_CSV, index=False)
    log("[ok] Saved per-fold results.")

    summary_df = aggregate_predictive_performance(per_fold_df)
    log(f"[stage] Writing predictive summary -> {SUMMARY_CSV}")
    summary_df.to_csv(SUMMARY_CSV, index=False)
    log("[ok] Saved predictive summary.\n")

    log("[preview] Predictive summary:")
    print(summary_df.sort_values(["ratio", "model"]).to_string(index=False), flush=True)

    log("[stage] Plotting results with baselines...")
    plot_step2_results(per_fold_df, prevalence=prevalence)

    log("[ok] Saved plots:")
    log(f" - {os.path.join(OUT_DIR, 'step2_auc_boxplot.png')}")
    log(f" - {os.path.join(OUT_DIR, 'step2_pr_auc_boxplot.png')}")
    log(f" - {os.path.join(OUT_DIR, 'step2_accuracy_boxplot.png')}")


if __name__ == "__main__":
    main()