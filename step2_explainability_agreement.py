from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from scipy.stats import spearmanr

from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

import xgboost as xgb

from german_credit import load_german_credit


# ============================================================
# Step 3: Explainability agreement (SHAP vs PFI) + stability
# Fold-safe one-hot encoding (consistent with updated Step 2)
# ============================================================

ModelType = Literal["tree", "linear"]

OUT_DIR = "./results"
PER_FOLD_OUT = os.path.join(OUT_DIR, "step3_per_fold_explain.csv")
SUMMARY_OUT = os.path.join(OUT_DIR, "step3_summary_explain.csv")
STABILITY_OUT = os.path.join(OUT_DIR, "step3_stability.csv")

AUC_PLOT_OUT = os.path.join(OUT_DIR, "step3_auc_boxplot.png")
SPEARMAN_PLOT_OUT = os.path.join(OUT_DIR, "step3_spearman_boxplot.png")
TOP5_PLOT_OUT = os.path.join(OUT_DIR, "step3_top5_overlap_boxplot.png")


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
    X_raw, y = load_german_credit(path, target_positive="bad")
    return X_raw, y


def onehot_align_train_test(
    X_train_raw: pd.DataFrame,
    X_test_raw: pd.DataFrame,
    *,
    drop_first: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fold-safe one-hot:
      - one-hot encode train and test separately
      - align test columns to train columns
      - fill unseen categories in test with 0
    """
    X_train_oh = pd.get_dummies(X_train_raw, drop_first=drop_first)
    X_test_oh = pd.get_dummies(X_test_raw, drop_first=drop_first)

    X_test_oh = X_test_oh.reindex(columns=X_train_oh.columns, fill_value=0)

    return X_train_oh.astype(float), X_test_oh.astype(float)


# -----------------------------
# 2) Resample TRAIN only to ratio
# -----------------------------
def resample_train_to_ratio(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    pos_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.Series]:
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
    return [{k: rng.choice(grid[k]).item() for k in keys} for _ in range(n_samples)]


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
) -> dict:
    Xn = X_train.to_numpy()
    yn = y_train.to_numpy()

    cands = sample_param_candidates(model_type, n_samples=n_candidates, seed=seed)
    best_score = -np.inf
    best_params: dict = {}

    for params in cands:
        score = mean_auc_inner_3fold(Xn, yn, model_type=model_type, params=params, seed=seed)
        if score > best_score:
            best_score = score
            best_params = params

    return best_params


# -----------------------------
# 4) Explainability methods
# -----------------------------
def shap_global_importance_pred_contribs(
    clf: xgb.XGBClassifier,
    X: np.ndarray,
    feature_names: list[str],
) -> np.ndarray:
    """
    XGBoost pred_contribs gives SHAP-like additive contributions.
    Returns mean(|contrib|) per feature (excluding bias term).
    """
    booster = clf.get_booster()
    dmat = xgb.DMatrix(X, feature_names=feature_names)
    contrib = booster.predict(dmat, pred_contribs=True)  # (n, p+1) includes bias
    contrib_no_bias = contrib[:, :-1]
    return np.mean(np.abs(contrib_no_bias), axis=0)


def pfi_importance_auc(
    clf: xgb.XGBClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    n_repeats: int = 10,
) -> np.ndarray:
    """
    Permutation Feature Importance using ROC-AUC scoring.
    """
    res = permutation_importance(
        clf,
        X_test,
        y_test,
        scoring="roc_auc",
        n_repeats=n_repeats,
        random_state=seed,
        n_jobs=-1,
    )
    return res.importances_mean


def topk_overlap(a: np.ndarray, b: np.ndarray, k: int = 5) -> float:
    ia = set(np.argsort(-a)[:k])
    ib = set(np.argsort(-b)[:k])
    return len(ia & ib) / float(k)


def safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    corr = spearmanr(a, b).correlation
    return 0.0 if (corr is None or np.isnan(corr)) else float(corr)


# -----------------------------
# 5) Outputs
# -----------------------------
@dataclass(frozen=True)
class FoldExplainRow:
    ratio: float
    model: str
    repeat: int
    fold: int
    split_id: int
    auc: float
    spearman_shap_pfi: float
    top5_overlap: float
    best_params: str
    seconds: float


def stability_from_importances(imps: np.ndarray) -> tuple[np.ndarray, float]:
    var = np.var(imps, axis=0, ddof=1)
    return var, float(np.mean(var))


# -----------------------------
# 6) Plotting
# -----------------------------
def grouped_boxplot(
    df: pd.DataFrame,
    metric: str,
    title: str,
    out_path: str,
    *,
    baseline: float | None = None,
    baseline_label: str | None = None,
) -> None:
    import matplotlib.pyplot as plt

    ratios = sorted(df["ratio"].unique().tolist())
    models = ["tree", "linear"]

    data: list[np.ndarray] = []
    labels: list[str] = []
    positions: list[float] = []

    pos = 1.0
    for r in ratios:
        for m in models:
            vals = df.loc[(df["ratio"] == r) & (df["model"] == m), metric].to_numpy(dtype=float)
            data.append(vals)
            labels.append(f"{r:.1f}\n{m}")
            positions.append(pos)
            pos += 1.0
        pos += 0.75

    plt.figure(figsize=(10, 5))
    plt.boxplot(data, positions=positions, widths=0.6, showfliers=True)
    plt.xticks(positions, labels)
    plt.ylabel(metric)
    plt.title(title)
    plt.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)

    if baseline is not None:
        plt.axhline(baseline, linestyle="--", linewidth=1)
        if baseline_label:
            x_right = max(positions) + 0.5
            plt.text(x_right, baseline, baseline_label, va="bottom")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def chance_top5_overlap_fraction(p_features: int, k: int = 5) -> float:
    # Expected overlap fraction ≈ k/p
    return 0.0 if p_features <= 0 else float(k) / float(p_features)


# -----------------------------
# 7) Runner
# -----------------------------
def run_step3_explainability(
    *,
    data_path: str | None = None,
    ratios: list[float] = [0.10, 0.30, 0.50],
    model_types: list[ModelType] = ["tree", "linear"],
    n_splits: int = 5,
    n_repeats: int = 5,
    hpo_candidates: int = 10,
    pfi_repeats: int = 10,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    """
    Returns:
      - per_fold_df: fold-level AUC + agreement metrics
      - summary_df: aggregated mean±std by (ratio, model)
      - stability_df: per-feature SHAP/PFI variance across folds by (ratio, model)
      - p_features: number of one-hot features (for chance baseline)
    """
    t_all = time.time()

    # Load RAW
    X_raw, y = load_german_raw(data_path)

    # Build a GLOBAL one-hot column list ONCE for stable variance comparisons across folds
    # (This uses category presence across the dataset; it does NOT train on test data values.)
    global_feature_names = list(pd.get_dummies(X_raw, drop_first=False).columns)
    p_features = len(global_feature_names)

    log(
        f"[stage] Data loaded: X_raw={X_raw.shape} "
        f"y_pos_rate={y.mean():.3f} global_onehot_features(p)={p_features}"
    )

    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    outer_splits = list(rskf.split(X_raw, y))
    total_splits = len(outer_splits)

    rows: list[FoldExplainRow] = []
    stability_rows: list[pd.DataFrame] = []

    log(f"[stage] Step 3: explainability across {total_splits} outer splits")
    log(f"[stage] Ratios={ratios} Models={model_types} HPO_cands={hpo_candidates} PFI_repeats={pfi_repeats}")

    for ratio in ratios:
        for model_type in model_types:
            shap_imps_all: list[np.ndarray] = []
            pfi_imps_all: list[np.ndarray] = []

            log(f"\n[block] ratio={ratio:.2f} model={model_type}")

            for split_id, (tr_idx, te_idx) in enumerate(outer_splits, start=1):
                t0 = time.time()

                repeat = (split_id - 1) // n_splits + 1
                fold = (split_id - 1) % n_splits + 1

                # Raw folds
                X_tr0_raw = X_raw.iloc[tr_idx].reset_index(drop=True)
                y_tr0 = y.iloc[tr_idx].reset_index(drop=True)
                X_te_raw = X_raw.iloc[te_idx].reset_index(drop=True)
                y_te = y.iloc[te_idx].reset_index(drop=True)

                # Fold-safe one-hot + align test->train
                X_tr0, X_te = onehot_align_train_test(X_tr0_raw, X_te_raw, drop_first=False)

                # Reindex BOTH to the global column list so:
                #  - SHAP/PFI arrays always have the same length
                #  - variance across folds is well-defined per feature
                X_tr0 = X_tr0.reindex(columns=global_feature_names, fill_value=0.0)
                X_te = X_te.reindex(columns=global_feature_names, fill_value=0.0)

                # Resample training only
                X_tr, y_tr = resample_train_to_ratio(
                    X_tr0,
                    y_tr0,
                    pos_ratio=ratio,
                    seed=seed + 10_000 * split_id + int(ratio * 1000),
                )

                # Nested HPO (inner 3-fold AUC)
                best_params = tune_hpo_inner(
                    X_tr,
                    y_tr,
                    model_type=model_type,
                    n_candidates=hpo_candidates,
                    seed=seed + 20_000 * split_id + int(ratio * 1000) + (1 if model_type == "tree" else 2),
                )

                X_tr_np = X_tr.to_numpy()
                y_tr_np = y_tr.to_numpy()
                X_te_np = X_te.to_numpy()
                y_te_np = y_te.to_numpy()

                if model_type == "linear":
                    scaler = StandardScaler()
                    X_tr_np = scaler.fit_transform(X_tr_np)
                    X_te_np = scaler.transform(X_te_np)

                clf = xgb.XGBClassifier(**make_xgb_params(model_type, best_params, random_seed=seed))
                clf.fit(X_tr_np, y_tr_np)

                p = clf.predict_proba(X_te_np)[:, 1]
                auc = float(roc_auc_score(y_te_np, p))

                # Importances on test fold (must use the SAME feature_names length/order)
                shap_imp = shap_global_importance_pred_contribs(
                    clf, X_te_np, feature_names=global_feature_names
                )
                pfi_imp = pfi_importance_auc(
                    clf, X_te_np, y_te_np, seed=seed + 30_000 + split_id, n_repeats=pfi_repeats
                )

                shap_imps_all.append(shap_imp)
                pfi_imps_all.append(pfi_imp)

                sp = safe_spearman(shap_imp, pfi_imp)
                ov5 = float(topk_overlap(shap_imp, pfi_imp, k=5))

                rows.append(
                    FoldExplainRow(
                        ratio=float(ratio),
                        model=str(model_type),
                        repeat=int(repeat),
                        fold=int(fold),
                        split_id=int(split_id),
                        auc=auc,
                        spearman_shap_pfi=sp,
                        top5_overlap=ov5,
                        best_params=json.dumps(best_params, sort_keys=True),
                        seconds=float(time.time() - t0),
                    )
                )

                if split_id % 5 == 0:
                    log(f"  [progress] split {split_id}/{total_splits} done (repeat={repeat}, fold={fold})")

            # Stability (variance across folds) for this (ratio, model)
            shap_mat = np.vstack(shap_imps_all)  # (25, p)
            pfi_mat = np.vstack(pfi_imps_all)

            shap_var, shap_avgvar = stability_from_importances(shap_mat)
            pfi_var, pfi_avgvar = stability_from_importances(pfi_mat)

            stab_df = pd.DataFrame(
                {
                    "ratio": ratio,
                    "model": model_type,
                    "feature": global_feature_names,
                    "shap_var": shap_var,
                    "pfi_var": pfi_var,
                }
            )
            stab_df["shap_avgvar"] = shap_avgvar
            stab_df["pfi_avgvar"] = pfi_avgvar
            stability_rows.append(stab_df)

    per_fold_df = pd.DataFrame([r.__dict__ for r in rows])

    summary_df = (
        per_fold_df.groupby(["ratio", "model"], as_index=False)
        .agg(
            auc_mean=("auc", "mean"),
            auc_std=("auc", "std"),
            spearman_mean=("spearman_shap_pfi", "mean"),
            spearman_std=("spearman_shap_pfi", "std"),
            top5_mean=("top5_overlap", "mean"),
            top5_std=("top5_overlap", "std"),
            n=("auc", "count"),
        )
        .fillna(0.0)
    )

    stability_df = pd.concat(stability_rows, ignore_index=True)

    log(f"\n[stage] Step 3 complete in {fmt_secs(time.time() - t_all)}")
    return per_fold_df, summary_df, stability_df, p_features

OUT_DIR = "./results"
def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    log("[start] Step 3: SHAP vs PFI agreement + stability")
    per_fold_df, summary_df, stability_df, p_features = run_step3_explainability(
        data_path=None,
        ratios=[0.10, 0.30, 0.50],
        model_types=["tree", "linear"],
        n_splits=5,
        n_repeats=5,
        hpo_candidates=10,
        pfi_repeats=10,
        seed=42,
    )

    # Save CSVs
    per_fold_df.to_csv(PER_FOLD_OUT, index=False)
    summary_df.to_csv(SUMMARY_OUT, index=False)
    stability_df.to_csv(STABILITY_OUT, index=False)

    log("[ok] Saved:")
    log(f" - {PER_FOLD_OUT}")
    log(f" - {SUMMARY_OUT}")
    log(f" - {STABILITY_OUT}")

    # Checks
    log("\n[check] rows per (ratio, model):")
    counts = per_fold_df.groupby(["ratio", "model"]).size().reset_index(name="n")
    print(counts.sort_values(["ratio", "model"]).to_string(index=False), flush=True)

    expected_per_group = 25
    bad = counts[counts["n"] != expected_per_group]
    if len(bad) > 0:
        log(f"[warn] Some groups are not {expected_per_group} folds. Check run completeness.")
        print(bad.to_string(index=False), flush=True)

    log("\n[preview] Summary (mean±std):")
    print(summary_df.sort_values(["ratio", "model"]).to_string(index=False), flush=True)

    # Plots with baselines
    chance_ov5 = chance_top5_overlap_fraction(p_features=p_features, k=5)
    log(f"\n[stage] Plotting... onehot_features(p)={p_features} -> chance top5_overlap baseline ≈ {chance_ov5:.3f}")

    grouped_boxplot(
        per_fold_df,
        metric="auc",
        title="Step 3: Test ROC-AUC across outer folds (by ratio & model)",
        out_path=AUC_PLOT_OUT,
        baseline=0.5,
        baseline_label="AUC=0.5 baseline",
    )

    grouped_boxplot(
        per_fold_df,
        metric="spearman_shap_pfi",
        title="Step 3: Spearman(SHAP vs PFI) across outer folds (by ratio & model)",
        out_path=SPEARMAN_PLOT_OUT,
        baseline=0.0,
        baseline_label="0 baseline",
    )

    grouped_boxplot(
        per_fold_df,
        metric="top5_overlap",
        title="Step 3: Top-5 overlap(SHAP vs PFI) across outer folds (by ratio & model)",
        out_path=TOP5_PLOT_OUT,
        baseline=chance_ov5,
        baseline_label=f"chance≈{chance_ov5:.3f}",
    )

    log("\n[ok] Saved plots:")
    log(f" - {AUC_PLOT_OUT}")
    log(f" - {SPEARMAN_PLOT_OUT}")
    log(f" - {TOP5_PLOT_OUT}")


if __name__ == "__main__":
    main()