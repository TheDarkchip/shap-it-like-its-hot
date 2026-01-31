from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd

from scipy.stats import spearmanr

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from sklearn.inspection import permutation_importance

import xgboost as xgb

try:
    import joblib
except ImportError:
    joblib = None  # handled later

from german_credit import load_german_credit


ModelType = Literal["tree", "linear"]

OUT_DIR = "./results"

# Inputs
STEP2_SUMMARY = os.path.join(OUT_DIR, "step2_predictive_summary.csv")  # if you created it
STEP2_PER_FOLD = os.path.join(OUT_DIR, "step2_per_fold.csv")          # fallback
STEP3_SUMMARY = os.path.join(OUT_DIR, "step3_summary_explain.csv")

# Outputs
MERGED_OUT = os.path.join(OUT_DIR, "step4_merged_summary.csv")
RANKED_OUT = os.path.join(OUT_DIR, "step4_ranked_configs.csv")
BEST_CONFIG_OUT = os.path.join(OUT_DIR, "step4_best_config.json")
FINAL_METRICS_OUT = os.path.join(OUT_DIR, "step4_final_holdout_metrics.json")

MODEL_OUT = os.path.join(OUT_DIR, "step4_final_model.joblib")
SCALER_OUT = os.path.join(OUT_DIR, "step4_final_scaler.joblib")  # only for linear

FINAL_IMPORTANCE_OUT = os.path.join(OUT_DIR, "step4_final_feature_importance.csv")


# -----------------------------
# Logging
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
# Data (one-hot)
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
    X_train_oh = pd.get_dummies(X_train_raw, drop_first=drop_first)
    X_test_oh = pd.get_dummies(X_test_raw, drop_first=drop_first)

    # align test columns to train columns
    X_test_oh = X_test_oh.reindex(columns=X_train_oh.columns, fill_value=0.0)

    return X_train_oh.astype(float), X_test_oh.astype(float)


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
# Model + HPO (tight grid for linear)
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


def mean_auc_inner_kfold(
    X: np.ndarray,
    y: np.ndarray,
    model_type: ModelType,
    params: dict,
    seed: int,
    n_splits: int = 5,
) -> float:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
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


def tune_hpo(
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
        score = mean_auc_inner_kfold(Xn, yn, model_type=model_type, params=params, seed=seed, n_splits=5)
        if score > best_score:
            best_score = score
            best_params = params

    return best_params


# -----------------------------
# Explainability helpers
# -----------------------------
def shap_global_importance_pred_contribs(
    clf: xgb.XGBClassifier,
    X: np.ndarray,
    feature_names: list[str],
) -> np.ndarray:
    booster = clf.get_booster()
    dmat = xgb.DMatrix(X, feature_names=feature_names)
    contrib = booster.predict(dmat, pred_contribs=True)  # (n, p+1)
    contrib_no_bias = contrib[:, :-1]
    return np.mean(np.abs(contrib_no_bias), axis=0)


def pfi_importance_auc(
    clf: xgb.XGBClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    n_repeats: int = 20,
) -> np.ndarray:
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


def safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    corr = spearmanr(a, b).correlation
    return 0.0 if (corr is None or np.isnan(corr)) else float(corr)


# -----------------------------
# Step 4: Load summaries
# -----------------------------
def load_step2_summary() -> pd.DataFrame:
    if os.path.exists(STEP2_SUMMARY):
        df = pd.read_csv(STEP2_SUMMARY)
        required = {"ratio", "model", "auc_mean", "auc_std", "pr_auc_mean", "pr_auc_std", "acc_mean", "acc_std", "n"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{STEP2_SUMMARY} missing columns: {sorted(missing)}")
        return df

    if not os.path.exists(STEP2_PER_FOLD):
        raise FileNotFoundError(f"Need either {STEP2_SUMMARY} or {STEP2_PER_FOLD}")

    per = pd.read_csv(STEP2_PER_FOLD)
    required = {"ratio", "model", "auc", "pr_auc", "accuracy"}
    missing = required - set(per.columns)
    if missing:
        raise ValueError(f"{STEP2_PER_FOLD} missing columns: {sorted(missing)}")

    summ = (
        per.groupby(["ratio", "model"], as_index=False)
        .agg(
            auc_mean=("auc", "mean"),
            auc_std=("auc", "std"),
            pr_auc_mean=("pr_auc", "mean"),
            pr_auc_std=("pr_auc", "std"),
            acc_mean=("accuracy", "mean"),
            acc_std=("accuracy", "std"),
            n=("auc", "count"),
        )
        .fillna(0.0)
    )
    return summ


def load_step3_summary_for_merge() -> pd.DataFrame:
    """
    Step 3 summary often includes auc_mean/auc_std/n as well. For Step 4 we only need:
      - spearman_mean/std
      - top5_mean/std
    So we DROP auc_* and n if present to avoid merge collisions.
    """
    if not os.path.exists(STEP3_SUMMARY):
        raise FileNotFoundError(f"Missing {STEP3_SUMMARY}. Run Step 3 first.")

    df = pd.read_csv(STEP3_SUMMARY)

    required = {"ratio", "model", "spearman_mean", "spearman_std", "top5_mean", "top5_std"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{STEP3_SUMMARY} missing columns: {sorted(missing)}")

    drop_cols = [c for c in ["auc_mean", "auc_std", "n"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    return df


def merge_and_rank(step2: pd.DataFrame, step3: pd.DataFrame) -> pd.DataFrame:
    merged = pd.merge(step2, step3, on=["ratio", "model"], how="inner")
    if len(merged) == 0:
        raise ValueError("Merge produced 0 rows. Check that (ratio, model) match in Step 2 and Step 3 outputs.")

    # Rank configs: max AUC -> max spearman -> max top5
    merged = merged.copy()
    merged["rank_auc"] = merged["auc_mean"].rank(ascending=False, method="min")
    merged["rank_spearman"] = merged["spearman_mean"].rank(ascending=False, method="min")
    merged["rank_top5"] = merged["top5_mean"].rank(ascending=False, method="min")
    merged["rank_sum"] = merged["rank_auc"] + merged["rank_spearman"] + merged["rank_top5"]

    merged = merged.sort_values(
        ["auc_mean", "spearman_mean", "top5_mean"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    return merged


# -----------------------------
# Final training on best config
# -----------------------------
@dataclass(frozen=True)
class BestConfig:
    ratio: float
    model: ModelType


def train_finalize_and_export(
    *,
    best: BestConfig,
    data_path: str | None,
    seed: int,
    hpo_candidates: int,
    pfi_repeats: int,
) -> None:
    if joblib is None:
        raise ImportError("joblib is required to save the model. Install with: pip install joblib")

    t0 = time.time()
    X_raw, y = load_german_raw(data_path)

    # Holdout split on RAW (not one-hot)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    tr_idx, te_idx = next(sss.split(X_raw, y))

    X_tr0_raw = X_raw.iloc[tr_idx].reset_index(drop=True)
    y_tr0 = y.iloc[tr_idx].reset_index(drop=True)
    X_te_raw = X_raw.iloc[te_idx].reset_index(drop=True)
    y_te = y.iloc[te_idx].reset_index(drop=True)

    # Fold-safe one-hot on holdout split
    X_tr0, X_te = onehot_align_train_test(X_tr0_raw, X_te_raw, drop_first=False)

    features = list(X_tr0.columns)
    FEATURES_OUT = os.path.join(OUT_DIR, "step4_feature_columns.json")

    with open(FEATURES_OUT, "w", encoding="utf-8") as f:
        json.dump({"feature_columns": features}, f, indent=2)


    # Resample TRAIN only to selected ratio
    X_tr, y_tr = resample_train_to_ratio(X_tr0, y_tr0, pos_ratio=best.ratio, seed=seed + 12345)

    log(f"[final] Best config: ratio={best.ratio:.2f} model={best.model}")
    log(f"[final] Holdout: train={len(y_tr0)} test={len(y_te)} test_prevalence={y_te.mean():.3f}")
    log(f"[final] Resampled train prevalence={y_tr.mean():.3f}")

    # HPO on resampled train
    log(f"[final] HPO candidates={hpo_candidates} ...")
    best_params = tune_hpo(X_tr, y_tr, model_type=best.model, n_candidates=hpo_candidates, seed=seed + 22222)
    log(f"[final] Best params: {json.dumps(best_params, sort_keys=True)}")

    # Fit final model
    X_tr_np = X_tr.to_numpy()
    y_tr_np = y_tr.to_numpy()
    X_te_np = X_te.to_numpy()
    y_te_np = y_te.to_numpy()

    scaler: Optional[StandardScaler] = None
    if best.model == "linear":
        scaler = StandardScaler()
        X_tr_np = scaler.fit_transform(X_tr_np)
        X_te_np = scaler.transform(X_te_np)

    clf = xgb.XGBClassifier(**make_xgb_params(best.model, best_params, random_seed=seed))
    clf.fit(X_tr_np, y_tr_np)

    # Evaluate
    p = clf.predict_proba(X_te_np)[:, 1]
    yhat = (p >= 0.5).astype(int)

    auc = float(roc_auc_score(y_te_np, p))
    pr_auc = float(average_precision_score(y_te_np, p))
    acc = float(accuracy_score(y_te_np, yhat))

    pr_baseline = float(y_te_np.mean())
    acc_baseline = float(max(pr_baseline, 1.0 - pr_baseline))

    # Explainability on holdout test
    shap_imp = shap_global_importance_pred_contribs(clf, X_te_np, feature_names=features)
    pfi_imp = pfi_importance_auc(clf, X_te_np, y_te_np, seed=seed + 33333, n_repeats=pfi_repeats)
    sp = safe_spearman(shap_imp, pfi_imp)

    metrics = {
        "best_ratio": best.ratio,
        "best_model": best.model,
        "holdout_auc": auc,
        "holdout_pr_auc": pr_auc,
        "holdout_accuracy": acc,
        "pr_auc_baseline": pr_baseline,
        "accuracy_baseline": acc_baseline,
        "shap_pfi_spearman_holdout": sp,
        "best_params": best_params,
        "seed": seed,
    }

    imp_df = pd.DataFrame(
        {
            "feature": features,
            "shap_mean_abs": shap_imp,
            "pfi_auc_drop": pfi_imp,
        }
    ).sort_values("shap_mean_abs", ascending=False).reset_index(drop=True)

    # Save artifacts
    joblib.dump(clf, MODEL_OUT)
    if scaler is not None:
        joblib.dump(scaler, SCALER_OUT)

    with open(FINAL_METRICS_OUT, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    imp_df.to_csv(FINAL_IMPORTANCE_OUT, index=False)

    log(f"[final] Holdout: AUC={auc:.4f} PR-AUC={pr_auc:.4f} Acc={acc:.4f}")
    log(f"[final] Baselines: PR-AUC={pr_baseline:.4f} Acc={acc_baseline:.4f}")
    log(f"[final] SHAP vs PFI Spearman (holdout): {sp:.4f}")

    log("[ok] Saved:")
    log(f" - {MODEL_OUT}")
    if scaler is not None:
        log(f" - {SCALER_OUT}")
    log(f" - {FINAL_METRICS_OUT}")
    log(f" - {FINAL_IMPORTANCE_OUT}")
    log(f"[done] Step 4 finished in {fmt_secs(time.time() - t0)}")

OUT_DIR = "./results"
def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    log("[start] Step 4: merge Step2+Step3, pick best config, train final model, export artifacts")

    step2 = load_step2_summary()
    step3 = load_step3_summary_for_merge()

    merged = merge_and_rank(step2, step3)
    merged.to_csv(MERGED_OUT, index=False)

    ranked = merged.sort_values(["rank_sum", "auc_mean"], ascending=[True, False]).reset_index(drop=True)
    ranked.to_csv(RANKED_OUT, index=False)

    best_ratio = float(merged.loc[0, "ratio"])
    best_model = str(merged.loc[0, "model"])
    if best_model not in ("tree", "linear"):
        raise ValueError(f"Unexpected model: {best_model}")

    best = BestConfig(ratio=best_ratio, model=best_model)  # type: ignore[arg-type]

    with open(BEST_CONFIG_OUT, "w", encoding="utf-8") as f:
        json.dump({"ratio": best.ratio, "model": best.model}, f, indent=2)

    log("\n[preview] merged summary (sorted by AUC → Spearman → Top5):")
    cols = ["ratio", "model", "auc_mean", "auc_std", "pr_auc_mean", "acc_mean", "spearman_mean", "top5_mean"]
    print(merged[cols].to_string(index=False), flush=True)

    log("\n[ok] Saved tables:")
    log(f" - {MERGED_OUT}")
    log(f" - {RANKED_OUT}")
    log(f" - {BEST_CONFIG_OUT}")

    # Finalize
    train_finalize_and_export(
        best=best,
        data_path=None,
        seed=42,
        hpo_candidates=30,
        pfi_repeats=20,
    )


if __name__ == "__main__":
    main()