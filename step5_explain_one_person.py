from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb
import joblib

# ---- paths (match your Step 4 outputs)
OUT_DIR = "./results"
MODEL_PATH = os.path.join(OUT_DIR, "step4_final_model.joblib")
SCALER_PATH = os.path.join(OUT_DIR, "step4_final_scaler.joblib")  # may not exist for tree
FEATURES_PATH = os.path.join(OUT_DIR, "step4_feature_columns.json")

# If you want to explain someone from the dataset:
from german_credit import load_german_credit


def sigmoid(z: float) -> float:
    return float(1.0 / (1.0 + np.exp(-z)))


def load_artifacts():
    clf = joblib.load(MODEL_PATH)

    scaler: Optional[object] = None
    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)

    with open(FEATURES_PATH, "r", encoding="utf-8") as f:
        feature_columns = json.load(f)["feature_columns"]

    return clf, scaler, feature_columns


def onehot_single_row(raw_row: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """
    raw_row: DataFrame with ONE ROW in the same raw column format as training (categorical + numeric).
    Returns: one-hot row aligned to training columns.
    """
    X_oh = pd.get_dummies(raw_row, drop_first=False).astype(float)
    X_oh = X_oh.reindex(columns=feature_columns, fill_value=0.0)
    return X_oh


def shap_like_contribs(clf, X_row_np: np.ndarray, feature_columns: list[str]) -> tuple[np.ndarray, float]:
    """
    Uses xgboost pred_contribs to get per-feature contributions + bias term.
    Returns (contribs_per_feature, bias).
    """
    booster = clf.get_booster()
    dmat = xgb.DMatrix(X_row_np, feature_names=feature_columns)
    contrib = booster.predict(dmat, pred_contribs=True)  # shape (1, p+1)
    contrib = contrib[0]
    bias = float(contrib[-1])
    per_feat = contrib[:-1]
    return per_feat, bias


def explain_person(raw_row: pd.DataFrame, top_k: int = 8) -> None:
    clf, scaler, feature_columns = load_artifacts()

    # 1) preprocess
    X_row = onehot_single_row(raw_row, feature_columns)
    X_np = X_row.to_numpy()

    # 2) scale if linear model used scaler in Step 4
    if scaler is not None:
        X_np = scaler.transform(X_np)

    # 3) prediction
    pd_hat = float(clf.predict_proba(X_np)[0, 1])

    # 4) SHAP-like contributions (log-odds units)
    contribs, bias = shap_like_contribs(clf, X_np, feature_columns)

    # 5) show top positive / negative drivers
    s = pd.Series(contribs, index=feature_columns)
    top_up = s.sort_values(ascending=False).head(top_k)
    top_down = s.sort_values(ascending=True).head(top_k)

    # Optional: show how it sums
    margin = bias + float(contribs.sum())
    pd_from_margin = sigmoid(margin)

    print("\n=== One-person explanation ===")
    print(f"Predicted PD (model): {pd_hat:.4f}")
    print(f"Predicted PD (from contrib sum check): {pd_from_margin:.4f}")
    print(f"Bias (baseline, log-odds): {bias:.4f}")
    print(f"Total margin (bias + sum contribs, log-odds): {margin:.4f}")

    print("\nTop factors pushing risk UP (positive contributions):")
    for name, val in top_up.items():
        print(f"  {name:40s}  +{val:.4f}")

    print("\nTop factors pushing risk DOWN (negative contributions):")
    for name, val in top_down.items():
        print(f"  {name:40s}  {val:.4f}")


if __name__ == "__main__":
    # Example: explain the first row of the dataset
    X_raw, y = load_german_credit(None, target_positive="bad")
    i = 0
    raw_row = X_raw.iloc[[i]]  # keep as DataFrame (1 row)

    print(f"Explaining row index {i}, true label (bad=1): {int(y.iloc[i])}")
    explain_person(raw_row, top_k=8)
