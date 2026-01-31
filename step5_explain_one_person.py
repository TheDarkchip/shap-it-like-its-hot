from __future__ import annotations

import argparse
import json
import os
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
import joblib

from german_credit import load_german_credit


# ------------------------------------------------------------
# Paths (match Step 4 outputs)
# ------------------------------------------------------------
OUT_DIR = "./results"
MODEL_PATH = os.path.join(OUT_DIR, "step4_final_model.joblib")
SCALER_PATH = os.path.join(OUT_DIR, "step4_final_scaler.joblib")  # exists only if best model was linear
FEATURES_PATH = os.path.join(OUT_DIR, "step4_feature_columns.json")


# ------------------------------------------------------------
# Codebook for German Credit (A-codes -> human labels)
# ------------------------------------------------------------
CODEBOOK: Dict[str, Dict[str, str]] = {
    "checking_status": {
        "A11": "< 0 DM",
        "A12": "0–<200 DM",
        "A13": "≥200 DM / salary assignments ≥1 year",
        "A14": "no checking account",
    },
    "credit_history": {
        "A30": "no credits / all paid back duly",
        "A31": "all credits at this bank paid back duly",
        "A32": "existing credits paid back duly till now",
        "A33": "delay in paying off in the past",
        "A34": "critical account / other credits elsewhere",
    },
    "purpose": {
        "A40": "car (new)",
        "A41": "car (used)",
        "A42": "furniture/equipment",
        "A43": "radio/television",
        "A44": "domestic appliances",
        "A45": "repairs",
        "A46": "education",
        "A48": "retraining",
        "A49": "business",
        "A410": "others",
    },
    "savings_status": {
        "A61": "< 100 DM",
        "A62": "100–<500 DM",
        "A63": "500–<1000 DM",
        "A64": "≥ 1000 DM",
        "A65": "unknown / no savings account",
    },
    "employment_since": {
        "A71": "unemployed",
        "A72": "< 1 year",
        "A73": "1–<4 years",
        "A74": "4–<7 years",
        "A75": "≥ 7 years",
    },
    "personal_status_sex": {
        "A91": "male: divorced/separated",
        "A92": "female: divorced/separated/married",
        "A93": "male: single",
        "A94": "male: married/widowed",
        "A95": "female: single",
    },
    "other_debtors": {"A101": "none", "A102": "co-applicant", "A103": "guarantor"},
    "property": {
        "A121": "real estate",
        "A122": "building society savings / life insurance",
        "A123": "car or other",
        "A124": "unknown / no property",
    },
    "other_installment_plans": {"A141": "bank", "A142": "stores", "A143": "none"},
    "housing": {"A151": "rent", "A152": "own", "A153": "for free"},
    "job": {
        "A171": "unemployed/unskilled (non-resident)",
        "A172": "unskilled (resident)",
        "A173": "skilled employee/official",
        "A174": "management/self-employed/highly qualified",
    },
    "telephone": {"A191": "none", "A192": "yes (registered in customer’s name)"},
    "foreign_worker": {"A201": "yes", "A202": "no"},
}


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------
def read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def sigmoid(z: float) -> float:
    return float(1.0 / (1.0 + np.exp(-z)))


def humanize_raw_value(feature: str, value) -> str:
    s = str(value)
    if feature in CODEBOOK and s in CODEBOOK[feature]:
        return f"{s} ({CODEBOOK[feature][s]})"
    return s


def load_artifacts() -> Tuple[xgb.XGBClassifier, Optional[object], List[str]]:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Missing model file: {MODEL_PATH}. Run Step 4 first.")
    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError(f"Missing feature columns file: {FEATURES_PATH}. Run Step 4 after saving feature columns.")

    clf: xgb.XGBClassifier = joblib.load(MODEL_PATH)

    scaler: Optional[object] = None
    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)

    feature_columns = read_json(FEATURES_PATH)["feature_columns"]
    if not isinstance(feature_columns, list) or len(feature_columns) == 0:
        raise ValueError(f"{FEATURES_PATH} does not contain a valid feature_columns list.")

    return clf, scaler, feature_columns


def onehot_single_row(raw_row: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
    X_oh = pd.get_dummies(raw_row, drop_first=False).astype(float)
    return X_oh.reindex(columns=feature_columns, fill_value=0.0)


def shap_like_contribs(
    clf: xgb.XGBClassifier,
    X_row_np: np.ndarray,
    feature_columns: List[str],
) -> Tuple[np.ndarray, float]:
    booster = clf.get_booster()
    dmat = xgb.DMatrix(X_row_np, feature_names=feature_columns)
    contrib = booster.predict(dmat, pred_contribs=True)  # (1, p+1)
    contrib = contrib[0]
    bias = float(contrib[-1])
    per_feat = contrib[:-1]
    return per_feat, bias


def build_group_map(feature_columns: List[str], raw_col_names: List[str]) -> Dict[str, str]:
    raw_set = set(raw_col_names)
    raw_sorted = sorted(raw_col_names, key=len, reverse=True)

    mapping: Dict[str, str] = {}
    for col in feature_columns:
        if col in raw_set:
            mapping[col] = col
            continue

        base = None
        for raw in raw_sorted:
            if col.startswith(raw + "_"):
                base = raw
                break

        mapping[col] = base if base is not None else col.split("_", 1)[0]

    return mapping


def group_contribs(contribs: pd.Series, group_map: Dict[str, str]) -> pd.Series:
    groups: Dict[str, float] = {}
    for col, val in contribs.items():
        g = group_map.get(col, col.split("_", 1)[0])
        groups[g] = groups.get(g, 0.0) + float(val)
    return pd.Series(groups).sort_values(ascending=False)


def pick_top_groups(grouped: pd.Series, k: int, direction: str) -> List[str]:
    if direction == "up":
        sel = grouped[grouped > 0].sort_values(ascending=False).head(k)
    else:
        sel = grouped[grouped < 0].sort_values(ascending=True).head(k)
    return sel.index.tolist()


def describe_feature_value(raw_row: pd.DataFrame, feature: str) -> str:
    """
    Turn a raw feature into a human phrase using this person's value.
    Example:
      checking_status -> "checking account status (< 0 DM)"
      installment_rate -> "high installment rate (4)"
    """
    if feature not in raw_row.columns:
        return feature

    v = raw_row.iloc[0][feature]
    hv = humanize_raw_value(feature, v)

    pretty_name = feature.replace("_", " ")

    # Custom nicer phrases for a few common ones
    if feature == "checking_status":
        return f"checking account status ({CODEBOOK['checking_status'].get(str(v), str(v))})"
    if feature == "installment_rate":
        return f"installment rate ({v})"
    if feature == "duration_months":
        return f"loan duration ({v} months)"
    if feature == "credit_amount":
        return f"credit amount ({v})"
    if feature == "existing_credits":
        return f"number of existing credits ({v})"
    if feature == "residence_since":
        return f"time at current residence ({v})"
    if feature in CODEBOOK:
        # categorical
        return f"{pretty_name} ({CODEBOOK[feature].get(str(v), str(v))})"

    return f"{pretty_name} ({hv})"


def risk_bucket(pd_hat: float) -> str:
    """
    Simple descriptive bucket (you can change thresholds if you want).
    """
    if pd_hat < 0.10:
        return "low risk"
    if pd_hat < 0.25:
        return "moderate risk"
    return "high risk"


def make_paragraph(
    raw_row: pd.DataFrame,
    pd_hat: float,
    grouped: pd.Series,
    *,
    up_k: int = 3,
    down_k: int = 3,
) -> str:
    up = pick_top_groups(grouped, up_k, "up")
    down = pick_top_groups(grouped, down_k, "down")

    up_phrases = [describe_feature_value(raw_row, f) for f in up]
    down_phrases = [describe_feature_value(raw_row, f) for f in down]

    pd_pct = pd_hat * 100.0
    bucket = risk_bucket(pd_hat)

    # Build readable sentences
    s1 = f"The model predicts a probability of default (PD) of {pd_pct:.1f}%, which indicates {bucket}."

    if len(up_phrases) > 0:
        s2 = (
            "The main factors that increased the predicted risk were "
            + ", ".join(up_phrases[:2])
            + (f". Smaller risk-increasing factors were {', '.join(up_phrases[2:])}." if len(up_phrases) > 2 else ".")
        )
    else:
        s2 = "No strong risk-increasing factors were identified among the top drivers."

    if len(down_phrases) > 0:
        s3 = (
            "The main factors that reduced the predicted risk were "
            + ", ".join(down_phrases[:2])
            + (f". Additional protective factors were {', '.join(down_phrases[2:])}." if len(down_phrases) > 2 else ".")
        )
    else:
        s3 = "No strong risk-reducing factors were identified among the top drivers."

    s4 = "Note: These are model drivers (patterns learned from data), not proof of causality."

    return "\n\n".join([s1, s2, s3, s4])


# ------------------------------------------------------------
# Main explain function
# ------------------------------------------------------------
def explain_index(i: int, topk: int, up_k: int, down_k: int) -> None:
    clf, scaler, feature_columns = load_artifacts()
    X_raw, y = load_german_credit(None, target_positive="bad")
    raw_col_names = list(X_raw.columns)

    if i < 0 or i >= len(X_raw):
        raise ValueError(f"--index must be in [0, {len(X_raw)-1}]")

    raw_row = X_raw.iloc[[i]].copy()
    true_label = int(y.iloc[i])

    # preprocess
    X_oh = onehot_single_row(raw_row, feature_columns)
    X_np = X_oh.to_numpy()

    if scaler is not None:
        X_np = scaler.transform(X_np)

    # predict
    pd_hat = float(clf.predict_proba(X_np)[0, 1])

    # contribs
    contrib_arr, bias = shap_like_contribs(clf, X_np, feature_columns)
    contribs = pd.Series(contrib_arr, index=feature_columns)

    # group
    group_map = build_group_map(feature_columns, raw_col_names)
    grouped = group_contribs(contribs, group_map)

    # print raw + paragraph
    print(f"\nExplaining dataset row index {i} (true label bad=1): {true_label}\n")

    print("Raw input values (human-friendly):")
    for col in raw_row.columns:
        print(f"  {col:25s}: {humanize_raw_value(col, raw_row.iloc[0][col])}")

    print("\nPrediction:")
    print(f"  Predicted PD: {pd_hat:.4f} (~{pd_hat*100:.1f}%)")

    # show top reasons (short)
    up_feats = pick_top_groups(grouped, topk, "up")
    down_feats = pick_top_groups(grouped, topk, "down")

    print("\nTop reasons pushing risk UP:")
    for f in up_feats:
        print(f"  - {describe_feature_value(raw_row, f)}")

    print("\nTop reasons pushing risk DOWN:")
    for f in down_feats:
        print(f"  - {describe_feature_value(raw_row, f)}")

    # paragraph
    paragraph = make_paragraph(raw_row, pd_hat, grouped, up_k=up_k, down_k=down_k)
    print("\n==============================")
    print(" Paste-ready explanation text ")
    print("==============================\n")
    print(paragraph)


def main() -> None:
    ap = argparse.ArgumentParser(description="Explain one person and print a paste-ready paragraph explanation.")
    ap.add_argument("--index", type=int, default=0, help="Dataset row index to explain.")
    ap.add_argument("--topk", type=int, default=5, help="List this many UP and DOWN reasons (bullets).")
    ap.add_argument("--up_k", type=int, default=3, help="How many UP drivers to use in paragraph.")
    ap.add_argument("--down_k", type=int, default=3, help="How many DOWN drivers to use in paragraph.")
    args = ap.parse_args()

    explain_index(int(args.index), topk=int(args.topk), up_k=int(args.up_k), down_k=int(args.down_k))


if __name__ == "__main__":
    main()