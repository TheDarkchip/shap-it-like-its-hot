import numpy as np
import pandas as pd
import xgboost as xgb
from german_credit import load_german_credit

def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + np.exp(-z))

def test_pred_contribs_sum_matches_predict_proba():
    X_raw, y = load_german_credit(None, target_positive="bad")
    X = pd.get_dummies(X_raw, drop_first=False).astype(float)

    # Train small model quickly
    clf = xgb.XGBClassifier(
        booster="gbtree",
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="auc",
        random_state=0,
        n_jobs=1,
        verbosity=0,
    )
    clf.fit(X.iloc[:500].to_numpy(), y.iloc[:500].to_numpy())

    # One row explanation check
    x1 = X.iloc[[0]]
    p = float(clf.predict_proba(x1.to_numpy())[0, 1])

    booster = clf.get_booster()
    dmat = xgb.DMatrix(x1.to_numpy(), feature_names=list(X.columns))
    contrib = booster.predict(dmat, pred_contribs=True)[0]  # (p+1,)
    bias = float(contrib[-1])
    per_feat = contrib[:-1]

    margin = bias + float(per_feat.sum())
    p_check = float(sigmoid(margin))

    assert abs(p - p_check) < 1e-6