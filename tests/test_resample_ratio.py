import numpy as np
from german_credit import load_german_credit
import pandas as pd

# Import your function from your step file:
from step1_predictive_performance import resample_train_to_ratio  # adjust if needed

def test_resample_train_to_ratio_keeps_size_and_ratio():
    X_raw, y = load_german_credit(None, target_positive="bad")
    X = pd.get_dummies(X_raw, drop_first=False).astype(float)

    X_tr = X.iloc[:400].reset_index(drop=True)
    y_tr = y.iloc[:400].reset_index(drop=True)

    X_rs, y_rs = resample_train_to_ratio(X_tr, y_tr, pos_ratio=0.30, seed=123)

    assert len(X_rs) == len(X_tr)
    assert len(y_rs) == len(y_tr)

    # Ratio is approximate because rounding, allow tolerance
    assert abs(float(y_rs.mean()) - 0.30) < 0.05