import pandas as pd
from german_credit import load_german_credit

# Import your function from wherever you defined it:
# from utils_preprocess import onehot_align_train_test
from step3_select_best_and_finalize_model import onehot_align_train_test  # adjust filename if needed

def test_onehot_align_train_test_same_columns():
    X_raw, y = load_german_credit(None, target_positive="bad")

    X_tr_raw = X_raw.iloc[:200].copy()
    X_te_raw = X_raw.iloc[200:260].copy()

    X_tr, X_te = onehot_align_train_test(X_tr_raw, X_te_raw, drop_first=False)

    assert list(X_tr.columns) == list(X_te.columns)
    assert X_tr.isna().sum().sum() == 0
    assert X_te.isna().sum().sum() == 0
    assert (X_tr.dtypes == float).all()
    assert (X_te.dtypes == float).all()