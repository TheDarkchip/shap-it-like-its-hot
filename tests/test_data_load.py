import pandas as pd
from german_credit import load_german_credit

def test_load_german_credit_shapes_and_labels():
    X, y = load_german_credit(None, target_positive="bad")

    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert len(X) == len(y)
    assert set(y.unique()).issubset({0, 1})
    assert y.nunique() == 2  # both classes present

    # Spot-check key raw columns exist
    for col in ["checking_status", "duration_months", "credit_amount", "age_years"]:
        assert col in X.columns