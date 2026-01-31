import os
import pandas as pd
import pytest

@pytest.mark.slow
def test_step4_outputs_exist():
    # assumes you've already run step3/step4 once locally
    assert os.path.exists("./results/step4_final_model.joblib")
    assert os.path.exists("./results/step4_feature_columns.json")
    assert os.path.exists("./results/step4_final_holdout_metrics.json")
    assert os.path.exists("./results/step4_final_feature_importance.csv")

    df = pd.read_csv("./results/step4_final_feature_importance.csv")
    assert "feature" in df.columns