"""Run the full MVS baseline with inner-CV HPO."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import json

import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agreement_metrics import write_agreement_summary  # noqa: E402
from experiment_utils import configure_logging  # noqa: E402
from experiment_utils import create_run_metadata  # noqa: E402
from experiment_utils import generate_run_id  # noqa: E402
from experiment_utils import set_global_seed  # noqa: E402
from german_credit import load_german_credit  # noqa: E402
from hpo_utils import tune_and_train  # noqa: E402
from pfi_utils import compute_pfi_importance  # noqa: E402
from resampling import resample_train_fold  # noqa: E402
from results_io import ResultRecord, append_record_csv  # noqa: E402
from shap_utils import compute_tree_shap  # noqa: E402
from stability_metrics import write_stability_summary  # noqa: E402
from xgboost_wrapper import predict_proba  # noqa: E402
from nested_cv import iter_outer_folds  # noqa: E402


PARAM_GRID = {
    "n_estimators": [100, 200],
    "max_depth": [3, 4],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}
AGREEMENT_TOP_K = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline MVS with HPO")
    parser.add_argument("--output-dir", default="results", help="Base output dir")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--outer-repeats", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--pfi-repeats", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    run_id = generate_run_id(prefix="mvs-hpo")
    results_dir = Path(args.output_dir) / run_id
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / "run.log"

    logger = configure_logging(
        run_id=run_id,
        seed=args.seed,
        log_file=log_path,
        force=True,
        logger_name="shap-it-like-its-hot",
    )

    X_raw, y = load_german_credit()
    X = pd.get_dummies(X_raw, drop_first=False)
    X = X.reindex(sorted(X.columns), axis=1)

    ratios = [0.1, 0.3, 0.5]
    results_path = results_dir / "results.csv"

    for outer in iter_outer_folds(
        X,
        y,
        outer_folds=args.outer_folds,
        outer_repeats=args.outer_repeats,
        seed=args.seed,
    ):
        logger.info(
            "Outer fold repeat=%s fold=%s seed=%s",
            outer.repeat_id,
            outer.fold_id,
            outer.seed,
        )
        X_train = X.iloc[outer.train_idx]
        y_train = y.iloc[outer.train_idx]
        X_test = X.iloc[outer.test_idx]
        y_test = y.iloc[outer.test_idx]

        for ratio in ratios:
            logger.info("Resampling ratio=%.2f", ratio)
            resampled = resample_train_fold(
                X_train,
                y_train,
                target_positive_ratio=ratio,
                random_state=outer.seed,
            )
            achieved_ratio = resampled.positive_count / (
                resampled.positive_count + resampled.negative_count
            )
            hpo = tune_and_train(
                resampled.X,
                resampled.y,
                param_grid=PARAM_GRID,
                metric_name="roc_auc",
                inner_folds=args.inner_folds,
                seed=outer.seed,
            )
            logger.info("Best HPO score=%.4f", hpo.best_score)
            proba = predict_proba(hpo.model, X_test)
            shap_result = compute_tree_shap(hpo.model, X_test)
            pfi_importance = compute_pfi_importance(
                hpo.model,
                X_test,
                y_test,
                metric_name="roc_auc",
                n_repeats=args.pfi_repeats,
                random_state=outer.seed,
            )
            if pd.Series(y_test).nunique() < 2:
                roc_auc = float("nan")
            else:
                roc_auc = float(roc_auc_score(y_test, proba))
            metrics = {
                "accuracy": float(accuracy_score(y_test, proba >= 0.5)),
                "roc_auc": roc_auc,
            }

            record = ResultRecord(
                fold_id=outer.fold_id,
                repeat_id=outer.repeat_id,
                seed=outer.seed,
                model_name="xgboost",
                class_ratio=ratio,
                metrics=metrics,
                shap_importance=shap_result.global_importance.to_dict(),
                pfi_importance=pfi_importance.to_dict(),
            )
            append_record_csv(results_path, record)

    frame = pd.read_csv(results_path)
    ratios = sorted(frame["class_ratio"].unique())
    write_stability_summary(frame, ratios=ratios, output_path=results_dir / "stability_summary.csv")
    write_agreement_summary(
        frame,
        ratios=ratios,
        output_path=results_dir / "agreement_summary.csv",
        top_k=AGREEMENT_TOP_K,
    )

    metadata = create_run_metadata(
        run_id=run_id,
        seed=args.seed,
        extra={
            "outer_folds": args.outer_folds,
            "outer_repeats": args.outer_repeats,
            "inner_folds": args.inner_folds,
            "pfi_repeats": args.pfi_repeats,
            "param_grid": PARAM_GRID,
            "agreement_top_k": AGREEMENT_TOP_K,
        },
    )
    (results_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    logger.info("Results written to %s", results_dir)
    print(f"Wrote baseline results to {results_dir}")


if __name__ == "__main__":
    main()
