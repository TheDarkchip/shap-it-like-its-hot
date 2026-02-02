"""Run the full MVS baseline with inner-CV HPO."""

from __future__ import annotations

import argparse
from pathlib import Path

import json

import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

from shap_stability.metrics.agreement import write_agreement_summary  # noqa: E402
from shap_stability.experiment_utils import configure_logging  # noqa: E402
from shap_stability.experiment_utils import create_run_metadata  # noqa: E402
from shap_stability.experiment_utils import generate_run_id  # noqa: E402
from shap_stability.experiment_utils import set_global_seed  # noqa: E402
from shap_stability.data import load_german_credit, one_hot_encode_train_test  # noqa: E402
from shap_stability.modeling.hpo_utils import HPOResult, tune_and_train  # noqa: E402
from shap_stability.explain.pfi_utils import compute_pfi_importance  # noqa: E402
from shap_stability.resampling import resample_train_fold  # noqa: E402
from shap_stability.metrics.results_io import ResultRecord, append_record_csv  # noqa: E402
from shap_stability.explain.shap_utils import compute_tree_shap  # noqa: E402
from shap_stability.metrics.stability import write_stability_summary  # noqa: E402
from shap_stability.modeling.xgboost_wrapper import predict_proba, train_xgb_classifier  # noqa: E402
from shap_stability.nested_cv import iter_outer_folds  # noqa: E402

PARAM_SPACE = {
    "n_estimators": {"type": "int", "low": 100, "high": 500},
    "max_depth": {"type": "int", "low": 2, "high": 6},
    "learning_rate": {"type": "float", "low": 0.01, "high": 0.2, "log": True},
    "subsample": {"type": "float", "low": 0.7, "high": 1.0},
    "colsample_bytree": {"type": "float", "low": 0.7, "high": 1.0},
    "min_child_weight": {"type": "float", "low": 1.0, "high": 20.0, "log": True},
    "reg_lambda": {"type": "float", "low": 1e-3, "high": 10.0, "log": True},
}

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
    parser.add_argument("--hpo-budget", type=int, default=64)
    parser.add_argument("--optimizer", default="grid", choices=["grid", "sobol", "smac"])
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

    ratios = [0.1, 0.3, 0.5]
    results_path = results_dir / "results.csv"

    for outer in iter_outer_folds(
        X_raw,
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
        X_train_raw = X_raw.iloc[outer.train_idx]
        y_train = y.iloc[outer.train_idx]
        X_test_raw = X_raw.iloc[outer.test_idx]
        y_test = y.iloc[outer.test_idx]
        X_train, X_test = one_hot_encode_train_test(X_train_raw, X_test_raw)

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
            use_space_opt = args.optimizer in ("sobol", "smac")
            hpo = tune_and_train(
                resampled.X,
                resampled.y,
                param_grid=PARAM_GRID,
                metric_name="roc_auc",
                inner_folds=args.inner_folds,
                seed=outer.seed,
                optimizer=args.optimizer,
                budget=args.hpo_budget if use_space_opt else None,
                param_space=PARAM_SPACE if use_space_opt else None,
                smac_output_dir=str(results_dir / "smac3_output") if args.optimizer == "smac" else None,
            )

            logger.info("Best HPO score=%.4f", hpo.best_score)
            X_train_enc, X_test_enc = one_hot_encode_train_test(
                resampled.X, X_test_raw
            )
            proba = predict_proba(hpo.model, X_test_enc)
            shap_result = compute_tree_shap(hpo.model, X_test_enc)
            pfi_result = compute_pfi_importance(
                hpo.model,
                X_test_enc,
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
                pfi_importance=pfi_result.mean.to_dict(),
                pfi_importance_std=pfi_result.std.to_dict(),
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
            "optimizer": args.optimizer,
            "hpo_budget": args.hpo_budget,
            "param_space": PARAM_SPACE,
        },
    )
    (results_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    logger.info("Results written to %s", results_dir)
    print(f"Wrote baseline results to {results_dir}")


if __name__ == "__main__":
    main()
