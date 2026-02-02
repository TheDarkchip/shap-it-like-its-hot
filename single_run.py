"""Run a single end-to-end experiment and persist artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[0]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shap_stability.config_loader import load_config, validate_config
from shap_stability.experiment_utils import (
    configure_logging,
    create_run_metadata,
    generate_run_id,
    set_global_seed,
)
from shap_stability.data import load_german_credit
from shap_stability.metrics.metrics_utils import parse_metrics_config
from shap_stability.explain.pfi_utils import compute_pfi_importance
from shap_stability.resampling import resample_train_fold
from shap_stability.metrics.results_io import ResultRecord, append_record_csv
from shap_stability.explain.shap_utils import compute_tree_shap
from shap_stability.modeling.xgboost_wrapper import predict_proba, train_xgb_classifier


@dataclass(frozen=True)
class RunArtifacts:
    results_path: Path
    metadata_path: Path
    log_path: Path


def _prepare_features(X: pd.DataFrame) -> pd.DataFrame:
    frame = pd.DataFrame(X).copy()
    encoded = pd.get_dummies(frame, drop_first=False)
    return encoded.reindex(sorted(encoded.columns), axis=1)


def _compute_metrics(y_true: pd.Series, y_proba: np.ndarray) -> dict[str, float]:
    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_proba >= 0.5)),
    }

    if pd.Series(y_true).nunique() < 2:
        metrics["roc_auc"] = float("nan")
    else:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))

    return metrics


def _iter_splits(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    outer_folds: int,
    outer_repeats: int,
    seed: int,
) -> Iterable[tuple[int, int, np.ndarray, np.ndarray, int]]:
    for repeat_id in range(outer_repeats):
        repeat_seed = seed + repeat_id
        splitter = StratifiedKFold(
            n_splits=outer_folds,
            shuffle=True,
            random_state=repeat_seed,
        )
        for fold_id, (train_idx, test_idx) in enumerate(splitter.split(X, y)):
            yield repeat_id, fold_id, train_idx, test_idx, repeat_seed


def run_single_experiment(
    cfg: dict[str, Any],
    *,
    output_dir: str | Path | None = None,
    data: tuple[pd.DataFrame, pd.Series] | None = None,
) -> RunArtifacts:
    validate_config(cfg)

    experiment_cfg = cfg["experiment"]
    run_name = experiment_cfg.get("name", "run")
    seed = int(experiment_cfg.get("random_seed", 0))

    run_id = generate_run_id(prefix=run_name)
    if output_dir:
        base_dir = Path(output_dir) / run_id
    else:
        base_dir = Path("artifacts") / run_id
    base_dir.mkdir(parents=True, exist_ok=True)

    results_path = base_dir / "results.csv"
    metadata_path = base_dir / "run_metadata.json"
    log_path = base_dir / "run.log"

    logger = configure_logging(
        run_id=run_id,
        seed=seed,
        log_file=log_path,
        force=True,
        logger_name="shap-it-like-its-hot",
    )

    set_global_seed(seed)

    if data is None:
        X_raw, y = load_german_credit()
    else:
        X_raw, y = data

    X = _prepare_features(X_raw)

    outer_folds = int(cfg["cv"].get("outer_folds", 3))
    outer_repeats = int(cfg["cv"].get("outer_repeats", 1))
    ratios = list(cfg["resampling"].get("target_positive_ratios", []))
    model_cfg = cfg.get("model", {})
    model_name = model_cfg.get("name", "xgboost")
    model_params = model_cfg.get("params", {})
    metrics_cfg = parse_metrics_config(cfg["metrics"])

    if model_name != "xgboost":
        raise ValueError("Only model.name='xgboost' is supported for now")

    if not ratios:
        raise ValueError("resampling.target_positive_ratios must be non-empty")

    logger.info(
        "Starting run: folds=%s repeats=%s ratios=%s model=%s",
        outer_folds,
        outer_repeats,
        ratios,
        model_name,
    )

    for repeat_id, fold_id, train_idx, test_idx, repeat_seed in _iter_splits(
        X,
        y,
        outer_folds=outer_folds,
        outer_repeats=outer_repeats,
        seed=seed,
    ):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]

        for ratio in ratios:
            resampled = resample_train_fold(
                X_train,
                y_train,
                target_positive_ratio=float(ratio),
                random_state=repeat_seed,
            )
            achieved_ratio = resampled.positive_count / (
                resampled.positive_count + resampled.negative_count
            )
            train_result = train_xgb_classifier(
                resampled.X,
                resampled.y,
                params=model_params,
                random_state=repeat_seed,
            )
            proba = predict_proba(train_result.model, X_test)
            metrics = _compute_metrics(y_test, proba)
            shap_result = compute_tree_shap(train_result.model, X_test)
            pfi_importance = compute_pfi_importance(
                train_result.model,
                X_test,
                y_test,
                metric_name=metrics_cfg.primary,
                n_repeats=int(cfg["metrics"].get("pfi_repeats", 5)),
                random_state=repeat_seed,
            )

            record = ResultRecord(
                fold_id=fold_id,
                repeat_id=repeat_id,
                seed=repeat_seed,
                model_name=model_name,
                class_ratio=achieved_ratio,
                metrics=metrics,
                shap_importance=shap_result.global_importance.to_dict(),
                pfi_importance=pfi_importance.to_dict(),
            )
            append_record_csv(results_path, record)

    metadata = create_run_metadata(
        run_id=run_id,
        seed=seed,
        extra={"experiment": run_name},
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    logger.info("Results written to %s", results_path)

    return RunArtifacts(
        results_path=results_path,
        metadata_path=metadata_path,
        log_path=log_path,
    )


def run_from_config(
    config_path: str | Path,
    *,
    output_dir: str | Path | None = None,
) -> RunArtifacts:
    cfg = load_config(config_path)
    return run_single_experiment(cfg, output_dir=output_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run a single experiment")
    parser.add_argument("config", help="Path to YAML config file")
    parser.add_argument("--output-dir", help="Directory for artifacts")
    args = parser.parse_args()

    artifacts = run_from_config(args.config, output_dir=args.output_dir)
    print(f"Wrote results to {artifacts.results_path}")
