from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter, UniformFloatHyperparameter,
    UniformIntegerHyperparameter,)

from smac import HyperparameterOptimizationFacade, Scenario
from smac.initial_design.sobol_design import SobolInitialDesign
from smac.runhistory.dataclasses import TrialValue

from .hpo_utils import _evaluate_params, _metric_greater_is_better


def _space_to_configspace(param_space: dict[str, dict[str, Any]]) -> ConfigurationSpace:
    cs = ConfigurationSpace()
    for name, spec in param_space.items():
        t = spec["type"]

        if t == "float":
            hp = UniformFloatHyperparameter(
                name=name,
                lower=float(spec["low"]),
                upper=float(spec["high"]),
                log=bool(spec.get("log", False)),
            )
        elif t == "int":
            hp = UniformIntegerHyperparameter(
                name=name,
                lower=int(spec["low"]),
                upper=int(spec["high"]),
            )
        elif t == "cat":
            hp = CategoricalHyperparameter(
                name=name,
                choices=list(spec["choices"]),
            )
        else:
            raise ValueError(f"Unknown spec type for {name}: {t}")

        cs.add_hyperparameter(hp)

    return cs


def select_best_params_smac(
    X,
    y,
    *,
    param_space: dict[str, dict[str, Any]],
    metric_name: str,
    inner_folds: int,
    seed: int,
    budget: int,
    base_params: dict[str, Any] | None = None,
    output_directory: str | Path | None = None,
    init_n_configs: int | None = None,
    init_max_ratio: float = 0.25,
) -> tuple[dict[str, Any], float]:
    if budget < 2:
        raise ValueError("SMAC budget must be >= 2")

    cs = _space_to_configspace(param_space)
    greater_is_better = _metric_greater_is_better(metric_name)

    def objective(config, seed: int = 0) -> float:
        params = dict(base_params or {})
        params.update(config.get_dictionary() if hasattr(config, "get_dictionary") else dict(config))

        score = _evaluate_params(
            X,
            y,
            params=params,
            metric_name=metric_name,
            inner_folds=inner_folds,
            seed=seed, 
        )

        if not np.isfinite(score):
            return float("inf")

        if greater_is_better:
            return 1.0 - float(score)
        
        return float(score)

    outdir = Path(output_directory) if output_directory is not None else Path("results/smac3_output")
    outdir.mkdir(parents=True, exist_ok=True)

    scenario = Scenario(
        configspace=cs,
        n_trials=budget,
        deterministic=True,
        seed=seed,
        output_directory=outdir,
    )

    initial_design = SobolInitialDesign(
        scenario,
        n_configs=init_n_configs,
        max_ratio=init_max_ratio,
        seed=seed,
    )

    intensifier = HyperparameterOptimizationFacade.get_intensifier(
        scenario,
        max_config_calls=1,  )

    smac = HyperparameterOptimizationFacade(
        scenario=scenario,
        target_function=objective,
        initial_design=initial_design,
        intensifier=intensifier,
        overwrite=True,
    )

    best_params: dict[str, Any] | None = None
    best_score: float | None = None

    for _ in range(budget):
        info = smac.ask()

        cost = objective(info.config)  
        value = TrialValue(cost=cost, time=0.0)

        smac.tell(info, value)

        score = 1.0 - cost if greater_is_better else cost
        params = dict(base_params or {})
        params.update(info.config.get_dictionary())

        if best_score is None:
            best_score = score
            best_params = params
        else:
            if greater_is_better and score > best_score:
                best_score = score
                best_params = params
            if (not greater_is_better) and score < best_score:
                best_score = score
                best_params = params

    if best_params is None or best_score is None:
        raise RuntimeError("SMAC produced no valid trials")

    return best_params, float(best_score)