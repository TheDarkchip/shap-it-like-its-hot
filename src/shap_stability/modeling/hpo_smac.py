from __future__ import annotations

from typing import Any
import time

from ConfigSpace import ConfigurationSpace
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformIntegerHyperparameter,
    UniformFloatHyperparameter,
)


from smac import HyperparameterOptimizationFacade, Scenario
from smac.runhistory.dataclasses import TrialValue

from .hpo_utils import _evaluate_params, _metric_greater_is_better 

def _space_to_configspace(param_space: dict[str, dict[str, Any]]) -> ConfigurationSpace:
    cs = ConfigurationSpace()
    for name, spec in param_space.items():
        t = spec["type"]
        if t == "float":
            hp = UniformFloatHyperparameter(
                name,
                lower=float(spec["low"]),
                upper=float(spec["high"]),
                log=bool(spec.get("log", False)),
            )
        elif t == "int":
            hp = UniformIntegerHyperparameter(
                name,
                lower=int(spec["low"]),
                upper=int(spec["high"]),
            )
        elif t == "cat":
            hp = CategoricalHyperparameter(
                name,
                choices=list(spec["choices"]),
            )
        else:
            raise ValueError(f"Unknown spec type: {t}")
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
) -> tuple[dict[str, Any], float]:
    cs = _space_to_configspace(param_space)
    greater_is_better = _metric_greater_is_better(metric_name)

    def objective(config, seed: int | None = None) -> float:
        params = dict(base_params or {})
        params.update(dict(config))
        score = _evaluate_params(
            X, y,
            params=params,
            metric_name=metric_name,
            inner_folds=inner_folds,
            seed=(seed if seed is not None else 0),
        )
        return -score if greater_is_better else score

    scenario = Scenario(cs, n_trials=budget, seed=seed, deterministic=False, output_directory= "results/smac3_output")
    intensifier = HyperparameterOptimizationFacade.get_intensifier(scenario, max_config_calls=1)

    smac = HyperparameterOptimizationFacade(
        scenario,
        objective,
        intensifier=intensifier,
        overwrite=True,
    )

    best_params = None
    best_score = None

    for _ in range(budget):
        info = smac.ask()
        cost = objective(info.config, seed=info.seed)
        smac.tell(info, TrialValue(cost=cost, time=0.0))

        score = -cost if greater_is_better else cost
        params = dict(base_params or {})
        params.update(dict(info.config))
        if best_score is None or (score > best_score if greater_is_better else score < best_score):
            best_score = score
            best_params = params

    if best_params is None or best_score is None:
        raise RuntimeError("SMAC produced no valid trials")

    return best_params, float(best_score)