from __future__ import annotations

from typing import Any, Dict, List
import math

try:
    from scipy.stats import qmc
except ImportError as e:
    raise ImportError("Sobol sampler requires scipy. Install scipy or use optimizer='grid'.") from e


def _map_u(u: float, spec: Dict[str, Any]) -> Any:
    t = spec["type"]
    if t == "float":
        lo, hi = float(spec["low"]), float(spec["high"])
        if spec.get("log", False):
            lo, hi = math.log(lo), math.log(hi)
            val = math.exp(lo + u * (hi - lo))
        else:
            val = lo + u * (hi - lo)
        return float(val)

    if t == "int":
        lo, hi = int(spec["low"]), int(spec["high"])
        return int(lo + math.floor(u * (hi - lo + 1)))

    if t == "cat":
        choices = list(spec["choices"])
        idx = min(int(math.floor(u * len(choices))), len(choices) - 1)
        return choices[idx]

    raise ValueError(f"Unknown param spec type: {t}")


def suggest_configs(param_space: Dict[str, Dict[str, Any]], budget: int, seed: int) -> List[Dict[str, Any]]:
    keys = list(param_space.keys())
    d = len(keys)
    if d == 0:
        raise ValueError("param_space must be non-empty for Sobol")

    m = int(math.ceil(math.log2(max(budget, 1))))
    engine = qmc.Sobol(d=d, scramble=True, seed=seed)
    U = engine.random_base2(m=m)[:budget]

    out: List[Dict[str, Any]] = []
    for row in U:
        cfg: Dict[str, Any] = {}
        for k, u in zip(keys, row):
            cfg[k] = _map_u(float(u), param_space[k])
        out.append(cfg)
    return out