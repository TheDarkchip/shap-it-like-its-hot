# shap-it-like-its-hot

This repository will contain the code and experiments for a student project on the **stability and reliability of global SHAP explanations** in a credit-scoring context.

The project empirically evaluates global SHAP feature importances under controlled experimental conditions, including:
- varying class imbalance,
- correlated and duplicated features,
- comparison with permutation feature importance (PFI).

Experiments are based on cross-validated evaluations using the German Credit dataset.

**Status:** Baseline MVS complete.  
The core MVS baseline runs, metrics, and plots are available under `results/` and summarized in `docs/mvs_results_summary.md`.

## Setup

This project uses `uv` for dependency management.

### Prerequisites

- Python >= 3.13
- `uv` installed

### Install dependencies

```bash
uv sync
```

### Common commands

```bash
# Run tests
uv run pytest

# Lint and format
uv run ruff check
uv run ruff format

# Validate a config
uv run python scripts/run_config.py configs/example.yaml

# Run a single experiment (writes artifacts under ./artifacts/<run_id>/)
uv run python single_run.py configs/example.yaml

# Run the MVS baseline with HPO (writes under ./results/<run_id>/)
uv run python scripts/run_mvs_hpo.py

# Regenerate MVS tables/plots for a given run
uv run python scripts/generate_mvs_report.py results/<run_id>
```

### Layout

- Reusable code lives under `src/shap_stability/`.
- CLI scripts live under `scripts/`.
- Tests live under `tests/`.

### Data cache

The German Credit dataset is downloaded on demand and cached at `./data/raw/` by default.
Set `SHAP_IT_DATA_DIR` to override the cache directory.

### Artifacts

Single-run experiments create a run folder with:
- `results.csv` (metrics + SHAP + PFI importances)
- `run_metadata.json` (seed, environment, run_id)
- `run.log` (structured logs with run id/seed)

MVS runs additionally generate:
- `stability_summary.csv`, `agreement_summary.csv`
- `stability_table.csv`, `agreement_table.csv`
- plots under `results/<run_id>/plots/`

### Protocol

See `docs/protocol.md` for the nested CV protocol, metrics, and evaluation flow.
