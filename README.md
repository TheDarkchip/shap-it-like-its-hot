# shap-it-like-its-hot

This repository will contain the code and experiments for a student project on the **stability and reliability of global SHAP explanations** in a credit-scoring context.

The project empirically evaluates global SHAP feature importances under controlled experimental conditions, including:
- varying class imbalance,
- correlated and duplicated features,
- comparison with permutation feature importance (PFI).

Experiments are based on cross-validated evaluations using the German Credit dataset.

**Status:** Work in progress.  
At the time of writing, no finalized experiments or results are included. Code and documentation will be added as the project is implemented.

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

# Run a single experiment (writes artifacts under ./artifacts/<run_id>/)
uv run python scripts/run_single.py configs/example.yaml

# Override the output directory (still creates a run_id subfolder)
uv run python scripts/run_single.py configs/example.yaml --output-dir artifacts
```

### Data cache

The German Credit dataset is downloaded on demand and cached at `./data/raw/` by default.
Set `SHAP_IT_DATA_DIR` to override the cache directory.

### Artifacts

Single-run experiments create a run folder with:
- `results.csv` (metrics + SHAP + PFI importances)
- `run_metadata.json` (seed, environment, run_id)
- `run.log` (structured logs with run id/seed)
