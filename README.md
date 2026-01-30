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
```

### Data cache

The German Credit dataset is downloaded on demand and cached at `./data/raw/` by default.
Set `SHAP_IT_DATA_DIR` to override the cache directory.
