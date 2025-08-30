# nocturne_bloom
NVIDIA (NVDA) stock price prediction pipeline and Dash dashboard.

<p align="center">
   <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="license" />
   <img src="https://img.shields.io/badge/python-%3E%3D3.10-blue.svg" alt="python" />
</p>

<p align="center">Built with the tools and technologies:</p>

<p align="center">
<img src="https://img.shields.io/badge/Dash-Plotly-1f77b4?logo=dash&logoColor=white" alt="Dash"/>
<img src="https://img.shields.io/badge/Plotly-visuals-3f6ad8?logo=plotly&logoColor=white" alt="Plotly"/>
<img src="https://img.shields.io/badge/pandas-data-150458?logo=pandas&logoColor=white" alt="pandas"/>
<img src="https://img.shields.io/badge/NumPy-numeric-013243?logo=numpy&logoColor=white" alt="NumPy"/>
<img src="https://img.shields.io/badge/scikit--learn-ml-f7931e?logo=scikit-learn&logoColor=white" alt="scikit-learn"/>
<img src="https://img.shields.io/badge/PyTorch-deep%20learning-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch"/>
<br>
<img src="https://img.shields.io/badge/Lightning-PyTorch-7d3cff?logo=github&logoColor=white" alt="Lightning"/>
<img src="https://img.shields.io/badge/Ray-distributed-00b2ff?logo=ray&logoColor=white" alt="Ray"/>
<img src="https://img.shields.io/badge/Optuna-hpo-7b61ff?logo=optuna&logoColor=white" alt="Optuna"/>
<img src="https://img.shields.io/badge/pytest-testing-1e90ff?logo=pytest&logoColor=white" alt="pytest"/>
<img src="https://img.shields.io/badge/pylint-linter-4b32c3?logo=pylint&logoColor=white" alt="pylint"/>
</p>

A complete repository for building, evaluating, backtesting, and visualising next-day closing-price predictions for NVIDIA (NVDA). The repo combines data collection and preparation, feature engineering, an ensemble of predictive models (including CNN components), hyperparameter optimisation, backtesting, and a production-oriented Dash app for exploration.

## Table of contents
- [What this repo does](#what-this-repo-does)
- [Quick start](#quick-start)
- [Installation](#installation-notes)
- [Running the app](#quick-start)
- [Project structure](#project-structure)
- Important workflow notes:
   - [Running common workflows](#running-common-workflows)
   - [Environment variables and secrets](#environment-variables-and-secrets)
   - [Long-running operations & resource notes](#long-running-operations--resource-notes)
- [License](#license)

## What this repo does

- Loads and prepares historical OHLCV data for NVDA (and supporting tickers).
- Builds an ensemble prediction system (CNN components + other models) to forecast next-day closing price.
- Runs hyperparameter optimisation (Ray Tune / Optuna) and logs experiments using Weights & Biases.
- Evaluates model predictions using standard regression and trading metrics and supports rolling backtests.
- Exposes an interactive Dash dashboard at `app/app.py` for data/metrics/visualisation and to orchestrate pipeline steps from the UI - can be run from `run.py`.

## Quick start

1) Clone the repository and change into the project directory:

```bash
git clone https://github.com/rayanjoshi/nocturne_bloom.git
cd nocturne_bloom
```

2) Create a virtual environment and install dependencies (from `pyproject.toml`):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.lock
```

3) Provide required secrets in a `.env` at the repository root (MANDATORY before running scripts or the app):

Use the provided `.env.example` as a template. You can either copy it and fill values, or populate secrets from the Dash UI (recommended):

```bash
cp .env.example .env
```

4) Run the Dash app:

```bash
python run.py

```

Then open the Dash UI in your browser at the printed local address (default: http://127.0.0.1:8050).

## Installation notes

- This repository lists dependencies in `pyproject.toml` and a `requirements.lock` file. Installing from `requirements.lock` will give a stable, reproducible environment.
- The project uses PyTorch/PyTorch Lightning, Ray Tune/Optuna, Dash + dash-bootstrap-components, and financial/data helper libraries such as `wrds`, `pandas-ta` and `backtrader`.

## Running common workflows

- Data preparation (from the app UI): use the Data Preparation page to supply WRDS credentials and click "Prepare Data" — the app runs `src/data_loader.py`, `src/feature_engineering.py` and `src/data_module.py` in sequence.
- Hyperparameter tuning: the Training page triggers `scripts/tune_model.py` via the UI (Ray Tune / Optuna).
- Model training: the Training page triggers `scripts/train_model.py`.
- Backtesting / evaluation: the Backtesting page runs `scripts/run_backtest.py` and reads results into `data/predictions/`.

If you prefer CLI, these scripts live under `scripts/` and can be invoked with the repository Python environment. Example:

```bash
python scripts/train_model.py
python scripts/run_backtest.py
```

## Project structure

```
nocturne_bloom
├─ LICENSE
├─ README.md
├─ app
│  ├─ __init__.py
│  ├─ app.py
│  └─ assets
│     └─ model_architecture.png
├─ cache
├─ configs
│  ├─ backtest.yaml
│  ├─ config.yaml
│  ├─ data_loader.yaml
│  ├─ data_module.yaml
│  ├─ feature_engineering.yaml
│  ├─ model_ensemble.yaml
│  └─ trainer.yaml
├─ data
│  ├─ predictions
│  ├─ preprocessing
│  ├─ processed
│  └─ raw
├─ models
│  └─ scalers
├─ pyproject.toml
├─ requirements.lock
├─ run.py
├─ scripts
│  ├─ __init__.py
│  ├─ feature_correlation.py
│  ├─ feature_importance.py
│  ├─ logging_config.py
│  ├─ run_backtest.py
│  ├─ train_model.py
│  └─ tune_model.py
└─ src
   ├─ WRDS_query.sql
   ├─ __init__.py
   ├─ data_loader.py
   ├─ data_module.py
   ├─ feature_engineering.py
   └─ model_ensemble.py

```

## Environment variables and secrets


This project requires a `.env` file at the repository root. A safe-to-commit template is included as `.env.example`.

   Use the Dash UI to pass secrets (expected flow for interactive users): the Data Preparation and Training pages include input fields for WRDS credentials and the W&B API key — submitting those forms will write the values into the repo-root `.env` file so the app and the scripts it launches can access them.

### How the UI writes `.env`

When you submit credentials in the Dash UI the app's callbacks read the existing `.env` (if present), update or append the submitted keys (`WRDS_USERNAME`, `WRDS_PASSWORD`, `WANDB_API_KEY`), and rewrite the file at the repository root using simple key=value lines. Values are stored in plain text.

## Long-running operations & resource notes

- Hyperparameter tuning (Ray Tune / Optuna) and model training are CPU/GPU heavy. Run these on a machine with a GPU or increase timeouts accordingly.
- The UI executes training/tuning via `subprocess` calls to scripts in `scripts/`. Expect those operations to take minutes–hours depending on configuration.

## License

This repository is released under the MIT License — see `LICENSE`.