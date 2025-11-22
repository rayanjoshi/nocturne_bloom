from dash import html
from pathlib import Path
from dash import register_page

script_dir = Path(__file__).parent  # /path/to/repo/app/pages/
repo_root = script_dir.parent.parent  # /path/to/repo/

# Explicit page path for Dash pages discovery
path = "/"

# Ensure Dash registers this page at import time
register_page(__name__, path=path)

layout = html.Div(
    [
        html.H1(
            "NVDA Stock Prediction Dashboard",
            className="mb-4",
            style={"color": "white"},
        ),
        html.H3(
            "Harness AI for Financial Insights",
            className="mb-4",
            style={"color": "white"},
        ),
        html.Div(
            [
                html.H5(
                    "1. Data Preparation", className="mb-2", style={"color": "white"}
                ),
                html.P(
                    "Access Wharton Research Data Services (WRDS) to retrieve Nvidia (NVDA) "
                    "stock data (2004-2022) "
                    "and market indices (SPY, QQQ, VIX).",
                    className="mb-3",
                    style={"color": "white"},
                ),
                html.P(
                    "Transform data into predictive features and prepare it for "
                    "model training using PyTorch Lightning.",
                    className="mb-3",
                    style={"color": "white"},
                ),
                html.H5(
                    "2. Model Training", className="mb-2", style={"color": "white"}
                ),
                html.P(
                    "Train an ensemble model with a multihead CNN for pattern recognition, Ridge "
                    "regression for price "
                    "forecasting, and an LSTM for trend prediction, unified by a meta-learner "
                    "for enhanced accuracy.",
                    className="mb-3",
                    style={"color": "white"},
                ),
                html.P(
                    "Optimise hyperparameters using Ray Tune and Optuna, with Weights & Biases "
                    "for experiment tracking.",
                    className="mb-3",
                    style={"color": "white"},
                ),
                html.H5("3. Backtesting", className="mb-2", style={"color": "white"}),
                html.P(
                    "Evaluate the model on NVDA data (2023-2024) with metrics like Sharpe ratio "
                    "and maximum drawdown.",
                    className="mb-3",
                    style={"color": "white"},
                ),
                html.P(
                    "Validate trading strategies to ensure robust, data-driven decisions.",
                    className="mb-3",
                    style={"color": "white"},
                ),
            ]
        ),
    ]
)
