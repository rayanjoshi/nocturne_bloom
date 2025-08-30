"""
Machine Learning Stock Prediction Dashboard.

This module implements a Dash web application for stock price prediction using
machine learning techniques. The application provides an interactive interface to
manage data preparation, model training, hyperparameter optimization, and backtesting
for Nvidia (NVDA) stock data. It integrates with Wharton Research Data Services (WRDS)
for data retrieval, uses PyTorch Lightning for model training, and employs Ray Tune
and Optuna for hyperparameter optimization. The dashboard includes visualizations
and performance metrics to evaluate model predictions.

Key features include:
- Data preparation with WRDS login and data processing pipelines
- Ensemble model training with CNN, Ridge Regression, LSTM, and a meta-learner
- Hyperparameter optimization with Ray Tune and Optuna
- Backtesting with performance metrics (e.g., Sharpe ratio, MAPE)
- Interactive UI with Bootstrap styling and Plotly visualizations

Dependencies:
    pathlib, subprocess, sys, json, numpy, dash, dash_bootstrap_components,
    pandas, diskcache, plotly.express, plotly.io
"""
from pathlib import Path
import subprocess
import sys
import json
import numpy as np
from dash import Dash, html, dcc, Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import dash
import plotly.express as px
import plotly.io as pio
import pandas as pd
import diskcache

script_dir = Path(__file__).parent  # /path/to/repo/app
repo_root = script_dir.parent  # /path/to/repo/

cache_location = Path(repo_root / "cache").resolve()
cache = diskcache.Cache(cache_location)
background_callback_manager = dash.DiskcacheManager(cache)

app = Dash(__name__,
            use_pages=True,
            external_stylesheets=[dbc.themes.BOOTSTRAP,
                dbc.icons.FONT_AWESOME],
            suppress_callback_exceptions=True,
            prevent_initial_callbacks=True,
            pages_folder=""
            )

app.title = "ML Stock Prediction"

server = app.server

dash.register_page(
    "Home",
    path="/",
    layout=html.Div([
        html.H1("NVDA Stock Prediction Dashboard", className="mb-4", style={"color": "white"}),
        html.H3("Harness AI for Financial Insights", className="mb-4", style={"color": "white"}),
        html.Div([
            html.H5("1. Data Preparation", className="mb-2", style={"color": "white"}),
            html.P(
                "Access Wharton Research Data Services (WRDS) to retrieve Nvidia (NVDA) "
                "stock data (2004-2022) "
                "and market indices (SPY, QQQ, VIX).",
                className="mb-3",
                style={"color": "white"},
            ),
            html.P("Transform data into predictive features and prepare it for "
                "model training using PyTorch Lightning.",
                className="mb-3",
                style={"color": "white"},
            ),
            html.H5("2. Model Training", className="mb-2", style={"color": "white"}),
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
        ])
    ])
)


data_load_save_location = Path(repo_root / "data/raw/nvda_raw_data.csv").resolve()
pio.templates.default = "plotly_dark"
data_load_df = pd.read_csv(data_load_save_location)
data_load_fig = px.line(data_load_df,
                            x="date",
                            y="Close",
                            title="NVDA Daily Closing Prices",
                            )
data_load_fig.update_layout(
    yaxis_title_text="Closing Price / $",
    xaxis_title_text="Date",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)"
)
data_load_fig.update_traces(line_color="#0076b9")

dash.register_page("Data Preparation", path="/data-preparation",
    layout=html.Div([
        html.H1("Data Preparation",
                style={"color": "white", "text-align": "left", "margin-bottom": "1.25rem"}
        ),
        html.H3("Access WRDS to retrieve stock data and prepare for model training",
                style={"color": "white", "margin-bottom": "1.875rem"}
        ),
        html.Div([
            # Left column (WRDS login & information)
            html.Div([
                html.H5("WRDS LOGIN", style={"color": "white", "margin-bottom": "0.625rem"}),
                html.P("USERNAME", style={"color": "white", "margin-bottom": "0.3125rem"}),
                dcc.Input(placeholder="Enter your WRDS username",
                    type="text",
                    className="mb-3",
                    id="wrds-username",
                    style={"width": "100%", "margin-bottom": "0.625rem"}),
                html.P("PASSWORD", style={"color": "white", "margin-bottom": "0.3125rem"}),
                dcc.Input(placeholder="Enter your WRDS password",
                    type="password",
                    className="mb-3",
                    id="wrds-password",
                    style={"width": "100%", "margin-bottom": "0.9375rem"}),
                html.Button("LOGIN",
                    id="login-button",
                    className="btn btn-primary data-load-button",
                    ),
                html.P("Passes login details to .env file",
                    style={"color": "white", "font-size": "0.875rem", "line-height": "1.4"}
                    ),
                html.Div(id="login-status", style={"margin-bottom": "0.625rem"}),
                html.Div([
                    html.H6("Information",
                    style={"color": "white", "margin-bottom": "0.625rem"}),
                    html.P("Date Range: 2004-10-31 - 2022-12-31",
                    style={"color": "white", "font-size": "0.875rem", "line-height": "1.4"}),
                    html.P("Stock: NVDA",
                    style={"color": "white", "font-size": "0.875rem", "line-height": "1.4"}),
                    html.P("Other Tickers: SPY, QQQ, VIXY ",
                    style={"color": "white", "font-size": "0.875rem", "line-height": "1.4"}),
                    html.P("Economic Indicators: 10 Year Treasury Yield",
                    style={"color": "white", "font-size": "0.875rem", "line-height": "1.4"}),
                    html.P("Some data were not available in the date range, so were imputed "
                            "using existing data. The date range was extended back to 2004, "
                            "so PE and PB ratios could also be calculated for early 2005 data.",
                    style={"color": "white", "font-size": "0.875rem", "line-height": "1.4"})
                ],
                style={"margin-top": "2.5rem"})
            ], style={
                "width": "35%",
                "padding": "1.25rem",
                "border-right": "0.125rem solid #333",
                "min-height": "31.25rem"
            }),

            # Right column (graph, buttons, status)
            html.Div([
                html.Div([
                    dcc.Graph(figure=data_load_fig, style={"height": "25rem"})
                ], style={"margin-bottom": "0.9375rem"}),

                html.Div([
                    html.Button(
                        "Prepare Data",
                        id="prepare-data-button",
                        className="btn btn-primary data-load-button",
                    )
                ]),

                html.Progress(id="data-loading-progress",
                                value="0",
                                style={"height": "0.25rem"}
                            ),

                html.Div(id="process-status", style={"margin-top": "0.3125rem"}),
            ], style={"width": "65%", "padding": "1.25rem", "text-align": "center"})
        ], style={
            "display": "flex",
            "width": "100%",
            "margin-top": "1.25rem"
        })
    ])
)

@app.callback(
    Output('login-status', 'children'),
    Input('login-button', 'n_clicks'),
    [State('wrds-username', 'value'),
    State('wrds-password', 'value')],
    prevent_initial_call=True
)
def update_env_file(n_clicks, username, password):
    """
    Update the .env file with WRDS credentials and provide user feedback.

    This function handles the updating of WRDS credentials in the .env file
    located at the repository root. It reads existing environment variables,
    updates them with the provided username and password, and writes them back
    to the .env file. It returns an HTML Div component with a success or error
    message to display to the user.

    Args:
        n_clicks (Optional[int]): Number of times the submit button was clicked.
            If None, the update is prevented.
        username (str): WRDS username to be saved in the .env file.
        password (str): WRDS password to be saved in the .env file.

    Returns:
        html.Div: A Dash HTML Div component containing a success or error message
            with appropriate styling.

    Raises:
        PreventUpdate: If n_clicks is None, indicating no user action.
        OSError: If there are issues reading from or writing to the .env file.
        ValueError: If the .env file contains malformed key-value pairs.
    """
    if n_clicks is None:
        raise PreventUpdate

    if not username or not password:
        return html.Div("Please enter both username and password",
                        style={"color": "#b90076", "font-size": "0.75rem"})

    try:
        # Define the path to the .env file (in the repo root)
        env_path = Path(repo_root / ".env").resolve()

        # Read existing .env file if it exists
        env_vars = {}
        if env_path.exists():
            with open(env_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip() and not line.startswith('#'):
                        key, value = line.strip().split('=', 1)
                        env_vars[key] = value

        # Update with new credentials
        env_vars['WRDS_USERNAME'] = username
        env_vars['WRDS_PASSWORD'] = password

        # Write back to .env file
        with open(env_path, 'w', encoding='utf-8') as f:
            for key, value in env_vars.items():
                f.write(f"{key}={value}\n")

        return html.Div("✓ Credentials saved successfully!",
                        style={"color": "#76B900", "font-size": "0.75rem"})

    except (OSError, ValueError) as e:
        return html.Div(f"Error saving credentials: {str(e)}",
                        style={"color": "#b90076", "font-size": "0.75rem"})

@app.callback(
    Output('process-status', 'children'),
    Output('prepare-data-button', 'disabled'),
    Input('prepare-data-button', 'n_clicks'),
    prevent_initial_call=True,
    background=True,
    manager=background_callback_manager,
    running=[
        (Output('prepare-data-button', 'disabled'), True, False)
    ],
    progress=[Output("data-loading-progress", "value"), Output("data-loading-progress", "max")],
)
def process_data_pipeline(set_progress, n_clicks):
    """
    Execute a sequence of data processing scripts and return status messages.

    This function runs a predefined list of Python scripts in the specified order
    from the 'src' directory. It captures the output of each script and generates
    styled HTML status messages indicating success or failure.

    Args:
        n_clicks (int or None): Number of times the button was clicked. If None,
            the function raises PreventUpdate to halt execution.

    Returns:
        dash.html.Div: A Div containing styled HTML status messages for each script
            execution, including success, failure, or error messages.

    Raises:
        PreventUpdate: If n_clicks is None, preventing the pipeline from running.
        OSError: If there is an issue accessing or running the script files.
        ValueError: If there is an issue with the script execution parameters.
        RuntimeError: If an unexpected error occurs during script execution.
    """
    if n_clicks is None:
        raise PreventUpdate

    # Get the src directory path
    local_script_dir = Path(__file__).parent
    local_repo_root = local_script_dir.parent
    src_dir = local_repo_root / 'src'

    # Define the scripts to run in order
    scripts = [
        ('Data Loader', 'data_loader.py'),
        ('Feature Engineering', 'feature_engineering.py'),
        ('Data Module', 'data_module.py')
    ]

    total_scripts = len(scripts)
    status_messages = []
    set_progress((str(0), str(total_scripts)))

    try:
        for i, (script_name, script_file) in enumerate(scripts, 1):
            status_messages.append(
                html.P(f"Running {script_name}...",
                        style={"color": "yellow", "font-size": "0.75rem", "margin": "0.0625rem 0"}))

            script_path = src_dir / script_file
            all_success = True
            # Run the script
            result = subprocess.run([
                sys.executable, str(script_path)
            ], capture_output=True, text=True, check=False, cwd=src_dir.parent)
            set_progress((str(i), str(total_scripts)))

            if result.returncode == 0:
                status_messages.append(
                    html.P(f"✓ {script_name} completed successfully",
                        style={"color": "#76B900", "font-size": "0.75rem",
                                "margin": "0.0625rem 0"}))
            else:
                error_msg = result.stderr if result.stderr else result.stdout
                status_messages.append(
                    html.P(f"✗ {script_name} failed",
                        style={"color": "#b90076", "font-size": "0.75rem",
                                "margin": "0.0625rem 0"}))
                status_messages.append(
                    html.P(f"Error: {error_msg[:200]}...",
                        style={"color": "#b90076", "font-size": "0.75rem", "margin": "0.0626rem 0",
                                "font-family": "monospace"}))
                all_success = False
                break
        if all_success:
            status_messages.clear()
            status_messages.append(
                html.P("All data processing completed successfully!",
                    style={"color": "#76B900", "font-size": "0.875rem",
                            "margin": "0.625rem 0", "font-weight": "bold"}))
    except (OSError, ValueError, RuntimeError) as e:
        status_messages.append(
            html.P(f"Pipeline Error: {str(e)}",
                style={"color": "#b90076", "font-size": "0.75rem", "margin": "0.0625rem 0"}))
        return (
            html.Div(status_messages),
            False
        )
    return (
        html.Div(status_messages),
        False
    )

dash.register_page("Training", path="/training",
    layout=html.Div([
        html.Div([
            html.H1("Model Training",
                style={"color": "white", "text-align": "left",
                        "margin-bottom": "1.25rem"}),
            html.H3("Train and Fine-tune the Ensemble Model",
                style={"color": "white", "margin-bottom": "1.875rem"})
        ]),

        html.Div([
            html.Div([
                html.Img(src="/assets/model_architecture.png",
                        style={"width": "100%", "height": "auto",
                                "border": "2px solid #555"})
            ], style={
                "width": "48%",
                "display": "inline-block",
                "vertical-align": "top"
            }),

            # Right column - Text and Buttons
            html.Div([
                html.Div([
                    html.H6("Information",
                style={"color": "white", "margin-bottom": "0.5rem"}),
            html.P("Goal: To produce direction-aware price predictions",
                style={"color": "white", "font-size": "0.875rem",
                        "line-height": "1.4"}),
            html.P("Ensemble Model Components: Multihead CNN, "
                    "Ridge Regression, LSTM, Meta-Learner",
                style={"color": "white", "font-size": "0.875rem",
                        "line-height": "1.4"}),
            html.P("Hyperparameter Optimisation: Ray Tune with Optuna",
                style={"color": "white", "font-size": "0.875rem",
                        "line-height": "1.4"}),
            html.P("Hyperparameter Optimisation will take time, please be patient.",
                style={"color": "white", "font-size": "0.875rem",
                        "line-height": "1.4"}),
            html.P("There are no loading indicators, due to the asynchronous "
                    "nature of the training process.",
                style={"color": "white", "font-size": "0.875rem",
                        "line-height": "1.4"}),

            ], style={"border-bottom": "1px solid #555",
                            "padding-bottom": "0rem", "margin-bottom": "1.5rem"}),

                html.Button("Optimise Hyperparameters",
                            id="optimise-hyperparameters",
                            className="btn btn-primary train-button"),
                html.Button("Train Model",
                            id="train-model",
                            className="btn btn-primary train-button"),
                html.Button("Train with optimised hyperparameters",
                            id="train-optimised",
                            disabled=True,
                            className="btn btn-primary train-button"),
                html.Div(id="train-status", style={"margin-bottom": "0.625rem"}),
            ], style={
                "width": "48%",
                "display": "inline-block",
                "vertical-align": "top",
                "margin-left": "4%"
            })
        ], style={"margin-bottom": "1rem"}),

        html.Div([
            html.H5("Weights and Biases API Key",
                style={"color": "white", "margin-bottom": "1rem"}),
            html.Div([
                dcc.Input(placeholder="Enter your W&B API key",
                        type="password",
                        id="wandb-api-key",
                        className="mb-3 api-input",
                        style={"width": "70%", "margin-right": "2%",
                                "height": "48px",
                                "box-sizing": "border-box"}),
                html.Button("Submit",
                            id="submit-wandb-api-key",
                            className="btn btn-primary train-button",
                            style={"width": "25%",
                                    "height": "48px", "box-sizing": "border-box"}),
            ], style={"display": "flex", "align-items": "stretch",
                        "margin-bottom": "1rem"}),
            html.Div(id="wandb-api-key-status", style={"margin-bottom": "0.625rem"})
        ])
    ])
)

@app.callback(
    Output("wandb-api-key-status", "children"),
    Input("submit-wandb-api-key", "n_clicks"),
    [State("wandb-api-key", "value")],
    prevent_initial_call=True
)
def save_wandb_api_key(n_clicks, api_key):
    """
    Save the provided Weights & Biases API key to the .env file.

    Args:
        n_clicks (int): Number of times the submit button has been clicked.
        api_key (str): The API key entered by the user.

    Returns:
        html.Div: A Div component containing a success or error message with appropriate styling.

    Raises:
        PreventUpdate: If the function is called without a button click.
        OSError: If there is an issue reading or writing to the .env file.
        ValueError: If the .env file contains invalid key-value pairs.
    """
    if n_clicks is None:
        raise PreventUpdate

    if not api_key:
        return html.Div("Please enter a valid API key",
                            style={"color": "#b90076", "font-size": "0.75rem"})
    try:
        env_path = Path(repo_root / ".env").resolve()
        env_vars = {}
        if env_path.exists():
            with open(env_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip() and not line.startswith('#'):
                        key, value = line.strip().split('=', 1)
                        env_vars[key] = value

        # Update with new API key
        env_vars['WANDB_API_KEY'] = api_key
        with open(env_path, 'w', encoding='utf-8') as f:
            for key, value in env_vars.items():
                f.write(f"{key}={value}\n")

        return html.Div("✓ API key saved successfully!",
                        style={"color": "#76B900", "font-size": "0.75rem"})

    except (OSError, ValueError) as e:
        return html.Div(f"Error saving API key: {str(e)}",
                        style={"color": "#b90076", "font-size": "0.75rem"})

@app.callback(
    Output("train-status", "children"),
    Output("optimise-hyperparameters", "disabled"),
    Output("train-optimised", "disabled"),
    Input("optimise-hyperparameters", "n_clicks"),
    prevent_initial_call=True,
    background=True,
    manager=background_callback_manager,
    running=[
        (Output('optimise-hyperparameters', 'disabled'), True, False),
        (Output('train-model', 'disabled'), True, False)
    ],
)
def optimise_hyperparameters(n_clicks):
    """
    Run the hyperparameter optimization script and update the UI with status.

    Args:
        n_clicks (int): Number of times the optimize button has been clicked.

    Returns:
        tuple: Contains:
            - html.Div: A Div component with status messages for the optimization process.
            - bool: False to enable the optimize button.
            - bool: False to enable the train-optimized button.

    Raises:
        PreventUpdate: If the function is called without a button click.
        OSError: If there is an issue accessing the script directory or file.
        ValueError: If there is an issue with script execution parameters.
        RuntimeError: If the script execution encounters an unexpected error.
    """
    if n_clicks is None:
        raise PreventUpdate

    local_script_dir = Path(__file__).parent
    local_repo_root = local_script_dir.parent
    src_dir = local_repo_root / 'scripts'

    scripts = [
        ('Model Tuning', 'tune_model.py')
    ]
    status_messages = []

    try:
        all_success = True
        for script_name, script_file in scripts:
            status_messages.append(
                html.P(f"Running {script_name}...",
                        style={"color": "yellow", "font-size": "0.75rem", "margin": "0.0625rem 0"}))

            script_path = src_dir / script_file
            # Run the script
            result = subprocess.run([
                sys.executable, str(script_path)
            ], capture_output=True, text=True, check=False, cwd=src_dir.parent)

            if result.returncode == 0:
                status_messages.append(
                    html.P(f"✓ {script_name} completed successfully",
                        style={"color": "#76B900", "font-size": "0.75rem",
                                "margin": "0.0625rem 0"}))
            else:
                error_msg = result.stderr if result.stderr else result.stdout
                status_messages.append(
                    html.P(f"✗ {script_name} failed",
                        style={"color": "#b90076", "font-size": "0.75rem",
                                "margin": "0.0625rem 0"}))
                status_messages.append(
                    html.P(f"Error: {error_msg[:200]}...",
                        style={"color": "#b90076", "font-size": "0.75rem", "margin": "0.0626rem 0",
                                "font-family": "monospace"}))
                all_success = False
                break

        if all_success:
            status_messages.clear()
            status_messages.append(
                html.P("All hyperparameter tuning completed successfully!",
                    style={"color": "#76B900", "font-size": "0.875rem",
                            "margin": "0.625rem 0", "font-weight": "bold"}))
    except (OSError, ValueError, RuntimeError) as e:
        status_messages.append(
            html.P(f"Pipeline Error: {str(e)}",
                style={"color": "#b90076", "font-size": "0.75rem", "margin": "0.0625rem 0"}))

    return (
        html.Div(status_messages),
        False,
        False,
    )

@app.callback(
    Output("train-status", "children", allow_duplicate=True),
    Input("train-model", "n_clicks"),
    prevent_initial_call=True,
    background=True,
    manager=background_callback_manager,
    running=[
        (Output('train-model', 'disabled'), True, False),
        (Output('optimise-hyperparameters', 'disabled'), True, False),
        (Output('train-optimised', 'disabled'), True, False)
    ],
)
def train_model(n_clicks):
    """
    Train the model using the specified script and update the UI with status.

    Args:
        n_clicks (int): Number of times the train button has been clicked.

    Returns:
        tuple: Contains:
            - html.Div: A Div component with status messages for the training process.
            - bool: False to enable the train button.

    Raises:
        PreventUpdate: If the function is called without a button click.
        OSError: If there is an issue accessing the script directory or file.
        ValueError: If there is an issue with script execution parameters.
        RuntimeError: If the script execution encounters an unexpected error.
    """
    if n_clicks is None:
        raise PreventUpdate

    local_script_dir = Path(__file__).parent
    local_repo_root = local_script_dir.parent
    src_dir = local_repo_root / 'scripts'

    scripts = [
        ('Model Training', 'train_model.py')
    ]
    status_messages = []

    try:
        all_success = True
        for script_name, script_file in scripts:
            status_messages.append(
                html.P(f"Running {script_name}...",
                        style={"color": "yellow", "font-size": "0.75rem", "margin": "0.0625rem 0"}))

            script_path = src_dir / script_file
            # Run the script
            result = subprocess.run([
                sys.executable, str(script_path)
            ], capture_output=True, text=True, check=False, cwd=src_dir.parent)

            if result.returncode == 0:
                status_messages.append(
                    html.P(f"✓ {script_name} completed successfully",
                        style={"color": "#76B900", "font-size": "0.75rem",
                                "margin": "0.0625rem 0"}))
            else:
                error_msg = result.stderr if result.stderr else result.stdout
                status_messages.append(
                    html.P(f"✗ {script_name} failed",
                        style={"color": "#b90076", "font-size": "0.75rem",
                                "margin": "0.0625rem 0"}))
                status_messages.append(
                    html.P(f"Error: {error_msg[:200]}...",
                        style={"color": "#b90076", "font-size": "0.75rem", "margin": "0.0626rem 0",
                                "font-family": "monospace"}))
                all_success = False
                break

        if all_success:
            status_messages.clear()
            status_messages.append(
                html.P("Model training completed successfully!",
                    style={"color": "#76B900", "font-size": "0.875rem",
                            "margin": "0.625rem 0", "font-weight": "bold"}))
    except (OSError, ValueError, RuntimeError) as e:
        status_messages.append(
            html.P(f"Pipeline Error: {str(e)}",
                style={"color": "#b90076", "font-size": "0.75rem", "margin": "0.0625rem 0"}))

    return (
        html.Div(status_messages),
        False,
    )

@app.callback(
    Output("train-status", "children", allow_duplicate=True),
    Input("train-optimised", "n_clicks"),
    prevent_initial_call=True,
    background=True,
    manager=background_callback_manager,
    running=[
        (Output('train-model', 'disabled'), True, False),
        (Output('optimise-hyperparameters', 'disabled'), True, False),
        (Output('train-optimised', 'disabled'), True, False)
    ],
)
def train_optimised_model(n_clicks):
    """
    Train the model using optimized hyperparameters and update the UI with status.

    Args:
        n_clicks (int): Number of times the train-optimized button has been clicked.

    Returns:
        tuple: Contains:
            - html.Div: A Div component with status messages for the training process.

    Raises:
        PreventUpdate: If the function is called without a button click.
        OSError: If there is an issue accessing the script directory or file.
        ValueError: If there is an issue with script execution parameters.
        RuntimeError: If the script execution encounters an unexpected error.
        subprocess.CalledProcessError: If the script execution fails.
    """
    if n_clicks is None:
        raise PreventUpdate

    local_script_dir = Path(__file__).parent
    local_repo_root = local_script_dir.parent
    src_dir = local_repo_root / 'scripts'
    config_location = "outputs/best_config_optuna_multi_objective_score.yaml"
    config_path = Path(local_repo_root / config_location).resolve()

    scripts = [
        ('Optimized Model Training', 'train_model.py')
    ]
    status_messages = []

    try:
        all_success = True
        for script_name, script_file in scripts:
            status_messages.append(
                html.P(f"Running {script_name}...",
                        style={"color": "yellow", "font-size": "0.75rem", "margin": "0.0625rem 0"}))

            script_path = src_dir / script_file
            # Run the script
            command = [sys.executable, str(script_path), "--configs", str(config_path)]
            result = subprocess.run(command, check=True, capture_output=True)

            if result.returncode == 0:
                status_messages.append(
                    html.P(f"✓ {script_name} completed successfully",
                        style={"color": "#76B900", "font-size": "0.75rem",
                                "margin": "0.0625rem 0"}))
            else:
                error_msg = result.stderr if result.stderr else result.stdout
                status_messages.append(
                    html.P(f"✗ {script_name} failed",
                        style={"color": "#b90076", "font-size": "0.75rem",
                                "margin": "0.0625rem 0"}))
                status_messages.append(
                    html.P(f"Error: {error_msg[:200]}...",
                        style={"color": "#b90076", "font-size": "0.75rem", "margin": "0.0626rem 0",
                                "font-family": "monospace"}))
                all_success = False
                break

        if all_success:
            status_messages.clear()
            status_messages.append(
                html.P("Model training completed successfully!",
                    style={"color": "#76B900", "font-size": "0.875rem",
                            "margin": "0.625rem 0", "font-weight": "bold"}))
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as e:
        status_messages.append(
            html.P(f"Pipeline Error: {str(e)}",
                style={"color": "#b90076", "font-size": "0.75rem", "margin": "0.0625rem 0"}))

    return (
        html.Div(status_messages),
    )

evaluate_save_location = Path(repo_root / "data/predictions/nvda_predictions.csv").resolve()
pio.templates.default = "plotly_dark"
evaluate_df = pd.read_csv(evaluate_save_location)
evaluate_fig = px.line(evaluate_df,
                            x="Time",
                            y=["Close", "Predicted"],
                            title="NVDA Predicted versus Actual Closing Prices",
                            color_discrete_map={
                                "Predicted": "#0076b9",
                                "Close": "#b90076"
                            }
                            )
evaluate_fig.update_layout(
    yaxis_title_text="Closing Price / $",
    xaxis_title_text="Date",
    legend_title_text="Legend",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)"
)


dash.register_page(
    "Backtesting",
    path="/backtesting",
    layout=html.Div([
        html.Div([
            html.H1("Backtesting", style={"color": "white", "text-align": "left",
                        "margin-bottom": "1.25rem"}),
            html.H3("Model evaluation on unseen historical data.", style={"color": "white"}),
            html.P("Evaluate model to calculate performance metrics.",
                    style={"color": "white", "font-size": "0.875rem", "line-height":"1.4"}),
        ]),
        html.Div([
            html.Div([
                dcc.Graph(figure=evaluate_fig,
                            style={"width": "100%", "height": "auto"}),
                html.Button(
                    "Evaluate Model",
                    id="evaluate-model-button",
                    className="btn btn-primary evaluate-button",
                    style={"margin-top": "0.5rem", "width": "100%"}
                ),
                html.Div(id="evaluate-status",
                            style={"margin-bottom": "0.625rem", "color": "white"}),
            ], style={"display": "inline-block", "vertical-align": "top", "width": "70%"}),
            # Right sidebar - 4 metric cards vertically
            html.Div([
                html.Div([
                    html.H4("0.000",
                            id="sortino-metric",
                            style={"color": "#0076b9", "margin": "0"}),
                    html.P("Sortino Ratio",
                            style={"color": "white", "margin": "0.5rem 0"})
                ], className="metric-card",
                            style={"margin-bottom": "1rem"}),
                html.Div([
                    html.H4("0.000",
                            id="drawdown-metric",
                            style={"color": "#0076b9", "margin": "0"}),
                    html.P("Max Drawdown",
                            style={"color": "white", "margin": "0.5rem 0"})
                ], className="metric-card",
                            style={"margin-bottom": "1rem"}),
                html.Div([
                    html.H4("0.000",
                            id="annual-return-metric",
                            style={"color": "#0076b9", "margin": "0"}),
                    html.P("Annual Return",
                            style={"color": "white", "margin": "0.5rem 0"})
                ], className="metric-card",
                            style={"margin-bottom": "1rem"}),
                html.Div([
                html.H4("0.000",
                        id="mape-metric",
                        style={"color": "#0076b9", "margin": "0"}),
                html.P("Mean Absolute Percentage Error",
                        style={"color": "white", "margin": "0.5rem 0"})
            ], className="metric-card",
                        style={"display": "inline-block", "margin-right": "1rem"}),
            ], style={"display": "inline-block", "vertical-align": "top",
                        "width": "30%", "padding-left": "3.75rem"}),
        ]),
        # Bottom row of 4 metric cards spanning 100% width
        html.Div([
            html.Div([
                html.H4("0.000",
                        id="mae-metric",
                        style={"color": "#0076b9", "margin": "0"}),
                html.P("Mean Absolute Error",
                        style={"color": "white", "margin": "0.5rem 0"})
            ], className="metric-card",
                        style={"display": "inline-block", "margin-right": "1rem"}),
            html.Div([
                    html.H4("0.000",
                            id="win-rate-metric",
                            style={"color": "#0076b9", "margin": "0"}),
                    html.P("Win Rate",
                            style={"color": "white", "margin": "0.5rem 0"})
                ], className="metric-card",
                        style={"display": "inline-block", "margin-right": "1rem"}),
            html.Div([
                html.H4("0.000",
                        id="directional-accuracy-metric",
                        style={"color": "#0076b9", "margin": "0"}),
                html.P("Directional Accuracy",
                        style={"color": "white", "margin": "0.5rem 0"})
            ], className="metric-card",
                        style={"display": "inline-block", "margin-right": "1rem"}),
            html.Div([
                html.H4("0.000",
                        id="sharpe-metric",
                        style={"color": "#0076b9", "margin": "0"}),
                html.P("Sharpe Ratio",
                        style={"color": "white", "margin": "0.5rem 0"})
            ], className="metric-card",
                        style={"display": "inline-block"}),
        ], style={"display": "flex", "justify-content": "space-between", "width": "100%"}),
    ])
)

@app.callback(
    Output("evaluate-status", "children"),
    Output("evaluate-model-button", "disabled"),
    Output("mae-metric", "children"),
    Output("mape-metric", "children"),
    Output("directional-accuracy-metric", "children"),
    Output("sharpe-metric", "children"),
    Output("sortino-metric", "children"),
    Output("drawdown-metric", "children"),
    Output("annual-return-metric", "children"),
    Output("win-rate-metric", "children"),
    Input("evaluate-model-button", "n_clicks"),
    prevent_initial_call=True,
    background=True,
    manager=background_callback_manager,
    running=[
        (Output('evaluate-model-button', 'disabled'), True, False)
    ]
)
def evaluate_model(n_clicks):
    """
    Evaluate a model by running a backtest script and calculating performance metrics.

    This function executes a model evaluation script, processes its output, and computes
    various performance metrics from prediction and trading data. It returns formatted
    HTML status messages and metric values for display.

    Args:
        n_clicks (int or None): Number of times a button has been clicked to trigger
            the evaluation. If None, raises PreventUpdate to halt execution.

    Returns:
        tuple: Contains the following elements:
            - html.Div or html.P: Status messages indicating the progress or outcome
              of the evaluation process.
            - bool: False, indicating whether a loading state should be active.
            - str: Mean Absolute Error (MAE) formatted to three decimal places.
            - str: Mean Absolute Percentage Error (MAPE) formatted to three decimal
              places with a percentage sign.
            - str: Directional accuracy formatted to three decimal places with a
              percentage sign.
            - str: Sharpe ratio formatted to three decimal places.
            - str: Sortino ratio formatted to three decimal places.
            - str: Maximum drawdown formatted to three decimal places with a
              percentage sign.
            - str: Annual return formatted to three decimal places with a percentage
              sign.
            - str: Win rate formatted to three decimal places with a percentage sign.
    """
    if n_clicks is None:
        raise PreventUpdate

    status_messages = []
    local_script_dir = Path(__file__).parent
    local_repo_root = local_script_dir.parent
    src_dir = Path(local_repo_root / 'scripts').resolve()

    scripts = [
        ('Model Evaluation', 'run_backtest.py')
    ]

    try:
        for script_name, script_file in scripts:
            status_messages.append(
                html.P(f"Running {script_name}...",
                        style={"color": "yellow", "font-size": "0.75rem", "margin": "0.0625rem 0"})
            )
            script_path = src_dir / script_file
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                check=False,
                cwd=src_dir.parent
            )

            if result.returncode == 0:
                status_messages.append(
                    html.P(f"✓ {script_name} completed successfully",
                            style={"color": "#76B900", "font-size": "0.75rem",
                                    "margin": "0.0625rem 0"})
                )
            else:
                error_msg = result.stderr if result.stderr else result.stdout
                status_messages.append(
                    html.P(f"✗ {script_name} failed: {error_msg[:200]}",
                            style={"color": "#b90076", "font-size": "0.75rem",
                                    "margin": "0.0625rem 0"})
                )
                return (
                    html.Div(status_messages), False,
                    "0.000", "0.000", "0.000", "0.000", "0.000", "0.000", "0.000", "0.000"
                )

        # Load and calculate prediction metrics
        predictions_path = Path(local_repo_root / "data/predictions/nvda_predictions.csv").resolve()
        df = pd.read_csv(predictions_path)
        mae = np.abs(df['Predicted'] - df['Close']).mean()
        mape = (np.abs((df['Predicted'] - df['Close']) / df['Close'])).mean() * 100
        directional_accuracy = (df['Direction_Match'] == 'yes').mean() * 100

        # Load trading metrics
        metrics_path = Path(local_repo_root / "data/predictions/trading_metrics.json").resolve()
        try:
            with open(metrics_path, 'r', encoding='utf-8') as f:
                trading_metrics = json.load(f)
            sharpe_ratio = trading_metrics.get('sharpe_ratio', 0.0)
            # Calculate Sortino ratio (not in JSON, compute from portfolio values if available)
            sortino_ratio = trading_metrics.get('sortino_ratio', 0.0)  # Placeholder
            max_drawdown = trading_metrics.get('max_drawdown', 0.0) * 100  # Convert to %
            annual_return = trading_metrics.get('annual_return', 0.0) * 100  # Convert to %
            win_rate = trading_metrics.get('win_rate', 0.0) * 100  # Convert to %
        except FileNotFoundError:
            status_messages.append(
                html.P("Trading metrics file not found",
                        style={"color": "#b90076", "font-size": "0.75rem",
                                "margin": "0.0625rem 0"})
            )
            sharpe_ratio = sortino_ratio = max_drawdown = annual_return = win_rate = 0.0

        # Format metrics for display
        return (
            html.P("Model evaluation completed successfully!",
                    style={"color": "#76B900", "font-size": "0.875rem",
                            "margin": "0.625rem 0"}),
            False,
            f"{mae:.3f}",
            f"{mape:.3f}%",
            f"{directional_accuracy:.3f}%",
            f"{sharpe_ratio:.3f}",
            f"{sortino_ratio:.3f}",
            f"{max_drawdown:.3f}%",
            f"{annual_return:.3f}%",
            f"{win_rate:.3f}%"
        )

    except (OSError, subprocess.SubprocessError, KeyError, ValueError) as e:
        # Catch specific, expected errors and return a clear pipeline error message.
        status_messages.append(
            html.P(f"Pipeline Error: {str(e)}",
                    style={"color": "#b90076", "font-size": "0.75rem",
                            "margin": "0.0625rem 0"})
        )
        return (
            html.Div(status_messages), False,
            "0.000", "0.000", "0.000", "0.000", "0.000", "0.000", "0.000", "0.000"
        )

# Create the sidebar
sidebar = html.Div([
    html.Div([
        html.I(className="fas fa-chart-line fa-2x mb-2"),
        html.Hr(className="sidebar_Hr"),
        html.H3("ML Stock", className="mb-0"),
        html.H3("Prediction", className="mb-4")
    ], className="text-center mb-4", style={"color": "white"}),

    html.Hr(className="sidebar_Hr"),

    html.Div([
        html.Div([
            dcc.Link([
                html.Div([
                    html.Span("Home")
                ], className="nav-link-content")
            ], href="/", className="sidebar-link", id="home-link")
        ], className="nav-item"),

        html.Div([
            dcc.Link([
                html.Div([
                    html.Span("1. Data Preparation")
                ], className="nav-link-content")
            ], href="/data-preparation", className="sidebar-link")
        ], className="nav-item"),

        html.Div([
            dcc.Link([
                html.Div([
                    html.Span("2. Training")
                ], className="nav-link-content")
            ], href="/training", className="sidebar-link")
        ], className="nav-item"),

        html.Div([
            dcc.Link([
                html.Div([
                    html.Span("3. Backtesting")
                ], className="nav-link-content")
            ], href="/backtesting", className="sidebar-link")
        ], className="nav-item"),
    ])
], id="sidebar")

# Main content area
content = html.Div(id="page-content", children=[
    dash.page_container
])

# App layout
app.layout = html.Div([
    sidebar,
    content
], id="main-container")

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            html, body {
                height: 100vh;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                background-color: #1a1a1a;
                color: white;
            }
            
            #react-entry-point {
                height: 100vh;
            }
            
            #main-container {
                display: flex;
                height: 100vh;
                width: 100vw;
            }
            
            #sidebar {
                width: 17.5rem;
                background: linear-gradient(135deg, #76B900 0%, #1a1a1a 100%);
                border-top-right-radius: 0.75rem;
                border-bottom-right-radius: 0.75rem;
                padding: 1.875rem 1.25rem;
                box-shadow: 0.125rem 0 0.635rem rgba(0,0,0,0.3);
                height: 100vh;
                overflow-y: auto;
                border-right: 0.125rem solid #2d2d2d;
                position: fixed;
                left: 0;
                top: 0;
                z-index: 1000;
            }
            
            #page-content {
                margin-left: 17.5rem;
                padding: 2.5rem;
                border-top-left-radius: 0.75rem;
                border-bottom-left-radius: 0.75rem;
                flex-grow: 1;
                background: linear-gradient(200deg, #000000 -30%, #1a1a1a 100%);
                min-height: 100vh;
                width: calc(100vw - 17.5rem);
                word-wrap: break-word;
                overflow-wrap: break-word;
                box-sizing: border-box;
                text-align: justify;
                overflow-y: auto;
            }
            
            .nav-item {
                margin-bottom: 0.5rem;
            }
            
            .sidebar-link {
                display: block;
                padding: 0.75rem 1.25rem;
                text-decoration: none;
                color: white;
                border-radius: 0.5rem;
                transition: all 0.3s ease;
                border-left: 0.25rem solid transparent;
            }
            
            .sidebar-link:hover {
                background-color: #0076b985;
                color: white;
                text-decoration: none;
                border-left-color: #b90076;
                transform: translateX(0.3125rem);
            }
            
            .nav-link-content {
                font-weight: 500;
                font-size: 1rem;
            }
            
            #home-link {
                background-color: #0076b985;
                color: white;
                border-left-color: #b90076;
            }
            
            h1, h2, h3 {
                color: white;
            }
            
            .fas {
                color: #0076b9;
            }
            
            .sidebar_Hr {
                border: 0.0625rem solid #0076b9;
            }
            
            .data-load-button {
                background-color: #0076b9;
                border: none;
                border-radius: 1.5625rem;
                padding: 0.625rem 1.25rem;
                margin-top: 0.625rem;
                margin-bottom: 0.3125rem;
                width: 100%;
            }

            .train-button {
                background-color: #0076b9;
                border: none;
                border-radius: 1.5625rem;
                padding: 0.625rem 1.25rem;
                margin-top: 0.625rem;
                margin-bottom: 0.3125rem;
                width: 100%;
            }
            
            .evaluate-button {
                background-color: #0076b9;
                border: none;
                border-radius: 1.5625rem;
                padding: 0.625rem 1.25rem;
                margin-top: 0.625rem;
                margin-bottom: 0.3125rem;
                width: 100%;
                display: block;
            }
            .metric-card {
                background: #2d3748;
                padding: 0.5rem;
                border-radius: 1rem;
                text-align: center;
                flex: 1 1 16.5rem;
                margin: 0.5rem;
                max-width: 16rem;
            }
            
            /* Scrollbar styling for sidebar */
            #sidebar::-webkit-scrollbar {
                width: 0.375rem;
            }
            
            #sidebar::-webkit-scrollbar-track {
                background: rgba(255,255,255,0.1);
                border-radius: 0.1875rem;
            }
            
            #sidebar::-webkit-scrollbar-thumb {
                background: rgba(74, 158, 255, 0.5);
                border-radius: 0.1875rem;
            }
            
            #sidebar::-webkit-scrollbar-thumb:hover {
                background: rgba(74, 158, 255, 0.7);
            }
            
            /* Responsive design */
            @media (max-width: 47.5rem) {
                #sidebar {
                    width: 100%;
                    height: auto;
                    position: relative;
                }
                
                #page-content {
                    margin-left: 0;
                }
                
                #main-container {
                    flex-direction: column;
                }
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

if __name__ == "__main__":
    app.run(debug=False)
