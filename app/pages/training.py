from dash import html, dcc, Input, Output, State, register_page
from pathlib import Path
import subprocess
import sys
from dash.exceptions import PreventUpdate
import dash
import diskcache

script_dir = Path(__file__).parent  # /path/to/repo/app/pages/
repo_root = script_dir.parent.parent  # /path/to/repo/

cache_location = Path(repo_root / "cache").resolve()
cache = diskcache.Cache(cache_location)
background_callback_manager = dash.DiskcacheManager(cache)

# Explicit page path for Dash pages discovery
path = "/training"

# Ensure Dash registers this page at import time
register_page(__name__, path=path)

layout = html.Div(
    [
        html.Div(
            [
                html.H1(
                    "Model Training",
                    style={
                        "color": "white",
                        "text-align": "left",
                        "margin-bottom": "1.25rem",
                    },
                ),
                html.H3(
                    "Train and Fine-tune the Ensemble Model",
                    style={"color": "white", "margin-bottom": "1.875rem"},
                ),
            ],
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.Img(
                            src="/assets/model_architecture.png",
                            style={
                                "width": "100%",
                                "height": "auto",
                                "border": "2px solid #555",
                            },
                        )
                    ],
                    style={
                        "width": "48%",
                        "display": "inline-block",
                        "vertical-align": "top",
                    },
                ),
                # Right column - Text and Buttons
                html.Div(
                    [
                        html.Div(
                            [
                                html.H6(
                                    "Information",
                                    style={"color": "white", "margin-bottom": "0.5rem"},
                                ),
                                html.P(
                                    "Goal: To produce direction-aware price predictions",
                                    style={
                                        "color": "white",
                                        "font-size": "0.875rem",
                                        "line-height": "1.4",
                                    },
                                ),
                                html.P(
                                    "Ensemble Model Components: Multihead CNN, Ridge Regression, LSTM, Meta-Learner",
                                    style={
                                        "color": "white",
                                        "font-size": "0.875rem",
                                        "line-height": "1.4",
                                    },
                                ),
                                html.P(
                                    "Hyperparameter Optimisation: Ray Tune with Optuna",
                                    style={
                                        "color": "white",
                                        "font-size": "0.875rem",
                                        "line-height": "1.4",
                                    },
                                ),
                                html.P(
                                    "Hyperparameter Optimisation will take time, please be patient.",
                                    style={
                                        "color": "white",
                                        "font-size": "0.875rem",
                                        "line-height": "1.4",
                                    },
                                ),
                                html.P(
                                    "There are no loading indicators, due to the asynchronous nature of the training process.",
                                    style={
                                        "color": "white",
                                        "font-size": "0.875rem",
                                        "line-height": "1.4",
                                    },
                                ),
                            ],
                            style={
                                "border-bottom": "1px solid #555",
                                "padding-bottom": "0rem",
                                "margin-bottom": "1.5rem",
                            },
                        ),
                        html.Button(
                            "Optimise Hyperparameters",
                            id="optimise-hyperparameters",
                            className="btn btn-primary train-button",
                        ),
                        html.Button(
                            "Train Model",
                            id="train-model",
                            className="btn btn-primary train-button",
                        ),
                        html.Button(
                            "Train with optimised hyperparameters",
                            id="train-optimised",
                            disabled=True,
                            className="btn btn-primary train-button",
                        ),
                        html.Div(
                            id="train-status", style={"margin-bottom": "0.625rem"}
                        ),
                    ],
                    style={
                        "width": "48%",
                        "display": "inline-block",
                        "vertical-align": "top",
                        "margin-left": "4%",
                    },
                ),
            ],
            style={"margin-bottom": "1rem"},
        ),
        html.Div(
            [
                html.H5(
                    "Weights and Biases API Key",
                    style={"color": "white", "margin-bottom": "1rem"},
                ),
                html.Div(
                    [
                        dcc.Input(
                            placeholder="Enter your W&B API key",
                            type="password",
                            id="wandb-api-key",
                            className="mb-3 api-input",
                            style={
                                "width": "70%",
                                "margin-right": "2%",
                                "height": "48px",
                                "box-sizing": "border-box",
                            },
                        ),
                        html.Button(
                            "Submit",
                            id="submit-wandb-api-key",
                            className="btn btn-primary train-button",
                            style={
                                "width": "25%",
                                "height": "48px",
                                "box-sizing": "border-box",
                            },
                        ),
                    ],
                    style={
                        "display": "flex",
                        "align-items": "stretch",
                        "margin-bottom": "1rem",
                    },
                ),
                html.Div(
                    id="wandb-api-key-status", style={"margin-bottom": "0.625rem"}
                ),
            ],
        ),
    ]
)


@dash.callback(
    Output("wandb-api-key-status", "children"),
    Input("submit-wandb-api-key", "n_clicks"),
    [State("wandb-api-key", "value")],
    prevent_initial_call=True,
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
        return html.Div(
            "Please enter a valid API key",
            style={"color": "#b90076", "font-size": "0.75rem"},
        )
    try:
        env_path = Path(repo_root / ".env").resolve()
        env_vars = {}
        if env_path.exists():
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip() and not line.startswith("#"):
                        key, value = line.strip().split("=", 1)
                        env_vars[key] = value

        # Update with new API key
        env_vars["WANDB_API_KEY"] = api_key
        with open(env_path, "w", encoding="utf-8") as f:
            for key, value in env_vars.items():
                f.write(f"{key}={value}\n")

        return html.Div(
            "✓ API key saved successfully!",
            style={"color": "#76B900", "font-size": "0.75rem"},
        )

    except (OSError, ValueError) as e:
        return html.Div(
            f"Error saving API key: {str(e)}",
            style={"color": "#b90076", "font-size": "0.75rem"},
        )


@dash.callback(
    Output("train-status", "children"),
    Output("optimise-hyperparameters", "disabled"),
    Output("train-optimised", "disabled"),
    Input("optimise-hyperparameters", "n_clicks"),
    prevent_initial_call=True,
    background=True,
    manager=background_callback_manager,
    running=[
        (Output("optimise-hyperparameters", "disabled"), True, False),
        (Output("train-model", "disabled"), True, False),
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

    src_dir = Path(repo_root / "scripts").resolve()

    scripts = [("Model Tuning", "tune_model.py")]
    status_messages = []

    try:
        all_success = True
        for script_name, script_file in scripts:
            status_messages.append(
                html.P(
                    f"Running {script_name}...",
                    style={
                        "color": "yellow",
                        "font-size": "0.75rem",
                        "margin": "0.0625rem 0",
                    },
                )
            )

            script_path = src_dir / script_file
            # Run the script
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                check=False,
                cwd=src_dir.parent,
            )

            if result.returncode == 0:
                status_messages.append(
                    html.P(
                        f"✓ {script_name} completed successfully",
                        style={
                            "color": "#76B900",
                            "font-size": "0.75rem",
                            "margin": "0.0625rem 0",
                        },
                    )
                )
            else:
                error_msg = result.stderr if result.stderr else result.stdout
                status_messages.append(
                    html.P(
                        f"✗ {script_name} failed",
                        style={
                            "color": "#b90076",
                            "font-size": "0.75rem",
                            "margin": "0.0625rem 0",
                        },
                    )
                )
                status_messages.append(
                    html.P(
                        f"Error: {error_msg[:200]}...",
                        style={
                            "color": "#b90076",
                            "font-size": "0.75rem",
                            "margin": "0.0626rem 0",
                            "font-family": "monospace",
                        },
                    )
                )
                all_success = False
                break

        if all_success:
            status_messages.clear()
            status_messages.append(
                html.P(
                    "All hyperparameter tuning completed successfully!",
                    style={
                        "color": "#76B900",
                        "font-size": "0.875rem",
                        "margin": "0.625rem 0",
                        "font-weight": "bold",
                    },
                )
            )
    except (OSError, ValueError, RuntimeError) as e:
        status_messages.append(
            html.P(
                f"Pipeline Error: {str(e)}",
                style={
                    "color": "#b90076",
                    "font-size": "0.75rem",
                    "margin": "0.0625rem 0",
                },
            )
        )

    return (
        html.Div(status_messages),
        False,
        False,
    )


@dash.callback(
    Output("train-status", "children", allow_duplicate=True),
    Input("train-model", "n_clicks"),
    prevent_initial_call=True,
    background=True,
    manager=background_callback_manager,
    running=[
        (Output("train-model", "disabled"), True, False),
        (Output("optimise-hyperparameters", "disabled"), True, False),
        (Output("train-optimised", "disabled"), True, False),
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

    src_dir = Path(repo_root / "scripts").resolve()

    scripts = [("Model Training", "train_model.py")]
    status_messages = []

    try:
        all_success = True
        for script_name, script_file in scripts:
            status_messages.append(
                html.P(
                    f"Running {script_name}...",
                    style={
                        "color": "yellow",
                        "font-size": "0.75rem",
                        "margin": "0.0625rem 0",
                    },
                )
            )

            script_path = src_dir / script_file
            # Run the script
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                check=False,
                cwd=src_dir.parent,
            )

            if result.returncode == 0:
                status_messages.append(
                    html.P(
                        f"✓ {script_name} completed successfully",
                        style={
                            "color": "#76B900",
                            "font-size": "0.75rem",
                            "margin": "0.0625rem 0",
                        },
                    )
                )
            else:
                error_msg = result.stderr if result.stderr else result.stdout
                status_messages.append(
                    html.P(
                        f"✗ {script_name} failed",
                        style={
                            "color": "#b90076",
                            "font-size": "0.75rem",
                            "margin": "0.0625rem 0",
                        },
                    )
                )
                status_messages.append(
                    html.P(
                        f"Error: {error_msg[:200]}...",
                        style={
                            "color": "#b90076",
                            "font-size": "0.75rem",
                            "margin": "0.0626rem 0",
                            "font-family": "monospace",
                        },
                    )
                )
                all_success = False
                break

        if all_success:
            status_messages.clear()
            status_messages.append(
                html.P(
                    "Model training completed successfully!",
                    style={
                        "color": "#76B900",
                        "font-size": "0.875rem",
                        "margin": "0.625rem 0",
                        "font-weight": "bold",
                    },
                )
            )
    except (OSError, ValueError, RuntimeError) as e:
        status_messages.append(
            html.P(
                f"Pipeline Error: {str(e)}",
                style={
                    "color": "#b90076",
                    "font-size": "0.75rem",
                    "margin": "0.0625rem 0",
                },
            )
        )

    return (
        html.Div(status_messages),
        False,
    )


@dash.callback(
    Output("train-status", "children", allow_duplicate=True),
    Input("train-optimised", "n_clicks"),
    prevent_initial_call=True,
    background=True,
    manager=background_callback_manager,
    running=[
        (Output("train-model", "disabled"), True, False),
        (Output("optimise-hyperparameters", "disabled"), True, False),
        (Output("train-optimised", "disabled"), True, False),
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

    src_dir = repo_root / "scripts"
    config_location = "outputs/best_config_optuna_multi_objective_score.yaml"
    config_path = Path(repo_root / config_location).resolve()

    scripts = [("Optimized Model Training", "train_model.py")]
    status_messages = []

    try:
        all_success = True
        for script_name, script_file in scripts:
            status_messages.append(
                html.P(
                    f"Running {script_name}...",
                    style={
                        "color": "yellow",
                        "font-size": "0.75rem",
                        "margin": "0.0625rem 0",
                    },
                )
            )

            script_path = src_dir / script_file
            # Run the script
            command = [sys.executable, str(script_path), "--configs", str(config_path)]
            result = subprocess.run(command, check=True, capture_output=True)

            if result.returncode == 0:
                status_messages.append(
                    html.P(
                        f"✓ {script_name} completed successfully",
                        style={
                            "color": "#76B900",
                            "font-size": "0.75rem",
                            "margin": "0.0625rem 0",
                        },
                    )
                )
            else:
                error_msg = result.stderr if result.stderr else result.stdout
                status_messages.append(
                    html.P(
                        f"✗ {script_name} failed",
                        style={
                            "color": "#b90076",
                            "font-size": "0.75rem",
                            "margin": "0.0625rem 0",
                        },
                    )
                )
                status_messages.append(
                    html.P(
                        f"Error: {error_msg[:200]}...",
                        style={
                            "color": "#b90076",
                            "font-size": "0.75rem",
                            "margin": "0.0626rem 0",
                            "font-family": "monospace",
                        },
                    )
                )
                all_success = False
                break

        if all_success:
            status_messages.clear()
            status_messages.append(
                html.P(
                    "Model training completed successfully!",
                    style={
                        "color": "#76B900",
                        "font-size": "0.875rem",
                        "margin": "0.625rem 0",
                        "font-weight": "bold",
                    },
                )
            )
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as e:
        status_messages.append(
            html.P(
                f"Pipeline Error: {str(e)}",
                style={
                    "color": "#b90076",
                    "font-size": "0.75rem",
                    "margin": "0.0625rem 0",
                },
            )
        )

    return (html.Div(status_messages),)
