from dash import html, dcc, Input, Output, register_page
from pathlib import Path
import subprocess
import sys
import json
import numpy as np
from dash.exceptions import PreventUpdate
import dash
import pandas as pd
import plotly.express as px
import plotly.io as pio
import diskcache

script_dir = Path(__file__).parent  # /path/to/repo/app
repo_root = script_dir.parent.parent  # /path/to/repo/

cache_location = Path(repo_root / "cache").resolve()
cache = diskcache.Cache(cache_location)
background_callback_manager = dash.DiskcacheManager(cache)

# Explicit page path for Dash pages discovery
path = "/backtesting"

# Ensure Dash registers this page at import time
register_page(__name__, path=path)


evaluate_save_location = Path(
    repo_root / "data/predictions/nvda_predictions.csv"
).resolve()
pio.templates.default = "plotly_dark"
evaluate_df = pd.read_csv(evaluate_save_location)
evaluate_fig = px.line(
    evaluate_df,
    x="Time",
    y=["Close", "Predicted"],
    title="NVDA Predicted versus Actual Closing Prices",
    color_discrete_map={"Predicted": "#0076b9", "Close": "#b90076"},
)
evaluate_fig.update_layout(
    yaxis_title_text="Closing Price / $",
    xaxis_title_text="Date",
    legend_title_text="Legend",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
)

layout = html.Div(
    [
        html.Div(
            [
                html.H1(
                    "Backtesting",
                    style={
                        "color": "white",
                        "text-align": "left",
                        "margin-bottom": "1.25rem",
                    },
                ),
                html.H3(
                    "Model evaluation on unseen historical data.",
                    style={"color": "white"},
                ),
                html.P(
                    "Evaluate model to calculate performance metrics.",
                    style={
                        "color": "white",
                        "font-size": "0.875rem",
                        "line-height": "1.4",
                    },
                ),
            ]
        ),
        html.Div(
            [
                html.Div(
                    [
                        dcc.Graph(
                            figure=evaluate_fig,
                            style={"width": "100%", "height": "auto"},
                        ),
                        html.Button(
                            "Evaluate Model",
                            id="evaluate-model-button",
                            className="btn btn-primary evaluate-button",
                            style={"margin-top": "0.5rem", "width": "100%"},
                        ),
                        html.Div(
                            id="evaluate-status",
                            style={"margin-bottom": "0.625rem", "color": "white"},
                        ),
                    ],
                    style={
                        "display": "inline-block",
                        "vertical-align": "top",
                        "width": "70%",
                    },
                ),
                # Right sidebar - 4 metric cards vertically
                html.Div(
                    [
                        html.Div(
                            [
                                html.H4(
                                    "0.000",
                                    id="sortino-metric",
                                    style={"color": "#0076b9", "margin": "0"},
                                ),
                                html.P(
                                    "Sortino Ratio",
                                    style={"color": "white", "margin": "0.5rem 0"},
                                ),
                            ],
                            className="metric-card",
                            style={"margin-bottom": "1rem"},
                        ),
                        html.Div(
                            [
                                html.H4(
                                    "0.000",
                                    id="drawdown-metric",
                                    style={"color": "#0076b9", "margin": "0"},
                                ),
                                html.P(
                                    "Max Drawdown",
                                    style={"color": "white", "margin": "0.5rem 0"},
                                ),
                            ],
                            className="metric-card",
                            style={"margin-bottom": "1rem"},
                        ),
                        html.Div(
                            [
                                html.H4(
                                    "0.000",
                                    id="annual-return-metric",
                                    style={"color": "#0076b9", "margin": "0"},
                                ),
                                html.P(
                                    "Annual Return",
                                    style={"color": "white", "margin": "0.5rem 0"},
                                ),
                            ],
                            className="metric-card",
                            style={"margin-bottom": "1rem"},
                        ),
                        html.Div(
                            [
                                html.H4(
                                    "0.000",
                                    id="mape-metric",
                                    style={"color": "#0076b9", "margin": "0"},
                                ),
                                html.P(
                                    "Mean Absolute Percentage Error",
                                    style={"color": "white", "margin": "0.5rem 0"},
                                ),
                            ],
                            className="metric-card",
                            style={"display": "inline-block", "margin-right": "1rem"},
                        ),
                    ],
                    style={
                        "display": "inline-block",
                        "vertical-align": "top",
                        "width": "30%",
                        "padding-left": "3.75rem",
                    },
                ),
            ]
        ),
        # Bottom row of 4 metric cards spanning 100% width
        html.Div(
            [
                html.Div(
                    [
                        html.H4(
                            "0.000",
                            id="mae-metric",
                            style={"color": "#0076b9", "margin": "0"},
                        ),
                        html.P(
                            "Mean Absolute Error",
                            style={"color": "white", "margin": "0.5rem 0"},
                        ),
                    ],
                    className="metric-card",
                    style={"display": "inline-block", "margin-right": "1rem"},
                ),
                html.Div(
                    [
                        html.H4(
                            "0.000",
                            id="win-rate-metric",
                            style={"color": "#0076b9", "margin": "0"},
                        ),
                        html.P(
                            "Win Rate", style={"color": "white", "margin": "0.5rem 0"}
                        ),
                    ],
                    className="metric-card",
                    style={"display": "inline-block", "margin-right": "1rem"},
                ),
                html.Div(
                    [
                        html.H4(
                            "0.000",
                            id="directional-accuracy-metric",
                            style={"color": "#0076b9", "margin": "0"},
                        ),
                        html.P(
                            "Directional Accuracy",
                            style={"color": "white", "margin": "0.5rem 0"},
                        ),
                    ],
                    className="metric-card",
                    style={"display": "inline-block", "margin-right": "1rem"},
                ),
                html.Div(
                    [
                        html.H4(
                            "0.000",
                            id="sharpe-metric",
                            style={"color": "#0076b9", "margin": "0"},
                        ),
                        html.P(
                            "Sharpe Ratio",
                            style={"color": "white", "margin": "0.5rem 0"},
                        ),
                    ],
                    className="metric-card",
                    style={"display": "inline-block"},
                ),
            ],
            style={
                "display": "flex",
                "justify-content": "space-between",
                "width": "100%",
            },
        ),
    ]
)


@dash.callback(
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
    running=[(Output("evaluate-model-button", "disabled"), True, False)],
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
    src_dir = Path(repo_root / "scripts").resolve()

    scripts = [("Model Evaluation", "run_backtest.py")]

    try:
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
                        f"✗ {script_name} failed: {error_msg[:200]}",
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
                    "0.000",
                    "0.000",
                    "0.000",
                    "0.000",
                    "0.000",
                    "0.000",
                    "0.000",
                    "0.000",
                )

        # Load and calculate prediction metrics
        predictions_path = Path(
            repo_root / "data/predictions/nvda_predictions.csv"
        ).resolve()
        df = pd.read_csv(predictions_path)
        mae = np.abs(df["Predicted"] - df["Close"]).mean()
        mape = (np.abs((df["Predicted"] - df["Close"]) / df["Close"])).mean() * 100
        directional_accuracy = (df["Direction_Match"] == "yes").mean() * 100

        # Load trading metrics
        metrics_path = Path(
            repo_root / "data/predictions/trading_metrics.json"
        ).resolve()
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                trading_metrics = json.load(f)
            sharpe_ratio = trading_metrics.get("sharpe_ratio", 0.0)
            # Calculate Sortino ratio (not in JSON, compute from portfolio values if available)
            sortino_ratio = trading_metrics.get("sortino_ratio", 0.0)  # Placeholder
            max_drawdown = (
                trading_metrics.get("max_drawdown", 0.0) * 100
            )  # Convert to %
            annual_return = (
                trading_metrics.get("annual_return", 0.0) * 100
            )  # Convert to %
            win_rate = trading_metrics.get("win_rate", 0.0) * 100  # Convert to %
        except FileNotFoundError:
            status_messages.append(
                html.P(
                    "Trading metrics file not found",
                    style={
                        "color": "#b90076",
                        "font-size": "0.75rem",
                        "margin": "0.0625rem 0",
                    },
                )
            )
            sharpe_ratio = sortino_ratio = max_drawdown = annual_return = win_rate = 0.0

        # Format metrics for display
        return (
            html.P(
                "Model evaluation completed successfully!",
                style={
                    "color": "#76B900",
                    "font-size": "0.875rem",
                    "margin": "0.625rem 0",
                },
            ),
            False,
            f"${mae:.2f}",
            f"{mape:.3f}%",
            f"{directional_accuracy:.3f}%",
            f"{sharpe_ratio:.3f}",
            f"{sortino_ratio:.3f}",
            f"{max_drawdown:.3f}%",
            f"{annual_return:.3f}%",
            f"{win_rate:.3f}%",
        )

    except (OSError, subprocess.SubprocessError, KeyError, ValueError) as e:
        # Catch specific, expected errors and return a clear pipeline error message.
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
            "0.000",
            "0.000",
            "0.000",
            "0.000",
            "0.000",
            "0.000",
            "0.000",
            "0.000",
        )
