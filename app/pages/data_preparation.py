from dash import html, dcc, Input, Output, State, register_page
from pathlib import Path
import subprocess
import sys
from dash.exceptions import PreventUpdate
import dash
import plotly.express as px
import plotly.io as pio
import pandas as pd
import diskcache


repo_root = Path(__file__).parent.parent.parent  # /path/to/repo/

cache_location = Path(repo_root / "cache").resolve()
cache = diskcache.Cache(cache_location)
background_callback_manager = dash.DiskcacheManager(cache)

# Explicit page path for Dash pages discovery
path = "/data-preparation"

# Ensure Dash registers this page at import time
register_page(__name__, path=path)

data_load_save_location = Path(repo_root / "data/raw/nvda_raw_data.csv").resolve()
pio.templates.default = "plotly_dark"
data_load_df = pd.read_csv(data_load_save_location)
data_load_fig = px.line(
    data_load_df,
    x="date",
    y="Close",
    title="NVDA Daily Closing Prices",
)
data_load_fig.update_layout(
    yaxis_title_text="Closing Price / $",
    xaxis_title_text="Date",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
)
data_load_fig.update_traces(line_color="#0076b9")

layout = html.Div(
    [
        html.H1(
            "Data Preparation",
            style={"color": "white", "text-align": "left", "margin-bottom": "1.25rem"},
        ),
        html.H3(
            "Access WRDS to retrieve stock data and prepare for model training",
            style={"color": "white", "margin-bottom": "1.875rem"},
        ),
        html.Div(
            [
                # Left column (WRDS login & information)
                html.Div(
                    [
                        html.H5(
                            "WRDS LOGIN",
                            style={"color": "white", "margin-bottom": "0.625rem"},
                        ),
                        html.P(
                            "USERNAME",
                            style={"color": "white", "margin-bottom": "0.3125rem"},
                        ),
                        dcc.Input(
                            placeholder="Enter your WRDS username",
                            type="text",
                            className="mb-3",
                            id="wrds-username",
                            style={"width": "100%", "margin-bottom": "0.625rem"},
                        ),
                        html.P(
                            "PASSWORD",
                            style={"color": "white", "margin-bottom": "0.3125rem"},
                        ),
                        dcc.Input(
                            placeholder="Enter your WRDS password",
                            type="password",
                            className="mb-3",
                            id="wrds-password",
                            style={"width": "100%", "margin-bottom": "0.9375rem"},
                        ),
                        html.Button(
                            "LOGIN",
                            id="login-button",
                            className="btn btn-primary data-load-button",
                        ),
                        html.P(
                            "Passes login details to .env file",
                            style={
                                "color": "white",
                                "font-size": "0.875rem",
                                "line-height": "1.4",
                            },
                        ),
                        html.Div(
                            id="login-status", style={"margin-bottom": "0.625rem"}
                        ),
                        html.Div(
                            [
                                html.H6(
                                    "Information",
                                    style={
                                        "color": "white",
                                        "margin-bottom": "0.625rem",
                                    },
                                ),
                                html.P(
                                    "Date Range: 2004-10-31 - 2022-12-31",
                                    style={
                                        "color": "white",
                                        "font-size": "0.875rem",
                                        "line-height": "1.4",
                                    },
                                ),
                                html.P(
                                    "Stock: NVDA",
                                    style={
                                        "color": "white",
                                        "font-size": "0.875rem",
                                        "line-height": "1.4",
                                    },
                                ),
                                html.P(
                                    "Other Tickers: SPY, QQQ, VIXY ",
                                    style={
                                        "color": "white",
                                        "font-size": "0.875rem",
                                        "line-height": "1.4",
                                    },
                                ),
                                html.P(
                                    "Economic Indicators: 10 Year Treasury Yield",
                                    style={
                                        "color": "white",
                                        "font-size": "0.875rem",
                                        "line-height": "1.4",
                                    },
                                ),
                                html.P(
                                    "Some data were not available in the date range, so were imputed "
                                    "using existing data. The date range was extended back to 2004, "
                                    "so PE and PB ratios could also be calculated for early 2005 data.",
                                    style={
                                        "color": "white",
                                        "font-size": "0.875rem",
                                        "line-height": "1.4",
                                    },
                                ),
                            ],
                            style={"margin-top": "2.5rem"},
                        ),
                    ],
                    style={
                        "width": "35%",
                        "padding": "1.25rem",
                        "border-right": "0.125rem solid #333",
                        "min-height": "31.25rem",
                    },
                ),
                # Right column (graph, buttons, status)
                html.Div(
                    [
                        html.Div(
                            [
                                dcc.Graph(
                                    figure=data_load_fig, style={"height": "25rem"}
                                )
                            ],
                            style={"margin-bottom": "0.9375rem"},
                        ),
                        html.Div(
                            [
                                html.Button(
                                    "Prepare Data",
                                    id="prepare-data-button",
                                    className="btn btn-primary data-load-button",
                                )
                            ]
                        ),
                        html.Progress(
                            id="data-loading-progress",
                            value="0",
                            style={"height": "0.25rem"},
                        ),
                        html.Div(
                            id="process-status", style={"margin-top": "0.3125rem"}
                        ),
                    ],
                    style={
                        "width": "65%",
                        "padding": "1.25rem",
                        "text-align": "center",
                    },
                ),
            ],
            style={"display": "flex", "width": "100%", "margin-top": "1.25rem"},
        ),
    ]
)


@dash.callback(
    Output("login-status", "children"),
    Input("login-button", "n_clicks"),
    [State("wrds-username", "value"), State("wrds-password", "value")],
    prevent_initial_call=True,
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
        return html.Div(
            "Please enter both username and password",
            style={"color": "#b90076", "font-size": "0.75rem"},
        )

    try:
        # Define the path to the .env file (in the repo root)
        env_path = Path(repo_root / ".env").resolve()

        # Read existing .env file if it exists
        env_vars = {}
        if env_path.exists():
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip() and not line.startswith("#"):
                        key, value = line.strip().split("=", 1)
                        env_vars[key] = value

        # Update with new credentials
        env_vars["WRDS_USERNAME"] = username
        env_vars["WRDS_PASSWORD"] = password

        # Write back to .env file
        with open(env_path, "w", encoding="utf-8") as f:
            for key, value in env_vars.items():
                f.write(f"{key}={value}\n")

        return html.Div(
            "✓ Credentials saved successfully!",
            style={"color": "#76B900", "font-size": "0.75rem"},
        )

    except (OSError, ValueError) as e:
        return html.Div(
            f"Error saving credentials: {str(e)}",
            style={"color": "#b90076", "font-size": "0.75rem"},
        )


@dash.callback(
    Output("process-status", "children"),
    Output("prepare-data-button", "disabled"),
    Input("prepare-data-button", "n_clicks"),
    prevent_initial_call=True,
    background=True,
    manager=background_callback_manager,
    running=[(Output("prepare-data-button", "disabled"), True, False)],
    progress=[
        Output("data-loading-progress", "value"),
        Output("data-loading-progress", "max"),
    ],
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
    src_dir = Path(repo_root / "src").resolve()

    # Define the scripts to run in order
    scripts = [
        ("Data Loader", "data_loader.py"),
        ("Feature Engineering", "feature_engineering.py"),
        ("Data Module", "data_module.py"),
    ]

    total_scripts = len(scripts)
    status_messages = []
    set_progress((str(0), str(total_scripts)))

    try:
        all_success = True
        for i, (script_name, script_file) in enumerate(scripts, 1):
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
            set_progress((str(i), str(total_scripts)))

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
                    "All data processing completed successfully!",
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
        return (html.Div(status_messages), False)
    return (html.Div(status_messages), False)
