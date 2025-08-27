from pathlib import Path
import subprocess
import sys
from dash import Dash, html, dcc, Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import dash
import plotly.express as px
import plotly.io as pio
import pandas as pd

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
                "Optimize hyperparameters using Ray Tune and Optuna, with Weights & Biases "
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

script_dir = Path(__file__).parent  # /path/to/repo/app
repo_root = script_dir.parent  # /path/to/repo/
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
    xaxis_title_text="Date"
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
                    className="btn btn-primary data_load_button",
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
                        className="btn btn-primary data_load_button",
                    )
                ]),

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
        env_path = repo_root / ".env"

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
    Input('prepare-data-button', 'n_clicks'),
    prevent_initial_call=True
)
def process_data_pipeline(n_clicks):
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

    status_messages = []

    try:
        for script_name, script_file in scripts:
            status_messages.append(
                html.P(f"Running {script_name}...",
                        style={"color": "yellow", "font-size": "0.75rem", "margin": "0.0625rem 0"}))

            script_path = src_dir / script_file
            all_success = True
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
                html.P("All data processing completed successfully!",
                    style={"color": "#76B900", "font-size": "0.875rem",
                            "margin": "0.625rem 0", "font-weight": "bold"}))

    except (OSError, ValueError, RuntimeError) as e:
        status_messages.append(
            html.P(f"Pipeline Error: {str(e)}",
                style={"color": "#b90076", "font-size": "0.75rem", "margin": "0.0625rem 0"}))
    return html.Div(status_messages)

dash.register_page("Training", path="/training",
                    layout=html.Div([
                        html.H1("Model Training", style={"color": "white"}),
                        html.P("Train your machine learning models here.",
                            style={"color": "white"})
                    ]))

dash.register_page("Backtesting", path="/backtesting",
                    layout=html.Div([
                        html.H1("Backtesting", style={"color": "white"}),
                        html.P("Test your models on historical data.",
                            style={"color": "white"})
                    ]))

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
            
            .data_load_button {
                background-color: #0076b9;
                border: none;
                padding: 0.625rem 1.25rem;
                margin-top: 0.625rem;
                margin-bottom: 0.3125rem;
                width: 100%;
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
    app.run(debug=True)
