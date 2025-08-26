from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
import dash


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
        html.H1("Welcome!", className="mb-4", style={"color": "white"}),
        html.H3("App Overview", className="mb-4", style={"color": "white"}),
        html.Div([
            html.H5("1) Data Preparation", className="mb-2",
                   style={"color": "white"}),
            html.P(
                "Connect to WRDS with your credentials to load financial data "
                "from 2004-2022. Perform feature engineering and create "
                "Lightning data modules for model training.",
                className="mb-3",
                style={"color": "white"},
            ),
            html.H5("2) Model Training", className="mb-2",
                   style={"color": "white"}),
            html.P(
                "Train machine learning models on your prepared data. Choose "
                "from various algorithms and configure parameters.",
                className="mb-3",
                style={"color": "white"},
            ),
            html.H5("3) Backtesting", className="mb-2",
                   style={"color": "white"}),
            html.P(
                "Test your trained models on historical data to evaluate "
                "performance and validate predictions.",
                className="mb-3",
                style={"color": "white"},
            ),
            html.H5("4) Prediction", className="mb-2",
                   style={"color": "white"}),
            html.P(
                "Generate future predictions using your trained and validated "
                "models.",
                className="mb-3",
                style={"color": "white"},
            ),
        ])
    ])
)

dash.register_page("Data Preparation", path="/data-preparation",
                    layout=html.Div([
                        html.H1("Data Preparation", style={"color": "white"}),
                        html.P("Prepare your data for machine learning models.",
                            style={"color": "white"})
                    ]))

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
                height: 100vw;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                background-color: #1a1a1a;
                color: white;
                overflow: hidden;
            }
            
            #react-entry-point {
                height: 100vw;
            }
            
            #main-container {
                display: flex;
                height: 100vh;
                width: 100vw;
                overflow-x: hidden;
            }
            
            #sidebar {
                width: 280px;
                background: linear-gradient(135deg, #76B900 0%, #1a1a1a 100%);
                padding: 30px 20px;
                box-shadow: 2px 0 10px rgba(0,0,0,0.3);
                height: 100vh;
                overflow-y: auto;
                border-right: 2px solid #2d2d2d;
                position: fixed;
                left: 0;
                top: 0;
                z-index: 1000;
            }
            
            #page-content {
                margin-left: 280px;
                padding: 40px;
                flex-grow: 1;
                background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
                min-height: 100vh;
                width: calc(100vh - 280px);
                word-wrap: break-word;
                overflow-wrap: break-word;
                box-sizing: border-box;
            }
            
            .nav-item {
                margin-bottom: 8px;
            }
            
            .sidebar-link {
                display: block;
                padding: 12px 20px;
                text-decoration: none;
                color: white;
                border-radius: 8px;
                transition: all 0.3s ease;
                border-left: 4px solid transparent;
            }
            
            .sidebar-link:hover {
                background-color: rgba(45, 90, 107, 0.3);
                color: white;
                text-decoration: none;
                border-left-color: #b90076;
                transform: translateX(5px);
            }
            
            .nav-link-content {
                font-weight: 500;
                font-size: 16px;
            }
            
            #home-link {
                background-color: rgba(45, 90, 107, 0.5);
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
                border: 1px solid #0076b9;
            }
            
            /* Scrollbar styling for sidebar */
            #sidebar::-webkit-scrollbar {
                width: 6px;
            }
            
            #sidebar::-webkit-scrollbar-track {
                background: rgba(255,255,255,0.1);
                border-radius: 3px;
            }
            
            #sidebar::-webkit-scrollbar-thumb {
                background: rgba(74, 158, 255, 0.5);
                border-radius: 3px;
            }
            
            #sidebar::-webkit-scrollbar-thumb:hover {
                background: rgba(74, 158, 255, 0.7);
            }
            
            /* Responsive design */
            @media (max-width: 768px) {
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
