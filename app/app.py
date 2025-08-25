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

dash.register_page("About", path="/", layout=html.Div([html.H1("About")]))
dash.register_page("Data Preparation", path="/data-preparation",
                    layout=html.Div([html.H1("Data Preparation")]))
dash.register_page("Training", path="/training", layout=html.Div([html.H1("Training")]))
dash.register_page("Backtesting", path="/backtesting", layout=html.Div([html.H1("Backtesting")]))

sidebar = html.Div(
    [html.H2("Navigation", className="display-4")] +
    [html.Div(dcc.Link(f"{page['name']}", href=page["relative_path"]))
    for page in dash.page_registry.values()]
)

app.layout = html.Div([
    sidebar,
    dash.page_container,
])

if __name__ == "__main__":
    app.run(debug=True)
