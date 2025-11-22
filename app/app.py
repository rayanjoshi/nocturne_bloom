"""Dash app entrypoint for Nocturne Bloom.

Creates the Dash application, registers pages, sets external stylesheets,
composes the layout with the sidebar and page container, and exposes `server`
for deployment.
"""

from dash import Dash, html
import dash_bootstrap_components as dbc
import dash

# Use absolute import so the module can be executed as a script (run.py)
from app.layout.sidebar import sidebar

app = Dash(
    __name__,
    use_pages=True,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        dbc.icons.FONT_AWESOME,
        "/assets/css/custom.css",  # ← now loaded as real file
    ],
    suppress_callback_exceptions=True,
    pages_folder="pages",
)

app.title = "Nocturne Bloom — NVDA Prediction Dashboard"
server = app.server
app.title = "ML Stock Prediction"

server = app.server

# Main content area that renders the registered pages
content = html.Div(id="page-content", children=[dash.page_container])

# Compose the app layout with the sidebar and the page container
app.layout = html.Div([sidebar, content], id="main-container")

if __name__ == "__main__":
    app.run(debug=False)
