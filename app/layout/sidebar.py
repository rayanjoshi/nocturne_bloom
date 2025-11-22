from dash import html, dcc

sidebar = html.Div(
    [
        html.Div(
            [
                html.I(className="fas fa-chart-line fa-2x mb-2"),
                html.Hr(className="sidebar_Hr"),
                html.H3("ML Stock", className="mb-0"),
                html.H3("Prediction", className="mb-4"),
            ],
            className="text-center mb-4",
            style={"color": "white"},
        ),
        html.Hr(className="sidebar_Hr"),
        html.Div(
            [
                html.Div(
                    [
                        dcc.Link(
                            [
                                html.Div(
                                    [html.Span("Home")], className="nav-link-content"
                                )
                            ],
                            href="/",
                            className="sidebar-link",
                            id="home-link",
                        )
                    ],
                    className="nav-item",
                ),
                html.Div(
                    [
                        dcc.Link(
                            [
                                html.Div(
                                    [html.Span("1. Data Preparation")],
                                    className="nav-link-content",
                                )
                            ],
                            href="/data-preparation",
                            className="sidebar-link",
                        )
                    ],
                    className="nav-item",
                ),
                html.Div(
                    [
                        dcc.Link(
                            [
                                html.Div(
                                    [html.Span("2. Training")],
                                    className="nav-link-content",
                                )
                            ],
                            href="/training",
                            className="sidebar-link",
                        )
                    ],
                    className="nav-item",
                ),
                html.Div(
                    [
                        dcc.Link(
                            [
                                html.Div(
                                    [html.Span("3. Backtesting")],
                                    className="nav-link-content",
                                )
                            ],
                            href="/backtesting",
                            className="sidebar-link",
                        )
                    ],
                    className="nav-item",
                ),
            ]
        ),
    ],
    id="sidebar",
)
