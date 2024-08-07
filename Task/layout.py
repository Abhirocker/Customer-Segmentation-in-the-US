from dash import Input, Output, dcc, html
import libraries
import data_wrangle

# Presentation Layer
## Application Layout
layout = html.Div(
    [
        # Application title
        html.H1("Survey of Consumer Finances"),
        # Bar chart element
        html.H2("High Variance Features"),
        # Bar chart graph
        dcc.Graph(id="bar-chart"), #dcc(dash core component)
        dcc.RadioItems(
            options = [
                {"label": "Trimmed", "value": True},
                {"label": "Not Trimmed", "value": False}
            ],
            value = True,
            id = "trim-button"
        ),
        html.H2("K-means Clustering"),
        html.H3("Number of Clusters (k)"),
        dcc.Slider(min=2, max=12, step=1, value=2, id="k-slider"),
        html.Div(id="metrics"),
        # PCA Scatter plot
        dcc.Graph(id="pca-scatter")
    ]
)