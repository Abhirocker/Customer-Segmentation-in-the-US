import plotly.express as px
import data_wrangle
from dash import Input, Output, dcc, html

from business_layer import get_high_var_features
from business_layer import get_model_metrics
from business_layer import get_pca_labels

# Service Layer
## Application Layout
def register_callbacks(app):
    
    @app.callback(
        Output("bar-chart", "figure"), Input("trim-button", "value")
    )
    def serve_bar_chart(trimmed=True):

        """Returns a horizontal bar chart of five highest-variance features.

        Parameters
        ----------
        trimmed : bool, default=True
            If ``True``, calculates trimmed variance, removing bottom and top 10%
            of observations.
        """
        # Get features
        top_five_features = get_high_var_features(trimmed=trimmed, return_feat_names=False)

        # Build bar chart
        fig = px.bar(x=top_five_features, y=top_five_features.index, orientation="h")
        fig.update_layout(xaxis_title="Variance", yaxis_title="Features")

        return fig

    serve_bar_chart(trimmed=False)

    # K-means Slider and Metrics
    @app.callback(
        Output("metrics", "children"),
        Input("trim-button", "value"),
        Input("k-slider", "value")
    )
    def serve_metrics(trimmed=True, k=2):

        """Returns list of ``H3`` elements containing inertia and silhouette score
        for ``KMeans`` model.

        Parameters
        ----------
        trimmed : bool, default=True
            If ``True``, calculates trimmed variance, removing bottom and top 10%
            of observations.

        k : int, default=2
            Number of clusters.
        """
        # Get metrics
        metrics = get_model_metrics(trimmed=trimmed, k=k, return_metrics=True)

        # Add metrics to HTML elements
        text = [
            html.H3(f"Inertia: {metrics['inertia']}"),
            html.H3(f"Silhouette Score: {metrics['silhouette_score']}")
        ]

        return text

    serve_metrics(k=20)
    #Output [H3('Inertia: 1478'), H3('Silhouette Score: 0.495')]

    # PCA Scatter Plot
    @app.callback(
        Output("pca-scatter", "figure"),
        Input("trim-button", "value"),
        Input("k-slider", "value")
    )
    def serve_scatter_plot(trimmed=True, k=2):

        """Build 2D scatter plot of ``df`` with ``KMeans`` labels.

        Parameters
        ----------
        trimmed : bool, default=True
            If ``True``, calculates trimmed variance, removing bottom and top 10%
            of observations.

        k : int, default=2
            Number of clusters.
        """
        fig = px.scatter(
            data_frame = get_pca_labels(trimmed=trimmed, k=k),
            x="PC1",
            y="PC2",
            color="labels",
            title="PCA representation of Clusters"
        )
        fig.update_layout(xaxis_title="PC1", yaxis_title="PC2")

        return fig

    serve_scatter_plot(trimmed=False, k=5)