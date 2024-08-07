from jupyter_dash import JupyterDash
from scipy.stats.mstats import trimmed_var
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd
import data_wrangle
from data_wrangle import df

# Business Layer
## Application Layout
def get_high_var_features(trimmed=True, return_feat_names=True):

    """Returns the five highest-variance features of ``df``.

    Parameters
    ----------
    trimmed : bool, default=True
        If ``True``, calculates trimmed variance, removing bottom and top 10%
        of observations.

    return_feat_names : bool, default=False
        If ``True``, returns feature names as a ``list``. If ``False``
        returns ``Series``, where index is feature names and values are
        variances.
    """
    # Calculate variance
    if trimmed:
        top_five_features = df.apply(trimmed_var).sort_values().tail(5)
    else:
        top_five_features = df.var().sort_values().tail(5)
    
    # Extract names
    if return_feat_names:
        top_five_features = top_five_features.index.to_list()
    
    return top_five_features

get_high_var_features()
#Output ['DEBT', 'NETWORTH', 'HOUSES', 'NFIN', 'ASSET']

# K-means Slider and Metrics
def get_model_metrics(trimmed=True, k=2, return_metrics=False):

    """Build ``KMeans`` model based on five highest-variance features in ``df``.

    Parameters
    ----------
    trimmed : bool, default=True
        If ``True``, calculates trimmed variance, removing bottom and top 10%
        of observations.

    k : int, default=2
        Number of clusters.

    return_metrics : bool, default=False
        If ``False`` returns ``KMeans`` model. If ``True`` returns ``dict``
        with inertia and silhouette score.

    """
    # Get high var feqtures
    features = get_high_var_features(trimmed=trimmed, return_feat_names=True)
    # Create feature metrics
    X = df[features]
    # Build model
    model = make_pipeline(StandardScaler(), KMeans(n_clusters=k, random_state=42))
    model.fit(X)
    
    if return_metrics:
        # Calculate inertia
        i = model.named_steps["kmeans"].inertia_
        # Claculate silhouette score
        ss = silhouette_score(X, model.named_steps["kmeans"].labels_)
        # Put results into dictionary
        metrics = {
            "inertia": round(i),
            "silhouette_score": round(ss, 3)
        }
        # Return dictionary to user
        return metrics
    
    return model

get_model_metrics(trimmed=True, k=20, return_metrics=False)
# Output Pipeline(steps=[('standardscaler', StandardScaler()),('kmeans', KMeans(n_clusters=20, random_state=42))])

# PCA Scatter Plot
def get_pca_labels(trimmed=True, k=2):

    """
    ``KMeans`` labels.

    Parameters
    ----------
    trimmed : bool, default=True
        If ``True``, calculates trimmed variance, removing bottom and top 10%
        of observations.

    k : int, default=2
        Number of clusters.
    """
    # Create feature metrics
    features = get_high_var_features(trimmed=trimmed, return_feat_names=True)
    X = df[features]
    
    # Build transformer
    transformer = PCA(n_components=2, random_state=42)
    
    # Transform data
    X_t = transformer.fit_transform(X)
    X_pca =pd.DataFrame(X_t, columns=["PC1", "PC2"])
    
    # Add labels
    model = get_model_metrics(trimmed=trimmed, k=k, return_metrics=False)
    X_pca["labels"] = model.named_steps["kmeans"].labels_.astype(str)
    X_pca.sort_values("labels", inplace=True)
    
    return X_pca

get_pca_labels().tail()

#   	PC1          	PC2         	labels
# 1570	-229796.419844	-14301.836873	1
# 1571	-229805.583716	-14250.840322	1
# 1572	-229814.747589	-14199.843771	1
# 1611	-213724.571420	-39060.460885	1
# 4417	334191.956229	-186450.064242	1