# Importing Libraries

import pandas as pd
import plotly.express as px
from dash import Input, Output, dcc, html

from jupyter_dash import JupyterDash
from scipy.stats.mstats import trimmed_var
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

JupyterDash.infer_jupyter_proxy_config()














