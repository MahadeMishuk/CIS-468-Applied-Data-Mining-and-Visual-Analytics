# src/unsupervised.py

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage


def run_kmeans(df_features: pd.DataFrame, n_clusters=3, random_state=42):
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    labels = km.fit_predict(df_features)
    sil = silhouette_score(df_features, labels)
    return km, labels, sil


def run_agglomerative(df_features: pd.DataFrame, n_clusters=3):
    agg = AgglomerativeClustering(n_clusters=n_clusters)
    labels = agg.fit_predict(df_features)
    return agg, labels


def compute_pca(df_features: pd.DataFrame, n_components=2):
    pca = PCA(n_components=n_components)
    vals = pca.fit_transform(df_features)
    return pca, vals


def sample_linkage(df_features: pd.DataFrame, max_samples=100, method="ward"):
    """
    Compute linkage for a sample of the data for dendrogram plotting.
    """
    n = len(df_features)
    sample_idx = np.random.choice(n, size=min(max_samples, n), replace=False)
    linked = linkage(df_features.iloc[sample_idx], method=method)
    return linked


def compare_macro_distributions(daily, food, legacy):
    """
    Simple helper: return summary statistics for macro columns
    so we can compare across datasets in the notebook.
    """
    summary = {}
    for name, df in [("Daily Log", daily), ("Food Composition", food), ("Legacy", legacy)]:
        cols = [c for c in df.columns if c in ["Calories", "Protein (g)", "Carbohydrates (g)", "Fat (g)", "Fiber (g)"]]
        summary[name] = df[cols].describe()
    return summary
