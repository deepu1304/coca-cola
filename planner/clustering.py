from sklearn.cluster import KMeans
import pandas as pd

def cluster_skus(df, n_clusters=4):
    # Aggregate features by SKU
    sku_features = df.groupby('sku')[['demand', 'allocated']].mean().reset_index()
    n_samples = sku_features.shape[0]

    # Adjust cluster count based on sample size
    if n_samples < n_clusters:
        n_clusters = max(1, n_samples)

    if n_samples == 0:
        sku_features['cluster'] = []
        return sku_features

    # Initialize and fit KMeans
    kmeans = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init=50,
        max_iter=500,
        random_state=42
    )
    sku_features['cluster'] = kmeans.fit_predict(sku_features[['demand', 'allocated']])
    return sku_features

