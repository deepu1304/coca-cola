from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd

def cluster_skus(df, n_clusters=4):
    if df.empty or 'sku' not in df.columns:
        return pd.DataFrame()
        
    sku_features = df.groupby('sku').agg({
        'demand': ['mean', 'std', 'sum'],
        'allocated': ['mean', 'sum']
    }).reset_index()
    
    sku_features.columns = ['sku', 'avg_demand', 'demand_volatility', 'total_demand', 'avg_allocated', 'total_allocated']
    sku_features['demand_volatility'] = sku_features['demand_volatility'].fillna(0)
    
    n_samples = sku_features.shape[0]
    
    if n_samples == 0:
        return pd.DataFrame(columns=['sku', 'avg_demand', 'total_demand', 'cluster'])
    
    if n_samples == 1:
        sku_features['cluster'] = 0
        return sku_features
    
    n_clusters = min(n_clusters, n_samples)
    
    try:
        features_for_clustering = ['total_demand', 'demand_volatility']
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(sku_features[features_for_clustering])
        
        kmeans = KMeans(n_clusters=n_clusters, init="k-means++", n_init=10, max_iter=300, random_state=42)
        sku_features['cluster'] = kmeans.fit_predict(scaled_features)
        
    except Exception as e:
        sku_features['cluster'] = pd.cut(sku_features['total_demand'], bins=n_clusters, labels=False)
    
    return sku_features



