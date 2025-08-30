from sklearn.ensemble import IsolationForest
import pandas as pd

def detect_anomalies(df):
    if len(df) < 10:
        df['anomaly'] = False
        return df
    
    try:
        features = []
        if 'demand' in df.columns and df['demand'].sum() > 0:
            features.append('demand')
        if 'allocated' in df.columns and df['allocated'].sum() > 0:
            features.append('allocated')
        if 'shipped' in df.columns and df['shipped'].sum() > 0:
            features.append('shipped')
        
        if len(features) < 2:
            df['anomaly'] = False
            return df
            
        contamination = min(0.05, max(0.01, 3/len(df)))
        
        clf = IsolationForest(contamination=contamination, random_state=42)
        preds = clf.fit_predict(df[features])
        df['anomaly'] = preds == -1
        
    except Exception as e:
        df['anomaly'] = False
    
    return df


