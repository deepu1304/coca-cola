from sklearn.ensemble import IsolationForest

def detect_anomalies(df):
    clf = IsolationForest(contamination=0.05)
    preds = clf.fit_predict(df[['demand', 'allocated']])
    df['anomaly'] = preds == -1
    return df
