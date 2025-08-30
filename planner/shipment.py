import pandas as pd
import numpy as np

def enhanced_truck_planning(df, truck_size=5000, strategy='partial', partial_threshold=0.6, safety_stock=5000):
    if df.empty:
        return df
    
    df = df.copy()
    
    if 'allocated' not in df.columns:
        df['allocated'] = df.get('demand', 0)
    
    df['allocated'] = pd.to_numeric(df['allocated'], errors='coerce').fillna(0)
    
    if strategy == 'partial_trucks' or strategy == 'partial':
        df['full_trucks'] = (df['allocated'] // truck_size).astype(int)
        df['remaining_units'] = df['allocated'] % truck_size
        df['use_partial'] = df['remaining_units'] >= (truck_size * partial_threshold)
        df['partial_trucks'] = df['use_partial'].astype(int)
        df['total_trucks'] = df['full_trucks'] + df['partial_trucks']
        
        df['shipped'] = (df['full_trucks'] * truck_size + 
                        df['use_partial'] * df['remaining_units']).astype(int)
        
        df['truck_utilization'] = np.where(
            df['total_trucks'] > 0,
            (df['shipped'] / (df['total_trucks'] * truck_size) * 100).round(2),
            0
        )
    else:
        df['total_trucks'] = (df['allocated'] // truck_size).astype(int)
        df['shipped'] = df['total_trucks'] * truck_size
        df['truck_utilization'] = np.where(df['total_trucks'] > 0, 100.0, 0.0)
    
    df['unshipped'] = df['allocated'] - df['shipped']
    df['safety_met'] = df['shipped'] >= safety_stock
    
    return df




