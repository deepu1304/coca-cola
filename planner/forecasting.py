import pandas as pd

def forecast_demand(history_df, year=2025, periods=6):
    if history_df.empty:
        return pd.DataFrame()
        
    forecast_results = []
    
    for (sku, dc), group in history_df.groupby(['sku', 'dc']):
        try:
            avg_demand = max(1000, group['demand'].mean())
            last_week = group['week'].max()
            
            for i in range(periods):
                week = last_week + i + 1
                demand = max(500, int(avg_demand * (0.9 + 0.2 * (i % 3))))
                forecast_results.append(pd.DataFrame({
                    'sku': [sku], 'dc': [dc], 'week': [week], 'demand': [demand]
                }))
                
        except Exception as e:
            continue
    
    return pd.concat(forecast_results, ignore_index=True) if forecast_results else pd.DataFrame()



