import pandas as pd

def calculate_metrics(fulldata):
    if fulldata.empty:
        return {
            'total_demand': 0,
            'total_shipped': 0,
            'service_level': 0,
            'truck_utilization': 0,
            'all_safety_met': False
        }
    
    metrics = {}
    metrics['total_demand'] = int(fulldata['demand'].sum())
    metrics['total_shipped'] = int(fulldata.get('shipped', fulldata.get('allocated', 0)).sum())
    metrics['total_allocated'] = int(fulldata.get('allocated', fulldata.get('demand', 0)).sum())
    
    if metrics['total_demand'] > 0:
        metrics['service_level'] = 100 * metrics['total_shipped'] / metrics['total_demand']
        metrics['allocation_efficiency'] = 100 * metrics['total_allocated'] / metrics['total_demand']
    else:
        metrics['service_level'] = 0
        metrics['allocation_efficiency'] = 0
    
    metrics['all_safety_met'] = fulldata.get('safety_met', pd.Series([True])).all()
    
    if 'truck_utilization' in fulldata.columns and not fulldata['truck_utilization'].isna().all():
        metrics['truck_utilization'] = fulldata['truck_utilization'].mean()
    else:
        metrics['truck_utilization'] = 0
    
    return metrics

def highlight_violations(df):
    def highlight(row):
        if not row.get('safety_met', True):
            return ['background-color: #fce4e4'] * len(row)
        if row.get('allocated', 0) < row.get('demand', 0):
            return ['background-color: #fff4c2'] * len(row)
        if row.get('truck_utilization', 100) < 70:
            return ['background-color: #e6f3ff'] * len(row)
        return [''] * len(row)
    return df.style.apply(highlight, axis=1)



