import pandas as pd

def get_weekly_capacity(week, base_capacity=150000):
    if week >= 4:
        return int(base_capacity * 0.85)
    return base_capacity

def allocate_production(demand_df, max_capacity=150000):
    if demand_df.empty:
        return demand_df
    
    demand_df = demand_df.copy()
    week_data = []
    
    for week in sorted(demand_df['week'].unique()):
        week_df = demand_df[demand_df['week'] == week].copy()
        total_demand = week_df['demand'].sum()
        
        weekly_capacity = get_weekly_capacity(week, max_capacity)
        
        if total_demand <= weekly_capacity:
            week_df['allocated'] = week_df['demand']
        elif total_demand > 0:
            week_df['allocated'] = (week_df['demand'] / total_demand) * weekly_capacity
        else:
            week_df['allocated'] = 0
            
        week_df['allocated'] = week_df['allocated'].astype(int)
        week_data.append(week_df)
    
    return pd.concat(week_data, ignore_index=True)




