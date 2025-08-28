import pulp
import pandas as pd

def allocate_production(demand_df, max_capacity=150000):
    week_data = []
    for week in demand_df['week'].unique():
        week_df = demand_df[demand_df['week'] == week].copy()
        total_demand = week_df['demand'].sum()
        if total_demand <= max_capacity:
            week_df['allocated'] = week_df['demand']
        else:
            week_df['allocated'] = (week_df['demand'] / total_demand) * max_capacity
        week_df['allocated'] = week_df['allocated'].astype(int)
        week_data.append(week_df)
    return pd.concat(week_data, ignore_index=True)


def optimize_allocation(demand_df, max_capacity, costs, truck_size, safety_stock):
    prob = pulp.LpProblem("ProductionShipmentOptimization", pulp.LpMinimize)

    # Decision variables
    alloc_vars = {}
    slack_vars = {}

    for idx, row in demand_df.iterrows():
        key = (row['sku'], row['dc'], row['week'])
        alloc_vars[key] = pulp.LpVariable(f"alloc_{key}", lowBound=0, cat='Integer')
        slack_vars[key] = pulp.LpVariable(f"slack_{key}", lowBound=0, cat='Continuous')

    # Objective function: Minimize total cost = production + transport + inventory holding
    prob += pulp.lpSum([
        costs['production'] * alloc_vars[key] +
        costs['transport'] * (alloc_vars[key] / truck_size) +
        costs['inventory'] * slack_vars[key]
        for key in alloc_vars.keys()
    ])

    # Constraints
    # 1. Production capacity per week
    for wk in demand_df['week'].unique():
        prob += pulp.lpSum(
            [alloc_vars[(row['sku'], row['dc'], row['week'])] for idx, row in demand_df[demand_df['week'] == wk].iterrows()]
        ) <= max_capacity, f"Capacity_Week_{wk}"

    # 2. Allocation cannot exceed demand
    for idx, row in demand_df.iterrows():
        key = (row['sku'], row['dc'], row['week'])
        prob += alloc_vars[key] <= row['demand'], f"DemandConstr_{key}"
        # slack >= safety_stock - allocation (to model max(safety_stock-alloc,0))
        prob += slack_vars[key] >= safety_stock - alloc_vars[key], f"SlackConstr1_{key}"
        prob += slack_vars[key] >= 0, f"SlackConstr2_{key}"

    # Solve the optimization problem
    prob.solve()

    # Assign optimized allocation back to DataFrame
    for idx, row in demand_df.iterrows():
        key = (row['sku'], row['dc'], row['week'])
        val = pulp.value(alloc_vars[key])
        demand_df.at[idx, 'allocated'] = int(val) if val is not None else 0

    return demand_df


