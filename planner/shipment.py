def plan_shipments(df, truck_size=10000, safety_stock=5000, lead_time_map=None):
    if lead_time_map is None:
        lead_time_map = {"North": 1, "South": 2}
    df['trucks'] = (df['allocated'] // truck_size).astype(int)
    df['shipped'] = df['trucks'] * truck_size
    df['safety_met'] = df['shipped'] >= safety_stock
    df['arrival_week'] = df.apply(lambda row: row['week'] + lead_time_map.get(row['dc'], 1), axis=1)
    return df


