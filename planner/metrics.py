def calculate_metrics(fulldata):
    metrics = {}
    metrics['total_demand'] = int(fulldata['demand'].sum())
    metrics['total_shipped'] = int(fulldata['shipped'].sum())
    metrics['service_level'] = 100 * metrics['total_shipped'] / max(metrics['total_demand'], 1)
    metrics['all_safety_met'] = fulldata['safety_met'].all()
    truck_count = fulldata['trucks'].sum()
    metrics['truck_utilization'] = 100 * metrics['total_shipped'] / max((truck_count * 10000), 1) if truck_count > 0 else 0
    return metrics

def highlight_violations(df):
    def highlight(row):
        if not row['safety_met']:
            return ['background-color: #fce4e4'] * len(row)
        if row['allocated'] < row['demand']:
            return ['background-color: #fff4c2'] * len(row)
        return [''] * len(row)
    return df.style.apply(highlight, axis=1)

