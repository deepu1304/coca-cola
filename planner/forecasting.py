import pandas as pd

def forecast_demand(history_df, year=2025, periods=8):
    history_df = history_df.copy()
    history_df['week_str'] = history_df['week'].apply(lambda x: f"{int(x):02d}")
    history_df['ds'] = pd.to_datetime(str(year) + '-' + history_df['week_str'] + '-1', format='%G-%V-%u')
    history_df = history_df.rename(columns={'demand': 'y'})

    from prophet import Prophet
    model = Prophet()
    model.fit(history_df[['ds', 'y']])

    future = model.make_future_dataframe(periods=periods, freq='W-MON')
    forecast = model.predict(future)
    forecast_df = forecast[['ds', 'yhat']].tail(periods).copy()
    forecast_df['week'] = forecast_df['ds'].dt.isocalendar().week
    forecast_df = forecast_df.rename(columns={'yhat': 'demand_forecast'})[['week', 'demand_forecast']]

    return forecast_df

