import pandas as pd

from src.models import naive_forecast, ets_forecast, lgbm_forecast
from src.backtest import rmse, mape

df = pd.read_csv("data/processed/monthly.csv", index_col=0, parse_dates=True).asfreq("MS")
train, test = df.iloc[:-12], df.iloc[-12:]
actual = test["REVOLSL"].to_numpy()

for name, fn in [("naive", naive_forecast), ("champion_ets", ets_forecast), ("challenger_lgbm", lgbm_forecast)]:
    pred = fn(train, 12)
    print(f"{name:16s} RMSE = {rmse(actual, pred):>10,.0f}   MAPE = {mape(actual, pred):.2f}%")
