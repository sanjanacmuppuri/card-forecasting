import os
import warnings
import pandas as pd

from src.models import naive_forecast, ets_forecast, lgbm_forecast
from src.backtest import rolling_backtest

warnings.filterwarnings("ignore")
os.makedirs("docs", exist_ok=True)

df = pd.read_csv("data/processed/monthly.csv", index_col=0, parse_dates=True).asfreq("MS")

models = {
    "naive": naive_forecast,
    "champion_ets": ets_forecast,
    "challenger_lgbm": lgbm_forecast,
}
res = rolling_backtest(df, models)
res.to_csv("docs/backtest_results.csv", index=False)

print(f"Windows: {res['origin'].nunique()}  "
      f"(first origin {res['origin'].min().date()}, last {res['origin'].max().date()})\n")

print("Average scorecard (all windows):")
print(res.groupby("model")[["rmse", "mape"]].mean().round(2), "\n")

by_window = res.pivot(index="origin", columns="model", values="mape")
share = (by_window["challenger_lgbm"] < by_window["champion_ets"]).mean()
print(f"Challenger beats champion in {share:.0%} of windows\n")

stress_mask = (
    res["origin"].between(pd.Timestamp("2007-01-01"), pd.Timestamp("2010-12-01"))
    | res["origin"].between(pd.Timestamp("2019-06-01"), pd.Timestamp("2021-06-01"))
)
print("Stress periods only:")
print(res[stress_mask].groupby("model")[["rmse", "mape"]].mean().round(2))
