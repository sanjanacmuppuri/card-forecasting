import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.models import naive_forecast, ets_forecast
from src.backtest import rmse, mape

os.makedirs("docs/figures", exist_ok=True)

df = pd.read_csv("data/processed/monthly.csv", index_col=0, parse_dates=True).asfreq("MS")

# Hide the last 12 months, forecast them, then compare to what happened
train, test = df.iloc[:-12], df.iloc[-12:]
actual = test["REVOLSL"].to_numpy()

preds = {
    "naive": naive_forecast(train, 12),
    "champion_ets": ets_forecast(train, 12),
}

for name, pred in preds.items():
    print(f"{name:14s} RMSE = {rmse(actual, pred):>10,.0f}   MAPE = {mape(actual, pred):.2f}%")

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(df.index[-60:], df["REVOLSL"].iloc[-60:], label="actual")
for name, pred in preds.items():
    ax.plot(test.index, pred, linestyle="--", label=name)
ax.axvline(test.index[0], color="gray", linewidth=0.8)
ax.set_title("Last 12 months: forecasts vs actual")
ax.legend()
fig.tight_layout()
fig.savefig("docs/figures/04_champion_check.png", dpi=150)
