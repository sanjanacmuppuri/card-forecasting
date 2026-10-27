import warnings
import numpy as np
import pandas as pd

from src.models import ets_forecast, lgbm_forecast
from src.backtest import horizon_errors, mape
from src.monitor import psi, needs_review

warnings.filterwarnings("ignore")

# Set these to match your Step 6 decision
CHOSEN = ets_forecast
CHOSEN_NAME = "champion_ets"      # or "challenger_lgbm"

df = pd.read_csv("data/processed/monthly.csv", index_col=0, parse_dates=True).asfreq("MS")

# ---- Part A: forecast range from past backtest errors ----
errs = horizon_errors(df, CHOSEN)
half = len(errs) // 2
lo, hi = np.quantile(errs[:half], [0.1, 0.9], axis=0)   # set the range on older windows
coverage = ((errs[half:] >= lo) & (errs[half:] <= hi)).mean()  # test it on newer windows
print(f"Backtest windows: {len(errs)} (older {half} set the range, newer {len(errs) - half} test it)")
print(f"Share of newer actuals inside the 80% range: {coverage:.0%}\n")

point = CHOSEN(df, 12)
idx = pd.date_range(df.index[-1], periods=13, freq="MS")[1:]
out = pd.DataFrame({"forecast": point, "lower_80": point * np.exp(lo),
                    "upper_80": point * np.exp(hi)}, index=idx).round(0)
out.to_csv("docs/forecast_with_range.csv")
print(out, "\n")

# ---- Part B: drift and error monitor ----
growth = np.log(df["REVOLSL"]).diff().dropna()
psi_value = psi(growth.iloc[:-12], growth.iloc[-12:])

train, test = df.iloc[:-12], df.iloc[-12:]
recent_mape = mape(test["REVOLSL"].to_numpy(), CHOSEN(train, 12))
bt = pd.read_csv("docs/backtest_results.csv")
backtest_mape = bt[bt["model"] == CHOSEN_NAME]["mape"].mean()

print(f"PSI (last 12 months of growth vs history): {psi_value:.3f}")
print(f"Recent 12-month MAPE: {recent_mape:.2f}%   Backtest average MAPE: {backtest_mape:.2f}%")
print("Needs review:", needs_review(recent_mape, backtest_mape, psi_value))
