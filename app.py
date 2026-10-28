import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from src.models import ets_forecast, lgbm_forecast

st.title("Revolving credit forecast")


@st.cache_data
def load() -> pd.DataFrame:
    # use the cleaned file, so the October 2025 fill is kept
    return pd.read_csv("data/processed/monthly.csv", index_col=0, parse_dates=True).asfreq("MS")


@st.cache_data
def forecasts(h: int):
    df = load()
    return ets_forecast(df, h), lgbm_forecast(df, h)


df = load()
h = st.slider("Months ahead", 1, 12, 6)
champ_vals, chall_vals = forecasts(h)

last_date, last_val = df.index[-1], df["REVOLSL"].iloc[-1]
idx = pd.date_range(last_date, periods=h + 1, freq="MS")  # starts at last actual, so lines connect
champ = pd.Series([last_val, *champ_vals], index=idx)
chall = pd.Series([last_val, *chall_vals], index=idx)

hist = df["REVOLSL"].iloc[-60:]
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(hist.index, hist.values, label="actual")
ax.plot(champ.index, champ.values, "--", label="champion (ETS), model of record")
ax.plot(chall.index, chall.values, ":", label="challenger (LightGBM), not adopted")
low = min(hist.min(), champ.min(), chall.min())
high = max(hist.max(), champ.max(), chall.max())
ax.set_ylim(low * 0.97, high * 1.03)
ax.set_ylabel("$ millions")
ax.legend()
fig.tight_layout()
st.pyplot(fig)

st.caption(
    "Champion kept after a rolling backtest, see docs/model_card.md. "
    "Personal project using public Federal Reserve data via FRED. Not investment advice."
)